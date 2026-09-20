"""Pendências resolvidas no motor: peso manual por coluna, score da fila com pesos, datas do cliente, análise
exploratória e cancelamento cooperativo do treino.

Testes leves: só a base INOVAAPPS (`dados/bases/inovaapps/arquivos/`) e `validar=False` (cada treino leva ~3 s).
"""

from __future__ import annotations

import copy
import json
import threading
from pathlib import Path

import pytest

from motor import Config, Mapeamento, TreinoCancelado, ler_arquivos, treinar
from motor.exploratoria import analisar
from motor.painel import preparar
from motor.score import formula, pontuar, reordenar, validar_pesos

RAIZ = Path(__file__).resolve().parents[1]
XLSX = RAIZ / "dados" / "bases" / "inovaapps" / "arquivos" / "INOVAAPPS_base_de_dados.xlsx"
PESOS = {"risco": 0.6, "valor": 0.3, "delta": 0.1}


@pytest.fixture(scope="module")
def mapeamento():
    d = json.loads((RAIZ / "dados" / "bases" / "inovaapps" / "mapeamento.json").read_text(encoding="utf-8"))
    d["colunas_ignoradas"] = []
    return Mapeamento.from_dict(d)


@pytest.fixture(scope="module")
def tabelas():
    return ler_arquivos([XLSX])


@pytest.fixture(scope="module")
def padrao(tabelas, mapeamento):
    return treinar(tabelas, mapeamento, config=Config(validar=False), gerado_em="2026-01-01")


def _sel(r):
    return {v["id"]: v for v in r.pesos["variaveis"] if v["selecionada"]}


# --------------------------------------------------------------------------- peso manual
def test_peso_zero_tira_a_coluna(tabelas, mapeamento, padrao):
    assert any(v["coluna"] == "nota_nps" for v in _sel(padrao).values())     # sem peso, o NPS entra
    r = treinar(tabelas, mapeamento, config=Config(validar=False, pesos_colunas={"pesquisas_nps.nota_nps": 0}),
                gerado_em="2026-01-01")
    nps = [v for v in r.pesos["variaveis"] if v["coluna"] == "nota_nps"]
    assert nps and not any(v["selecionada"] for v in nps)
    assert {v["motivo_descarte"] for v in nps} == {"peso 0 definido pelo usuário"}
    assert all(v["peso_usuario"] == 0 for v in nps)
    assert r.pesos["regras"]["pesos_manuais"]


def test_peso_baixo_encarece_a_variavel(tabelas, mapeamento, padrao):
    top = max(_sel(padrao).values(), key=lambda v: v["frequencia_selecao"] or 0)
    r = treinar(tabelas, mapeamento, gerado_em="2026-01-01",
                config=Config(validar=False, pesos_colunas={f"{top['tabela']}.{top['coluna']}": 0.15}))
    depois = {v["id"]: v for v in r.pesos["variaveis"]}[top["id"]]
    assert depois["frequencia_selecao"] < top["frequencia_selecao"]
    assert depois["peso_usuario"] == 0.15


@pytest.mark.parametrize("ruim", [{"x": 2}, {"x": -0.1}, {"x": "a"}])
def test_peso_invalido_falha_antes_do_treino(tabelas, mapeamento, ruim):
    with pytest.raises(ValueError, match="peso de 'x'"):
        treinar(tabelas, mapeamento, config=Config(validar=False, pesos_colunas=ruim))


# --------------------------------------------------------------------------- score
def test_score_ordena_a_fila(tabelas, mapeamento):
    r = treinar(tabelas, mapeamento, config=Config(validar=False, pesos_score=PESOS), gerado_em="2026-01-01")
    cl = r.clientes["clientes"]
    sc = [c["score"] for c in cl]
    assert sc == sorted(sc, reverse=True) and all(0 <= s <= 100 for s in sc)
    assert [c["prioridade"] for c in cl] == list(range(1, len(cl) + 1))
    assert r.clientes["pesos_score"] == PESOS and "score" in r.clientes["modelo"]["formula_prioridade"]
    # conferido à mão nas 3 primeiras linhas
    vmax = max(c["valor_mensal"] for c in cl)
    ds = [c["delta_risco"] or 0 for c in cl]
    lo, hi = min(min(ds), 0), max(max(ds), 0)
    for c in cl[:3]:
        esperado = 100 * (0.6 * c["risco"] + 0.3 * c["valor_mensal"] / vmax + 0.1 * ((c["delta_risco"] or 0) - lo) / (hi - lo))
        assert c["score"] == pytest.approx(esperado, abs=0.06)


def test_sem_pesos_a_ordem_e_a_de_sempre(padrao):
    cl = padrao.clientes["clientes"]
    assert all(c["score"] is None for c in cl) and padrao.clientes["pesos_score"] is None
    perdas = [c["perda_anual_ajustada"] for c in cl]
    assert perdas == sorted(perdas, reverse=True)


def test_reordenar_refaz_sem_retreinar(tabelas, mapeamento, padrao):
    com = treinar(tabelas, mapeamento, config=Config(validar=False, pesos_score=PESOS), gerado_em="2026-01-01")
    doc = reordenar(copy.deepcopy(padrao.clientes), PESOS)
    assert [c["cliente_id"] for c in doc["clientes"]] == [c["cliente_id"] for c in com.clientes["clientes"]]
    assert doc["modelo"]["formula_prioridade"] == com.clientes["modelo"]["formula_prioridade"]
    volta = reordenar(doc, None)                                   # sem pesos: volta ao padrão do treino
    assert [c["cliente_id"] for c in volta["clientes"]] == [c["cliente_id"] for c in padrao.clientes["clientes"]]
    assert volta["clientes"][0]["score"] is None and volta["pesos_score"] is None


def test_pesos_do_score_invalidos():
    for ruim in ({"risco": -1}, {"risco": 0, "valor": 0, "delta": 0}, {"foo": 1}, {"risco": "a"}):
        with pytest.raises(ValueError):
            validar_pesos(ruim)
    assert validar_pesos({}) is None and validar_pesos(None) is None
    assert validar_pesos({"risco": 1}) == {"risco": 1.0, "valor": 0.0, "delta": 0.0}
    assert "60% risco" in formula(PESOS, True) and "valor" not in formula(PESOS, False)
    # sem coluna de valor, o peso do valor é ignorado: só risco e delta contam
    s = pontuar([0.5, 0.2], None, [0.0, 0.0], {"risco": 1, "valor": 5, "delta": 0})
    assert list(s) == pytest.approx([50.0, 20.0])


# --------------------------------------------------------------------------- datas do cliente
def test_datas_do_cliente_chegam_na_fila(padrao):
    assert padrao.clientes["modelo"]["colunas_data"] == [{"coluna": "inicio_contrato", "rotulo": "Início contrato"}]
    datas = [c["datas"]["inicio_contrato"] for c in padrao.clientes["clientes"]]
    assert all(d and len(d) == 7 and d[4] == "-" for d in datas)      # AAAA-MM
    assert all("inicio_contrato" not in c["atributos"] for c in padrao.clientes["clientes"])


# --------------------------------------------------------------------------- análise exploratória
def test_analise_exploratoria_bate_com_o_notebook(tabelas, mapeamento):
    a = analisar(preparar(tabelas, mapeamento, Config()))
    assert (a["cancelados"], a["ativos"], a["referencia"]) == (22, 58, "2026-06")
    por = {s["coluna"]: s for s in a["series"]}
    assert por["uso_plataforma_pct"]["antecedencia"] == 3 and por["tempo_medio_resolucao_h"]["antecedencia"] == 3
    assert por["pct_sla_cumprido"]["antecedencia"] == 2 and por["chamados_criticos"]["antecedencia"] == 2
    assert por["chamados_abertos"]["antecedencia"] == 1 and por["chamados_reabertos"]["antecedencia"] is None
    assert por["uso_plataforma_pct"]["rotulo"] == "Uso plataforma (%)"
    aucs = [abs(s["auc"] - 0.5) for s in a["series"]]
    assert aucs == sorted(aucs, reverse=True)                        # mais separa, primeiro
    pts = por["uso_plataforma_pct"]["pontos"]
    assert [p["rel"] for p in pts] == list(range(-12, 0))
    assert pts[-1]["c"]["n"] == 22 and pts[-1]["a"]["n"] == 58
    assert pts[-1]["c"]["p25"] <= pts[-1]["c"]["med"] <= pts[-1]["c"]["p75"]
    json.dumps(a, allow_nan=False)


def test_analise_respeita_colunas_ignoradas(tabelas, mapeamento):
    d = mapeamento.to_dict()
    d["colunas_ignoradas"] = ["atendimento_mensal.uso_plataforma_pct"]
    a = analisar(preparar(tabelas, Mapeamento.from_dict(d), Config()))
    assert "uso_plataforma_pct" not in {s["coluna"] for s in a["series"]}


# --------------------------------------------------------------------------- cancelar
def test_cancelar_interrompe_o_treino(tabelas, mapeamento):
    parar = threading.Event()

    def progresso(etapa, fracao):
        if fracao >= 0.06:               # chegou em "gerando variáveis candidatas"
            parar.set()

    with pytest.raises(TreinoCancelado):
        treinar(tabelas, mapeamento, config=Config(validar=False), progresso=progresso, cancelar=parar)


def test_cancelar_antes_de_comecar(tabelas, mapeamento):
    with pytest.raises(TreinoCancelado):
        treinar(tabelas, mapeamento, config=Config(validar=False), cancelar=lambda: True)
