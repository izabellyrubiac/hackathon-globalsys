"""(c) base sintética pequena com um sinal plantado: o motor precisa achar a variável certa sozinho."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from motor import Config, Mapeamento, inspecionar, ler_arquivos, treinar


def gerar_base(pasta, n=60, n_canc=15, semente=7):
    rng = np.random.default_rng(semente)
    ids = [f"K{i:03d}" for i in range(1, n + 1)]
    meses = pd.period_range("2024-01", "2025-06", freq="M")
    canc = set(rng.choice(ids, n_canc, replace=False))
    saida = {c: meses[int(rng.integers(7, len(meses)))] + 1 for c in canc}   # sai entre ago/2024 e jul/2025
    clientes = pd.DataFrame({"codigo": ids, "regiao": rng.choice(["Norte", "Sul", "Leste"], n),
                             "mrr": rng.integers(500, 5000, n).astype(float),
                             "inicio": [f"{int(rng.integers(1, 28)):02d}/0{int(rng.integers(1, 9))}/2022" for _ in ids]})
    situ = pd.DataFrame({"codigo": ids, "churn": [int(c in canc) for c in ids],
                         "data_saida": [saida[c].strftime("%Y-%m-%d") if c in canc else None for c in ids]})
    uso, fin = [], []
    for c in ids:
        base = rng.uniform(40, 90)
        for m in meses:
            if c in canc and m >= saida[c]:
                break
            k = (saida[c] - m).n if c in canc else 99
            queda = max(0, 5 - k) * 5 if c in canc else 0            # sinal plantado: cai nos 4 meses antes da saída
            uso.append({"codigo": c, "mes": str(m), "engajamento": base - queda + rng.normal(0, 7),
                        "ruido_a": rng.normal(10, 2), "ruido_b": rng.poisson(3)})
            fin.append({"cliente": c, "data": f"{int(rng.integers(1, 28)):02d}/{m.month:02d}/{m.year}",
                        "desconto": rng.uniform(0, 10), "atraso": rng.poisson(2)})
    eventos = pd.DataFrame({"codigo": rng.choice(ids, 200), "tipo": rng.choice(["duvida", "bug", "pedido"], 200),
                            "nota": rng.integers(1, 6, 200)})
    tabs = {"clientes": clientes, "situacao": situ, "uso_mensal": pd.DataFrame(uso),
            "financeiro": pd.DataFrame(fin), "chamados": eventos}
    caminhos = []
    for nome, df in tabs.items():
        p = pasta / f"{nome}.csv"
        df.to_csv(p, index=False)
        caminhos.append(p)
    return caminhos


@pytest.fixture(scope="module")
def sintetica(tmp_path_factory):
    t = ler_arquivos(gerar_base(tmp_path_factory.mktemp("sint")))
    insp = inspecionar(t)
    return t, insp


def test_sintetica_inspecao(sintetica):
    _, insp = sintetica
    s = insp["sugestao"]
    assert s["tabela_clientes"] == "clientes" and s["coluna_id"] == "codigo"
    assert s["alvo"]["tabela"] == "situacao" and s["alvo"]["coluna_situacao"] == "churn"
    assert str(s["alvo"]["valor_cancelado"]) == "1" and s["alvo"]["coluna_data_saida"] == "data_saida"
    assert s["coluna_valor"] == {"tabela": "clientes", "coluna": "mrr"}
    papeis = {x["nome"]: x["papel_sugerido"] for x in insp["tabelas"]}
    assert papeis == {"clientes": "clientes", "situacao": "estatica", "uso_mensal": "temporal",
                      "financeiro": "temporal", "chamados": "eventos"}


def test_sintetica_acha_sinal_plantado(sintetica):
    t, insp = sintetica
    m = Mapeamento.de_sugestao(insp, acoes={"engajamento": "Ligar para o cliente"})
    r = treinar(t, m, config=Config(dobras=5), gerado_em="2026-01-01")
    for obj in (r.clientes, r.validacao, r.pesos):
        json.dumps(obj, allow_nan=False)
    sel = [v for v in r.pesos["variaveis"] if v["selecionada"]]
    assert sel and sel[0]["coluna"] == "engajamento"                  # a mais estável é a plantada
    assert all(v["coluna"] == "engajamento" for v in sel)             # e nenhuma variável de ruído entra
    assert all(v["direcao"] == -1 for v in sel if v["transformacao"] != "persistencia")   # queda = risco, aprendido
    auc = {(a["score"], a["k"]): a["auc"] for a in r.validacao["auc_por_mes"]}
    assert auc[("motor", 1)] > 0.8
    assert len(r.clientes["clientes"]) == 45
    assert any(c["acao_sugerida"] == "Ligar para o cliente" for c in r.clientes["clientes"])


def test_fotografia_sem_tabela_temporal(sintetica):
    """Base sem nenhuma tabela com data: o motor treina uma fotografia (sem antecedência) e avisa."""
    t, insp = sintetica
    so_estaticas = {k: v for k, v in t.items() if k in ("clientes", "situacao", "chamados")}
    m = Mapeamento.de_sugestao(inspecionar(so_estaticas))
    r = treinar(so_estaticas, m, config=Config(dobras=3), gerado_em="2026-01-01")
    assert r.clientes["mes_referencia"] is None
    assert len(r.clientes["clientes"]) == 45
    assert any("fotografia" in a for a in r.avisos)


def test_sintetica_alerta_vazamento(sintetica):
    """Colunas que 'contam' o desfecho são excluídas e marcadas no pesos.json."""
    t, insp = sintetica
    t2 = {k: v.copy() for k, v in t.items()}
    canc = set(t2["situacao"].loc[t2["situacao"]["churn"] == 1, "codigo"])
    rng = np.random.default_rng(1)
    # fixa por cliente e preenchida só para quem saiu (ex.: código do motivo de saída)
    t2["clientes"]["codigo_motivo"] = [float(rng.integers(1, 5)) if c in canc else 0.0 for c in t2["clientes"]["codigo"]]
    # mensal que só muda no último mês antes da saída (ex.: aviso prévio registrado)
    u = t2["uso_mensal"]
    saida = t2["situacao"].set_index("codigo")["data_saida"].dropna().map(lambda d: pd.Period(d[:7], "M") - 1)
    u["aviso_previo"] = [1.0 if c in saida.index and pd.Period(m, "M") == saida[c] else 0.0
                         for c, m in zip(u["codigo"], u["mes"])]
    m = Mapeamento.de_sugestao(inspecionar(t2))
    r = treinar(t2, m, config=Config(validar=False), gerado_em="2026-01-01")
    pv = {v["id"]: v for v in r.pesos["variaveis"]}
    assert pv["codigo_motivo__estatico"]["vazamento"] and not pv["codigo_motivo__estatico"]["selecionada"]
    assert "vazamento" in pv["codigo_motivo__estatico"]["motivo_descarte"]
    vaz = [v for v in r.pesos["variaveis"] if v["coluna"] == "aviso_previo"]
    assert vaz and any(v["vazamento"] for v in vaz) and not any(v["selecionada"] for v in vaz)
    assert set(r.pesos["resumo"]["suspeitas_vazamento"]) >= {"codigo_motivo__estatico"}
