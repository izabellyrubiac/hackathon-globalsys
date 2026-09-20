"""Rotas das pendências: PATCH da versão (renomear e pesos do score), cancelar treino e análise exploratória.

Tudo sobre a base INOVAAPPS de verdade (`dados/bases/inovaapps/arquivos/`), enviada pela própria API e treinada
uma vez, sem validação cruzada (~3 s). Nenhuma base sintética.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from api.tests.conftest import esperar, esperar_execucao

RAIZ = Path(__file__).resolve().parents[2]
XLSX = RAIZ / "dados" / "bases" / "inovaapps" / "arquivos" / "INOVAAPPS_base_de_dados.xlsx"
MAPEAMENTO = {**json.loads((RAIZ / "dados" / "bases" / "inovaapps" / "mapeamento.json").read_text(encoding="utf-8")),
              "colunas_ignoradas": []}
RAPIDO = {"validar": False, "semente": 0}
PESOS = {"risco": 0.6, "valor": 0.3, "delta": 0.1}


def _treinar(cliente, base_id, rotulo=None, **opcoes):
    corpo = {"mapeamento": MAPEAMENTO, "opcoes": {**RAPIDO, **opcoes}}
    if rotulo:
        corpo["rotulo"] = rotulo
    return cliente.post(f"/api/bases/{base_id}/treinar", json=corpo)


@pytest.fixture(scope="module")
def inova(cliente):
    """A INOVAAPPS enviada e treinada uma vez. Devolve (base_id, execucao_id)."""
    with open(XLSX, "rb") as f:
        r = cliente.post("/api/bases", files=[("arquivos", (XLSX.name, f.read(),
                          "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"))])
    assert r.status_code == 201, r.text
    base_id = r.json()["base_id"]
    t = _treinar(cliente, base_id, rotulo="original")
    assert t.status_code == 202, t.text
    eid = t.json()["execucao_id"]
    assert esperar_execucao(cliente, base_id, eid)["estado"] == "pronta"
    yield base_id, eid
    cliente.delete(f"/api/bases/{base_id}")


def _ordem(cliente, base_id, eid):
    r = cliente.get(f"/api/bases/{base_id}/execucoes/{eid}/clientes?limite=5000&incluir_historico=false").json()
    return r, [c["cliente_id"] for c in r["clientes"]]


# --------------------------------------------------------------------------- renomear e pesos do score
def test_renomear_troca_so_o_rotulo(cliente, inova):
    base_id, eid = inova
    r = cliente.patch(f"/api/bases/{base_id}/execucoes/{eid}", json={"rotulo": "  Nome novo  "})
    assert r.status_code == 200, r.text
    assert r.json()["rotulo"] == "Nome novo" and r.json()["execucao_id"] == eid
    assert r.json()["resumo"]["rotulo"] == "Nome novo"
    lista = cliente.get(f"/api/bases/{base_id}/execucoes").json()["execucoes"]
    assert [e["rotulo"] for e in lista if e["execucao_id"] == eid] == ["Nome novo"]


def test_patch_valida_o_corpo(cliente, inova):
    base_id, eid = inova
    rota = f"/api/bases/{base_id}/execucoes/{eid}"
    assert cliente.patch(rota, json={}).status_code == 400
    assert cliente.patch(rota, json={"campo": 1}).status_code == 422
    assert cliente.patch(rota, json={"pesos_score": {"risco": -1}}).status_code == 422
    assert cliente.patch(rota, json={"pesos_score": {"risco": 0, "valor": 0, "delta": 0}}).status_code == 422
    assert cliente.patch(f"/api/bases/{base_id}/execucoes/inexistente", json={"rotulo": "x"}).status_code == 404


def test_pesos_do_score_reordenam_e_voltam(cliente, inova):
    base_id, eid = inova
    rota = f"/api/bases/{base_id}/execucoes/{eid}"
    doc0, padrao = _ordem(cliente, base_id, eid)
    assert all(c["score"] is None for c in doc0["clientes"])

    r = cliente.patch(rota, json={"pesos_score": PESOS})
    assert r.status_code == 200, r.text
    assert r.json()["opcoes"]["pesos_score"] == PESOS
    doc, com_score = _ordem(cliente, base_id, eid)
    sc = [c["score"] for c in doc["clientes"]]
    assert sc == sorted(sc, reverse=True) and [c["prioridade"] for c in doc["clientes"]] == list(range(1, len(sc) + 1))
    assert "score" in doc["modelo"]["formula_prioridade"]
    assert sorted(com_score) == sorted(padrao)                       # os mesmos clientes, só a ordem muda

    # só o valor pesa: o score é o valor relativo ao maior contrato (empates do score exibido saem pelo risco)
    cliente.patch(rota, json={"pesos_score": {"risco": 0, "valor": 1, "delta": 0}})
    doc, _ = _ordem(cliente, base_id, eid)
    cl = doc["clientes"]
    vmax = max(c["valor_mensal"] for c in cl)
    assert cl[0]["valor_mensal"] == vmax and cl[0]["score"] == 100.0
    assert all(c["score"] == pytest.approx(100 * c["valor_mensal"] / vmax, abs=0.06) for c in cl)
    assert [c["score"] for c in cl] == sorted((c["score"] for c in cl), reverse=True)

    r = cliente.patch(rota, json={"pesos_score": None})              # volta à ordem padrão do treino
    assert r.status_code == 200
    doc, volta = _ordem(cliente, base_id, eid)
    assert volta == padrao and all(c["score"] is None for c in doc["clientes"])


def test_treino_ja_com_pesos_do_score_e_pesos_de_coluna(cliente, inova):
    base_id, _ = inova
    r = _treinar(cliente, base_id, rotulo="com pesos", pesos_score=PESOS,
                 pesos_colunas={"pesquisas_nps.nota_nps": 0})
    assert r.status_code == 202, r.text
    eid = r.json()["execucao_id"]
    assert esperar_execucao(cliente, base_id, eid)["estado"] == "pronta"
    doc, _ = _ordem(cliente, base_id, eid)
    assert doc["pesos_score"] == PESOS and doc["clientes"][0]["score"] >= doc["clientes"][-1]["score"]
    pesos = cliente.get(f"/api/bases/{base_id}/execucoes/{eid}/pesos").json()
    assert not any(v["selecionada"] for v in pesos["variaveis"] if v["coluna"] == "nota_nps")
    assert doc["clientes"][0]["datas"]["inicio_contrato"]
    assert cliente.delete(f"/api/bases/{base_id}/execucoes/{eid}").status_code == 204


def test_opcoes_de_peso_invalidas(cliente, inova):
    base_id, _ = inova
    assert _treinar(cliente, base_id, pesos_score={"risco": 0, "valor": 0, "delta": 0}).status_code == 422
    assert _treinar(cliente, base_id, pesos_colunas={"clientes.plano": 3}).status_code == 422


# --------------------------------------------------------------------------- cancelar
def test_cancelar_um_treino(cliente, inova):
    base_id, _ = inova
    r = _treinar(cliente, base_id, rotulo="vai cancelar")
    assert r.status_code == 202
    eid = r.json()["execucao_id"]
    c = cliente.post(f"/api/bases/{base_id}/execucoes/{eid}/cancelar")
    assert c.status_code == 202, c.text
    e = esperar_execucao(cliente, base_id, eid)
    assert e["estado"] == "cancelada" and "cancelado" in e["mensagem"].lower() and e["resumo"] is None
    assert cliente.get(f"/api/bases/{base_id}/execucoes/{eid}/clientes").status_code == 404   # sem resultados
    assert esperar(cliente, base_id)["estado"] != "treinando"
    # já terminou: não há o que cancelar
    assert cliente.post(f"/api/bases/{base_id}/execucoes/{eid}/cancelar").status_code == 409
    # versão cancelada não vira ativa e pode ser apagada
    assert cliente.post(f"/api/bases/{base_id}/execucoes/{eid}/ativar").status_code == 409
    assert cliente.delete(f"/api/bases/{base_id}/execucoes/{eid}").status_code == 204


def test_cancelar_versao_pronta_e_409(cliente, inova):
    base_id, eid = inova
    assert cliente.post(f"/api/bases/{base_id}/execucoes/{eid}/cancelar").status_code == 409


def test_pesos_do_score_so_em_versao_pronta(cliente, inova):
    base_id, _ = inova
    eid = _treinar(cliente, base_id, rotulo="cancelada").json()["execucao_id"]
    cliente.post(f"/api/bases/{base_id}/execucoes/{eid}/cancelar")
    assert esperar_execucao(cliente, base_id, eid)["estado"] == "cancelada"
    r = cliente.patch(f"/api/bases/{base_id}/execucoes/{eid}", json={"pesos_score": PESOS})
    assert r.status_code == 409
    cliente.delete(f"/api/bases/{base_id}/execucoes/{eid}")


# --------------------------------------------------------------------------- análise exploratória
def test_analise_exploratoria_pela_api(cliente, inova):
    base_id, _ = inova
    r = cliente.post(f"/api/bases/{base_id}/analise", json=MAPEAMENTO)
    assert r.status_code == 200, r.text
    a = r.json()
    assert (a["cancelados"], a["ativos"], a["referencia"]) == (22, 58, "2026-06")
    por = {s["coluna"]: s for s in a["series"]}
    assert por["uso_plataforma_pct"]["antecedencia"] == 3 and por["chamados_reabertos"]["antecedencia"] is None
    assert len(por["uso_plataforma_pct"]["pontos"]) == 12
    aucs = [abs(s["auc"] - 0.5) for s in a["series"]]
    assert aucs == sorted(aucs, reverse=True)


def test_analise_so_com_a_situacao_infere_a_saida(cliente, inova):
    """Sem a coluna do mês da saída o motor usa o fim do histórico: a análise continua valendo."""
    base_id, _ = inova
    r = cliente.post(f"/api/bases/{base_id}/analise", json={**MAPEAMENTO, "coluna_data_saida": None})
    assert r.status_code == 200, r.text
    assert r.json()["cancelados"] == 22 and r.json()["series"]


def test_analise_sem_cancelamento_e_422(cliente, inova):
    base_id, _ = inova
    r = cliente.post(f"/api/bases/{base_id}/analise",
                     json={**MAPEAMENTO, "coluna_data_saida": None, "coluna_situacao": None})
    assert r.status_code == 422
    assert "situação" in json.dumps(r.json(), ensure_ascii=False)
    assert cliente.post("/api/bases/nao-existe/analise", json=MAPEAMENTO).status_code == 404
