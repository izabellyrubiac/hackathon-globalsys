"""(a) INOVAAPPS de ponta a ponta pelo motor genérico."""

from __future__ import annotations

import json

import numpy as np


def test_leitura_ignora_documentacao(tabelas_inova):
    assert set(tabelas_inova) == {"clientes", "atendimento_mensal", "pesquisas_nps", "situacao_clientes"}
    assert set(tabelas_inova.ignoradas) == {"Leia-me", "dicionario"}
    assert any("Leia-me" in a for a in tabelas_inova.avisos)


def test_inspecao_sugere_mapeamento(inova):
    insp, m, _ = inova
    json.dumps(insp, allow_nan=False)
    s = insp["sugestao"]
    assert s["tabela_clientes"] == "clientes" and s["coluna_id"] == "cliente_id"
    assert s["alvo"] == {"tabela": "situacao_clientes", "coluna_situacao": "situacao", "valor_cancelado": "Cancelado",
                         "coluna_data_saida": "mes_cancelamento"}
    assert s["coluna_valor"] == {"tabela": "clientes", "coluna": "valor_mensal"}
    papeis = {t["nome"]: t["papel_sugerido"] for t in insp["tabelas"]}
    assert papeis == {"clientes": "clientes", "atendimento_mensal": "temporal", "pesquisas_nps": "temporal",
                      "situacao_clientes": "estatica", "Leia-me": "ignorada", "dicionario": "ignorada"}
    tipos = {c["nome"]: c["tipo"] for t in insp["tabelas"] if t["nome"] == "atendimento_mensal" for c in t["colunas"]}
    assert tipos["cliente_id"] == "id" and tipos["mes_ref"] == "data" and tipos["uso_plataforma_pct"] == "numerica"


def test_fila_e_contrato(inova):
    _, m, r = inova
    c = r.clientes
    for obj in (r.clientes, r.validacao, r.pesos):
        json.dumps(obj, allow_nan=False)
    assert c["mes_referencia"] == "2026-06"
    cl = c["clientes"]
    assert len(cl) == 58 == c["resumo"]["clientes_ativos"]
    assert [x["prioridade"] for x in cl] == list(range(1, 59))
    perdas = [x["perda_anual_esperada"] for x in cl]
    assert perdas == sorted(perdas, reverse=True)
    assert all(abs(x["perda_anual_esperada"] - x["risco"] * x["valor_mensal"] * 12) <= 6e-5 * x["valor_mensal"] * 12
               for x in cl)          # risco sai arredondado em 4 casas
    assert {x["faixa_risco"] for x in cl} <= {"alto", "atencao", "baixo"}
    alto, aten = r.cortes["alto"], r.cortes["atencao"]
    for x in cl:
        esperado = "alto" if x["risco"] >= alto else "atencao" if x["risco"] >= aten else "baixo"
        assert x["faixa_risco"] == esperado
        assert len(x["historico"]) >= 4 and x["historico"][-1]["mes_ref"] == "2026-06"
    assert all(x["evidencias"] for x in cl if x["faixa_risco"] == "alto")


def test_alvo_nunca_vira_variavel(inova):
    _, _, r = inova
    tabelas = {v["tabela"] for v in r.pesos["variaveis"]}
    colunas = {v["coluna"] for v in r.pesos["variaveis"]}
    assert "situacao_clientes" not in tabelas
    assert not colunas & {"situacao", "mes_cancelamento", "cliente_id", "mes_ref"}


def test_selecao_e_validacao(inova):
    _, _, r = inova
    p = r.pesos
    sel = [v for v in p["variaveis"] if v["selecionada"]]
    assert 2 <= len(sel) <= p["resumo"]["limite_variaveis"]
    assert all(v["frequencia_selecao"] >= 0.6 and v["consistencia_sinal"] >= 0.9 for v in sel)
    assert all(v["motivo_descarte"] for v in p["variaveis"] if not v["selecionada"])
    # o motor encontra sozinho o sinal mais forte da base (uso da plataforma)
    assert any(v["coluna"] == "uso_plataforma_pct" for v in sel)
    auc = {(a["score"], a["k"]): a["auc"] for a in r.validacao["auc_por_mes"]}
    assert auc[("motor", 1)] > 0.9 and auc[("motor", 2)] > 0.85
    met = {m_["id"]: m_ for m_ in r.validacao["metodos"]}
    assert met["motor_alto_ou_atencao"]["canc_k1"] >= 18


def test_deterministico(tabelas_inova, inova):
    from motor import Config, treinar
    _, m, r = inova
    r2 = treinar(tabelas_inova, m, gerado_em="2026-01-01", config=Config(validar=False))
    r3 = treinar(tabelas_inova, m, gerado_em="2026-01-01", config=Config(validar=False))
    assert json.dumps(r2.clientes, sort_keys=True) == json.dumps(r3.clientes, sort_keys=True)
    assert json.dumps(r2.pesos, sort_keys=True) == json.dumps(r3.pesos, sort_keys=True)
    # o modelo final só usa a validação na consistência entre dobras: sem validação, a regra fica desligada e
    # a diferença de variáveis vem só do limite (quem entrou com a validação estava "fora do limite" sem ela)
    assert r2.pesos["regras"]["consistencia_dobras"].startswith("desligada")
    motivo2 = {v["id"]: v["motivo_descarte"] for v in r2.pesos["variaveis"]}
    for f in set(r.modelo.features) - set(r2.modelo.features):
        assert motivo2[f].startswith("fora do limite"), (f, motivo2[f])
    for f in set(r2.modelo.features) - set(r.modelo.features):
        assert r.oof.modelos["logistica"].frequencia_dobras().get(f, 0.0) < 0.5, f
    np.testing.assert_allclose(r2.modelo.prob(r2.X.loc[r2.fila["linha"]]), r2.fila["probabilidade"], atol=1e-12)
