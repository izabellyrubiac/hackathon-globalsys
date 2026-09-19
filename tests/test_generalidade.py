"""(b) a mesma base disfarçada e (e) a mesma base em CSV devem dar o mesmo resultado."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from motor import Mapeamento, inspecionar, ler_arquivos, treinar

from .conftest import GERADO, XLSX

RENOMEAR = {
    "cliente_id": "id_conta", "mes_ref": "competencia", "uso_plataforma_pct": "utilizacao_sistema",
    "nota_nps": "nota_recomendacao", "chamados_criticos": "tickets_graves", "valor_mensal": "mensalidade",
    "situacao": "status_conta", "mes_cancelamento": "data_encerramento", "tempo_medio_resolucao_h": "horas_solucao",
    "reclamacoes_formais": "queixas", "classificacao_nps": "perfil_recomendacao",
}
ABAS = {"clientes": "contas", "atendimento_mensal": "suporte", "pesquisas_nps": "pesquisa"}


def _traduz_id(vid: str) -> str:
    for a, b in RENOMEAR.items():
        vid = vid.replace(a, b)
    return vid


def _disfarcar() -> dict[str, pd.DataFrame]:
    """Colunas renomeadas, alvo dentro da tabela de clientes, meses como data completa (AAAA-MM-DD)."""
    abas = pd.read_excel(XLSX, sheet_name=None)
    cli = abas["clientes"].merge(abas["situacao_clientes"], on="cliente_id")
    cli["mes_cancelamento"] = cli["mes_cancelamento"].map(lambda m: f"{m}-15" if isinstance(m, str) else None)
    out = {"contas": cli}
    for velho, novo in (("atendimento_mensal", "suporte"), ("pesquisas_nps", "pesquisa")):
        d = abas[velho].copy()
        d["mes_ref"] = d["mes_ref"].map(lambda m: f"{m}-01")
        out[novo] = d
    return {k: v.rename(columns=RENOMEAR) for k, v in out.items()}


def _comparar(r_a, r_b):
    fa = {c["cliente_id"]: c for c in r_a.clientes["clientes"]}
    fb = {c["cliente_id"]: c for c in r_b.clientes["clientes"]}
    assert set(fa) == set(fb)
    assert [c["cliente_id"] for c in r_a.clientes["clientes"]] == [c["cliente_id"] for c in r_b.clientes["clientes"]]
    np.testing.assert_allclose([fa[c]["risco"] for c in fa], [fb[c]["risco"] for c in fa], atol=1e-9)
    assert [fa[c]["faixa_risco"] for c in fa] == [fb[c]["faixa_risco"] for c in fa]
    assert r_a.cortes == r_b.cortes
    sel_a = sorted(_traduz_id(v["id"]) for v in r_a.pesos["variaveis"] if v["selecionada"])
    sel_b = sorted(v["id"] for v in r_b.pesos["variaveis"] if v["selecionada"])
    assert sel_a == sel_b
    auc_a = [a["auc"] for a in r_a.validacao["auc_por_mes"]]
    auc_b = [a["auc"] for a in r_b.validacao["auc_por_mes"]]
    np.testing.assert_allclose(auc_a, auc_b, atol=1e-9)


@pytest.fixture(scope="module")
def disfarcada(tmp_path_factory):
    d = _disfarcar()
    caminho = tmp_path_factory.mktemp("b") / "outra_base.xlsx"
    with pd.ExcelWriter(caminho) as w:          # abas em outra ordem + aba de documentação
        d["pesquisa"].to_excel(w, sheet_name="pesquisa", index=False)
        pd.DataFrame({"Sobre esta base": ["Exportação do CRM, uma linha por conta e competência mensal.",
                                          "Valores em reais; notas de 0 a 10."]}).to_excel(w, sheet_name="LEIAME", index=False)
        d["suporte"].to_excel(w, sheet_name="suporte", index=False)
        d["contas"].to_excel(w, sheet_name="contas", index=False)
    return ler_arquivos([caminho])


def test_disfarcada_sugestao(disfarcada):
    assert set(disfarcada) == {"pesquisa", "suporte", "contas"} and "LEIAME" in disfarcada.ignoradas
    s = inspecionar(disfarcada)["sugestao"]
    assert s["tabela_clientes"] == "contas" and s["coluna_id"] == "id_conta"
    assert s["alvo"] == {"tabela": "contas", "coluna_situacao": "status_conta", "valor_cancelado": "Cancelado",
                         "coluna_data_saida": "data_encerramento"}
    assert s["coluna_valor"] == {"tabela": "contas", "coluna": "mensalidade"}


def test_disfarcada_equivalente(disfarcada, inova):
    _, _, r_a = inova
    m = Mapeamento.de_sugestao(inspecionar(disfarcada))
    r_b = treinar(disfarcada, m, gerado_em=GERADO)
    colunas = {v["coluna"] for v in r_b.pesos["variaveis"]}
    assert not colunas & {"status_conta", "data_encerramento"}
    _comparar(r_a, r_b)


def test_csv_equivalente(tmp_path, inova):
    """(e) várias tabelas como .csv (separador ';', decimal ',', latin-1) — mesmo resultado do .xlsx."""
    _, m, r_a = inova
    abas = pd.read_excel(XLSX, sheet_name=None)
    caminhos = []
    for nome in ("clientes", "atendimento_mensal", "pesquisas_nps", "situacao_clientes", "dicionario"):
        p = tmp_path / f"{nome}.csv"
        abas[nome].to_csv(p, sep=";", decimal=",", index=False, encoding="latin-1")
        caminhos.append(p)
    t = ler_arquivos(caminhos)
    assert set(t) == {"clientes", "atendimento_mensal", "pesquisas_nps", "situacao_clientes"}
    assert "dicionario" in t.ignoradas
    assert pd.api.types.is_numeric_dtype(t["atendimento_mensal"]["uso_plataforma_pct"])
    assert Mapeamento.de_sugestao(inspecionar(t)) == m
    r_e = treinar(t, m, gerado_em=GERADO)
    fa = [c["risco"] for c in r_a.clientes["clientes"]]
    fe = [c["risco"] for c in r_e.clientes["clientes"]]
    np.testing.assert_allclose(fa, fe, atol=1e-9)
    assert r_a.cortes == r_e.cortes
