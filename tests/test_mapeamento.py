"""(d) mapeamento inválido gera erro legível em pt-BR."""

from __future__ import annotations

import pytest

from motor import ErroMapeamento, Mapeamento, treinar

BASE = dict(tabela_clientes="clientes", coluna_id="cliente_id", alvo_tabela="situacao_clientes",
            coluna_situacao="situacao", valor_cancelado="Cancelado", coluna_data_saida="mes_cancelamento",
            valor_tabela="clientes", coluna_valor="valor_mensal")


def _erro(tabelas, **mudar) -> str:
    with pytest.raises(ErroMapeamento) as e:
        Mapeamento(**{**BASE, **mudar}).validar(tabelas)
    return str(e.value)


def test_valido(tabelas_inova):
    assert Mapeamento(**BASE).validar(tabelas_inova) == []


@pytest.mark.parametrize("mudar, trecho", [
    ({"tabela_clientes": "clientela"}, "A tabela de clientes 'clientela' não existe"),
    ({"coluna_id": "id"}, "A coluna de ID do cliente 'id' não existe na tabela 'clientes'"),
    ({"tabela_clientes": "atendimento_mensal"}, "ID(s) repetido(s)"),
    ({"valor_cancelado": "Churn"}, "O valor 'Churn' não aparece na coluna 'situacao'"),
    ({"coluna_situacao": "status"}, "A coluna de situação 'status' não existe"),
    ({"coluna_data_saida": "segmento", "alvo_tabela": "clientes", "coluna_situacao": None}, "não parece uma data"),
    ({"coluna_valor": "segmento"}, "não é numérica"),
    ({"horizonte_meses": 0}, "entre 1 e 12 meses"),
    ({"coluna_situacao": None, "coluna_data_saida": None}, "sem cancelamento não há como aprender"),
])
def test_erros_legiveis(tabelas_inova, mudar, trecho):
    assert trecho in _erro(tabelas_inova, **mudar)


def test_sem_cancelados(tabelas_inova):
    t = dict(tabelas_inova)
    s = t["situacao_clientes"].copy()
    s["situacao"] = "Ativo"
    s.loc[0, "situacao"] = "Encerrado"
    t["situacao_clientes"] = s
    msg = _erro(t, valor_cancelado="Cancelado")
    assert "não aparece" in msg
    msg = _erro(t, valor_cancelado="Encerrado")
    assert "pelo menos 5" in msg


def test_varios_erros_juntos(tabelas_inova):
    with pytest.raises(ErroMapeamento) as e:
        Mapeamento(**{**BASE, "coluna_id": "x", "valor_cancelado": "Churn", "horizonte_meses": 20}).validar(tabelas_inova)
    assert len(e.value.problemas) >= 3
    assert str(e.value).startswith("Mapeamento inválido:")


def test_treinar_valida_antes(tabelas_inova):
    with pytest.raises(ErroMapeamento):
        treinar(tabelas_inova, Mapeamento(**{**BASE, "alvo_tabela": "nao_existe"}))


def test_dicionario(tabelas_inova):
    m = Mapeamento(**BASE, acoes={"nota_nps": "Ligar"})
    assert Mapeamento.from_dict(m.to_dict()) == m
    with pytest.raises(ErroMapeamento, match="desconhecido"):
        Mapeamento.from_dict({**m.to_dict(), "tabela": "x"})
    with pytest.raises(ErroMapeamento, match="tabela_clientes"):
        Mapeamento.from_dict({"coluna_id": "cliente_id"})
