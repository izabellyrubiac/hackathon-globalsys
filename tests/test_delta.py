"""(d) `Config.delta_risco`: o motor em dois estágios roda, cria a candidata de delta e não vaza.

Base sintética de **ruído puro**: a saída é sorteada sem nenhuma relação com as variáveis. Se o delta
vazasse (probabilidade das linhas de teste vinda de um modelo que viu aquele cliente), a AUC fora da
amostra subiria para perto de 1 — com ruído puro ela tem de ficar perto de 0,5.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from motor import Config, Mapeamento, de_dataframes, treinar
from motor.delta import ID_DELTA, ID_DELTA_3M

GERADO = "2026-01-01"
CFG = dict(dobras=3, rodadas=8, rodadas_arvores=6, dobras_internas=3, delta_dobras=2, n_jobs=1, semente=0)


def base_ruido(n: int = 70, semente: int = 3) -> dict[str, pd.DataFrame]:
    """`n` clientes × até 14 meses; tudo ruído: o mês de saída é sorteado, nada o antecede."""
    rng = np.random.default_rng(semente)
    meses = pd.period_range("2025-01", periods=14, freq="M")
    ids = [f"R{i:03d}" for i in range(1, n + 1)]
    canc = sorted(rng.choice(ids, n // 3, replace=False))
    saida = {c: meses[int(rng.integers(5, len(meses)))] for c in canc}
    linhas = []
    for c in ids:
        for m in meses:
            if c in saida and m >= saida[c]:
                break
            linhas.append({"codigo": c, "mes": str(m), "ruido_a": float(rng.normal(50, 10)),
                           "ruido_b": float(rng.poisson(4)), "ruido_c": float(rng.uniform(0, 1))})
    return {
        "clientes": pd.DataFrame({"codigo": ids, "mrr": rng.integers(500, 5000, n).astype(float)}),
        "situacao": pd.DataFrame({"codigo": ids, "churn": [int(c in saida) for c in ids],
                                  "data_saida": [str(saida[c]) if c in saida else None for c in ids]}),
        "uso_mensal": pd.DataFrame(linhas),
    }


def mapa() -> Mapeamento:
    return Mapeamento(tabela_clientes="clientes", coluna_id="codigo", alvo_tabela="situacao",
                      coluna_situacao="churn", valor_cancelado=1, coluna_data_saida="data_saida",
                      valor_tabela="clientes", coluna_valor="mrr", horizonte_meses=3)


@pytest.fixture(scope="module")
def tabelas():
    return de_dataframes(base_ruido())


@pytest.fixture(scope="module")
def com_delta(tabelas):
    return treinar(tabelas, mapa(), config=Config(delta_risco=True, **CFG), gerado_em=GERADO)


@pytest.fixture(scope="module")
def sem_delta(tabelas):
    return treinar(tabelas, mapa(), config=Config(delta_risco=False, **CFG), gerado_em=GERADO)


def test_roda_e_entrega_fila(com_delta):
    r = com_delta
    assert len(r.fila) > 0
    assert r.painel.loc[r.painel["pontuavel"], "p_final"].between(0, 1).all()
    assert r.clientes["clientes"] and r.pesos["variaveis"]


def test_delta_entra_nas_candidatas(com_delta):
    """A candidata existe e é avaliada — pode ou não ser selecionada."""
    rel = com_delta.modelo.selecao.relatorio
    assert ID_DELTA in rel.index and ID_DELTA_3M in rel.index
    assert ID_DELTA in com_delta.meta and com_delta.meta[ID_DELTA].tabela == "modelo"
    ids_pesos = {v["id"] for v in com_delta.pesos["variaveis"]}
    assert {ID_DELTA, ID_DELTA_3M} <= ids_pesos
    # e o registro traz o que a decisão precisa: entrou ou não, com peso e em quantas dobras
    v = next(v for v in com_delta.pesos["variaveis"] if v["id"] == ID_DELTA)
    assert set(v) >= {"selecionada", "frequencia_dobras", "frequencia_selecao", "auc_univariada"}


def test_sem_delta_nao_cria_a_candidata(sem_delta):
    ids_pesos = {v["id"] for v in sem_delta.pesos["variaveis"]}
    assert not ({ID_DELTA, ID_DELTA_3M} & ids_pesos)
    assert not any(c.startswith("risco_previsto__") for c in sem_delta.X.columns)


def test_primeiro_mes_do_cliente_fica_nulo(com_delta):
    """Sem mês anterior não há delta: o valor é nulo (não zero)."""
    P = com_delta.painel
    if ID_DELTA not in com_delta.X.columns:
        pytest.skip("delta não foi selecionada nesta base")
    primeiro = P.groupby("cliente")["mes"].transform("min") == P["mes"]
    assert com_delta.X.loc[primeiro & P["pontuavel"], ID_DELTA].isna().all()


def test_sem_vazamento_com_ruido_puro(com_delta, sem_delta):
    """Com alvo sorteado, a AUC out-of-fold fica perto de 0,5 nos dois braços — o delta não 'adivinha'."""
    for r in (com_delta, sem_delta):
        P, p = r.painel, r.oof.p
        t = P["treino"] & p.notna()
        a_linhas = _auc(P.loc[t, "y"].to_numpy(), p[t].to_numpy())
        m = (P["k"] == 1) & P["pontuavel"] & p.notna()
        a_k1 = _auc(P.loc[m, "cancelado"].to_numpy(), p[m].to_numpy())
        assert a_linhas < 0.75, f"{r.config.delta_risco=}: AUC nas linhas {a_linhas:.3f} alta demais para ruído puro"
        assert a_k1 < 0.75, f"{r.config.delta_risco=}: AUC em −1 {a_k1:.3f} alta demais para ruído puro"


def _auc(y, s) -> float:
    from motor.util import auc
    return float(auc(np.asarray(y).astype(bool), np.asarray(s, dtype=float)))
