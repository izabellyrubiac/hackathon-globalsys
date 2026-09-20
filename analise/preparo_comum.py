"""Peças comuns aos scripts que preparam amostras reais de posto/rede para o motor.

As bases operacionais não têm campo de cancelamento. A regra de rótulo definida pelo time:

1. **Cancelou** = primeiro mês **sem nenhuma compra** depois da primeira compra do cliente
   (só venda não cancelada e cliente identificado).
2. **Sem reavaliação** depois da saída: os meses seguintes ficam fora (o motor já corta tudo a partir do
   mês de saída, inclusive se o cliente voltar a comprar).
3. Só **meses completos**: o último mês da exportação é incompleto e é descartado; a referência é o mês
   anterior. Ativo = comprou em todos os meses desde a primeira compra até o mês de referência.

O valor do contrato (gasto mensal médio antes da saída) usa meses posteriores a cada linha do painel:
serve só para ordenar a fila (`colunas_ignoradas` no mapeamento), nunca como variável.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def para_mes(s: pd.Series) -> pd.Series:
    """Texto de data (AAAA-MM-DD…) → Period mensal."""
    return pd.to_datetime(s.astype(str).str[:10], errors="coerce").dt.to_period("M")


def para_data(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s.astype(str).str[:10], errors="coerce")


def grupo_pagamento(f: pd.Series) -> pd.Series:
    """Formas de pagamento cruas → poucas categorias estáveis."""
    f = f.fillna("").astype(str).str.upper()
    return np.select(
        [f.str.contains("PRAZO|FATURA|NOTA"), f.str.contains("PIX"), f.str.contains("CREDITO|CARTAO|CARTOES"),
         f.str.contains("DEBITO"), f.str.contains("DINHEIRO"), f.str.contains("CHEQUE")],
        ["a_prazo", "pix", "credito", "debito", "dinheiro", "cheque"], "outros")


def rotular(compras: pd.DataFrame, mes_ref: pd.Period, col_cliente: str = "cliente_id") -> pd.DataFrame:
    """`cliente`, `situacao`, `mes_saida`, `primeira_compra` pela regra do 1º mês sem compra.

    `compras` = uma linha por compra (ou por cliente × mês), com `col_cliente` e a coluna `mes` (Period).
    Meses depois de `mes_ref` são ignorados.
    """
    c = compras.loc[compras["mes"] <= mes_ref, [col_cliente, "mes"]].drop_duplicates()
    meses = pd.period_range(c["mes"].min(), mes_ref, freq="M")
    pos = pd.Series(np.arange(len(meses)), index=meses)
    clientes = pd.Index(sorted(c[col_cliente].unique()), name=col_cliente)
    presente = np.zeros((len(clientes), len(meses)), dtype=bool)
    presente[clientes.get_indexer(c[col_cliente]), c["mes"].map(pos).to_numpy()] = True

    i0 = presente.argmax(axis=1)                                   # primeira compra
    depois = presente | (np.arange(len(meses))[None, :] < i0[:, None])   # antes da 1ª compra não conta como falta
    falta = ~depois
    tem_falta = falta.any(axis=1)
    i_saida = np.where(tem_falta, falta.argmax(axis=1), -1)
    return pd.DataFrame({
        col_cliente: clientes,
        "situacao": np.where(tem_falta, "Cancelado", "Ativo"),
        "mes_saida": [str(meses[i]) if i >= 0 else None for i in i_saida],
        "primeira_compra": [str(meses[i]) for i in i0],
    })


def valor_mensal_medio(compras: pd.DataFrame, saida: pd.Series, col_cliente: str = "cliente_id",
                       col_valor: str = "total_venda") -> pd.Series:
    """Gasto mensal médio do cliente **antes** da saída (só para a prioridade da fila)."""
    s = compras[col_cliente].map(saida)
    antes = compras[s.isna() | (compras["mes"] < s)]
    return antes.groupby([col_cliente, "mes"])[col_valor].sum().groupby(level=0).mean()
