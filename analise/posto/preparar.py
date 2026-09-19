"""Prepara a amostra real do posto (`amostra_1243d7ed/`) para o motor.

A base não tem campo de cancelamento. Regra definida pelo time:
1. Um cliente "cancela" no **primeiro mês sem nenhuma compra** depois da primeira compra dele.
2. Depois do cancelamento o cliente **não é reavaliado**: os meses seguintes (inclusive se ele voltar a comprar)
   ficam fora (o motor corta tudo a partir do mês da saída).
3. Só entram meses completos: o último mês da amostra (ago/2026, até dia 09) é descartado; referência = jul/2026.
   Ativo = comprou em todos os meses desde a primeira compra até jul/2026.

Também anexa `cliente_codigo` e data às tabelas que só têm `venda_codigo` (itens, pagamentos), liga títulos,
cheques e duplicatas pelo documento (hash) — elas usam outro código de cliente — e remove colunas que refletem o
estado **atual** (pendente/situação/pagamento), que vazariam o futuro. Tudo fica entre jul/2025 e jul/2026.

Uso: uv run python analise/posto/preparar.py  →  amostra_1243d7ed/base_motor/*.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
ORIGEM = RAIZ / "amostra_1243d7ed"
DESTINO = ORIGEM / "base_motor"
MES_INI = pd.Period("2025-07", "M")   # início das vendas (títulos/cheques têm registros antigos: ficam fora)
MES_REF = pd.Period("2026-07", "M")   # último mês completo

COMBUSTIVEIS = ("GASOLINA", "ETANOL", "DIESEL", "V-POWER", "GNV")


def _mes(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s.astype(str).str[:10], errors="coerce").dt.to_period("M")


def _data(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s.astype(str).str[:10], errors="coerce")


def _grupo_pagamento(f: pd.Series) -> pd.Series:
    f = f.fillna("").str.upper()
    return np.select(
        [f.str.contains("PRAZO|FATURA|NOTA"), f.str.contains("PIX"), f.str.contains("CREDITO"),
         f.str.contains("DEBITO"), f.str.contains("DINHEIRO"), f.str.contains("CHEQUE")],
        ["a_prazo", "pix", "credito", "debito", "dinheiro", "cheque"], "outros")


def rotular(vendas: pd.DataFrame) -> pd.DataFrame:
    """cliente_codigo, situacao, mes_saida — pela regra do 1º mês sem compra."""
    v = vendas[vendas["mes"] <= MES_REF]
    meses = pd.period_range(v["mes"].min(), MES_REF, freq="M")
    ativo = (v.groupby(["cliente_codigo", "mes"]).size().unstack(fill_value=0)
             .reindex(columns=meses, fill_value=0) > 0)
    linhas = []
    for cli, r in ativo.iterrows():
        a = r.to_numpy()
        i0 = int(a.argmax())                      # primeira compra
        gap = np.flatnonzero(~a[i0:])
        saida = meses[i0 + gap[0]] if len(gap) else None
        linhas.append({"cliente_codigo": cli, "situacao": "Cancelado" if saida is not None else "Ativo",
                       "mes_saida": str(saida) if saida is not None else None,
                       "primeira_compra": str(meses[i0])})
    return pd.DataFrame(linhas)


def main() -> None:
    DESTINO.mkdir(exist_ok=True)
    v = pd.read_csv(ORIGEM / "vendas.csv", usecols=["venda_codigo", "cliente_codigo", "cliente_documento", "data_hora",
                                                     "total_venda", "cancelada"])
    v = v[(v["cancelada"] == "N") & (v["cliente_codigo"] > 0)].copy()
    v["cliente_codigo"] = v["cliente_codigo"].astype("int64")
    v["data"] = _data(v["data_hora"])
    v["mes"] = v["data"].dt.to_period("M")
    v["hora"] = pd.to_datetime(v["data_hora"].str[:19], errors="coerce").dt.hour
    v = v[v["mes"] <= MES_REF]

    sit = rotular(v)
    clientes_ok = set(sit["cliente_codigo"])
    saida = sit.set_index("cliente_codigo")["mes_saida"].map(lambda x: pd.Period(x, "M") if x else None)

    # valor mensal (só para a prioridade — nunca vira variável): média do gasto nos meses antes da saída
    s_cli = v["cliente_codigo"].map(saida)
    antes = v[s_cli.isna() | (v["mes"] < s_cli)]
    gasto = antes.groupby(["cliente_codigo", "mes"])["total_venda"].sum().groupby(level=0).mean()

    cad = pd.read_csv(ORIGEM / "clientes.csv", usecols=["cliente_codigo", "tipo_pessoa"])
    cad = cad[cad["cliente_codigo"].isin(clientes_ok)].drop_duplicates("cliente_codigo")
    cad = pd.DataFrame({"cliente_codigo": sorted(clientes_ok)}).merge(cad, how="left")
    cad["valor_mensal_medio"] = cad["cliente_codigo"].map(gasto).round(2)

    vendas = v[["cliente_codigo", "data", "total_venda", "hora"]]

    # títulos/cheques/duplicatas usam outro código de cliente: liga pelo documento (hash, igual em todas as tabelas)
    doc = v.dropna(subset=["cliente_documento"]).drop_duplicates("cliente_documento").set_index("cliente_documento")["cliente_codigo"]

    def por_documento(df: pd.DataFrame, coluna_doc: str) -> pd.DataFrame:
        df = df.assign(cliente_codigo=df[coluna_doc].map(doc))
        return df[df["cliente_codigo"].notna()].astype({"cliente_codigo": "int64"})

    it = pd.read_csv(ORIGEM / "venda_itens.csv",
                     usecols=["venda_codigo", "produto_nome", "quantidade", "preco_custo", "total_venda"])
    it = it.merge(v[["venda_codigo", "cliente_codigo", "data"]], on="venda_codigo")
    nome = it["produto_nome"].fillna("").str.upper()
    it["categoria"] = np.where(nome.str.contains("|".join(COMBUSTIVEIS)), "combustivel", "conveniencia")
    it["margem"] = (it["total_venda"] - it["quantidade"] * it["preco_custo"]).round(2)
    itens = it[["cliente_codigo", "data", "categoria", "quantidade", "total_venda", "margem"]]

    pg = pd.read_csv(ORIGEM / "venda_pagamentos.csv", usecols=["venda_codigo", "forma_pagamento", "valor"])
    pg = pg.merge(v[["venda_codigo", "cliente_codigo", "data"]], on="venda_codigo")
    pg["forma"] = _grupo_pagamento(pg["forma_pagamento"])
    pagamentos = pg[["cliente_codigo", "data", "forma", "valor"]]

    tr = por_documento(pd.read_csv(ORIGEM / "titulos_receber.csv", usecols=[
        "cliente_documento", "data_movimento", "data_vencimento", "valor", "tipo"]), "cliente_documento")
    tr["data"] = _data(tr["data_movimento"])
    tr["prazo_dias"] = (_data(tr["data_vencimento"]) - tr["data"]).dt.days
    tr["tipo"] = _grupo_pagamento(tr["tipo"])
    titulos = tr[["cliente_codigo", "data", "valor", "prazo_dias", "tipo"]]

    ch = por_documento(pd.read_csv(ORIGEM / "cheques.csv", usecols=[
        "emitente_documento", "data_movimento", "bom_para", "valor"]), "emitente_documento")
    ch["data"] = _data(ch["data_movimento"])
    ch["prazo_dias"] = (_data(ch["bom_para"]) - ch["data"]).dt.days
    cheques = ch[["cliente_codigo", "data", "valor", "prazo_dias"]]

    du = por_documento(pd.read_csv(ORIGEM / "duplicatas.csv", usecols=[
        "cliente_documento", "data_movimento", "vencimento", "valor_duplicata"]), "cliente_documento")
    du["data"] = _data(du["data_movimento"])
    du["prazo_dias"] = (_data(du["vencimento"]) - du["data"]).dt.days
    duplicatas = du[["cliente_codigo", "data", "valor_duplicata", "prazo_dias"]]

    cm = pd.read_csv(ORIGEM / "compras_mensal.csv")   # mesmo código das vendas
    compras = cm.rename(columns={"mes": "data"})

    saidas = {"clientes": cad, "situacao": sit[["cliente_codigo", "situacao", "mes_saida"]], "vendas": vendas,
              "itens": itens, "pagamentos": pagamentos, "titulos": titulos, "cheques": cheques,
              "duplicatas": duplicatas, "compras_mensal": compras}
    for nome_t, df in saidas.items():
        if "data" in df.columns:
            m = _mes(df["data"])
            df = df[(m >= MES_INI) & (m <= MES_REF)]
        if "cliente_codigo" in df.columns and nome_t != "clientes":
            df = df[df["cliente_codigo"].isin(clientes_ok)]
        df.to_csv(DESTINO / f"{nome_t}.csv", index=False)
        print(f"{nome_t:15s} {len(df):>8,} linhas")
    n_c = int((sit["situacao"] == "Cancelado").sum())
    print(f"\nclientes: {len(sit)} · cancelados (1º mês sem compra): {n_c} · ativos em {MES_REF}: {len(sit) - n_c}")


if __name__ == "__main__":
    main()
