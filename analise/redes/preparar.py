"""Prepara a amostra multi-rede (`amostra_redes/`) para o motor.

Mesmas regras de rótulo da amostra de um posto só (ver `analise/preparo_comum.py`):
cancelou = 1º mês sem nenhuma compra depois da primeira compra · sem reavaliação depois da saída ·
só meses completos (ago/2026 é parcial → referência = jul/2026).

Particularidades desta base:
* **Chave do cliente = `cliente_id`** (`rede_id-cliente_codigo`); `cliente_codigo` repete entre redes.
  As compras de todos os postos da rede são do mesmo cliente.
* `titulos_receber`, `cheques` e `duplicatas` já trazem `cliente_id` (não precisa ligar pelo documento).
  `venda_itens` e `venda_pagamentos` só têm `venda_codigo` (único na base) → ligados às vendas.
* **Redes com exportação truncada**: 2 redes terminam em fev/2026 e mar/2026. Todos os clientes delas
  "parariam" no fim do arquivo — artefato da exportação, não cancelamento. Essas redes saem da base.
* Colunas de **estado atual** (pendente, situação, devolvido, data/valor de pagamento) são removidas:
  refletem o futuro em relação a cada linha do painel.

Uso: uv run python analise/redes/preparar.py  →  amostra_redes/base_motor/*.csv
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from analise.preparo_comum import (  # noqa: E402
    grupo_pagamento, para_data, para_mes, rotular, valor_mensal_medio,
)

ORIGEM = RAIZ / "amostra_redes"
DESTINO = ORIGEM / "base_motor"
MES_REF = pd.Period("2026-07", "M")      # último mês completo (ago/2026 é parcial)
MES_FIM_ESPERADO = MES_REF + 1           # rede íntegra tem venda em ago/2026

COMBUSTIVEIS = ("GASOLINA", "ETANOL", "DIESEL", "V-POWER", "GNV", "ALCOOL", "ARLA")


def carregar_vendas() -> tuple[pd.DataFrame, list[str]]:
    """Vendas válidas (não canceladas) das redes com exportação completa, até `MES_REF`."""
    v = pd.read_csv(ORIGEM / "vendas.csv",
                    usecols=["rede_id", "cliente_id", "venda_codigo", "data_hora", "total_venda", "cancelada"],
                    dtype={"rede_id": "string", "cliente_id": "string", "venda_codigo": "int64",
                           "data_hora": "string", "cancelada": "string"})
    v = v[(v["cancelada"] == "N") & v["cliente_id"].notna()].drop(columns="cancelada")
    v["mes"] = para_mes(v["data_hora"])
    fim_rede = v.groupby("rede_id", observed=True)["mes"].max()
    redes_ok = fim_rede.index[fim_rede >= MES_FIM_ESPERADO]
    truncadas = sorted(set(fim_rede.index) - set(redes_ok))
    v = v[v["rede_id"].isin(set(redes_ok))]
    v["hora"] = pd.to_datetime(v["data_hora"].str[:19], errors="coerce").dt.hour.astype("float32")
    v["data"] = para_data(v["data_hora"])
    v = v.drop(columns="data_hora")
    return v[v["mes"] <= MES_REF].reset_index(drop=True), truncadas


def main() -> None:
    t0 = time.perf_counter()
    DESTINO.mkdir(exist_ok=True)
    v, truncadas = carregar_vendas()
    print(f"[{time.perf_counter() - t0:5.1f}s] vendas válidas até {MES_REF}: {len(v):,} "
          f"({v['cliente_id'].nunique():,} clientes) · redes truncadas fora: {len(truncadas)}")

    sit = rotular(v, MES_REF)
    clientes_ok = set(sit["cliente_id"])
    saida = sit.set_index("cliente_id")["mes_saida"].map(lambda x: pd.Period(x, "M") if x else None)
    gasto = valor_mensal_medio(v, saida)

    cad = pd.read_csv(ORIGEM / "clientes.csv", usecols=["cliente_id", "rede_id", "tipo_pessoa"])
    cad = cad[cad["cliente_id"].isin(clientes_ok)].drop_duplicates("cliente_id")
    cad = pd.DataFrame({"cliente_id": sorted(clientes_ok)}).merge(cad, how="left")
    cad["rede_id"] = cad["rede_id"].fillna(cad["cliente_id"].str.split("-").str[0])
    cad["valor_mensal_medio"] = cad["cliente_id"].map(gasto).round(2)

    chave = v[["venda_codigo", "cliente_id", "data"]]
    vendas = v[["cliente_id", "data", "total_venda", "hora"]]

    it = pd.read_csv(ORIGEM / "venda_itens.csv",
                     usecols=["venda_codigo", "produto_nome", "quantidade", "preco_custo", "total_venda"],
                     dtype={"venda_codigo": "int64", "produto_nome": "string", "quantidade": "float32",
                            "preco_custo": "float32", "total_venda": "float32"})
    it = it.merge(chave, on="venda_codigo").drop(columns="venda_codigo")
    nome = it.pop("produto_nome").fillna("").str.upper()
    it["categoria"] = np.where(nome.str.contains("|".join(COMBUSTIVEIS)), "combustivel", "conveniencia")
    it["margem"] = (it["total_venda"] - it["quantidade"] * it["preco_custo"].fillna(0)).round(2)
    itens = it[["cliente_id", "data", "categoria", "quantidade", "total_venda", "margem"]]
    del it, nome

    pg = pd.read_csv(ORIGEM / "venda_pagamentos.csv", usecols=["venda_codigo", "forma_pagamento", "valor"],
                     dtype={"venda_codigo": "int64", "forma_pagamento": "string", "valor": "float32"})
    pg = pg.merge(chave, on="venda_codigo").drop(columns="venda_codigo")
    pg["forma"] = grupo_pagamento(pg["forma_pagamento"])
    pagamentos = pg[["cliente_id", "data", "forma", "valor"]]
    del pg

    # títulos/cheques/duplicatas já têm cliente_id; colunas de estado atual (pendente/situação/pagamento) saem
    tr = pd.read_csv(ORIGEM / "titulos_receber.csv",
                     usecols=["cliente_id", "data_movimento", "data_vencimento", "valor", "tipo"])
    tr["data"] = para_data(tr["data_movimento"])
    tr["prazo_dias"] = (para_data(tr["data_vencimento"]) - tr["data"]).dt.days
    tr["tipo"] = grupo_pagamento(tr["tipo"])
    titulos = tr[["cliente_id", "data", "valor", "prazo_dias", "tipo"]]

    ch = pd.read_csv(ORIGEM / "cheques.csv", usecols=["cliente_id", "data_movimento", "bom_para", "valor"])
    ch["data"] = para_data(ch["data_movimento"])
    ch["prazo_dias"] = (para_data(ch["bom_para"]) - ch["data"]).dt.days
    cheques = ch[["cliente_id", "data", "valor", "prazo_dias"]]

    du = pd.read_csv(ORIGEM / "duplicatas.csv",
                     usecols=["cliente_id", "data_movimento", "vencimento", "valor_duplicata"])
    du["data"] = para_data(du["data_movimento"])
    du["prazo_dias"] = (para_data(du["vencimento"]) - du["data"]).dt.days
    duplicatas = du[["cliente_id", "data", "valor_duplicata", "prazo_dias"]]

    cm = pd.read_csv(ORIGEM / "compras_mensal.csv",
                     usecols=["cliente_id", "mes", "n_compras", "valor", "dias_com_compra", "qtd_itens"])
    compras = cm.rename(columns={"mes": "data"})

    saidas = {"clientes": cad, "situacao": sit[["cliente_id", "situacao", "mes_saida"]], "vendas": vendas,
              "itens": itens, "pagamentos": pagamentos, "titulos": titulos, "cheques": cheques,
              "duplicatas": duplicatas, "compras_mensal": compras}
    linhas = {}
    for nome_t, df in saidas.items():
        if "data" in df.columns:
            m = para_mes(df["data"])
            df = df[m.notna() & (m <= MES_REF)]
        if "cliente_id" in df.columns and nome_t != "clientes":
            df = df[df["cliente_id"].isin(clientes_ok)]
        df.to_csv(DESTINO / f"{nome_t}.csv", index=False)
        linhas[nome_t] = len(df)
        print(f"  {nome_t:15s} {len(df):>9,} linhas")

    n_c = int((sit["situacao"] == "Cancelado").sum())
    print(f"\nclientes: {len(sit):,} · cancelados (1º mês sem compra): {n_c:,} · "
          f"ativos em {MES_REF}: {len(sit) - n_c:,}")
    print(f"redes truncadas descartadas: {truncadas}")
    print(f"tempo: {time.perf_counter() - t0:.1f}s · destino {DESTINO.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
