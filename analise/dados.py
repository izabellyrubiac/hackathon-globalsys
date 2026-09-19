"""Carga e limpeza da base INOVAAPPS 2026.

Uso:
    from dados import carregar
    d = carregar()          # d.clientes, d.atendimento, d.nps, d.situacao

Eixo de tempo alinhado (`mes_relativo`)
---------------------------------------
Na base, o histórico de um cliente cancelado termina no mês ANTERIOR ao
`mes_cancelamento` (ex.: cancelou em 2026-05, última linha em 2026-04).
Por isso o alinhamento é feito contra o "mês de saída":

* cancelados: mês de saída = `mes_cancelamento`
* ativos:     mês de saída = 2026-07 (primeiro mês ainda não observado,
              isto é, o mês que queremos prever)

`mes_relativo = mes_ref − mes_saida` → 0 é o mês da saída (nunca observado),
−1 é o último mês com dados para os dois grupos (ativos: jun/2026).
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ARQUIVO = Path(__file__).resolve().parent.parent / "INOVAAPPS_base_de_dados.xlsx"

MES_FINAL = pd.Period("2026-06", "M")          # último mês observado
MES_SAIDA_ATIVOS = MES_FINAL + 1               # jul/2026

ORDEM_STATUS = ["Cancelou", "Ativo"]
CORES = {"Cancelou": "#d03b3b", "Ativo": "#4e7896"}

ORDEM_NPS = ["Promotor", "Neutro", "Detrator", "Sem resposta"]
CORES_NPS = {
    "Promotor": "#2a78d6",
    "Neutro": "#c3c1b9",
    "Detrator": "#d03b3b",
    "Sem resposta": "#52514e",
}

# Métricas mensais usadas nas comparações (na ordem de exibição padrão)
METRICAS = [
    "uso_plataforma_pct",
    "chamados_abertos",
    "chamados_criticos",
    "chamados_reabertos",
    "pct_sla_cumprido",
    "tempo_medio_resolucao_h",
    "reclamacoes_formais",
    "dias_atraso_pagamento",
    "pct_reunioes_realizadas",
]

ROTULOS = {
    "cliente_id": "Cliente",
    "segmento": "Segmento",
    "porte": "Porte",
    "plano": "Plano",
    "valor_mensal": "Valor mensal (R$)",
    "sla_contratado_h": "SLA contratado (h)",
    "inicio_contrato": "Início do contrato",
    "status": "Status",
    "mes_ref": "Mês",
    "mes_cancelamento": "Mês do cancelamento",
    "mes_relativo": "Meses até a saída",
    "chamados_abertos": "Chamados abertos",
    "chamados_criticos": "Chamados críticos",
    "chamados_reabertos": "Chamados reabertos",
    "chamados_dentro_sla": "Chamados dentro do SLA",
    "pct_sla_cumprido": "SLA cumprido (%)",
    "tempo_medio_resolucao_h": "Tempo médio de resolução (h)",
    "reclamacoes_formais": "Reclamações formais",
    "uso_plataforma_pct": "Uso da plataforma (%)",
    "dias_atraso_pagamento": "Atraso de pagamento (dias)",
    "reunioes_previstas": "Reuniões previstas",
    "reunioes_realizadas": "Reuniões realizadas",
    "pct_reunioes_realizadas": "Reuniões realizadas (%)",
    "respondeu": "Respondeu NPS",
    "nota_nps": "Nota NPS",
    "classificacao_nps": "Classificação NPS",
}

# Formato d3 de cada métrica no hover
FORMATOS = {m: ",.1f" for m in METRICAS} | {
    "valor_mensal": ",.0f",
    "nota_nps": ",.1f",
}

_ACENTOS = {
    "Logistica": "Logística",
    "Educacao": "Educação",
    "Saude": "Saúde",
    "Servicos": "Serviços",
    "Industria": "Indústria",
    "Medio": "Médio",
    "Avancado": "Avançado",
}
ORDEM_CATEGORIAS = {
    "porte": ["Pequeno", "Médio", "Grande"],
    "plano": ["Essencial", "Avançado", "Enterprise"],
}

_MESES = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]


class Dados(NamedTuple):
    clientes: pd.DataFrame
    atendimento: pd.DataFrame
    nps: pd.DataFrame
    situacao: pd.DataFrame


# --------------------------------------------------------------------------- formatação
def fmt_mes(p) -> str:
    """Period/Timestamp → 'jun/2026'."""
    if p is None or pd.isna(p):
        return ""
    return f"{_MESES[p.month - 1]}/{p.year}"


def fmt_brl(v: float) -> str:
    """12345.6 → 'R$ 12.346'."""
    return "R$ " + f"{v:,.0f}".replace(",", ".")


def _periodo(s: pd.Series) -> pd.Series:
    return pd.PeriodIndex(s.astype("string"), freq="M").to_series(index=s.index)


# --------------------------------------------------------------------------- carga
def carregar(arquivo: str | Path = ARQUIVO) -> Dados:
    abas = pd.read_excel(
        arquivo,
        sheet_name=["clientes", "atendimento_mensal", "pesquisas_nps", "situacao_clientes"],
    )

    # situação ---------------------------------------------------------------
    sit = abas["situacao_clientes"].copy()
    sit["mes_cancelamento"] = _periodo(sit["mes_cancelamento"])
    sit["status"] = np.where(sit["situacao"].eq("Cancelado"), "Cancelou", "Ativo")
    sit["status"] = pd.Categorical(sit["status"], categories=ORDEM_STATUS)
    sit["mes_saida"] = sit["mes_cancelamento"].fillna(MES_SAIDA_ATIVOS)

    # clientes ---------------------------------------------------------------
    cli = abas["clientes"].copy()
    for col in ["segmento", "porte", "plano"]:
        cli[col] = cli[col].replace(_ACENTOS)
    for col, ordem in ORDEM_CATEGORIAS.items():
        cli[col] = pd.Categorical(cli[col], categories=ordem, ordered=True)
    cli["inicio_contrato"] = pd.to_datetime(cli["inicio_contrato"])
    cli = cli.merge(sit[["cliente_id", "status", "mes_cancelamento", "mes_saida"]], on="cliente_id")

    chaves = cli[["cliente_id", "status", "mes_saida", "segmento", "porte", "plano", "valor_mensal"]]

    # atendimento mensal -----------------------------------------------------
    at = abas["atendimento_mensal"].copy()
    at["mes_ref"] = _periodo(at["mes_ref"])
    at["pct_reunioes_realizadas"] = np.where(
        at["reunioes_previstas"] > 0,
        100 * at["reunioes_realizadas"] / at["reunioes_previstas"].where(at["reunioes_previstas"] > 0),
        np.nan,
    )
    at = at.merge(chaves, on="cliente_id", how="left")
    at["mes_relativo"] = _meses_entre(at["mes_ref"], at["mes_saida"])
    at = at.sort_values(["cliente_id", "mes_ref"]).reset_index(drop=True)

    # NPS --------------------------------------------------------------------
    nps = abas["pesquisas_nps"].copy()
    nps["mes_ref"] = _periodo(nps["mes_ref"])
    nps["respondeu"] = nps["respondeu"].astype(bool)
    nps["classificacao_nps"] = pd.Categorical(nps["classificacao_nps"], categories=ORDEM_NPS)
    nps = nps.merge(chaves, on="cliente_id", how="left")
    nps["mes_relativo"] = _meses_entre(nps["mes_ref"], nps["mes_saida"])
    # janelas de 3 meses: −1 = meses −3..−1, −2 = meses −6..−4, ...
    nps["trimestre_relativo"] = -((-nps["mes_relativo"] - 1) // 3 + 1)
    nps = nps.sort_values(["cliente_id", "mes_ref"]).reset_index(drop=True)

    return Dados(cli, at, nps, sit)


def _meses_entre(ini: pd.Series, fim: pd.Series) -> pd.Series:
    return (ini.dt.year - fim.dt.year) * 12 + (ini.dt.month - fim.dt.month)


# --------------------------------------------------------------------------- agregações
def perfil_relativo(df: pd.DataFrame, metrica: str, inicio: int = -12, fim: int = -1) -> pd.DataFrame:
    """Mediana, P25, P75, média e n por status × mes_relativo."""
    sub = df[df["mes_relativo"].between(inicio, fim)]
    g = sub.groupby(["status", "mes_relativo"], observed=True)[metrica]
    return g.agg(
        mediana="median",
        p25=lambda s: s.quantile(0.25),
        p75=lambda s: s.quantile(0.75),
        media="mean",
        n="count",
    ).reset_index()


def media_janela_final(at: pd.DataFrame, meses: int = 3, metricas: list[str] = METRICAS) -> pd.DataFrame:
    """Média por cliente dos `meses` finais observados (cancelados: antes da saída; ativos: até jun/2026)."""
    sub = at[at["mes_relativo"] >= -meses]
    out = sub.groupby("cliente_id")[metricas].mean()
    return out.join(at.groupby("cliente_id")[["status"]].first()).reset_index()


def auc_separacao(df: pd.DataFrame, metrica: str) -> float:
    """AUC de Cancelou vs Ativo usando a métrica crua (>0,5 → maior entre cancelados)."""
    sub = df[["status", metrica]].dropna()
    return float(roc_auc_score(sub["status"].eq("Cancelou"), sub[metrica]))


def suavizar(at: pd.DataFrame, metricas: list[str] = METRICAS, janela: int = 3) -> pd.DataFrame:
    """Média móvel retroativa de `janela` meses por cliente (usa só o passado → sem vazamento)."""
    out = at.sort_values(["cliente_id", "mes_ref"]).copy()
    out[metricas] = (
        out.groupby("cliente_id")[metricas]
        .rolling(janela, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    return out


def antecedencia(perfil: pd.DataFrame) -> int | None:
    """Maior k tal que, de −k até −1, a mediana de quem cancelou fica fora da faixa P25–P75 dos ativos.

    `perfil` é a saída de `perfil_relativo`. Devolve None se no mês −1 ainda não há separação.
    """
    p = perfil.pivot(index="mes_relativo", columns="status", values=["mediana", "p25", "p75"])
    fora = (p[("mediana", "Cancelou")] < p[("p25", "Ativo")]) | (p[("mediana", "Cancelou")] > p[("p75", "Ativo")])
    k = 0
    for m in sorted(fora.index, reverse=True):
        if not fora[m]:
            break
        k = -m
    return k or None
