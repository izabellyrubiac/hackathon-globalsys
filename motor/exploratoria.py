"""Análise exploratória: como cada métrica mensal evolui nos meses antes da saída, cancelados × ativos.

É uma olhada nos dados **antes de treinar** (não depende de modelo) para ajudar a decidir o mapeamento; a medida validada
por cliente está em `validacao.json`. Mesmo formato do tipo `Analise` do frontend:

    {series: [{tabela, coluna, rotulo, auc, antecedencia, pontos: [{rel, c, a}]}],
     cancelados, ativos, referencia ("AAAA-MM")}

* Séries = colunas numéricas das tabelas mensais (as do cancelamento e as `colunas_ignoradas` já ficam de fora).
* Por cliente, o valor do mês é a média das linhas do mês, e a série é a **média móvel de 3 meses corridos**.
* Alinhamento em meses até a saída (`rel`): cancelados = mês − mês da saída; ativos = mês − (mês de referência + 1). Assim
  −1 é o último mês com dados nos dois grupos.
* `c` (cancelou) e `a` (ativo) = {med, p25, p75, n} por mês relativo, de −12 a −1 (null sem cliente naquele mês).
* `antecedencia` = maior k tal que, de −k até −1, a mediana de quem cancelou fica fora da faixa P25–P75 dos ativos
  (null se já em −1 não separa). `auc` = separação dos dois grupos em −1.
* Ordem das séries = quanto separam os grupos, |AUC − 0,5| (o mesmo critério da pré-seleção do motor).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .mapeamento import ErroMapeamento
from .painel import Base
from .saida import limpo
from .util import auc, rotulo_legivel

MESES = 12
JANELA = 3


def rotulo_serie(base: Base, coluna: str) -> str:
    """Rótulo do mapeamento, se houver; senão o nome legível da coluna."""
    return base.rotulos.get(coluna) or rotulo_legivel(coluna)


def _resumo(v: np.ndarray) -> dict | None:
    v = v[~np.isnan(v)]
    if not len(v):
        return None
    return {"med": float(np.median(v)), "p25": float(np.quantile(v, 0.25)), "p75": float(np.quantile(v, 0.75)),
            "n": int(len(v))}


def antecedencia(pontos: list[dict]) -> int | None:
    k = 0
    for p in sorted(pontos, key=lambda x: x["rel"], reverse=True):
        c, a = p["c"], p["a"]
        if c is None or a is None or not (c["med"] < a["p25"] or c["med"] > a["p75"]):
            break
        k = -p["rel"]
    return k or None


def analisar(base: Base) -> dict:
    """Análise sobre uma `Base` já preparada (`motor.painel.preparar`). ErroMapeamento se não há como comparar."""
    if base.fotografia or base.mes_ref is None:
        raise ErroMapeamento(["A base não tem tabela mensal: sem histórico não há como comparar quem saiu com quem ficou."])
    canc = base.cancelado
    if not canc.any() or canc.all():
        raise ErroMapeamento(["É preciso ter clientes cancelados e clientes ativos para comparar."])
    saida = base.mes_saida.dropna().map(lambda p: p.ordinal)
    if saida.empty:
        raise ErroMapeamento(["Para comparar quem saiu com quem ficou é preciso a coluna com o mês da saída. "
                              "Escolha-a em \"O que prever\"."])
    fim = pd.Series(base.mes_ref.ordinal + 1, index=canc.index).where(~canc, saida).astype("int64")
    eh_canc = canc.to_numpy()
    n = len(canc)

    series, com_dado = [], np.zeros(n, dtype=bool)
    for t in base.temporais:
        for coluna, tipo in t.colunas.items():
            if tipo != "numerica" or coluna not in t.df.columns:
                continue
            m = t.df.groupby(["__cli", "__mes"])[coluna].mean().dropna().reset_index()
            m = m[m["__cli"].isin(canc.index)]
            if m.empty:
                continue
            m["o"] = m["__mes"].array.asi8
            omin, omax = int(m["o"].min()), int(m["o"].max())
            grade = (m.pivot(index="__cli", columns="o", values=coluna)
                      .reindex(index=canc.index, columns=range(omin, omax + 1)))
            movel = grade.T.rolling(JANELA, min_periods=1).mean().T.to_numpy()     # meses corridos, ignora vazios
            pontos, ultimo = [], None
            for rel in range(-MESES, 0):
                col = (fim.to_numpy() + rel - omin)
                ok = (col >= 0) & (col < movel.shape[1])
                v = np.full(n, np.nan)
                v[ok] = movel[np.flatnonzero(ok), col[ok]]
                com_dado |= ~np.isnan(v)
                pontos.append({"rel": rel, "c": _resumo(v[eh_canc]), "a": _resumo(v[~eh_canc])})
                if rel == -1:
                    ultimo = v
            if not any(p["c"] for p in pontos) or not any(p["a"] for p in pontos):
                continue
            vc, va = ultimo[eh_canc], ultimo[~eh_canc]
            vc, va = vc[~np.isnan(vc)], va[~np.isnan(va)]
            y = np.r_[np.ones(len(vc)), np.zeros(len(va))]
            a_ = float(auc(y, np.r_[vc, va])) if len(vc) and len(va) else 0.5
            series.append({"tabela": t.nome, "coluna": coluna, "rotulo": rotulo_serie(base, coluna), "auc": a_,
                           "antecedencia": antecedencia(pontos), "pontos": pontos})
    if not series:
        raise ErroMapeamento(["Nenhuma tabela mensal com métrica numérica para analisar (ou as colunas foram ignoradas)."])
    series.sort(key=lambda s: -abs(s["auc"] - 0.5))
    return limpo({"series": series, "cancelados": int((com_dado & eh_canc).sum()),
                  "ativos": int((com_dado & ~eh_canc).sum()), "referencia": str(base.mes_ref)}, 3)
