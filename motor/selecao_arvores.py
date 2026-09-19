"""Seleção de variáveis para os modelos de árvore (Random Forest e LightGBM), no estilo Boruta.

Reaproveita de `selecao.py` as etapas comuns (`pre_selecao`: nulos, quase constante, AUC univariada com
direção aprendida, vazamento, persistência, pré-filtro de escala e redundância). Depois:

1. Sombras: em `rodadas_arvores` (20) subamostras estratificadas de clientes (`frac_subamostra`; em base
   grande, até `razao_ativos_rodada` ativos por cancelado), cada candidata ganha uma cópia embaralhada
   (mesma distribuição, sem relação com o cancelamento). Um LightGBM rápido (100 árvores) é treinado com
   as reais + as sombras.
2. Uma variável "vence" a rodada se a importância dela (ganho) superar a MAIOR importância entre as sombras.
   Fica se vencer em ≥ `limiar_sombra` (60%) das rodadas.
3. Consistência entre dobras (só no modelo final, como na logística) e o mesmo limite de variáveis
   (mín(positivos/5, cancelados/2)), ordenado pela taxa de vitória e depois pelo ganho médio.

A direção de risco de cada variável (usada como restrição de monotonia no modelo) é a univariada da
pré-seleção. Tudo usa só as linhas de treino recebidas (roda dentro de cada dobra na validação).
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import Config
from .selecao import PreSelecao, Selecao, amostrar_clientes, aplicar_limite, pre_selecao

RODADA_ARVORES = 100


def _params_sombra(cfg: Config, semente: int) -> dict:
    return {"objective": "binary", "boosting": "gbdt", "learning_rate": 0.1, "num_leaves": cfg.folhas,
            "min_data_in_leaf": cfg.min_linhas_folha, "lambda_l2": 1.0, "feature_fraction": 0.8,
            "verbose": -1, "seed": semente, "deterministic": True, "force_col_wise": True, "num_threads": 1,
            "feature_pre_filter": False}


def selecionar_arvores(X: pd.DataFrame, grade: pd.DataFrame, meta: dict, treino: pd.Series, cfg: Config,
                       horizonte: int, pre: PreSelecao | None = None,
                       freq_dobras: pd.Series | None = None) -> tuple[Selecao, pd.DataFrame]:
    """Devolve (Selecao, X com persistências). Mesmo contrato de `selecao.selecionar`."""
    pre = pre or pre_selecao(X, grade, meta, treino, cfg, horizonte)
    rng = np.random.default_rng(cfg.semente)
    X = pre.X
    rel = pre.relatorio.copy()
    cols = list(pre.colunas)
    T = X.loc[treino]
    y = grade.loc[treino, "y"].to_numpy().astype(int)
    grupos = grade.loc[treino, "cliente"].to_numpy()
    A = T[cols].to_numpy(dtype=float)
    p = len(cols)

    cli_unicos = np.array(sorted(set(grupos)))
    canc_cli = pd.Series(y, index=grupos).groupby(level=0).max().reindex(cli_unicos).to_numpy().astype(bool)
    vitorias = np.zeros((cfg.rodadas_arvores, p))
    ganho = np.zeros((cfg.rodadas_arvores, p))
    feitas = np.zeros(cfg.rodadas_arvores, dtype=bool)
    for b in range(cfg.rodadas_arvores):
        amostra = amostrar_clientes(rng, cli_unicos, canc_cli, cfg)
        m = np.isin(grupos, amostra)
        if y[m].sum() == 0 or p == 0:
            continue
        Ab = A[m]
        sombras = rng.permuted(Ab, axis=0)
        ds = lgb.Dataset(np.hstack([Ab, sombras]), y[m], params={"feature_pre_filter": False, "verbose": -1})
        bst = lgb.train(_params_sombra(cfg, cfg.semente + b), ds, num_boost_round=RODADA_ARVORES)
        imp = bst.feature_importance(importance_type="gain")
        real, somb = imp[:p], imp[p:]
        vitorias[b] = real > somb.max()
        tot = imp.sum()
        ganho[b] = real / tot if tot > 0 else 0.0
        feitas[b] = True
    n = max(1, int(feitas.sum()))
    freq = vitorias[feitas].sum(axis=0) / n
    ganho_medio = ganho[feitas].sum(axis=0) / n
    for j, c in enumerate(cols):
        rel.at[c, "frequencia_selecao"] = freq[j]
        rel.at[c, "ganho_medio"] = ganho_medio[j]
        rel.at[c, "consistencia_sinal"] = np.nan
        rel.at[c, "sinal_l1"] = np.nan

    elegiveis = []
    for j, c in enumerate(cols):
        if freq[j] < cfg.limiar_sombra:
            rel.at[c, "motivo"] = (f"superou a maior sombra em só {100 * freq[j]:.0f}% das rodadas "
                                   f"(mínimo {100 * cfg.limiar_sombra:.0f}%)")
        else:
            elegiveis.append(c)
    sel = aplicar_limite(rel, elegiveis, cols, pre.limite,
                         lambda c: (-rel.at[c, "frequencia_selecao"], -rel.at[c, "ganho_medio"],
                                    -rel.at[c, "auc_oof"], cols.index(c)),
                         freq_dobras, cfg)
    sel = [c for c in X.columns if c in sel]
    return Selecao(sel, rel, pre.persistencias, None, pre.limite,
                   T[sel].median() if sel else pd.Series(dtype=float), metodo="sombras"), X
