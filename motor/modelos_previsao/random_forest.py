"""Random Forest pelo próprio LightGBM (`boosting="rf"`: bagging de 63,2% das linhas e 70% das variáveis por
árvore, `arvores_rf` árvores com média das saídas) — evita a dependência `shap`, porque o LightGBM já
calcula TreeSHAP exato. Monotonia, calibração de Platt fora da amostra e contribuições em `_arvores.py`.
"""

from __future__ import annotations

from ._arvores import ModeloArvore, ajustar_arvore

NOME = "random_forest"


def ajustar(X, grade, meta, treino, cfg, horizonte, pre=None, freq_dobras=None, n_jobs=1) -> tuple[ModeloArvore, object]:
    """Seleção por sombras + Random Forest monotônica calibrada nas linhas `treino`."""
    return ajustar_arvore(NOME, X, grade, meta, treino, cfg, horizonte, pre=pre, freq_dobras=freq_dobras)
