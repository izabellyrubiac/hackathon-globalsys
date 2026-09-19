"""LightGBM (gradient boosting, `boosting="gbdt"`) com monotonia na direção de risco aprendida, parada
antecipada por GroupKFold (clientes inteiros), calibração de Platt fora da amostra e TreeSHAP exato.
Detalhes comuns às árvores em `_arvores.py`; seleção por sombras em `motor/selecao_arvores.py`.

Este arquivo se chama `lightgbm.py`, mas os imports do pacote são absolutos: `import lightgbm` aqui dentro
carrega a biblioteca instalada, não este módulo (testado em `tests/test_modelos.py`).
"""

from __future__ import annotations

import lightgbm as lgb          # a biblioteca (import absoluto)

from ._arvores import ModeloArvore, ajustar_arvore

NOME = "lightgbm"
VERSAO_LIGHTGBM = lgb.__version__


def ajustar(X, grade, meta, treino, cfg, horizonte, pre=None, freq_dobras=None, n_jobs=1) -> tuple[ModeloArvore, object]:
    """Seleção por sombras + LightGBM monotônico calibrado nas linhas `treino`."""
    return ajustar_arvore(NOME, X, grade, meta, treino, cfg, horizonte, pre=pre, freq_dobras=freq_dobras)
