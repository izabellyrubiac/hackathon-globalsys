"""Modelos de previsão do motor — um arquivo por modelo.

* `base.py`          protocolo `ModeloPrevisao`, calibração de Platt, utilidades e registro `MODELOS`;
* `logistica.py`     regressão logística L2 (padrão; stability selection);
* `random_forest.py` Random Forest via LightGBM `boosting="rf"`;
* `lightgbm.py`      LightGBM `boosting="gbdt"`;
* `_arvores.py`      comum às árvores (monotonia, parada antecipada, calibração, TreeSHAP).

Qual modelo usar é decidido em `motor/modelos_classificacao/escolha.py`.
"""

from .base import MODELOS, InfoModelo, ModeloBase, ModeloPrevisao, ajustar_modelo, log_loss, platt, sigmoide

__all__ = ["MODELOS", "InfoModelo", "ModeloBase", "ModeloPrevisao", "ajustar_modelo", "log_loss", "platt", "sigmoide"]
