"""Compatibilidade: a regressão logística agora mora em `motor/modelos_previsao/logistica.py`
(os outros modelos — Random Forest e LightGBM — ficam na mesma pasta, um arquivo por modelo)."""

from .modelos_previsao.logistica import CS_L2, Modelo, ModeloLogistico, _ajustar_l2, _log_loss, ajustar

__all__ = ["CS_L2", "Modelo", "ModeloLogistico", "_ajustar_l2", "_log_loss", "ajustar"]
