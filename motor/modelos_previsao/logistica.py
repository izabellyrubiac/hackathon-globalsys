"""Regressão logística L2 nas variáveis da stability selection (`selecao.py`) — o modelo padrão do motor.

* Imputação pela mediana do treino → padronização (média/desvio do treino) → LogisticRegression L2,
  sem class_weight (a probabilidade sai calibrada na taxa real de cancelamento do painel).
* C escolhido por GroupKFold (5 dobras, clientes inteiros em cada dobra, log-loss).
* Se alguma variável sai com coeficiente de sinal oposto à direção aprendida na seleção, a de maior
  |coef| invertido sai e o modelo é reajustado (explicação nunca contradiz o dado univariado).
* Limiar de evidência de cada variável: corte de Youden no treino, arredondado para um número redondo.
* Contribuição = coef × valor padronizado (log-odds relativos ao cliente médio do treino); importância = |coef|.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from ..config import Config
from ..selecao import PreSelecao, _desvio, imputar, selecionar
from .base import ModeloBase, limiares_evidencia, log_loss

CS_L2 = np.logspace(-3, 2, 11)
_log_loss = log_loss          # compatibilidade


@dataclass(kw_only=True)
class Modelo(ModeloBase):
    medianas: pd.Series
    media: np.ndarray
    desvio: np.ndarray
    coef: np.ndarray
    C: float

    def z(self, X: pd.DataFrame) -> np.ndarray:
        if not self.features:
            return np.zeros((len(X), 0))
        return (imputar(X[self.features], self.medianas) - self.media) / self.desvio

    def logito(self, X: pd.DataFrame) -> np.ndarray:
        return self.z(X) @ self.coef + self.intercepto

    def contribuicoes(self, X: pd.DataFrame) -> pd.DataFrame:
        """coef × valor padronizado (log-odds relativos ao cliente médio do treino)."""
        return pd.DataFrame(self.z(X) * self.coef, index=X.index, columns=self.features)


ModeloLogistico = Modelo


def _ajustar_l2(Z: np.ndarray, y: np.ndarray, grupos: np.ndarray, cfg: Config) -> tuple[float, np.ndarray, float]:
    gkf = GroupKFold(n_splits=min(cfg.dobras_internas, len(np.unique(grupos))))
    perdas = np.zeros(len(CS_L2))
    for tr, te in gkf.split(Z, y, grupos):
        for i, C in enumerate(CS_L2):
            if len(np.unique(y[tr])) < 2:
                continue
            lr = LogisticRegression(C=C, max_iter=5000).fit(Z[tr], y[tr])
            perdas[i] += log_loss(y[te], lr.predict_proba(Z[te])[:, 1]) * len(te)
    C = float(CS_L2[int(np.argmin(perdas))])
    lr = LogisticRegression(C=C, max_iter=5000).fit(Z, y)
    return C, lr.coef_[0], float(lr.intercept_[0])


def ajustar(X: pd.DataFrame, grade: pd.DataFrame, meta: dict, treino: pd.Series, cfg: Config,
            horizonte: int, pre: PreSelecao | None = None, freq_dobras: pd.Series | None = None,
            n_jobs: int = 1) -> tuple[Modelo, pd.DataFrame]:
    """Seleção + L2 nas linhas `treino`. Devolve (modelo, X com persistências)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return _ajustar(X, grade, meta, treino, cfg, horizonte, pre, freq_dobras, n_jobs)


def _ajustar(X, grade, meta, treino, cfg, horizonte, pre, freq_dobras, n_jobs=1):
    sel, X = selecionar(X, grade, meta, treino, cfg, horizonte, pre=pre, freq_dobras=freq_dobras, n_jobs=n_jobs)
    rel = sel.relatorio
    T = X.loc[treino]
    y = grade.loc[treino, "y"].to_numpy().astype(int)
    grupos = grade.loc[treino, "cliente"].to_numpy()
    feats = list(sel.selecionadas)
    removidas = []
    while True:
        med = T[feats].median()
        A = imputar(T[feats], med)
        media, desvio = A.mean(axis=0), _desvio(A)
        C, coef, b0 = _ajustar_l2((A - media) / desvio, y, grupos, cfg)
        dirs = np.array([int(rel.at[f, "direcao"]) for f in feats])
        inv = [(abs(c), f) for c, f, d in zip(coef, feats, dirs) if np.sign(c) != d]
        if not inv or len(feats) == 1:
            break
        pior = max(inv)[1]
        removidas.append(pior)
        rel.at[pior, "motivo"] = "coeficiente com sinal invertido no modelo final (retirada)"
        rel.at[pior, "selecionada"] = False
        feats.remove(pior)
    direcoes = {f: int(rel.at[f, "direcao"]) for f in feats}
    m = Modelo(nome="logistica", features=feats, direcoes=direcoes, limiares=limiares_evidencia(T, y, feats, direcoes),
               selecao=sel, intercepto=b0, removidas=removidas, medianas=med, media=media, desvio=desvio,
               coef=np.asarray(coef, dtype=float), C=C)
    m.importancias = {f: float(abs(c)) for f, c in zip(feats, m.coef)}
    return m, X
