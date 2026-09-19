"""Interface comum dos modelos de previsão, calibração e registro `MODELOS`.

Todo modelo (logística, Random Forest, LightGBM) entrega o que `explicar.py`, `treino.py` e `saida.py` usam:

* `nome`, `features`, `direcoes` (+1: maior = mais risco), `limiares` (evidência), `persistencias`,
  `selecao`, `removidas`, `importancias` (média de |contribuição| nas linhas de treino);
* `preparar(X, grade)` → X com as persistências do modelo;
* `logito(X)`, `prob(X)`;
* `contribuicoes(X)` → DataFrame linha × variável, ADITIVO: `intercepto + soma das contribuições = logito`
  (logística: coef × valor padronizado; árvores: TreeSHAP exato, já na escala calibrada).

Calibração (árvores): Platt/sigmoide sobre o logito bruto de previsões FORA DA AMOSTRA das dobras internas
(GroupKFold por cliente) — p = σ(a·logito_bruto + b). A logística dispensa: sem class_weight, a
probabilidade já sai na taxa real do painel.

Registro: `MODELOS[nome]` = rótulo, complexidade (ordem do desempate: logística < RF < LightGBM) e o
módulo que implementa `ajustar(X, grade, meta, treino, cfg, horizonte, pre=None, freq_dobras=None, n_jobs=1)`.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ..config import Config
from ..selecao import PreSelecao, Selecao, adicionar_persistencias, limiar_youden


# --------------------------------------------------------------------------- protocolo
class ModeloPrevisao(Protocol):
    nome: str
    features: list[str]
    direcoes: dict[str, int]
    limiares: dict[str, float]
    selecao: Selecao
    removidas: list[str]
    importancias: dict[str, float]
    intercepto: float

    @property
    def persistencias(self) -> dict[str, dict]: ...

    def preparar(self, X: pd.DataFrame, grade: pd.DataFrame) -> pd.DataFrame: ...

    def logito(self, X: pd.DataFrame) -> np.ndarray: ...

    def prob(self, X: pd.DataFrame) -> np.ndarray: ...

    def contribuicoes(self, X: pd.DataFrame) -> pd.DataFrame: ...


@dataclass
class ModeloBase:
    """Campos e métodos comuns. Subclasses implementam `logito` e `contribuicoes`."""
    nome: str
    features: list[str]
    direcoes: dict[str, int]
    limiares: dict[str, float]
    selecao: Selecao
    intercepto: float
    removidas: list[str] = field(default_factory=list)
    importancias: dict[str, float] = field(default_factory=dict)
    texto_nulo: str = "sem dado (usada a mediana)"

    @property
    def persistencias(self) -> dict[str, dict]:
        return {k: v for k, v in self.selecao.persistencias.items() if k in self.features}

    def preparar(self, X: pd.DataFrame, grade: pd.DataFrame) -> pd.DataFrame:
        """X com as colunas de persistência usadas pelo modelo."""
        return adicionar_persistencias(X, grade, self.persistencias)

    def logito(self, X: pd.DataFrame) -> np.ndarray:  # pragma: no cover - abstrato
        raise NotImplementedError

    def contribuicoes(self, X: pd.DataFrame) -> pd.DataFrame:  # pragma: no cover - abstrato
        raise NotImplementedError

    def prob(self, X: pd.DataFrame) -> np.ndarray:
        return sigmoide(self.logito(X))

    def calcular_importancias(self, X: pd.DataFrame) -> None:
        C = self.contribuicoes(X)
        self.importancias = {f: float(C[f].abs().mean()) for f in self.features}


# --------------------------------------------------------------------------- utilidades comuns
def sigmoide(z) -> np.ndarray:
    return 1 / (1 + np.exp(-np.asarray(z, dtype=float)))


def log_loss(y, p) -> float:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def limiares_evidencia(T: pd.DataFrame, y: np.ndarray, feats: list[str], direcoes: dict[str, int]) -> dict[str, float]:
    """Limiar de evidência de cada variável: corte de Youden no treino, arredondado (persistência: inteiro ≥ 1)."""
    out = {}
    for f in feats:
        lim, _ = limiar_youden(T[f].to_numpy(dtype=float), y, int(direcoes[f]))
        if f.endswith("__persistencia") and not np.isnan(lim):
            lim = float(max(1, round(lim)))
        out[f] = lim
    return out


def platt(logito_oof: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """(a, b) de p = σ(a·logito + b), ajustados em previsões fora da amostra. a ≥ 0,01 (nunca inverte)."""
    z = np.asarray(logito_oof, dtype=float).reshape(-1, 1)
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2 or np.nanstd(z) == 0:
        taxa = float(np.clip(y.mean() if len(y) else 0.5, 1e-6, 1 - 1e-6))
        return 0.01, float(np.log(taxa / (1 - taxa)))
    lr = LogisticRegression(C=1e6, max_iter=5000).fit(z, y)
    return max(float(lr.coef_[0, 0]), 0.01), float(lr.intercept_[0])


# --------------------------------------------------------------------------- registro
@dataclass(frozen=True)
class InfoModelo:
    nome: str
    rotulo: str
    complexidade: int          # desempate: o mais simples fica se o mais complexo não ganhar por > 1 EP
    arvore: bool
    modulo: str


MODELOS: dict[str, InfoModelo] = {
    "logistica": InfoModelo("logistica", "Regressão logística L2", 0, False, "logistica"),
    "random_forest": InfoModelo("random_forest", "Random Forest (LightGBM rf)", 1, True, "random_forest"),
    "lightgbm": InfoModelo("lightgbm", "LightGBM (gradient boosting)", 2, True, "lightgbm"),
}


def ajustar_modelo(nome: str, X: pd.DataFrame, grade: pd.DataFrame, meta: dict, treino: pd.Series, cfg: Config,
                   horizonte: int, pre: PreSelecao | None = None, freq_dobras: pd.Series | None = None,
                   n_jobs: int = 1) -> tuple[ModeloPrevisao, pd.DataFrame]:
    """Seleção + ajuste do modelo `nome` nas linhas `treino`. Devolve (modelo, X com persistências).
    `n_jobs` só paraleliza ajustes independentes (mesmo resultado com qualquer valor)."""
    if nome not in MODELOS:
        raise ValueError(f"Modelo desconhecido: {nome!r} (opções: {', '.join(MODELOS)})")
    mod = importlib.import_module(f".{MODELOS[nome].modulo}", __package__)
    return mod.ajustar(X, grade, meta, treino, cfg, horizonte, pre=pre, freq_dobras=freq_dobras, n_jobs=n_jobs)
