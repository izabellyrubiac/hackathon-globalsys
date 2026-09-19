"""Código comum aos modelos de árvore (LightGBM `gbdt` e Random Forest via LightGBM `rf`).

* Seleção: `selecao_arvores.py` (sombras, estilo Boruta) sobre a pré-seleção comum.
* Monotonia: cada variável tem restrição de monotonia na direção de risco aprendida (+1: maior = mais
  risco), método "basic" do LightGBM — garante que o risco E a contribuição TreeSHAP de uma variável
  nunca andam contra a direção aprendida (a explicação não contradiz o dado).
* Nº de árvores: LightGBM — parada antecipada (50 rodadas sem melhora do log-loss) em cada dobra interna
  do GroupKFold por cliente; usa a média das melhores iterações. RF — `arvores_rf` fixas.
* Calibração: Platt (σ(a·logito_bruto + b)) nas previsões FORA DA AMOSTRA das dobras internas (com o nº de
  árvores final); o modelo final é treinado em todas as linhas de treino.
* Contribuições: TreeSHAP exato do LightGBM (`pred_contrib=True`) × a; intercepto = a·valor esperado + b,
  então intercepto + soma das contribuições = logito calibrado. No modo `rf` o LightGBM devolve a SOMA das
  árvores no logito bruto e nas contribuições (a probabilidade usa a média): dividimos por nº de árvores.
* Determinismo: `deterministic=True`, `force_col_wise`, 1 thread por modelo (as dobras rodam em paralelo)
  e semente fixa (`Config.semente`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb          # a biblioteca: import absoluto (não confunde com modelos_previsao/lightgbm.py)
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from ..config import Config
from ..selecao import PreSelecao
from ..selecao_arvores import selecionar_arvores
from .base import ModeloBase, limiares_evidencia, platt

PACIENCIA = 50
MAX_LINHAS_IMPORTANCIA = 5000


@dataclass(kw_only=True)
class ModeloArvore(ModeloBase):
    booster: lgb.Booster
    a: float                     # calibração de Platt
    b: float
    base_bruta: float            # valor esperado do logito bruto (TreeSHAP)
    n_arvores: int
    escala: float = 1.0          # RF: o LightGBM devolve a SOMA das árvores no logito bruto → escala = 1/nº de árvores
    params: dict = field(default_factory=dict)
    texto_nulo: str = "sem dado (o modelo usa o ramo aprendido para ausentes)"

    def _matriz(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.features].to_numpy(dtype=float)

    def logito_bruto(self, X: pd.DataFrame) -> np.ndarray:
        if len(X) == 0:
            return np.zeros(0)
        return self.escala * self.booster.predict(self._matriz(X), raw_score=True, num_iteration=self.n_arvores)

    def logito(self, X: pd.DataFrame) -> np.ndarray:
        return self.a * self.logito_bruto(X) + self.b

    def contribuicoes(self, X: pd.DataFrame) -> pd.DataFrame:
        """TreeSHAP exato na escala do logito calibrado (soma + intercepto = logito)."""
        if len(X) == 0:
            return pd.DataFrame(columns=self.features, dtype=float)
        C = self.booster.predict(self._matriz(X), pred_contrib=True, num_iteration=self.n_arvores)
        return pd.DataFrame(self.a * self.escala * C[:, :-1], index=X.index, columns=self.features)


def params_arvore(tipo: str, cfg: Config, direcoes: list[int]) -> dict:
    comum = {"objective": "binary", "verbose": -1, "seed": cfg.semente, "deterministic": True,
             "force_col_wise": True, "num_threads": 1, "min_data_in_leaf": cfg.min_linhas_folha,
             "lambda_l2": 1.0, "monotone_constraints": list(direcoes), "monotone_constraints_method": "basic",
             "feature_pre_filter": False}
    if tipo == "lightgbm":
        return {**comum, "boosting": "gbdt", "learning_rate": cfg.taxa_aprendizado, "num_leaves": cfg.folhas,
                "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1}
    if tipo == "random_forest":
        return {**comum, "boosting": "rf", "num_leaves": 2 * cfg.folhas + 1, "bagging_fraction": 0.632,
                "bagging_freq": 1, "feature_fraction": 0.7}
    raise ValueError(tipo)


def _dataset(M: np.ndarray, y: np.ndarray) -> lgb.Dataset:
    return lgb.Dataset(M, y, params={"feature_pre_filter": False, "verbose": -1}, free_raw_data=False)


def ajustar_arvore(tipo: str, X: pd.DataFrame, grade: pd.DataFrame, meta: dict, treino: pd.Series, cfg: Config,
                   horizonte: int, pre: PreSelecao | None = None,
                   freq_dobras: pd.Series | None = None) -> tuple[ModeloArvore, pd.DataFrame]:
    """Seleção por sombras + árvore monotônica calibrada nas linhas `treino`."""
    sel, X = selecionar_arvores(X, grade, meta, treino, cfg, horizonte, pre=pre, freq_dobras=freq_dobras)
    rel = sel.relatorio
    feats = list(sel.selecionadas)
    T = X.loc[treino]
    y = grade.loc[treino, "y"].to_numpy().astype(int)
    grupos = grade.loc[treino, "cliente"].to_numpy()
    direcoes = {f: int(rel.at[f, "direcao"]) for f in feats}
    params = params_arvore(tipo, cfg, [direcoes[f] for f in feats])
    M = T[feats].to_numpy(dtype=float)

    gkf = GroupKFold(n_splits=min(cfg.dobras_internas, len(np.unique(grupos))))
    dobras = [(tr, te) for tr, te in gkf.split(M, y, grupos) if len(np.unique(y[tr])) == 2]
    boosters = []
    if tipo == "lightgbm":
        melhores = []
        for tr, te in dobras:
            bst = lgb.train({**params, "metric": "binary_logloss"}, _dataset(M[tr], y[tr]),
                            num_boost_round=cfg.max_arvores, valid_sets=[_dataset(M[te], y[te])],
                            callbacks=[lgb.early_stopping(PACIENCIA, verbose=False)])
            melhores.append(bst.best_iteration or cfg.max_arvores)
            boosters.append(bst)
        n_arv = int(round(np.mean(melhores))) if melhores else 100
    else:
        n_arv = cfg.arvores_rf
        boosters = [lgb.train(params, _dataset(M[tr], y[tr]), num_boost_round=n_arv) for tr, _ in dobras]
    escala = 1.0 / n_arv if tipo == "random_forest" else 1.0
    bruto_oof = np.full(len(y), np.nan)
    for (tr, te), bst in zip(dobras, boosters):
        bruto_oof[te] = escala * bst.predict(M[te], raw_score=True, num_iteration=n_arv)
    ok = ~np.isnan(bruto_oof)
    a, b = platt(bruto_oof[ok], y[ok])

    final = lgb.train(params, _dataset(M, y), num_boost_round=n_arv)
    base_bruta = escala * float(final.predict(M[:1], pred_contrib=True, num_iteration=n_arv)[0, -1])
    m = ModeloArvore(nome=tipo, features=feats, direcoes=direcoes, limiares=limiares_evidencia(T, y, feats, direcoes),
                     selecao=sel, intercepto=a * base_bruta + b, booster=final, a=a, b=b, base_bruta=base_bruta,
                     n_arvores=n_arv, escala=escala, params=params)
    amostra = T if len(T) <= MAX_LINHAS_IMPORTANCIA else T.iloc[np.linspace(0, len(T) - 1, MAX_LINHAS_IMPORTANCIA).astype(int)]
    m.calcular_importancias(amostra)
    return m, X
