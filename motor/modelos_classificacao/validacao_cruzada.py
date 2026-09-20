"""Validação out-of-fold agrupada por cliente, com a seleção de variáveis ANINHADA, para vários modelos.

Em cada dobra externa (StratifiedKFold sobre os clientes, estratificado por cancelou), a pré-seleção comum
(filtro, AUC, vazamento, persistência, redundância) roda uma vez só com os clientes de treino; depois,
CADA modelo candidato refaz a própria seleção (stability selection ou sombras), o ajuste (C, parada
antecipada, calibração) e os limiares — também só com os clientes de treino. Os clientes de teste recebem
a probabilidade de cada modelo em todos os seus meses. Todos os modelos usam AS MESMAS dobras, então a
comparação é pareada por dobra (`metricas.diferenca_pareada`).

Registra as variáveis de cada dobra por modelo (coeficiente na logística, importância nas árvores) — é
daí que sai a consistência entre dobras do modelo final (`Config.min_frac_dobras`).

Baseline "melhor variável sozinha": em cada dobra, a candidata de maior AUC out-of-fold no treino,
orientada pela direção aprendida, com o limiar de Youden do treino.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold

from ..config import Config
from ..delta import ajustar_modelo_com_delta
from ..modelos_previsao.base import MODELOS
from ..selecao import limiar_youden, pre_selecao
from .metricas import log_loss_dobras


@dataclass
class ResultadoModelo:
    nome: str
    p: pd.Series                   # probabilidade out-of-fold (NaN fora das linhas pontuáveis)
    dobras: pd.DataFrame           # dobra × variável: coef (logística) / importância (árvores); NaN = não selecionada
    log_loss_dobras: np.ndarray    # log-loss de cada dobra (linhas de treino dos clientes de teste)
    n_dobras: int

    def frequencia_dobras(self) -> pd.Series:
        """Fração das dobras externas em que cada variável entrou no modelo."""
        return (self.dobras.notna().sum() / self.n_dobras) if len(self.dobras.columns) else pd.Series(dtype=float)


@dataclass
class ValidacaoOOF:
    p: pd.Series                   # do modelo escolhido (ver `usar`)
    base_score: pd.Series          # melhor variável sozinha (orientada: maior = mais risco)
    base_alarme: pd.Series         # melhor variável ≥ limiar de Youden do treino
    dobras: pd.DataFrame           # do modelo escolhido
    base_vars: list[str]           # melhor variável escolhida em cada dobra
    n_dobras: int
    modelos: dict[str, ResultadoModelo] = field(default_factory=dict)
    clientes_dobra: list = field(default_factory=list)     # clientes de teste de cada dobra
    escolhido: str | None = None

    def usar(self, nome: str) -> None:
        self.escolhido = nome
        self.p = self.modelos[nome].p
        self.dobras = self.modelos[nome].dobras


def _info_dobra(mod) -> dict[str, float]:
    if hasattr(mod, "coef"):
        return dict(zip(mod.features, map(float, mod.coef)))
    return {f: float(mod.importancias.get(f, 0.0)) for f in mod.features}


def _dobra(i, X, grade, meta, treino_cli, teste_cli, cfg, H, nomes):
    tr = grade["treino"] & grade["cliente"].isin(treino_cli)
    te = grade["pontuavel"] & grade["cliente"].isin(teste_cli)
    pre = pre_selecao(X, grade, meta, tr, cfg, H)
    por_modelo = {}
    for nome in nomes:
        mod, X2, _ = ajustar_modelo_com_delta(nome, X, grade, meta, tr, cfg, H, pre=pre)
        por_modelo[nome] = (mod.prob(X2.loc[te]), _info_dobra(mod))
    # baseline: melhor variável sozinha (AUC out-of-fold no treino da dobra)
    rel = pre.relatorio
    cand = rel[rel["auc_oof"].notna() & ~rel["vazamento"].astype(bool) & ~rel.index.str.endswith("__persistencia")]
    melhor = cand["auc_oof"].astype(float).idxmax()
    d = int(rel.at[melhor, "direcao"])
    med = X.loc[tr, melhor].median()
    lim, _ = limiar_youden(X.loc[tr, melhor].to_numpy(dtype=float), grade.loc[tr, "y"].to_numpy(), d)
    xb = X.loc[te, melhor].fillna(med)
    alarme = (xb >= lim) if d > 0 else (xb <= lim)
    return i, te[te].index, por_modelo, (d * xb).to_numpy(), alarme.to_numpy(), melhor


def dobras_externas(grade: pd.DataFrame, cfg: Config) -> list[tuple[int, pd.Index, pd.Index]]:
    """(i, clientes de treino, clientes de teste) — StratifiedKFold sobre os clientes, por cancelou."""
    cli = grade.groupby("cliente")["cancelado"].first()
    n_c, n_a = int(cli.sum()), int((~cli).sum())
    n = int(max(2, min(cfg.dobras, n_c, n_a)))
    skf = StratifiedKFold(n_splits=n, shuffle=True, random_state=cfg.semente)
    return [(i, cli.index[tr], cli.index[te]) for i, (tr, te) in enumerate(skf.split(cli.index, cli.to_numpy()))]


def validar(X: pd.DataFrame, grade: pd.DataFrame, meta: dict, cfg: Config, H: int, progresso=None,
            inicio: float = 0.1, fim: float = 0.8, modelos: list[str] | tuple[str, ...] = ("logistica",)) -> ValidacaoOOF:
    """Valida os `modelos` nas mesmas dobras externas. `oof.p`/`oof.dobras` = o primeiro da lista até `usar()`."""
    nomes = [m for m in modelos if m in MODELOS]
    if not nomes:
        raise ValueError(f"Nenhum modelo válido em {modelos!r}")
    tarefas = dobras_externas(grade, cfg)
    n = len(tarefas)
    ger = Parallel(n_jobs=cfg.n_jobs, return_as="generator_unordered")(
        delayed(_dobra)(i, X, grade, meta, tr, te, cfg, H, nomes) for i, tr, te in tarefas)
    res = []
    for feitos, r in enumerate(ger, start=1):
        res.append(r)
        if progresso:
            progresso(f"validação cruzada ({feitos}/{n} dobras, {len(nomes)} modelo(s))", inicio + (fim - inicio) * feitos / n)
    ps = {m: pd.Series(np.nan, index=grade.index, name="p_oof") for m in nomes}
    linhas = {m: [] for m in nomes}
    bs = pd.Series(np.nan, index=grade.index)
    ba = pd.Series(False, index=grade.index)
    bvars = []
    for i, idx, por_modelo, sb, ab, melhor in sorted(res, key=lambda r: r[0]):
        for m, (pp, info) in por_modelo.items():
            ps[m].loc[idx] = pp
            linhas[m].append({"dobra": i, **info})
        bs.loc[idx] = sb
        ba.loc[idx] = ab
        bvars.append(melhor)
    clientes = [list(te) for _, _, te in tarefas]
    resultados = {m: ResultadoModelo(m, ps[m], pd.DataFrame(linhas[m]).set_index("dobra"),
                                     log_loss_dobras(grade, ps[m], clientes), n) for m in nomes}
    primeiro = resultados[nomes[0]]
    oof = ValidacaoOOF(primeiro.p, bs, ba.astype(bool), primeiro.dobras, bvars, n, resultados, clientes)
    oof.escolhido = nomes[0]
    return oof
