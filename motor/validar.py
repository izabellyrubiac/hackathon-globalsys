"""Validação out-of-fold agrupada por cliente, com a seleção de variáveis ANINHADA.

Em cada dobra externa (StratifiedKFold sobre os clientes, estratificado por cancelou), o motor inteiro —
filtro, AUC, persistência, redundância, stability selection, C do L2, limiares — roda só com os clientes
de treino; os clientes de teste recebem probabilidade em todos os seus meses. Assim nenhuma decisão
enxergou o cliente avaliado.

Métricas (eixo k = meses até a saída; ativos: k = 1 é o mês de referência)
--------------------------------------------------------------------------
* detecção dos cancelados em −1/−2/−3 (probabilidade ≥ corte no mês −k);
* alarmes falsos: ativos com alarme no mês de referência e em algum dos últimos 12 meses;
  % dos meses-cliente dos ativos em alarme;
* antecedência mediana (alarme contínuo até −1);
* AUC Cancelou × Ativo por mês (−1 … −6), precisão no topo (top 10/20);
* baseline "melhor variável sozinha": em cada dobra, a candidata de maior AUC out-of-fold no treino,
  orientada pela direção aprendida, com o limiar de Youden do treino.
Os cortes das faixas são escolhidos sobre essas probabilidades out-of-fold (ver `escolher_cortes`) —
isso é a única decisão tomada depois de ver o resultado da validação; ela é leve (2 números) mas deixa
os números das faixas um pouco otimistas.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold

from .config import Config
from .modelo import ajustar
from .selecao import imputar, limiar_youden
from .util import auc

KS = (1, 2, 3)


@dataclass
class ValidacaoOOF:
    p: pd.Series                   # probabilidade out-of-fold (NaN fora das linhas pontuáveis)
    base_score: pd.Series          # melhor variável sozinha (orientada: maior = mais risco)
    base_alarme: pd.Series         # melhor variável ≥ limiar de Youden do treino
    dobras: pd.DataFrame           # dobra × variável: coeficiente (NaN = não selecionada)
    base_vars: list[str]           # melhor variável escolhida em cada dobra
    n_dobras: int


def _dobra(i, X, grade, meta, treino_cli, teste_cli, cfg, H):
    tr = grade["treino"] & grade["cliente"].isin(treino_cli)
    te = grade["pontuavel"] & grade["cliente"].isin(teste_cli)
    mod, X2 = ajustar(X, grade, meta, tr, cfg, H)
    p = mod.prob(X2.loc[te])
    # baseline: melhor variável sozinha (AUC out-of-fold no treino da dobra)
    rel = mod.selecao.relatorio
    cand = rel[rel["auc_oof"].notna() & ~rel["vazamento"].astype(bool) & ~rel.index.str.endswith("__persistencia")]
    melhor = cand["auc_oof"].astype(float).idxmax()
    d = int(rel.at[melhor, "direcao"])
    med = X.loc[tr, melhor].median()
    lim, _ = limiar_youden(X.loc[tr, melhor].to_numpy(dtype=float), grade.loc[tr, "y"].to_numpy(), d)
    xb = X.loc[te, melhor].fillna(med)
    alarme = (xb >= lim) if d > 0 else (xb <= lim)
    coefs = dict(zip(mod.features, mod.coef))
    return i, te[te].index, p, (d * xb).to_numpy(), alarme.to_numpy(), coefs, melhor


def validar(X: pd.DataFrame, grade: pd.DataFrame, meta: dict, cfg: Config, H: int, progresso=None,
            inicio: float = 0.1, fim: float = 0.8) -> ValidacaoOOF:
    cli = grade.groupby("cliente")["cancelado"].first()
    n_c, n_a = int(cli.sum()), int((~cli).sum())
    n = int(max(2, min(cfg.dobras, n_c, n_a)))
    skf = StratifiedKFold(n_splits=n, shuffle=True, random_state=cfg.semente)
    tarefas = [(i, cli.index[tr], cli.index[te]) for i, (tr, te) in enumerate(skf.split(cli.index, cli.to_numpy()))]
    p = pd.Series(np.nan, index=grade.index, name="p_oof")
    bs = pd.Series(np.nan, index=grade.index)
    ba = pd.Series(False, index=grade.index)
    linhas, bvars = [], []
    ger = Parallel(n_jobs=cfg.n_jobs, return_as="generator_unordered")(
        delayed(_dobra)(i, X, grade, meta, tr, te, cfg, H) for i, tr, te in tarefas)
    feitos = 0
    res = []
    for r in ger:
        res.append(r)
        feitos += 1
        if progresso:
            progresso(f"validação cruzada ({feitos}/{n} dobras)", inicio + (fim - inicio) * feitos / n)
    for i, idx, pp, sb, ab, coefs, melhor in sorted(res, key=lambda r: r[0]):
        p.loc[idx] = pp
        bs.loc[idx] = sb
        ba.loc[idx] = ab
        linhas.append({"dobra": i, **coefs})
        bvars.append(melhor)
    dob = pd.DataFrame(linhas).set_index("dobra")
    return ValidacaoOOF(p, bs, ba.astype(bool), dob, bvars, n)


# --------------------------------------------------------------------------- métricas
def desempenho(grade: pd.DataFrame, alarme: pd.Series, ks=KS, janela_hist: int = 12) -> dict:
    """Detecção por k nos cancelados; alarmes falsos nos ativos (hoje, em algum mês, % dos meses)."""
    x = grade[["cliente", "k", "cancelado", "pontuavel"]].assign(a=alarme.fillna(False).astype(bool) & grade["pontuavel"])
    canc, ativ = x[x["cancelado"]], x[~x["cancelado"]]
    out = {f"canc_k{k}": int(canc[canc["k"] == k]["a"].sum()) for k in ks}
    out["ativos_hoje"] = int(ativ[ativ["k"] == 1]["a"].sum())
    janela = ativ[(ativ["k"] <= janela_hist) & ativ["pontuavel"]]
    out["ativos_algum_mes"] = int(janela.groupby("cliente")["a"].any().sum())
    out["meses_alarme_ativos_pct"] = float(100 * janela["a"].mean()) if len(janela) else float("nan")
    ant = antecedencia(grade, alarme)
    a = ant[ant.index.isin(canc["cliente"]) & (ant > 0)]
    out["antecedencia_mediana"] = float(a.median()) if len(a) else 0.0
    return out


def antecedencia(grade: pd.DataFrame, alarme: pd.Series) -> pd.Series:
    """Por cliente: maior k com alarme em todos os meses −k … −1 (0 se sem alarme em −1)."""
    a = grade[["cliente", "k"]].assign(a=alarme.fillna(False).astype(bool).to_numpy())
    piv = a.pivot_table(index="cliente", columns="k", values="a", aggfunc="max", fill_value=False)
    piv = piv.reindex(columns=sorted(piv.columns)).astype(bool)
    return piv.cumprod(axis=1).sum(axis=1).rename("antecedencia")


def curva(grade: pd.DataFrame, p: pd.Series, grade_cortes=None) -> pd.DataFrame:
    """Para cada corte: `desempenho` + precisão/recall/taxa de alarme falso nas linhas de treino."""
    if grade_cortes is None:
        grade_cortes = np.round(np.arange(0.01, 0.96, 0.01), 2)
    t = grade["treino"] & p.notna()
    y = grade["y"]
    n1, n0 = int((t & (y == 1)).sum()), int((t & (y == 0)).sum())
    linhas = []
    for c in grade_cortes:
        a = (p >= c) & grade["pontuavel"]
        r = desempenho(grade, a)
        vp, fp = int((a & t & (y == 1)).sum()), int((a & t & (y == 0)).sum())
        rec = vp / n1 if n1 else np.nan
        fpr = fp / n0 if n0 else np.nan
        linhas.append({"corte": float(c), **r, "precisao_linhas": vp / (vp + fp) if vp + fp else np.nan,
                       "recall_linhas": rec, "fpr_linhas": fpr, "youden_linhas": rec - fpr})
    return pd.DataFrame(linhas)


def escolher_cortes(cv: pd.DataFrame) -> dict:
    """Cortes das faixas (regras genéricas; nenhuma depende da capacidade da equipe):

    * alto    = menor corte com precisão ≥ 50% nos meses-cliente de treino marcados (out-of-fold):
                pelo menos metade dos meses marcados como "alto" terminou em cancelamento nos H meses
                seguintes. Se nenhum corte chega a 50%, usa o de maior precisão.
    * atencao = corte de Youden nas linhas de treino (out-of-fold): maximiza sensibilidade − taxa de
                alarme falso por mês-cliente; empate → o corte mais alto. Se ficar ≥ alto, atencao = alto/2.
    """
    ok = cv[cv["precisao_linhas"] >= 0.5]
    alto = float(ok["corte"].min()) if len(ok) else float(cv.loc[cv["precisao_linhas"].idxmax(), "corte"])
    j = cv["youden_linhas"]
    youden = float(cv.loc[np.isclose(j, j.max()), "corte"].max())
    atencao = youden if youden < alto else round(alto / 2, 2)
    return {"alto": alto, "atencao": atencao, "youden": youden}


def faixa(p, cortes: dict) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return np.select([p >= cortes["alto"], p >= cortes["atencao"]], ["alto", "atencao"], "baixo")


def auc_por_mes(grade: pd.DataFrame, scores: dict[str, pd.Series], ks=range(1, 7)) -> pd.DataFrame:
    """AUC Cancelou × Ativo com o score no mês −k (um valor por cliente)."""
    linhas = []
    for k in ks:
        m = (grade["k"] == k) & grade["pontuavel"]
        y = grade.loc[m, "cancelado"].to_numpy()
        for nome, s in scores.items():
            v = s[m].to_numpy(dtype=float)
            ok = ~np.isnan(v)
            linhas.append({"k": int(k), "score": nome, "auc": auc(y[ok], v[ok]), "n": int(ok.sum()),
                           "n_canc": int(y[ok].sum())})
    return pd.DataFrame(linhas)


def precisao_topo(grade: pd.DataFrame, s: pd.Series, ks=KS, ns=(10, 20)) -> pd.DataFrame:
    """Cancelados entre os N primeiros quando todos os clientes são ordenados pelo score no mês −k."""
    linhas = []
    for k in ks:
        m = (grade["k"] == k) & grade["pontuavel"] & s.notna()
        q = grade.loc[m, ["cliente", "cancelado"]].assign(s=s[m]).sort_values(["s", "cliente"], ascending=[False, True])
        for n in ns:
            linhas.append({"k": int(k), "top": int(n), "cancelados": int(q.head(n)["cancelado"].sum())})
    return pd.DataFrame(linhas)
