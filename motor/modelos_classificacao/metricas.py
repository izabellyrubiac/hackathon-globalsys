"""Métricas de qualidade dos modelos, todas sobre probabilidades FORA DA AMOSTRA (validação agrupada por cliente).

Eixo k = meses até a saída; ativos: k = 1 é o mês de referência.

* `log_loss_dobras` / `comparar_modelos`: log-loss nas linhas de treino dos clientes de teste de cada dobra
  (mede calibração e separação ao mesmo tempo — é o critério da escolha do modelo), média ± erro-padrão
  entre dobras, AUC nas linhas e AUC Cancelou × Ativo em −1/−2/−3;
* `desempenho`: detecção dos cancelados em −1/−2/−3 (probabilidade ≥ corte no mês −k); alarmes falsos:
  ativos com alarme no mês de referência e em algum dos últimos 12 meses; % dos meses-cliente dos ativos
  em alarme; antecedência mediana (alarme contínuo até −1);
* `curva` + `escolher_cortes` + `faixa`: cortes das faixas alto/atenção (a única decisão tomada depois de
  ver a validação: 2 números, leve otimismo nas contagens por faixa);
* `auc_por_mes` (−1 … −6) e `precisao_topo` (top 10/20).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..modelos_previsao.base import log_loss
from ..util import auc

KS = (1, 2, 3)


# --------------------------------------------------------------------------- detecção × alarme falso
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


# --------------------------------------------------------------------------- cortes das faixas
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


# --------------------------------------------------------------------------- separação
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


# --------------------------------------------------------------------------- comparação de modelos
def log_loss_dobras(grade: pd.DataFrame, p: pd.Series, clientes_dobra: list) -> np.ndarray:
    """Log-loss de cada dobra externa: linhas de treino (rótulo conhecido) dos clientes de teste da dobra."""
    out = []
    for cli in clientes_dobra:
        m = grade["treino"] & grade["cliente"].isin(cli) & p.notna()
        out.append(log_loss(grade.loc[m, "y"].to_numpy(), p[m].to_numpy()) if m.any() else np.nan)
    return np.asarray(out, dtype=float)


def erro_padrao(v: np.ndarray) -> float:
    v = np.asarray(v, dtype=float)
    v = v[~np.isnan(v)]
    return float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")


def comparar_modelos(grade: pd.DataFrame, oof) -> pd.DataFrame:
    """Uma linha por modelo validado: log-loss (média das dobras ± erro-padrão e no conjunto), AUC nas
    linhas de treino e AUC Cancelou × Ativo em −1/−2/−3 — tudo fora da amostra, nas MESMAS dobras."""
    linhas = []
    t = grade["treino"]
    for nome, res in oof.modelos.items():
        p = res.p
        m = t & p.notna()
        a = auc_por_mes(grade, {nome: p}, ks=KS).set_index("k")["auc"]
        ll = res.log_loss_dobras
        linhas.append({"modelo": nome, "log_loss": float(np.nanmean(ll)), "log_loss_ep": erro_padrao(ll),
                       "log_loss_conjunto": log_loss(grade.loc[m, "y"].to_numpy(), p[m].to_numpy()),
                       "auc_linhas": auc(grade.loc[m, "y"].to_numpy(), p[m].to_numpy()),
                       **{f"auc_k{k}": float(a.get(k, np.nan)) for k in KS}})
    return pd.DataFrame(linhas).set_index("modelo") if linhas else pd.DataFrame()


def diferenca_pareada(oof, a: str, b: str) -> tuple[float, float]:
    """(média, erro-padrão) de log-loss(a) − log-loss(b) pareado por dobra (negativo = a melhor)."""
    d = oof.modelos[a].log_loss_dobras - oof.modelos[b].log_loss_dobras
    return float(np.nanmean(d)), erro_padrao(d)
