"""Seleção supervisionada automática das variáveis (o núcleo do motor).

Tudo aqui usa só as linhas de treino recebidas — na validação, esta função roda dentro de cada dobra
(seleção aninhada), então o desempenho reportado não é otimista pela escolha das variáveis.

Etapas (parâmetros em `Config`)
-------------------------------
1. Filtro: descarta variável com > 50% de nulos, quase constante (≥ 98% das linhas no mesmo valor).
2. AUC univariada out-of-fold agrupada por cliente (GroupKFold 5): a direção de risco (+1: maior = mais
   risco, −1: menor = mais risco) é aprendida na parte de treino de cada dobra e aplicada na de teste;
   a direção final é a do treino inteiro. Nada é definido à mão.
3. Vazamento: AUC univariada (no treino) > 0,98; variável fixa por cliente com AUC por cliente
   (cancelou × ativo) > 0,98; ou separa bem só no último mês antes da saída (AUC ≥ 0,90 em k = 1 e
   ≤ 0,55 em k = 2…H) → excluída e marcada em pesos.json com o motivo.
4. Persistência: para as 5 variáveis temporais de maior AUC, limiar de Youden (arredondado) e contagem
   de meses seguidos, até t, do lado de risco do limiar → novas candidatas.
5. Redundância: |Spearman| > 0,9 entre duas candidatas → fica a de maior AUC out-of-fold.
6. Stability selection: 50 subamostras estratificadas de 50% dos clientes; em cada uma, regressão
   logística elastic-net (l1_ratio 0,5, saga, classes balanceadas, variáveis padronizadas; `l1_ratio=1`
   = L1 puro) com C escolhido por validação agrupada (AUC; `regra_1ep` opcional). O elastic-net divide o
   peso entre variáveis correlacionadas em vez de sortear uma só — frequências mais estáveis.
   Selecionada = escolhida em ≥ 60% das rodadas, com o mesmo sinal em ≥ 90% delas, e sinal igual à
   direção univariada (senão a explicação ficaria contraditória).
7. Limite: no máximo mín(meses-cliente positivos/5, clientes cancelados/2) variáveis (mínimo 2) — as de
   maior frequência. Com poucos cancelados, poucas variáveis: evita sobreajuste (o L2 final ainda encolhe
   os coeficientes).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from .config import Config
from .util import auc, numero_redondo

CS_L1 = np.logspace(-2.5, 1, 15)


# --------------------------------------------------------------------------- utilidades
def imputar(X: pd.DataFrame, medianas: pd.Series) -> np.ndarray:
    return X.fillna(medianas).fillna(0.0).to_numpy(dtype=float)


def padronizar(A: np.ndarray, media: np.ndarray, desvio: np.ndarray) -> np.ndarray:
    return (A - media) / desvio


def _desvio(A: np.ndarray) -> np.ndarray:
    d = A.std(axis=0)
    d[d == 0] = 1.0
    return d


def limiar_youden(x: np.ndarray, y: np.ndarray, direcao: int, arredondar: bool = True) -> tuple[float, float]:
    """(limiar, J) do corte que maximiza sensibilidade − (1 − especificidade) com alarme = direcao·x ≥ direcao·limiar.
    O limiar é arredondado para um número 'redondo' se isso preservar ≥ 90% do J."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y).astype(bool)
    m = ~np.isnan(x)
    x, y = x[m], y[m]
    if y.sum() == 0 or (~y).sum() == 0:
        return float("nan"), 0.0
    s = direcao * x
    cands = np.unique(s)
    if len(cands) > 400:
        cands = np.unique(np.quantile(s, np.linspace(0, 1, 401)))

    def J(c):
        a = s >= c
        return a[y].mean() - a[~y].mean()

    js = np.array([J(c) for c in cands])
    i = int(np.argmax(js))
    best, jbest = float(cands[i]), float(js[i])
    lim = direcao * best
    if not arredondar:
        return lim, jbest
    inteiro = np.allclose(x, np.round(x))
    escala = float(np.nanstd(x)) or abs(lim) or 1.0
    passo = numero_redondo(escala / 4)
    if inteiro:
        passo = max(1.0, round(passo))
    for p in (passo, passo / 2, passo / 5, passo / 10):
        cand = round(lim / p) * p
        if inteiro and p < 1:
            break
        if J(direcao * cand) >= 0.9 * jbest:
            return float(round(cand, 6)), float(J(direcao * cand))
    return float(lim), jbest


def meses_seguidos(valor: np.ndarray, cliente: np.ndarray, ordem: np.ndarray, limiar: float, direcao: int) -> np.ndarray:
    """Nº de meses seguidos (terminando em cada linha) com o valor do lado de risco do limiar.
    `ordem` ordena as linhas por (cliente, mês)."""
    v = valor[ordem]
    c = cliente[ordem]
    lado = np.where(np.isnan(v), False, (v >= limiar) if direcao > 0 else (v <= limiar))
    out = np.zeros(len(v))
    run = 0
    for i in range(len(v)):
        if i > 0 and c[i] != c[i - 1]:
            run = 0
        run = run + 1 if lado[i] else 0
        out[i] = run
    res = np.empty(len(v))
    res[ordem] = out
    return res


# --------------------------------------------------------------------------- resultado
@dataclass
class Selecao:
    selecionadas: list[str]
    relatorio: pd.DataFrame                   # uma linha por candidata
    persistencias: dict[str, dict]            # id → {base, limiar, direcao} (todas as criadas; o modelo usa as selecionadas)
    C_l1: float
    limite: int
    medianas: pd.Series = field(default_factory=pd.Series)


def adicionar_persistencias(X: pd.DataFrame, grade: pd.DataFrame, specs: dict[str, dict]) -> pd.DataFrame:
    """Colunas de persistência para todas as linhas do painel (cada uma usa só o passado do próprio cliente)."""
    if not specs:
        return X
    ordem = np.lexsort((grade["mes"].map(lambda p: p.ordinal).to_numpy(), grade["cliente"].to_numpy()))
    cli = grade["cliente"].to_numpy()
    novas = {pid: meses_seguidos(X[s["base"]].to_numpy(dtype=float), cli, ordem, s["limiar"], s["direcao"])
             for pid, s in specs.items() if pid not in X.columns}
    if not novas:
        return X
    return pd.concat([X, pd.DataFrame(novas, index=X.index)], axis=1)


def _auc_oof(A: np.ndarray, y: np.ndarray, grupos: np.ndarray, n_dobras: int) -> tuple[np.ndarray, np.ndarray]:
    """AUC out-of-fold agrupada por cliente com direção aprendida no treino de cada dobra."""
    n, p = A.shape
    sc = np.full((n, p), np.nan)
    gkf = GroupKFold(n_splits=min(n_dobras, len(np.unique(grupos))))
    for tr, te in gkf.split(A, y, grupos):
        for j in range(p):
            a = auc(y[tr], A[tr, j])
            d = 1.0 if (np.isnan(a) or a >= 0.5) else -1.0
            sc[te, j] = d * A[te, j]
    aucs = np.array([auc(y, sc[:, j]) for j in range(p)])
    aucs_in = np.array([auc(y, A[:, j]) for j in range(p)])
    return aucs, aucs_in


def _escolher_C_l1(Z: np.ndarray, y: np.ndarray, grupos: np.ndarray, cfg: Config) -> float:
    gkf = GroupKFold(n_splits=min(cfg.dobras_internas, len(np.unique(grupos))))
    res = np.full((len(CS_L1), gkf.get_n_splits()), np.nan)
    for f, (tr, te) in enumerate(gkf.split(Z, y, grupos)):
        if y[tr].sum() == 0 or y[te].sum() == 0 or y[te].sum() == len(te):
            continue
        for i, C in enumerate(CS_L1):
            lr = _esparsa(C, cfg).fit(Z[tr], y[tr])
            res[i, f] = auc(y[te], lr.decision_function(Z[te]))
    media = np.nanmean(res, axis=1)
    if np.all(np.isnan(media)):
        return float(CS_L1[len(CS_L1) // 2])
    ep = np.nanstd(res, axis=1) / np.sqrt(np.sum(~np.isnan(res), axis=1))
    i = int(np.nanargmax(media))
    if not cfg.regra_1ep:
        return float(CS_L1[i])
    ok = np.flatnonzero(media >= media[i] - ep[i])
    return float(CS_L1[ok.min()])


def _esparsa(C: float, cfg: Config) -> LogisticRegression:
    if cfg.l1_ratio >= 1:
        return LogisticRegression(l1_ratio=1, solver="liblinear", C=C, class_weight="balanced", max_iter=2000,
                                  random_state=cfg.semente)
    return LogisticRegression(l1_ratio=cfg.l1_ratio, solver="saga", C=C, class_weight="balanced", max_iter=5000,
                              random_state=cfg.semente,
                              tol=1e-3)


# --------------------------------------------------------------------------- seleção
def selecionar(X: pd.DataFrame, grade: pd.DataFrame, meta: dict, treino: pd.Series, cfg: Config,
               horizonte: int) -> tuple[Selecao, pd.DataFrame]:
    """Devolve (Selecao, X com as colunas de persistência acrescentadas)."""
    rng = np.random.default_rng(cfg.semente)
    T = X.loc[treino]
    y = grade.loc[treino, "y"].to_numpy().astype(int)
    grupos = grade.loc[treino, "cliente"].to_numpy()
    k = grade.loc[treino, "k"].to_numpy()
    canc_cli_linha = grade.loc[treino, "cancelado"].to_numpy().astype(bool)
    prim = ~pd.Series(grupos).duplicated().to_numpy()          # uma linha por cliente
    rel = pd.DataFrame(index=pd.Index(X.columns, name="id"))
    rel["motivo"] = None
    rel["vazamento"] = False

    def filtrar(cols):
        sub = T[cols]
        nulos = sub.isna().mean()
        moda = pd.Series({c: float(sub[c].value_counts(normalize=True, dropna=False).iloc[0]) if len(sub) else 1.0
                          for c in cols})
        rel.loc[cols, "nulos_pct"] = (100 * nulos).round(1)
        rel.loc[cols, "moda_pct"] = (100 * moda).round(1)
        ok = []
        for c in cols:
            if nulos[c] > cfg.max_nulos:
                rel.at[c, "motivo"] = f"muitos nulos ({100 * nulos[c]:.0f}% das linhas de treino)"
            elif sub[c].nunique() <= 1:
                rel.at[c, "motivo"] = "constante nas linhas de treino"
            else:
                ok.append(c)
        return ok

    def sem_quase_constantes(cols):
        """Depois do alerta de vazamento (um indicador raro que só aparece antes da saída é vazamento, não ruído)."""
        ok = []
        for c in cols:
            if rel.at[c, "moda_pct"] >= 100 * cfg.max_moda:
                rel.at[c, "motivo"] = f"quase constante ({rel.at[c, 'moda_pct']:.0f}% das linhas com o mesmo valor)"
            else:
                ok.append(c)
        return ok

    def avaliar(cols):
        med = T[cols].median()
        A = imputar(T[cols], med)
        a_oof, a_in = _auc_oof(A, y, grupos, cfg.dobras_internas)
        for j, c in enumerate(cols):
            d = 1 if (np.isnan(a_in[j]) or a_in[j] >= 0.5) else -1
            rel.at[c, "direcao"] = d
            rel.at[c, "auc_treino"] = max(a_in[j], 1 - a_in[j]) if not np.isnan(a_in[j]) else np.nan
            rel.at[c, "auc_oof"] = a_oof[j]
            # vazamento
            if rel.at[c, "auc_treino"] > cfg.auc_vazamento:
                rel.at[c, "vazamento"] = True
                rel.at[c, "motivo"] = (f"possível vazamento: AUC univariada {rel.at[c, 'auc_treino']:.3f} > "
                                       f"{cfg.auc_vazamento} (separa quase perfeitamente)")
            elif c in meta and not meta[c].temporal:
                # variável fixa por cliente: AUC com um valor por cliente (rótulo = cancelou em algum momento)
                ac = auc(canc_cli_linha[prim], A[prim, j])
                ac = max(ac, 1 - ac) if not np.isnan(ac) else np.nan
                if ac > cfg.auc_vazamento:
                    rel.at[c, "vazamento"] = True
                    rel.at[c, "motivo"] = (f"possível vazamento: separa cancelados de ativos quase perfeitamente "
                                           f"(AUC por cliente {ac:.3f} > {cfg.auc_vazamento})")
            if not rel.at[c, "vazamento"] and horizonte >= 2 and (c not in meta or meta[c].temporal):
                neg = y == 0
                p1, p2 = (y == 1) & (k == 1), (y == 1) & (k >= 2)
                if p1.sum() >= 3 and p2.sum() >= 3:
                    s = d * A[:, j]
                    a1 = auc(np.r_[np.ones(p1.sum()), np.zeros(neg.sum())], np.r_[s[p1], s[neg]])
                    a2 = auc(np.r_[np.ones(p2.sum()), np.zeros(neg.sum())], np.r_[s[p2], s[neg]])
                    if a1 >= 0.9 and a2 <= 0.55:
                        rel.at[c, "vazamento"] = True
                        rel.at[c, "motivo"] = (f"possível vazamento: só separa no último mês antes da saída "
                                               f"(AUC {a1:.2f} em −1 × {a2:.2f} em −2…−{horizonte})")
        return med

    cols = filtrar(list(X.columns))
    avaliar(cols)
    cols = sem_quase_constantes([c for c in cols if not rel.at[c, "vazamento"]])

    # persistência --------------------------------------------------------------
    specs: dict[str, dict] = {}
    temporais = [c for c in cols if meta[c].temporal and meta[c].transformacao not in ("meses_desde", "nunca")]
    top, usadas = [], set()          # a melhor transformação de cada coluna de origem (diversifica)
    for c in sorted(temporais, key=lambda c: (-rel.at[c, "auc_oof"], cols.index(c))):
        origem = (meta[c].tabela, meta[c].coluna, meta[c].categoria)
        if origem not in usadas and len(top) < cfg.n_persistencia:
            usadas.add(origem)
            top.append(c)
    for c in top:
        if rel.at[c, "auc_oof"] < 0.55:
            continue
        d = int(rel.at[c, "direcao"])
        lim, _ = limiar_youden(T[c].to_numpy(dtype=float), y, d)
        if np.isnan(lim):
            continue
        specs[f"{c}__persistencia"] = {"base": c, "limiar": lim, "direcao": d}
    if specs:
        X = adicionar_persistencias(X, grade, specs)
        T = X.loc[treino]
        novas = list(specs)
        for pid in novas:
            rel.loc[pid, ["motivo", "vazamento"]] = [None, False]
        ok = filtrar(novas)
        avaliar(ok)
        cols += sem_quase_constantes([c for c in ok if not rel.at[c, "vazamento"]])

    # redundância ---------------------------------------------------------------
    med = T[cols].median()
    A = imputar(T[cols], med)
    R = pd.DataFrame(A).rank().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = np.corrcoef(R, rowvar=False)
    rho = np.nan_to_num(np.atleast_2d(rho))
    ordem = sorted(range(len(cols)), key=lambda j: (-rel.at[cols[j], "auc_oof"], j))
    manter: list[int] = []
    for j in ordem:
        par = next((i for i in manter if abs(rho[i, j]) > cfg.max_spearman), None)
        if par is None:
            manter.append(j)
        else:
            rel.at[cols[j], "motivo"] = f"redundante com '{meta_rotulo(meta, specs, cols[par])}' (|Spearman| = {abs(rho[par, j]):.2f})"
    cols = [cols[j] for j in sorted(manter)]

    # stability selection --------------------------------------------------------
    med = T[cols].median()
    A = imputar(T[cols], med)
    Z = padronizar(A, A.mean(axis=0), _desvio(A))
    C = _escolher_C_l1(Z, y, grupos, cfg)
    cli_unicos = np.array(sorted(set(grupos)))
    canc_cli = pd.Series(y, index=grupos).groupby(level=0).max().reindex(cli_unicos).to_numpy().astype(bool)
    escolhas = np.zeros((cfg.rodadas, len(cols)))
    sinais = np.zeros((cfg.rodadas, len(cols)))
    for b in range(cfg.rodadas):
        amostra = []
        for grupo in (canc_cli, ~canc_cli):
            ids = cli_unicos[grupo]
            n = max(1, int(round(cfg.frac_subamostra * len(ids))))
            amostra.extend(rng.choice(ids, size=n, replace=False))
        m = np.isin(grupos, amostra)
        if y[m].sum() == 0:
            continue
        Zb = Z[m]
        Zb = (Zb - Zb.mean(axis=0)) / _desvio(Zb)
        lr = _esparsa(C, cfg).fit(Zb, y[m])
        co = lr.coef_[0]
        escolhas[b] = co != 0
        sinais[b] = np.sign(co)
    freq = escolhas.mean(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        pos = (sinais > 0).sum(axis=0) / escolhas.sum(axis=0)
    consist = np.where(escolhas.sum(axis=0) > 0, np.maximum(pos, 1 - pos), np.nan)
    sinal_l1 = np.where(pos >= 0.5, 1, -1)
    for j, c in enumerate(cols):
        rel.at[c, "frequencia_selecao"] = freq[j]
        rel.at[c, "consistencia_sinal"] = consist[j]
        rel.at[c, "sinal_l1"] = sinal_l1[j] if freq[j] > 0 else np.nan

    n_pos = int(y.sum())
    n_canc = int(canc_cli.sum())
    limite = max(2, min(n_pos // cfg.linhas_por_variavel, n_canc // cfg.clientes_por_variavel))
    elegiveis = []
    for j, c in enumerate(cols):
        if freq[j] < cfg.limiar_frequencia:
            rel.at[c, "motivo"] = f"escolhida em só {100 * freq[j]:.0f}% das rodadas (mínimo {100 * cfg.limiar_frequencia:.0f}%)"
        elif consist[j] < cfg.consistencia_sinal:
            rel.at[c, "motivo"] = f"sinal instável entre as rodadas ({100 * consist[j]:.0f}% no sinal majoritário)"
        elif sinal_l1[j] != rel.at[c, "direcao"]:
            rel.at[c, "motivo"] = "no modelo conjunto o efeito tem direção oposta à univariada (efeito de supressão)"
        else:
            elegiveis.append(c)
    elegiveis.sort(key=lambda c: (-rel.at[c, "frequencia_selecao"], -rel.at[c, "auc_oof"], cols.index(c)))
    for c in elegiveis[limite:]:
        rel.at[c, "motivo"] = f"fora do limite de {limite} variáveis (frequência {100 * rel.at[c, 'frequencia_selecao']:.0f}%)"
    sel = elegiveis[:limite]
    if not sel and cols:           # nenhuma estável: fica a de maior frequência (ou AUC)
        melhor = sorted(cols, key=lambda c: (-rel.at[c, "frequencia_selecao"], -rel.at[c, "auc_oof"]))[0]
        rel.at[melhor, "motivo"] = None
        sel = [melhor]
    for c in sel:
        rel.at[c, "motivo"] = None
    rel["selecionada"] = rel.index.isin(sel)
    sel = [c for c in X.columns if c in sel]           # ordem estável
    return Selecao(sel, rel, specs, C, limite, T[sel].median() if sel else pd.Series(dtype=float)), X


def meta_rotulo(meta: dict, specs: dict, vid: str) -> str:
    if vid in meta:
        return meta[vid].rotulo
    if vid in specs:
        return rotulo_persistencia(meta, specs[vid])
    return vid


def rotulo_persistencia(meta: dict, spec: dict) -> str:
    return f"Meses seguidos com {meta[spec['base']].rotulo.lower()} na zona de risco"
