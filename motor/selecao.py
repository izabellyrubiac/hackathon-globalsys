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
7. Consistência entre dobras (só no modelo final): com a validação ligada, quem foi selecionada em menos de
   `min_frac_dobras` (50%) das dobras externas vai para o fim da fila do limite (`modo_dobras="priorizar"`:
   só entra se sobrar vaga; `"excluir"`: nunca entra) — a vaga vai para a próxima estável.
8. Limite: no máximo mín(meses-cliente positivos/5, clientes cancelados/2) variáveis (mínimo 2) — as de
   maior frequência. Com poucos cancelados, poucas variáveis: evita sobreajuste (o L2 final ainda encolhe
   os coeficientes).

As etapas 1–5 (e o pré-filtro de escala: com mais de `max_candidatas` (300), só as `pre_filtro_k` (100)
de maior AUC univariada seguem)
ficam em `pre_selecao()`, compartilhada com a seleção das árvores (`selecao_arvores.py`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from .config import Config
from .util import auc, auc_colunas, numero_redondo

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
    `ordem` ordena as linhas por (cliente, mês). Vetorizado: conta desde a última quebra (valor fora da
    zona de risco ou início do cliente)."""
    v = valor[ordem]
    c = cliente[ordem]
    n = len(v)
    if n == 0:
        return np.zeros(0)
    lado = np.where(np.isnan(v), False, (v >= limiar) if direcao > 0 else (v <= limiar))
    idx = np.arange(n)
    inicio = np.r_[True, c[1:] != c[:-1]]
    quebra = np.maximum(np.where(~lado, idx, -1), np.where(inicio, idx - 1, -1))
    out = (idx - np.maximum.accumulate(quebra)).astype(float)
    res = np.empty(n)
    res[ordem] = out
    return res


# --------------------------------------------------------------------------- resultado
@dataclass
class Selecao:
    selecionadas: list[str]
    relatorio: pd.DataFrame                   # uma linha por candidata
    persistencias: dict[str, dict]            # id → {base, limiar, direcao} (todas as criadas; o modelo usa as selecionadas)
    C_l1: float | None                        # C da stability selection (None nas árvores)
    limite: int
    medianas: pd.Series = field(default_factory=pd.Series)
    metodo: str = "stability"                 # "stability" (logística) | "sombras" (árvores, estilo Boruta)


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
        a = auc_colunas(y[tr], A[tr])
        d = np.where(np.isnan(a) | (a >= 0.5), 1.0, -1.0)
        sc[te] = d * A[te]
    return auc_colunas(y, sc), auc_colunas(y, A)


def _paralelo(n_jobs: int) -> Parallel:
    """Threads (o solver do scikit-learn libera o GIL): cada ajuste é independente e determinístico, então o
    resultado não depende de `n_jobs`. Dentro das dobras da validação `n_jobs = 1` (as dobras já são paralelas)."""
    return Parallel(n_jobs=n_jobs, prefer="threads")


def validar_pesos_colunas(pesos) -> dict[str, float]:
    """Pesos por coluna do usuário: {"tabela.coluna" | "coluna": 0–1}. ValueError em pt-BR."""
    out = {}
    for k, v in (pesos or {}).items():
        try:
            f = float(v)
        except (TypeError, ValueError):
            raise ValueError(f"O peso de '{k}' precisa ser um número entre 0 e 1.") from None
        if not (0.0 <= f <= 1.0):
            raise ValueError(f"O peso de '{k}' precisa estar entre 0 e 1 (recebi {v}).")
        out[str(k)] = f
    return out


def peso_da_coluna(tabela: str, coluna: str, cfg: Config) -> float:
    """Peso 0–1 dado pelo usuário à coluna (1 = normal, 0 = ignorar). `tabela.coluna` vence `coluna`."""
    p = cfg.pesos_colunas or {}
    for chave in (f"{tabela}.{coluna}", coluna):
        if chave in p:
            return float(p[chave])
    return 1.0


def peso_da_variavel(vid: str, meta: dict, specs: dict, cfg: Config) -> float:
    """Peso da variável `vid`; a persistência herda o da coluna da variável-base."""
    v = meta.get(vid) or (meta.get(specs[vid]["base"]) if vid in specs else None)
    return 1.0 if v is None else peso_da_coluna(v.tabela, v.coluna, cfg)


def _auc_C(Z, y, tr, te, C, cfg) -> float:
    lr = _esparsa(C, cfg).fit(Z[tr], y[tr])
    return auc(y[te], lr.decision_function(Z[te]))


def _escolher_C_l1(Z: np.ndarray, y: np.ndarray, grupos: np.ndarray, cfg: Config, n_jobs: int = 1) -> float:
    gkf = GroupKFold(n_splits=min(cfg.dobras_internas, len(np.unique(grupos))))
    res = np.full((len(CS_L1), gkf.get_n_splits()), np.nan)
    tarefas = [(i, f, tr, te, C) for f, (tr, te) in enumerate(gkf.split(Z, y, grupos))
               if not (y[tr].sum() == 0 or y[te].sum() == 0 or y[te].sum() == len(te))
               for i, C in enumerate(CS_L1)]
    aucs = _paralelo(n_jobs)(delayed(_auc_C)(Z, y, tr, te, C, cfg) for _, _, tr, te, C in tarefas)
    for (i, f, *_), a in zip(tarefas, aucs):
        res[i, f] = a
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


# --------------------------------------------------------------------------- pré-seleção (comum)
@dataclass
class PreSelecao:
    """Resultado das etapas comuns a todos os modelos (filtro, AUC, vazamento, persistência, pré-filtro de
    escala e redundância), calculado uma vez por conjunto de treino. `relatorio` é copiado por quem o usa."""
    relatorio: pd.DataFrame
    colunas: list[str]                       # candidatas que seguem para a seleção do modelo
    persistencias: dict[str, dict]
    X: pd.DataFrame                          # X com as persistências
    treino: pd.Series
    limite: int
    n_pos: int
    n_canc: int


def limite_variaveis(n_pos: int, n_canc: int, cfg: Config) -> int:
    return max(2, min(n_pos // cfg.linhas_por_variavel, n_canc // cfg.clientes_por_variavel))


def amostrar_clientes(rng: np.random.Generator, cli_unicos: np.ndarray, canc_cli: np.ndarray, cfg: Config) -> list:
    """Subamostra estratificada de clientes (fração `frac_subamostra` de cada grupo). Em base grande
    (`razao_ativos_rodada`), os ativos da rodada ficam limitados a N por cancelado sorteado."""
    canc, ativ = cli_unicos[canc_cli], cli_unicos[~canc_cli]
    n_c = min(len(canc), max(1, int(round(cfg.frac_subamostra * len(canc)))))
    n_a = min(len(ativ), max(1, int(round(cfg.frac_subamostra * len(ativ)))))
    if cfg.razao_ativos_rodada:
        n_a = min(n_a, max(1, int(cfg.razao_ativos_rodada * n_c)))
    return [*rng.choice(canc, size=n_c, replace=False), *rng.choice(ativ, size=n_a, replace=False)]


def pre_selecao(X: pd.DataFrame, grade: pd.DataFrame, meta: dict, treino: pd.Series, cfg: Config,
                horizonte: int) -> PreSelecao:
    """Etapas 1–5 da seleção (comuns à logística e às árvores), só com as linhas `treino`."""
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
        nunicos = sub.nunique()
        ok = []
        for c in cols:
            if c in meta and peso_da_coluna(meta[c].tabela, meta[c].coluna, cfg) == 0:
                rel.at[c, "motivo"] = "peso 0 definido pelo usuário"
            elif nulos[c] > cfg.max_nulos:
                rel.at[c, "motivo"] = f"muitos nulos ({100 * nulos[c]:.0f}% das linhas de treino)"
            elif nunicos[c] <= 1:
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
        neg = y == 0
        p1, p2 = (y == 1) & (k == 1), (y == 1) & (k >= 2)
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

    # pré-filtro de escala -----------------------------------------------------------
    if len(cols) > cfg.max_candidatas:
        k_top = min(cfg.pre_filtro_k, cfg.max_candidatas)
        ordem = sorted(cols, key=lambda c: (-rel.at[c, "auc_oof"], cols.index(c)))
        fora = set(ordem[k_top:])
        for c in fora:
            rel.at[c, "motivo"] = (f"pré-filtro de escala ({len(cols)} candidatas > {cfg.max_candidatas}): fora das "
                                   f"{k_top} de maior AUC univariada")
        cols = [c for c in cols if c not in fora]

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

    n_pos = int(y.sum())
    n_canc = int(pd.Series(y, index=grupos).groupby(level=0).max().sum())
    return PreSelecao(rel, cols, specs, X, treino, limite_variaveis(n_pos, n_canc, cfg), n_pos, n_canc)


def motivo_dobras(freq: float, cfg: Config) -> str:
    return (f"selecionada em só {100 * freq:.0f}% das dobras da validação (mínimo "
            f"{100 * cfg.min_frac_dobras:.0f}%): instável entre dobras")


def aplicar_limite(rel: pd.DataFrame, elegiveis: list[str], cols: list[str], limite: int, chave,
                   freq_dobras: pd.Series | None, cfg: Config) -> list[str]:
    """Consistência entre dobras (se houver `freq_dobras`) → ordena por `chave` → corta no limite.

    `modo_dobras="priorizar"`: as variáveis presentes em ≥ `min_frac_dobras` das dobras externas vêm antes
    no limite; uma inconsistente só entra se sobrar vaga. `"excluir"`: inconsistente nunca entra."""
    fd = (lambda c: float(freq_dobras.get(c, 0.0))) if freq_dobras is not None else (lambda c: 1.0)
    instavel = lambda c: fd(c) < cfg.min_frac_dobras  # noqa: E731
    if freq_dobras is not None and cfg.modo_dobras == "excluir":
        for c in elegiveis:
            if instavel(c):
                rel.at[c, "motivo"] = motivo_dobras(fd(c), cfg)
        elegiveis = [c for c in elegiveis if not instavel(c)]
    elegiveis = sorted(elegiveis, key=lambda c: (instavel(c), chave(c)))
    for c in elegiveis[limite:]:
        rel.at[c, "motivo"] = (motivo_dobras(fd(c), cfg) + f" e sem vaga no limite de {limite} variáveis" if instavel(c)
                               else f"fora do limite de {limite} variáveis (frequência "
                                    f"{100 * rel.at[c, 'frequencia_selecao']:.0f}%)")
    sel = elegiveis[:limite]
    if not sel and cols:           # nenhuma estável: fica a mais consistente entre dobras, depois a mais frequente
        melhor = sorted(cols, key=lambda c: (-fd(c), -np.nan_to_num(rel.at[c, "frequencia_selecao"]),
                                             -rel.at[c, "auc_oof"]))[0]
        sel = [melhor]
    for c in sel:
        rel.at[c, "motivo"] = None
    rel["selecionada"] = rel.index.isin(sel)
    return sel


# --------------------------------------------------------------------------- seleção (logística)
def _coef_rodada(Zb: np.ndarray, yb: np.ndarray, C: float, cfg: Config, w: np.ndarray | None = None) -> np.ndarray:
    Zb = (Zb - Zb.mean(axis=0)) / _desvio(Zb)
    if w is not None:
        Zb = Zb * w            # peso do usuário: escala da penalização (menor peso = coeficiente mais caro)
    return _esparsa(C, cfg).fit(Zb, yb).coef_[0]


def selecionar(X: pd.DataFrame, grade: pd.DataFrame, meta: dict, treino: pd.Series, cfg: Config,
               horizonte: int, pre: PreSelecao | None = None, freq_dobras: pd.Series | None = None,
               n_jobs: int = 1) -> tuple[Selecao, pd.DataFrame]:
    """Devolve (Selecao, X com as colunas de persistência acrescentadas). `pre` reaproveita a pré-seleção
    já calculada nas mesmas linhas; `freq_dobras` (fração das dobras externas em que cada variável entrou)
    liga a consistência entre dobras (modelo final)."""
    pre = pre or pre_selecao(X, grade, meta, treino, cfg, horizonte)
    rng = np.random.default_rng(cfg.semente)
    X = pre.X
    rel = pre.relatorio.copy()
    cols = list(pre.colunas)
    T = X.loc[treino]
    y = grade.loc[treino, "y"].to_numpy().astype(int)
    grupos = grade.loc[treino, "cliente"].to_numpy()

    # stability selection --------------------------------------------------------
    med = T[cols].median()
    A = imputar(T[cols], med)
    Z = padronizar(A, A.mean(axis=0), _desvio(A))
    w = np.array([peso_da_variavel(c, meta, pre.persistencias, cfg) for c in cols])
    C = _escolher_C_l1(Z * w, y, grupos, cfg, n_jobs)
    cli_unicos = np.array(sorted(set(grupos)))
    canc_cli = pd.Series(y, index=grupos).groupby(level=0).max().reindex(cli_unicos).to_numpy().astype(bool)
    escolhas = np.zeros((cfg.rodadas, len(cols)))
    sinais = np.zeros((cfg.rodadas, len(cols)))
    mascaras = [np.isin(grupos, amostrar_clientes(rng, cli_unicos, canc_cli, cfg)) for _ in range(cfg.rodadas)]
    rodadas = [(b, m) for b, m in enumerate(mascaras) if y[m].sum() > 0]
    coefs = _paralelo(n_jobs)(delayed(_coef_rodada)(Z[m], y[m], C, cfg, w) for _, m in rodadas)
    for (b, _), co in zip(rodadas, coefs):
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
    sel = aplicar_limite(rel, elegiveis, cols, pre.limite,
                         lambda c: (-rel.at[c, "frequencia_selecao"], -rel.at[c, "auc_oof"], cols.index(c)),
                         freq_dobras, cfg)
    sel = [c for c in X.columns if c in sel]           # ordem estável
    return Selecao(sel, rel, pre.persistencias, C, pre.limite, T[sel].median() if sel else pd.Series(dtype=float)), X


def meta_rotulo(meta: dict, specs: dict, vid: str) -> str:
    if vid in meta:
        return meta[vid].rotulo
    if vid in specs:
        return rotulo_persistencia(meta, specs[vid])
    return vid


def rotulo_persistencia(meta: dict, spec: dict) -> str:
    return f"Meses seguidos com {meta[spec['base']].rotulo.lower()} na zona de risco"
