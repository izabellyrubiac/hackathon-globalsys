"""Candidata **delta de risco**: o motor em dois estágios (`Config.delta_risco = True`).

Ideia: o nível do risco já é o que a fila usa; o que ele não diz é se o risco **subiu**. O delta é a
variação da própria probabilidade prevista de um mês para o outro, oferecida como mais uma candidata —
a seleção decide se ela fica.

Dois estágios, refeitos **dentro de cada dobra externa** e também no treino final:

1. **Estágio 1** — treina o modelo normal nas linhas de treino.
2. Prevê a probabilidade de todas as linhas pontuáveis (treino e teste da dobra).
3. Cria as candidatas `risco_previsto__delta` = p(t) − p(t−1) do mesmo cliente e (se `delta_risco_3m`)
   `risco_previsto__delta_3m` = p(t) − média dos 3 meses anteriores.
4. **Estágio 2** — refaz a pré-seleção, a seleção e o ajuste com essas candidatas entre as demais. Elas
   passam pelos mesmos filtros (nulos, redundância, vazamento, estabilidade, limite) e podem ser
   descartadas.

**Primeiro mês do cliente**: sem mês anterior, o delta fica **nulo** (NaN) — não é 0, porque "não houve
variação" e "não há com o que comparar" são coisas diferentes. Os nulos seguem o tratamento padrão do
motor (imputação pela mediana do treino; a variável é descartada se passar de `Config.max_nulos`). O
delta de 3 meses usa os meses anteriores disponíveis (mínimo 1), então no 2º mês ele coincide com o de 1
mês — a etapa de redundância (|Spearman| > 0,9) resolve se as duas sobrarem parecidas.

**Sem vazamento**: a probabilidade das linhas de teste de uma dobra vem sempre do modelo do estágio 1
treinado sem aqueles clientes. Para as linhas de **treino**, p vem de um *cross-fit* por cliente
(`Config.delta_dobras` dobras internas, GroupKFold): cada linha de treino recebe a previsão de um modelo
que não viu aquele cliente. Sem isso, o delta do treino viria de previsões dentro da amostra (mais
afiadas) e o do teste de previsões fora da amostra — distribuições diferentes, coeficiente enviesado e
AUC univariada inflada na seleção. `delta_dobras = 0` desliga o cross-fit (mais rápido, otimista).

**Braço de controle** (`Config.delta_placebo`): embaralha os valores do delta, mantendo a distribuição e
zerando a informação. Serve para medir quanto da diferença entre "com" e "sem" delta é só o ruído de
refazer a seleção com duas candidatas a mais — não é para uso em produção.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .config import Config
from .modelos_previsao.base import ModeloPrevisao, ajustar_modelo
from .variaveis import Variavel

TABELA = "modelo"
COLUNA = "risco_previsto"
ROTULO_COLUNA = "Risco previsto"
ID_DELTA = "risco_previsto__delta"
ID_DELTA_3M = "risco_previsto__delta_3m"


def ids(cfg: Config) -> list[str]:
    return [ID_DELTA] + ([ID_DELTA_3M] if cfg.delta_risco_3m else [])


def variaveis(cfg: Config) -> dict[str, Variavel]:
    """Metadados das candidatas de delta (entram no `meta` do motor, como qualquer outra variável)."""
    out = {
        ID_DELTA: Variavel(ID_DELTA, "Risco previsto (variação vs o mês anterior)", TABELA, COLUNA,
                           "delta_risco", ROTULO_COLUNA, janela=1),
        ID_DELTA_3M: Variavel(ID_DELTA_3M, "Risco previsto (variação vs a média dos 3 meses anteriores)", TABELA,
                              COLUNA, "delta_risco", ROTULO_COLUNA, janela=3),
    }
    return {k: out[k] for k in ids(cfg)}


def colunas(p: pd.Series, grade: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Colunas de delta alinhadas à grade, a partir da probabilidade `p` (NaN fora das pontuáveis)."""
    d = grade[["cliente", "mes"]].assign(p=np.asarray(p, dtype=float))
    d = d.sort_values(["cliente", "mes"], kind="stable")
    g = d.groupby("cliente", sort=False)["p"]
    ant = g.shift(1)
    out = pd.DataFrame(index=grade.index)
    out[ID_DELTA] = (d["p"] - ant).reindex(grade.index)
    if cfg.delta_risco_3m:
        m3 = g.transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        out[ID_DELTA_3M] = (d["p"] - m3).reindex(grade.index)
    if cfg.delta_placebo:
        rng = np.random.default_rng(cfg.semente)
        for c in out.columns:                      # mesma distribuição, informação zero (braço de controle)
            ok = out[c].notna().to_numpy()
            v = np.array(out.loc[ok, c], dtype=float)      # cópia: em paralelo o X chega só-leitura
            rng.shuffle(v)
            out.loc[ok, c] = v
    return out


def _p_estagio1(nome: str, X: pd.DataFrame, grade: pd.DataFrame, meta: dict, treino: pd.Series, cfg: Config,
                H: int, pre, n_jobs: int) -> pd.Series:
    """Probabilidade do estágio 1 em todas as linhas pontuáveis, sem vazamento (ver o cabeçalho)."""
    pts = grade["pontuavel"]
    p = pd.Series(np.nan, index=grade.index, name="p_estagio1")
    mod, X1 = ajustar_modelo(nome, X, grade, meta, treino, cfg, H, pre=pre, n_jobs=n_jobs)
    p.loc[pts] = mod.prob(X1.loc[pts])
    if not cfg.delta_dobras or not bool(treino.any()):
        return p
    unicos = np.array(sorted(set(grade.loc[treino, "cliente"])))
    n = int(min(cfg.delta_dobras, len(unicos)))
    if n < 2:
        return p
    gkf = GroupKFold(n_splits=n)
    for tr, te in gkf.split(unicos, groups=unicos):
        dentro = set(unicos[tr])
        sub = treino & grade["cliente"].isin(dentro)
        if not bool(grade.loc[sub, "y"].sum()):
            continue
        mi, Xi = ajustar_modelo(nome, X, grade, meta, sub, cfg, H, n_jobs=n_jobs)
        alvo = pts & grade["cliente"].isin(set(unicos[te]))
        if alvo.any():
            p.loc[alvo] = mi.prob(Xi.loc[alvo])
    return p


def ajustar(nome: str, X: pd.DataFrame, grade: pd.DataFrame, meta: dict, treino: pd.Series, cfg: Config,
            H: int, pre=None, freq_dobras: pd.Series | None = None,
            n_jobs: int = 1) -> tuple[ModeloPrevisao, pd.DataFrame, dict]:
    """Ajuste em dois estágios. Devolve (modelo, X com as candidatas de delta e as persistências, meta)."""
    p1 = _p_estagio1(nome, X, grade, meta, treino, cfg, H, pre, n_jobs)
    meta2 = {**meta, **variaveis(cfg)}
    Xd = pd.concat([X, colunas(p1, grade, cfg)], axis=1)
    mod, X2 = ajustar_modelo(nome, Xd, grade, meta2, treino, cfg, H, freq_dobras=freq_dobras, n_jobs=n_jobs)
    return mod, X2, meta2


def ajustar_modelo_com_delta(nome: str, X: pd.DataFrame, grade: pd.DataFrame, meta: dict, treino: pd.Series,
                             cfg: Config, H: int, pre=None, freq_dobras: pd.Series | None = None,
                             n_jobs: int = 1) -> tuple[ModeloPrevisao, pd.DataFrame, dict]:
    """`ajustar_modelo` com o delta quando `Config.delta_risco`; senão o ajuste normal (meta inalterado)."""
    if cfg.delta_risco:
        return ajustar(nome, X, grade, meta, treino, cfg, H, pre=pre, freq_dobras=freq_dobras, n_jobs=n_jobs)
    mod, X2 = ajustar_modelo(nome, X, grade, meta, treino, cfg, H, pre=pre, freq_dobras=freq_dobras, n_jobs=n_jobs)
    return mod, X2, meta
