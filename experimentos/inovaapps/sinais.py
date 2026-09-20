"""Sinais de risco de cancelamento: nível, variação, antecedência, persistência e combinação.

Uso:
    from dados import carregar
    from sinais import SINAIS, valores_ate, auc_sinal, disparos, contar_sinais
    d = carregar()
    v = valores_ate(d, k=1)             # valores por cliente usando só dados até o mês −1
    auc_sinal(v, "uso", "nivel")

Convenções
----------
* `k` = antecedência. `valores_ate(d, k)` usa apenas meses com `mes_relativo <= −k`
  (sem vazamento). k = 1 → dados até o último mês observado.
* **nível**    = média dos `janela` meses que terminam em −k (−k−janela+1 … −k).
* **variação** = nível − média de todos os meses anteriores à janela (baseline do próprio cliente).
  Sem nenhum mês de baseline → NaN.
* NPS é trimestral: nível = última nota respondida até −k; variação = essa nota − média
  das notas respondidas anteriores.
* `direcao` = +1 quando valor maior indica risco, −1 quando valor menor indica risco.
  As AUC são orientadas pela direção definida a priori (não se escolhe o lado olhando o resultado).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

FORMAS = ["nivel", "variacao"]
ROTULO_FORMA = {"nivel": "Nível", "variacao": "Variação"}


@dataclass(frozen=True)
class Sinal:
    coluna: str
    direcao: int          # +1: maior = risco · −1: menor = risco
    rotulo: str
    unidade: str          # unidade do nível (a variação usa `unidade_var`)
    unidade_var: str
    fonte: str = "atendimento"   # "atendimento" ou "nps"


SINAIS: dict[str, Sinal] = {
    "uso": Sinal("uso_plataforma_pct", -1, "Uso da plataforma", "%", "p.p."),
    "tempo": Sinal("tempo_medio_resolucao_h", +1, "Tempo médio de resolução", "h", "h"),
    "sla": Sinal("pct_sla_cumprido", -1, "SLA cumprido", "%", "p.p."),
    "criticos": Sinal("chamados_criticos", +1, "Chamados críticos", "/mês", "/mês"),
    "reclamacoes": Sinal("reclamacoes_formais", +1, "Reclamações formais", "/mês", "/mês"),
    "atraso": Sinal("dias_atraso_pagamento", +1, "Atraso de pagamento", "dias", "dias"),
    "nps": Sinal("nota_nps", -1, "Nota NPS", "", "pontos", fonte="nps"),
}


# --------------------------------------------------------------------------- valores por cliente
def _nivel_variacao_mensal(at: pd.DataFrame, coluna: str, k: int, janela: int) -> pd.DataFrame:
    sub = at[at["mes_relativo"] <= -k]
    ini = -k - janela + 1
    jan = sub[sub["mes_relativo"] >= ini].groupby("cliente_id")[coluna].mean()
    base = sub[sub["mes_relativo"] < ini].groupby("cliente_id")[coluna].mean()
    out = pd.DataFrame({"nivel": jan, "base": base})
    out["variacao"] = out["nivel"] - out["base"]
    return out


def _nivel_variacao_nps(nps: pd.DataFrame, k: int) -> pd.DataFrame:
    resp = nps[nps["respondeu"] & (nps["mes_relativo"] <= -k)].sort_values(["cliente_id", "mes_ref"])
    ult = resp.groupby("cliente_id").tail(1).set_index("cliente_id")
    ant = resp.drop(index=resp.groupby("cliente_id").tail(1).index)
    out = pd.DataFrame({"nivel": ult["nota_nps"], "base": ant.groupby("cliente_id")["nota_nps"].mean(),
                        "mes_relativo_nps": ult["mes_relativo"]})
    out["variacao"] = out["nivel"] - out["base"]
    return out


def valores_ate(d, k: int = 1, janela: int = 3, sinais: list[str] | None = None) -> pd.DataFrame:
    """Uma linha por cliente, colunas MultiIndex (sinal, forma) com forma ∈ {nivel, base, variacao}.

    Usa só dados com mes_relativo <= −k. Inclui colunas de contexto: status, valor_mensal.
    Clientes sem dados até −k ficam com NaN.
    """
    partes = {}
    for s in sinais or list(SINAIS):
        cfg = SINAIS[s]
        if cfg.fonte == "nps":
            partes[s] = _nivel_variacao_nps(d.nps, k)[["nivel", "base", "variacao"]]
        else:
            partes[s] = _nivel_variacao_mensal(d.atendimento, cfg.coluna, k, janela)
    v = pd.concat(partes, axis=1)
    ctx = d.clientes.set_index("cliente_id")[["status", "valor_mensal"]]
    ctx.columns = pd.MultiIndex.from_tuples([(c, "") for c in ctx.columns])
    return ctx.join(v)


# --------------------------------------------------------------------------- separação
def auc_orientada(y: pd.Series, x: pd.Series, direcao: int) -> tuple[float, int]:
    """AUC de `y` (True = cancelou) usando direcao·x (orientada a priori). Devolve (auc, n usados)."""
    m = x.notna()
    if y[m].nunique() < 2:
        return np.nan, int(m.sum())
    return float(roc_auc_score(y[m], direcao * x[m])), int(m.sum())


def auc_sinal(v: pd.DataFrame, sinal: str, forma: str) -> float:
    return auc_orientada(v["status"].eq("Cancelou"), v[(sinal, forma)], SINAIS[sinal].direcao)[0]


def tabela_auc(d, ks=range(1, 7), janela: int = 3, sinais: list[str] | None = None) -> pd.DataFrame:
    """AUC (orientada) por sinal × forma × antecedência k. Colunas: sinal, forma, k, auc, n, n_canc."""
    linhas = []
    for k in ks:
        v = valores_ate(d, k, janela, sinais)
        y = v["status"].eq("Cancelou")
        for s in sinais or list(SINAIS):
            for f in FORMAS:
                a, n = auc_orientada(y, v[(s, f)], SINAIS[s].direcao)
                linhas.append(dict(sinal=s, forma=f, k=k, auc=a, n=n, n_canc=int((y & v[(s, f)].notna()).sum())))
    return pd.DataFrame(linhas)


# --------------------------------------------------------------------------- limiares e disparos
@dataclass(frozen=True)
class Limiar:
    sinal: str
    forma: str
    valor: float
    rotulo: str | None = None     # texto próprio (ex.: "≤ 6 (Detrator)")

    @property
    def cfg(self) -> Sinal:
        return SINAIS[self.sinal]

    def texto(self) -> str:
        if self.rotulo:
            return self.rotulo
        c = self.cfg
        op = "≤" if c.direcao < 0 else "≥"
        v = f"{self.valor:g}".replace("-", "−").replace(".", ",")
        if self.forma == "variacao":
            u = c.unidade_var
            return f"variação {op} {'+' if self.valor > 0 else ''}{v} {u}".strip()
        u = c.unidade
        return f"{op} {v}{u if u in ('%',) else (' ' + u if u else '')}".strip()

    def dispara(self, x: pd.Series) -> pd.Series:
        """True quando o valor está do lado de risco do limiar (NaN → False)."""
        r = x <= self.valor if self.cfg.direcao < 0 else x >= self.valor
        return r & x.notna()


def disparos(v: pd.DataFrame, limiares: dict[str, Limiar]) -> pd.DataFrame:
    """Booleano cliente × sinal: o sinal disparou com os valores `v` (saída de `valores_ate`)."""
    return pd.DataFrame({s: lim.dispara(v[(lim.sinal, lim.forma)]) for s, lim in limiares.items()})


def contar_sinais(v: pd.DataFrame, limiares: dict[str, Limiar]) -> pd.Series:
    """Quantos sinais disparam por cliente."""
    return disparos(v, limiares).sum(axis=1).rename("n_sinais")


def _fmt(x: float, casas: int = 1) -> str:
    return f"{x:+.{casas}f}".replace("-", "−").replace(".", ",") if casas >= 0 else str(x)


def evidencias(v: pd.DataFrame, limiares: dict[str, Limiar]) -> pd.Series:
    """Por cliente, lista de textos dos sinais disparados com o valor observado
    (ex.: 'Uso da plataforma: −21,2 p.p. vs média anterior')."""
    dp = disparos(v, limiares)
    out = {}
    for cid, linha in dp.iterrows():
        itens = []
        for s, lim in limiares.items():
            if not linha[s]:
                continue
            c, x = lim.cfg, v.at[cid, (lim.sinal, lim.forma)]
            if lim.forma == "variacao":
                itens.append(f"{c.rotulo}: {_fmt(x)} {c.unidade_var} vs média anterior")
            else:
                num = f"{x:.1f}".replace(".", ",")
                itens.append(f"{c.rotulo}: {num}{c.unidade if c.unidade == '%' else (' ' + c.unidade if c.unidade else '')}")
        out[cid] = itens
    return pd.Series(out, name="evidencias")


def taxa_deteccao(v: pd.DataFrame, alerta: pd.Series) -> dict[str, float]:
    """% de cancelados com alerta e % de ativos com alerta (alarme falso)."""
    st = v["status"]
    return {"pct_canc": 100 * alerta[st == "Cancelou"].mean(), "pct_ativos": 100 * alerta[st == "Ativo"].mean(),
            "n_canc": int(alerta[st == "Cancelou"].sum()), "n_ativos": int(alerta[st == "Ativo"].sum())}


# --------------------------------------------------------------------------- persistência
def serie_janela(at: pd.DataFrame, coluna: str, janela: int = 3) -> pd.DataFrame:
    """Média móvel retroativa (só passado) de `coluna` por cliente, com mes_relativo e status."""
    out = at.sort_values(["cliente_id", "mes_ref"])[["cliente_id", "mes_ref", "mes_relativo", "status", coluna]].copy()
    out[coluna] = (out.groupby("cliente_id")[coluna].rolling(janela, min_periods=1).mean()
                      .reset_index(level=0, drop=True))
    return out


def episodios(serie: pd.DataFrame, coluna: str, limiar: float, direcao: int) -> pd.DataFrame:
    """Sequências de meses consecutivos do lado de risco do limiar, por cliente.

    `serie` tem cliente_id, mes_relativo, status e `coluna` (mensal ou já suavizada).
    Devolve cliente_id, status, inicio, fim (mes_relativo), duracao (meses) e em_curso
    (a sequência chega ao último mês observado, −1).
    """
    s = serie.sort_values(["cliente_id", "mes_relativo"]).copy()
    lado = (s[coluna] <= limiar) if direcao < 0 else (s[coluna] >= limiar)
    s["lado"] = lado & s[coluna].notna()
    quebra = (s["lado"] != s.groupby("cliente_id")["lado"].shift()) | s["cliente_id"].ne(s["cliente_id"].shift())
    s["seq"] = quebra.cumsum()
    ep = (s[s["lado"]].groupby("seq")
            .agg(cliente_id=("cliente_id", "first"), status=("status", "first"),
                 inicio=("mes_relativo", "min"), fim=("mes_relativo", "max"), duracao=("mes_relativo", "size"))
            .reset_index(drop=True))
    ultimo = serie.groupby("cliente_id")["mes_relativo"].max()
    ep["em_curso"] = ep["fim"].eq(ep["cliente_id"].map(ultimo))
    return ep


def episodios_sinal(h: pd.DataFrame, disparo: pd.Series) -> pd.DataFrame:
    """`episodios` aplicado a um disparo avaliado mês a mês (saída de `historico` + série booleana alinhada)."""
    m = h["tem_dados"]
    s = h.loc[m, ["cliente_id", "status"]].assign(mes_relativo=-h.loc[m, "k"], x=disparo[m].astype(float))
    return episodios(s, "x", 1, +1)


def meses_seguidos_ate(serie: pd.DataFrame, coluna: str, limiar: float, direcao: int, k: int = 1) -> pd.Series:
    """Nº de meses consecutivos do lado de risco terminando exatamente em −k (0 se o mês −k não está)."""
    sub = serie[serie["mes_relativo"] <= -k]
    ep = episodios(sub, coluna, limiar, direcao)
    ep = ep[ep["fim"].eq(-k)]
    return ep.set_index("cliente_id")["duracao"].reindex(serie["cliente_id"].unique(), fill_value=0)


# --------------------------------------------------------------------------- histórico de disparos
def historico(d, limiares: dict[str, Limiar], ks=range(1, 16), janela: int = 3,
              valores: dict[int, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Disparos de cada sinal avaliados mês a mês: uma linha por cliente × k (avaliação feita no mês −k,
    só com dados até −k). Colunas: cliente_id, k, status, valor_mensal, tem_dados, <sinais>, n_sinais.
    `valores` = cache opcional {k: valores_ate(d, k, janela)}."""
    partes = []
    for k in ks:
        v = valores[k] if valores is not None and k in valores else valores_ate(d, k, janela)
        dp = disparos(v, limiares)
        dp["n_sinais"] = dp[list(limiares)].sum(axis=1)
        dp["tem_dados"] = v[("uso", "nivel")].notna()          # cliente já tinha dados no mês −k
        dp.insert(0, "k", k)
        dp.insert(1, "status", v["status"])
        dp.insert(2, "valor_mensal", v["valor_mensal"])
        partes.append(dp.reset_index(names="cliente_id"))
    return pd.concat(partes, ignore_index=True)


# --------------------------------------------------------------------------- regra de alerta
@dataclass(frozen=True)
class Regra:
    """Condição do mês = (sinal `principal` disparado) OU (≥ `min_outros` dos demais sinais disparados).
    Sem principal (None): condição = ≥ `min_outros` sinais quaisquer.
    Alerta no mês −k = condição verdadeira em `persistencia` avaliações mensais seguidas (−k, −k−1, …)."""
    limiares: dict[str, Limiar]
    principal: str | None = "uso"
    min_outros: int = 2
    persistencia: int = 1

    def descricao(self) -> str:
        if self.principal:
            base = f"{SINAIS[self.principal].rotulo} ({self.limiares[self.principal].texto()}) ou ≥ {self.min_outros} outros sinais"
        else:
            base = f"≥ {self.min_outros} sinais"
        return base + (f", por {self.persistencia} meses seguidos" if self.persistencia > 1 else "")

    def condicao(self, h: pd.DataFrame) -> pd.Series:
        sinais = list(self.limiares)
        if self.principal is None:
            return h[sinais].sum(axis=1) >= self.min_outros
        outros = h[[s for s in sinais if s != self.principal]].sum(axis=1)
        return h[self.principal] | (outros >= self.min_outros)

    def alerta(self, h: pd.DataFrame) -> pd.Series:
        """Série booleana alinhada a `h` (saída de `historico`)."""
        c = self.condicao(h) & h["tem_dados"]
        h2 = h[["cliente_id", "k"]].assign(c=c)
        ok = c.copy()
        for j in range(1, self.persistencia):
            prox = h2.assign(k=h2["k"] - j).set_index(["cliente_id", "k"])["c"]   # condição em −(k+j)
            ok &= pd.Series(h2.set_index(["cliente_id", "k"]).index.map(prox), index=h.index).fillna(False).astype(bool)
        return ok.rename("alerta")


def antecedencia_alerta(h: pd.DataFrame, alerta: pd.Series) -> pd.Series:
    """Por cliente: maior k tal que o alerta vale em todos os meses −k … −1 (alerta contínuo até o fim).
    0 quando não há alerta em −1."""
    a = h[["cliente_id", "k"]].assign(a=alerta.values).pivot(index="cliente_id", columns="k", values="a").fillna(False)
    a = a.reindex(columns=sorted(a.columns)).astype(bool)
    return a.cumprod(axis=1).sum(axis=1).rename("antecedencia")


def desempenho(h: pd.DataFrame, alerta: pd.Series, ks=(1, 2, 3), janela_hist: int = 12) -> dict:
    """Resumo de uma regra: detectados entre cancelados em cada k, alarmes nos ativos em −1 (hoje)
    e % de ativos que teriam recebido alerta em algum mês dos últimos `janela_hist` meses."""
    x = h.assign(a=alerta.values)
    canc, ativ = x[x.status == "Cancelou"], x[x.status == "Ativo"]
    out = {f"canc_k{k}": int(canc[canc.k == k].a.sum()) for k in ks}
    out["ativos_hoje"] = int(ativ[ativ.k == 1].a.sum())
    out["ativos_algum_mes"] = int(ativ[ativ.k <= janela_hist].groupby("cliente_id").a.any().sum())
    out["meses_alarme_ativos_pct"] = 100 * ativ[ativ.k <= janela_hist].a.mean()
    ant = antecedencia_alerta(h, alerta)
    out["antecedencia_mediana"] = float(ant[ant.index.isin(canc.cliente_id) & (ant > 0)].median())
    return out
