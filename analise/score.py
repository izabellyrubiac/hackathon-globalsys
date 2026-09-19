"""Score de risco de cancelamento: regressão logística explicável sobre um painel cliente × mês.

Uso:
    from dados import carregar
    from score import montar_painel, ajustar, validar, prever_atual, explicar, priorizar
    d = carregar()
    P = montar_painel(d)                      # uma linha por cliente × mês (k = meses antes da saída)
    modelo = ajustar(P)                       # treino final em todos os dados (pipeline + variáveis mantidas)
    oof = validar(P)                          # probabilidades out-of-fold, deixando um cliente de fora por vez
    fila = priorizar(prever_atual(P, modelo), cortes)

Painel (sem vazamento)
----------------------
* Cada linha é um cliente avaliado no mês −k (eixo `mes_relativo` de `dados.py`), usando **só**
  dados até −k (`sinais.valores_ate`). Ativos: −1 = jun/2026. Cancelados: −1 = mês anterior à saída.
* Rótulo `y` = cancela nos próximos 3 meses = cancelado com k ∈ {1, 2, 3}. Demais linhas de
  cancelados (k ≥ 4) e todas as linhas de ativos = 0.
* Linha entra no **treino** só com histórico suficiente: janela de 3 meses completa + ≥ 1 mês
  anterior para a média de referência do uso (≥ 4 meses de dados). Linhas com 1–3 meses de dados
  ainda recebem score (valores faltantes imputados pela mediana do treino), mas não treinam.

Variáveis (todas avaliadas no mês −k)
-------------------------------------
uso           variação do uso da plataforma: média dos 3 meses − média anterior do cliente (p.p.)
tempo         tempo médio de resolução, média de 3 meses (h)
sla           SLA cumprido, média de 3 meses (%)
criticos      chamados críticos por mês, média de 3 meses
reclamacoes   reclamações formais por mês, média de 3 meses
atraso        dias de atraso de pagamento, média de 3 meses
nps           última nota NPS respondida até −k (NaN → mediana do treino)
nps_sem_nota  1 quando o cliente nunca respondeu o NPS até −k (indicador da imputação acima)
persistencia  meses seguidos, terminando em −k, com a condição da regra do notebook 02
              (queda de uso ≥ 10 p.p. OU ≥ 2 outros sinais)

Modelo
------
SimpleImputer(mediana) → StandardScaler → LogisticRegression(L2). C escolhido por validação
agrupada por cliente (GroupKFold 5, log-loss). Toda variável precisa ter o sinal de risco esperado
(`direcao`): a que inverter sai e o modelo é reajustado (`ajustar` repete até nenhuma inverter).

Explicação
----------
contribuição_i = coef_i × z_i (log-odds, relativo ao cliente médio do painel; z = valor padronizado).
Uma variável **dispara** quando contribuição > 0 E o valor passa do limiar do notebook 02
(`LIMIARES`, mesmos valores). Contribuições positivas abaixo do limiar = fatores secundários.

Prioridade
----------
perda_anual_esperada = probabilidade de cancelar nos próximos 3 meses × valor_mensal × 12.
A fila ordena todos os ativos por essa perda (desempate: probabilidade, cliente_id).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from dados import fmt_mes
from sinais import Limiar, Regra, auc_orientada, desempenho, historico, valores_ate

KS = range(1, 19)                  # 18 meses de histórico → avaliações de −1 a −18
HORIZONTE = 3                      # rótulo: cancela nos próximos 3 meses
CS = np.logspace(-3, 2, 11)        # grade de C (regularização L2: menor C = mais regularização)

# Mesmos limiares do notebook 02 (seção 3) — usados para "disparar" evidências e na variável de persistência
LIMIARES: dict[str, Limiar] = {
    "uso":         Limiar("uso", "variacao", -10, "queda ≥ 10 p.p. vs média anterior"),
    "tempo":       Limiar("tempo", "nivel", 28, "≥ 28 h (média 3 meses)"),
    "sla":         Limiar("sla", "nivel", 60, "≤ 60% (média 3 meses)"),
    "criticos":    Limiar("criticos", "nivel", 2, "≥ 2/mês (≥ 6 em 3 meses)"),
    "reclamacoes": Limiar("reclamacoes", "nivel", 1, "≥ 1/mês (≥ 3 em 3 meses)"),
    "atraso":      Limiar("atraso", "nivel", 5, "≥ 5 dias (média 3 meses)"),
    "nps":         Limiar("nps", "nivel", 6, "última nota ≤ 6 (Detrator)"),
}
REGRA = Regra(LIMIARES, "uso", 2, 2)          # regra combinada do notebook 02 (baseline)
CONDICAO = Regra(LIMIARES, "uso", 2, 1)       # condição mensal da regra (base da persistência)
MIN_PERSISTENCIA = 2                          # limiar de disparo da persistência (igual à regra)


@dataclass(frozen=True)
class Variavel:
    nome: str
    rotulo: str
    direcao: int                 # +1: maior = risco · −1: menor = risco
    sinal: str | None            # chave em SINAIS/LIMIARES (None para as variáveis derivadas)
    forma: str | None = None     # "nivel" ou "variacao" (valores_ate)
    coluna: str = ""             # nome do campo equivalente no JSON


VARIAVEIS: dict[str, Variavel] = {
    "uso": Variavel("uso", "Queda do uso vs média anterior", -1, "uso", "variacao", "uso_plataforma_pct"),
    "tempo": Variavel("tempo", "Tempo médio de resolução (3m)", +1, "tempo", "nivel", "tempo_medio_resolucao_h"),
    "sla": Variavel("sla", "SLA cumprido (3m)", -1, "sla", "nivel", "pct_sla_cumprido"),
    "criticos": Variavel("criticos", "Chamados críticos (3m)", +1, "criticos", "nivel", "chamados_criticos"),
    "reclamacoes": Variavel("reclamacoes", "Reclamações formais (3m)", +1, "reclamacoes", "nivel", "reclamacoes_formais"),
    "atraso": Variavel("atraso", "Atraso de pagamento (3m)", +1, "atraso", "nivel", "dias_atraso_pagamento"),
    "nps": Variavel("nps", "Última nota NPS", -1, "nps", "nivel", "nota_nps"),
    "nps_sem_nota": Variavel("nps_sem_nota", "Nunca respondeu o NPS", +1, None, None, "nps_sem_resposta"),
    "persistencia": Variavel("persistencia", "Meses seguidos em alerta", +1, None, None, "meses_em_alerta"),
}
FEATURES = list(VARIAVEIS)

ACOES = {
    "uso": "Agendar revisão de valor e adoção da plataforma com o decisor",
    "tempo": "Montar plano técnico de estabilização (tempo de resolução, SLA e chamados críticos)",
    "sla": "Montar plano técnico de estabilização (tempo de resolução, SLA e chamados críticos)",
    "criticos": "Montar plano técnico de estabilização (tempo de resolução, SLA e chamados críticos)",
    "reclamacoes": "Contato do gestor da conta para tratar as reclamações formais",
    "atraso": "Alinhamento financeiro sobre pagamentos em atraso",
    "nps": "Entrevista de escuta sobre a experiência (NPS)",
    "nps_sem_nota": "Entrevista de escuta sobre a experiência (NPS)",
}
ACAO_ROTINA = "Manter acompanhamento de rotina"


# --------------------------------------------------------------------------- painel
def _meses_seguidos(h: pd.DataFrame, cond: pd.Series) -> pd.Series:
    """Nº de avaliações mensais seguidas com `cond` verdadeira terminando em cada (cliente, k)."""
    x = h[["cliente_id", "k"]].assign(c=cond.astype(int).values).sort_values(["cliente_id", "k"], ascending=[True, False])
    bloco = (x["c"] == 0).groupby(x["cliente_id"]).cumsum()
    x["run"] = x["c"].groupby([x["cliente_id"], bloco]).cumsum() * x["c"]
    return x["run"].reindex(h.index)


def _ultima_nps(d, ks=KS) -> pd.DataFrame:
    """Mês (mes_relativo) e classificação da última pesquisa NPS respondida até cada −k."""
    resp = d.nps[d.nps["respondeu"]].sort_values(["cliente_id", "mes_ref"])
    partes = []
    for k in ks:
        ult = resp[resp["mes_relativo"] <= -k].groupby("cliente_id").tail(1)
        partes.append(ult[["cliente_id", "mes_ref", "classificacao_nps"]]
                      .rename(columns={"mes_ref": "nps_mes"}).assign(k=k))
    return pd.concat(partes, ignore_index=True)


def montar_painel(d, ks=KS, janela: int = 3) -> pd.DataFrame:
    """Uma linha por cliente × k (80 × 18). Colunas: cliente_id, k, mes_ref, status, valor_mensal,
    tem_dados, treino, y, as variáveis de FEATURES, os disparos dos 7 sinais (`disp_<sinal>`),
    n_sinais, condicao (condição mensal da regra), alerta_regra, alerta_uso e colunas de contexto
    (uso_nivel, uso_base, nps_mes, nps_classe) usadas nos textos das evidências."""
    V = {k: valores_ate(d, k, janela) for k in ks}
    H = historico(d, LIMIARES, ks, janela, valores=V)
    H["condicao"] = CONDICAO.condicao(H) & H["tem_dados"]
    H["alerta_regra"] = REGRA.alerta(H).values
    H["alerta_uso"] = H["uso"] & H["tem_dados"]
    H = H.rename(columns={s: f"disp_{s}" for s in LIMIARES})

    vals = []
    for k in ks:
        v = V[k]
        f = pd.DataFrame({n: v[(c.sinal, c.forma)] for n, c in VARIAVEIS.items() if c.sinal}, index=v.index)
        f["uso_nivel"], f["uso_base"] = v[("uso", "nivel")], v[("uso", "base")]
        vals.append(f.assign(k=k).reset_index(names="cliente_id"))
    P = H.merge(pd.concat(vals, ignore_index=True), on=["cliente_id", "k"], how="left")
    P = P.merge(_ultima_nps(d, ks), on=["cliente_id", "k"], how="left").rename(columns={"classificacao_nps": "nps_classe"})

    P["nps_sem_nota"] = (P["nps"].isna() & P["tem_dados"]).astype(float)
    P["persistencia"] = _meses_seguidos(P, P["condicao"]).astype(float)

    saida = d.clientes.set_index("cliente_id")["mes_saida"]
    P["mes_ref"] = [saida[c] - k for c, k in zip(P["cliente_id"], P["k"])]
    P["y"] = (P["status"].eq("Cancelou") & P["k"].le(HORIZONTE)).astype(int)
    P["treino"] = P["tem_dados"] & P["uso"].notna()
    P["status"] = P["status"].astype(str)
    return P.sort_values(["cliente_id", "k"]).reset_index(drop=True)


# --------------------------------------------------------------------------- modelo
def pipeline(C: float = 1.0, class_weight=None) -> Pipeline:
    return Pipeline([
        ("imputar", SimpleImputer(strategy="median")),
        ("padronizar", StandardScaler()),
        ("logistica", LogisticRegression(C=C, class_weight=class_weight, max_iter=5000)),
    ])


@dataclass
class Modelo:
    pipe: Pipeline
    features: list[str]
    C: float
    removidas: list[str]                # variáveis retiradas por inverter o sinal esperado

    def coeficientes(self) -> pd.DataFrame:
        lr = self.pipe.named_steps["logistica"]
        sc = self.pipe.named_steps["padronizar"]
        c = pd.DataFrame({"coef": lr.coef_[0], "media": sc.mean_, "desvio": sc.scale_}, index=self.features)
        c["direcao"] = [VARIAVEIS[f].direcao for f in self.features]
        c["sinal_ok"] = np.sign(c["coef"]) == c["direcao"]
        c["odds_ratio"] = np.exp(c["coef"])            # multiplicador das chances por +1 desvio-padrão
        c["rotulo"] = [VARIAVEIS[f].rotulo for f in self.features]
        return c

    def intercepto(self) -> float:
        return float(self.pipe.named_steps["logistica"].intercept_[0])

    def prob(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipe.predict_proba(X[self.features])[:, 1]


def _grid(X, y, g, class_weight, splits: int = 5) -> GridSearchCV:
    gs = GridSearchCV(pipeline(class_weight=class_weight), {"logistica__C": CS}, scoring="neg_log_loss",
                      cv=GroupKFold(n_splits=splits), n_jobs=1)
    return gs.fit(X, y, groups=g)


def ajustar(P: pd.DataFrame, features: list[str] | None = None, class_weight=None) -> Modelo:
    """Escolhe C por validação agrupada e ajusta nas linhas de treino. Remove, uma a uma, as variáveis
    cujo coeficiente sai com o sinal oposto ao esperado (a de maior |coef| invertido primeiro) e reajusta."""
    feats = list(features or FEATURES)
    T = P[P["treino"]]
    removidas = []
    while True:
        gs = _grid(T[feats], T["y"], T["cliente_id"], class_weight)
        coef = gs.best_estimator_.named_steps["logistica"].coef_[0]
        dirs = np.array([VARIAVEIS[f].direcao for f in feats])
        inv = [(abs(c), f) for c, f, dr in zip(coef, feats, dirs) if np.sign(c) != dr]
        if not inv:
            return Modelo(gs.best_estimator_, feats, float(gs.best_params_["logistica__C"]), removidas)
        pior = max(inv)[1]
        removidas.append(pior)
        feats.remove(pior)


def _dobra(P: pd.DataFrame, cid: str, idx, features, class_weight):
    m = ajustar(P[P["cliente_id"] != cid], features, class_weight)
    coefs = dict(zip(m.features, m.pipe.named_steps["logistica"].coef_[0]))
    return idx, m.prob(P.loc[idx]), {"cliente_id": cid, "C": m.C, "removidas": ",".join(m.removidas), **coefs}


def validar(P: pd.DataFrame, features: list[str] | None = None, class_weight=None,
            n_jobs: int = -1) -> tuple[pd.Series, pd.DataFrame]:
    """Validação deixando um cliente de fora (LOCO): para cada cliente, `ajustar` (escolha de C e
    checagem de sinal inclusas) nos outros 79 e prevê todas as linhas dele com dados.
    Devolve (probabilidade out-of-fold alinhada a P — NaN sem dados, tabela por dobra com C e coeficientes)."""
    base = P[P["tem_dados"]]
    tarefas = [(base["cliente_id"].iloc[t[0]], base.index[t])
               for _, t in LeaveOneGroupOut().split(base, groups=base["cliente_id"])]
    res = Parallel(n_jobs=n_jobs)(delayed(_dobra)(P, cid, idx, features, class_weight) for cid, idx in tarefas)
    oof = pd.Series(np.nan, index=P.index, name="p_oof")
    for idx, p, _ in res:
        oof.loc[idx] = p
    return oof, pd.DataFrame([r for *_, r in res]).sort_values("cliente_id").reset_index(drop=True)


def prever(P: pd.DataFrame, modelo: Modelo) -> pd.Series:
    """Probabilidade do modelo em todas as linhas com dados (NaN nas demais)."""
    p = pd.Series(np.nan, index=P.index, name="p")
    m = P["tem_dados"]
    p[m] = modelo.prob(P[m])
    return p


def prever_atual(P: pd.DataFrame, modelo: Modelo) -> pd.DataFrame:
    """Ativos avaliados em jun/2026 (k = 1) com a probabilidade do modelo final."""
    A = P[(P["status"] == "Ativo") & (P["k"] == 1)].copy()
    A["probabilidade"] = modelo.prob(A)
    return A.set_index("cliente_id")


# --------------------------------------------------------------------------- faixas e avaliação
def faixa(p, cortes: dict[str, float]):
    """'alto' se p ≥ cortes['alto'], 'atencao' se p ≥ cortes['atencao'], senão 'baixo'."""
    p = np.asarray(p, dtype=float)
    return np.select([p >= cortes["alto"], p >= cortes["atencao"]], ["alto", "atencao"], "baixo")


def curva(P: pd.DataFrame, p: pd.Series, cortes_grade=None) -> pd.DataFrame:
    """Para cada corte de probabilidade (alarme = p ≥ corte): cancelados detectados em −1/−2/−3 (de 22),
    ativos em alarme em jun/2026 (de 58) e em algum mês dos últimos 12 (colunas de `sinais.desempenho`),
    e, nas linhas de treino do painel (meses-cliente): precisão, recall, taxa de alarme falso e Youden."""
    grade = np.round(np.arange(0.02, 0.96, 0.01), 2) if cortes_grade is None else cortes_grade
    linhas = []
    for c in grade:
        a = (p >= c) & P["tem_dados"]
        r = desempenho(P, a)
        t = P["treino"]
        vp, fp = int((a & t & (P["y"] == 1)).sum()), int((a & t & (P["y"] == 0)).sum())
        rec, fpr = vp / int((t & (P["y"] == 1)).sum()), fp / int((t & (P["y"] == 0)).sum())
        linhas.append({"corte": float(c), **r, "precisao_linhas": vp / (vp + fp) if vp + fp else np.nan,
                       "recall_linhas": rec, "fpr_linhas": fpr, "youden_linhas": rec - fpr})
    return pd.DataFrame(linhas)


def auc_por_mes(P: pd.DataFrame, scores: dict[str, tuple[pd.Series, int]], ks=range(1, 7)) -> pd.DataFrame:
    """AUC Cancelou × Ativo com o score avaliado no mês −k (um valor por cliente, como no notebook 02).
    `scores` = {nome: (série alinhada a P, direção)}."""
    linhas = []
    for k in ks:
        m = (P["k"] == k) & P["tem_dados"]
        y = P.loc[m, "status"].eq("Cancelou")
        for nome, (s, dr) in scores.items():
            a, n = auc_orientada(y, s[m], dr)
            linhas.append({"k": k, "score": nome, "auc": a, "n": n, "n_canc": int((y & s[m].notna()).sum())})
    return pd.DataFrame(linhas)


def precisao_topo(P: pd.DataFrame, s: pd.Series, direcao: int = 1, ks=(1, 2, 3), ns=(10, 20)) -> pd.DataFrame:
    """Cancelados entre os N primeiros quando os 80 clientes são ordenados pelo score no mês −k
    (cancelados no seu próprio −k, ativos em −k contado de jul/2026)."""
    linhas = []
    for k in ks:
        m = (P["k"] == k) & P["tem_dados"] & s.notna()
        q = P.loc[m, ["cliente_id", "status"]].assign(s=direcao * s[m]).sort_values(["s", "cliente_id"], ascending=[False, True])
        for n in ns:
            linhas.append({"k": k, "top": n, "cancelados": int(q.head(n)["status"].eq("Cancelou").sum())})
    return pd.DataFrame(linhas)


# --------------------------------------------------------------------------- explicação
def contribuicoes(modelo: Modelo, X: pd.DataFrame) -> pd.DataFrame:
    """coef_i × z_i por linha (log-odds relativos ao cliente médio do painel de treino)."""
    imp = modelo.pipe.named_steps["imputar"]
    sc = modelo.pipe.named_steps["padronizar"]
    z = sc.transform(imp.transform(X[modelo.features]))
    coef = modelo.pipe.named_steps["logistica"].coef_[0]
    return pd.DataFrame(z * coef, index=X.index, columns=modelo.features)


def _num(x: float, casas: int = 1) -> str:
    return f"{x:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".").replace("-", "−")


def passa_limiar(nome: str, linha: pd.Series) -> bool:
    """A variável está do lado de risco do limiar do notebook 02?"""
    if nome in LIMIARES:                      # disparo calculado por sinais.disparos (via historico)
        return bool(linha[f"disp_{nome}"])
    if nome == "nps_sem_nota":
        return bool(linha[nome] >= 1)
    if nome == "persistencia":
        return bool(linha[nome] >= MIN_PERSISTENCIA)
    return False


def texto(nome: str, linha: pd.Series, disparou: bool) -> tuple[str, str]:
    """(título, detalhe) legíveis em pt-BR para uma variável do cliente."""
    x = linha[nome]
    limite = "limite" if disparou else "abaixo do limite de" if VARIAVEIS[nome].direcao > 0 else "limite"
    if nome == "uso":
        de_para = f"{_num(linha['uso_base'], 0)}% → {_num(linha['uso_nivel'], 0)}%"
        if x < 0:
            det = f"Caiu {_num(-x)} p.p. vs a média anterior do cliente ({de_para}); limite: queda de 10 p.p."
            return ("Uso da plataforma caiu" if disparou else "Uso da plataforma em leve queda"), det
        return "Uso da plataforma estável", f"Variação de +{_num(x)} p.p. vs a média anterior ({de_para})"
    if nome == "tempo":
        return "Tempo de resolução alto" if disparou else "Tempo de resolução acima da média", \
            f"{_num(x)} h em média nos últimos 3 meses ({limite} 28 h)"
    if nome == "sla":
        return "SLA abaixo do contratado" if disparou else "SLA abaixo da média", \
            f"{_num(x, 0)}% dos chamados no SLA nos últimos 3 meses (limite: 60%)"
    if nome == "criticos":
        return "Chamados críticos frequentes" if disparou else "Chamados críticos acima da média", \
            f"{_num(x)} chamados críticos por mês nos últimos 3 meses ({limite} 2)"
    if nome == "reclamacoes":
        return "Reclamações formais" if disparou else "Reclamações formais acima da média", \
            f"{_num(x)} reclamações formais por mês nos últimos 3 meses ({limite} 1)"
    if nome == "atraso":
        return "Atraso de pagamento" if disparou else "Atraso de pagamento acima da média", \
            f"{_num(x, 0)} dias de atraso médio nos últimos 3 meses ({limite} 5)"
    if nome == "nps":
        classe = str(linha.get("nps_classe", "")).lower()
        mes = fmt_mes(linha["nps_mes"]) if pd.notna(linha.get("nps_mes")) else ""
        return ("NPS detrator" if disparou else "NPS neutro"), \
            f"Última nota NPS {_num(x, 0)} — {classe} ({mes}); limite: nota ≤ 6"
    if nome == "nps_sem_nota":
        return "Nunca respondeu o NPS", "Nenhuma pesquisa NPS respondida até agora"
    if nome == "persistencia":
        n = int(x)
        return ("Alerta persistente" if disparou else "Alerta recente"), \
            f"{n} {'mês' if n == 1 else 'meses seguidos'} com queda de uso ou ≥ 2 outros sinais (alerta a partir de 2)"
    raise KeyError(nome)


def explicar(modelo: Modelo, A: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por cliente × variável: valor, contribuição (log-odds), passa do limiar, dispara
    (contribuição > 0 e passa do limiar), secundario (contribuição > 0 sem passar), título e detalhe."""
    C = contribuicoes(modelo, A)
    linhas = []
    for cid in A.index:
        pos = C.loc[cid].clip(lower=0).sum()
        for f in modelo.features:
            c = float(C.at[cid, f])
            passa = passa_limiar(f, A.loc[cid])
            disp = c > 0 and passa
            t, det = texto(f, A.loc[cid], disp)
            linhas.append({"cliente_id": cid, "variavel": f, "valor": A.at[cid, f], "contribuicao": c,
                           "passa_limiar": passa, "dispara": disp, "secundario": c > 0 and not passa,
                           "peso": c / pos if c > 0 and pos > 0 else 0.0, "titulo": t, "detalhe": det})
    E = pd.DataFrame(linhas)
    return E.sort_values(["cliente_id", "contribuicao"], ascending=[True, False]).reset_index(drop=True)


def acao_sugerida(E_cliente: pd.DataFrame, faixa_: str) -> tuple[str, str | None]:
    """(ação, variável dominante). Dominante = variável que dispara com maior contribuição (persistência
    não é acionável: passa para a próxima); sem disparo, a maior contribuição secundária fora da faixa baixa."""
    for col in ("dispara", "secundario"):
        cand = E_cliente[E_cliente[col] & E_cliente["variavel"].isin(list(ACOES))]
        if col == "secundario" and faixa_ == "baixo":
            break
        if len(cand):
            v = cand.iloc[0]["variavel"]
            return ACOES[v], v
    return ACAO_ROTINA, None


def priorizar(A: pd.DataFrame, cortes: dict[str, float]) -> pd.DataFrame:
    """Fila com todos os ativos: perda_anual_esperada = probabilidade × valor_mensal × 12, ordenada
    desc (desempate: probabilidade desc, cliente_id). Acrescenta faixa_risco e prioridade (1..N)."""
    F = A.copy()
    F["perda_anual_esperada"] = F["probabilidade"] * F["valor_mensal"] * 12
    F["faixa_risco"] = faixa(F["probabilidade"], cortes)
    F = F.reset_index().sort_values(["perda_anual_esperada", "probabilidade", "cliente_id"],
                                    ascending=[False, False, True])
    F["prioridade"] = np.arange(1, len(F) + 1)
    return F.set_index("cliente_id")


# --------------------------------------------------------------------------- cortes das faixas
def escolher_cortes(cv: pd.DataFrame, referencia: dict) -> dict:
    """Cortes das faixas a partir da curva out-of-fold (`curva` sobre as probabilidades OOF).

    * alto    = menor corte em que a precisão nas linhas do painel é ≥ 50%: entre os meses-cliente
                marcados como alto na validação, pelo menos metade terminou em cancelamento nos 3 meses
                seguintes.
    * atencao = maior corte que detecta, em −2 e em −3, pelo menos tantos cancelados quanto a
                `referencia` (desempenho da regra do notebook 02): o score não pode dar menos
                antecedência que a regra. Se nenhum corte atinge, usa o ponto de Youden das linhas.
    Nenhum dos dois depende de quantos clientes a equipe consegue atender.
    """
    alto = float(cv.loc[cv["precisao_linhas"] >= 0.5, "corte"].min())
    ok = (cv["canc_k2"] >= referencia["canc_k2"]) & (cv["canc_k3"] >= referencia["canc_k3"]) & (cv["corte"] < alto)
    youden = float(cv.loc[cv["youden_linhas"].idxmax(), "corte"])
    atencao = float(cv.loc[ok, "corte"].max()) if ok.any() else youden
    return {"alto": alto, "atencao": atencao, "youden": youden}


# --------------------------------------------------------------------------- tudo de uma vez
@dataclass
class Resultado:
    painel: pd.DataFrame          # P + p_oof + p_final
    modelo: Modelo                # treino final em todas as linhas de treino
    dobras: pd.DataFrame          # C e coeficientes por dobra da validação
    curva: pd.DataFrame           # curva OOF (corte × detecção × alarmes)
    cortes: dict                  # {"alto", "atencao", "youden"}
    baselines: dict               # desempenho de "só uso", "regra" e da logística nas faixas
    fila: pd.DataFrame            # ativos em jun/2026, ordenados (prioridade 1..58)
    explicacao: pd.DataFrame      # cliente × variável (contribuições, disparos, textos)


def rodar(d) -> Resultado:
    """Painel → validação LOCO → cortes → treino final → fila explicada."""
    P = montar_painel(d)
    P["p_oof"], dobras = validar(P)
    ref = desempenho(P, P["alerta_regra"])
    cv = curva(P, P["p_oof"])
    cortes = escolher_cortes(cv, ref)
    modelo = ajustar(P)
    P["p_final"] = prever(P, modelo)
    bl = {"so_uso": desempenho(P, P["alerta_uso"]), "regra": ref,
          "logistica_alto": desempenho(P, (P["p_oof"] >= cortes["alto"]) & P["tem_dados"]),
          "logistica_alto_ou_atencao": desempenho(P, (P["p_oof"] >= cortes["atencao"]) & P["tem_dados"])}
    fila = priorizar(prever_atual(P, modelo), cortes)
    E = explicar(modelo, fila)
    acoes = {c: acao_sugerida(E[E["cliente_id"] == c], fila.at[c, "faixa_risco"]) for c in fila.index}
    fila["acao_sugerida"] = [acoes[c][0] for c in fila.index]
    fila["variavel_dominante"] = [acoes[c][1] for c in fila.index]
    fila["n_evidencias"] = E[E["dispara"]].groupby("cliente_id").size().reindex(fila.index, fill_value=0)
    return Resultado(P, modelo, dobras, cv, cortes, bl, fila, E)
