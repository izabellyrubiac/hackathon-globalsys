"""Explicação de cada cliente da fila: contribuições, evidências, fatores secundários e ação sugerida.

* contribuição_i = `modelo.contribuicoes` (aditiva em log-odds): na logística coef_i × z_i (relativo ao cliente
  médio do treino; z = valor padronizado); nas árvores, TreeSHAP exato (relativo ao valor esperado).
* Uma variável DISPARA quando contribuição > 0 E o valor passou do limiar aprendido (Youden arredondado).
  Contribuição > 0 sem passar do limiar = fator secundário.
* peso = contribuição / soma das contribuições positivas do cliente (evidências + secundários somam 1).
* Ação sugerida: a da variável dominante (a evidência de maior contribuição cuja coluna tem ação no
  dicionário `Mapeamento.acoes`); sem dicionário ou sem ação mapeada, "Investigar <coluna dominante>".
  Sem evidência: faixa alto/atenção usa o maior fator secundário; faixa baixa → acompanhamento de rotina.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .util import fmt_mes, fmt_num, fmt_pct, minusc
from .variaveis import Variavel

ACAO_ROTINA = "Manter acompanhamento de rotina"


def meta_persistencias(meta: dict[str, Variavel], specs: dict[str, dict]) -> dict[str, Variavel]:
    """Metadados das variáveis de persistência criadas na seleção."""
    out = {}
    for pid, s in specs.items():
        b = meta[s["base"]]
        out[pid] = Variavel(pid, f"Meses seguidos com {minusc(b.rotulo)} na zona de risco", b.tabela, b.coluna,
                            "persistencia", b.rotulo_coluna, categoria=b.categoria,
                            extra={"base": s["base"], "limiar_base": s["limiar"], "direcao_base": s["direcao"]})
    return out


def passa(x: float, limiar: float, direcao: int) -> bool:
    if x is None or np.isnan(x) or limiar is None or np.isnan(limiar):
        return False
    return bool(x >= limiar) if direcao > 0 else bool(x <= limiar)


def _lim(lim: float, direcao: int, fracao: bool = False, sinal: bool = False) -> str:
    op = "≥" if direcao > 0 else "≤"
    if fracao:
        return f"{op} {fmt_pct(lim)}"
    t = fmt_num(lim)
    if sinal and lim > 0:
        t = "+" + t
    return f"{op} {t}"


def texto(v: Variavel, x: float, aux: dict, lim: float, direcao: int, dispara: bool, mes, meta: dict) -> tuple[str, str]:
    """(título, detalhe) em pt-BR para a variável `v` com valor `x` no mês `mes`."""
    R = v.rotulo_coluna
    r = minusc(R)
    ab = "alto" if direcao > 0 else "baixo"
    tr = v.transformacao
    L = _lim(lim, direcao, fracao=v.fracao)
    if tr in ("nivel", "media"):
        onde = "nos últimos" if tr == "nivel" else "nos registros dos últimos"
        return f"{R} {ab}", f"{R}: {fmt_num(x)} em média {onde} {v.janela} meses (limite {L})"
    if tr == "variacao":
        ref, niv = aux.get(f"{v.id}:ref"), aux.get(f"{v.id}:nivel")
        verbo = "caiu" if x < 0 else "subiu"
        base = "a média anterior do cliente" if v.id + ":desde" not in aux else "a média dos registros anteriores"
        de_para = f" ({fmt_num(ref)} → {fmt_num(niv)})" if ref is not None and not np.isnan(ref) else ""
        if (direcao < 0 and lim <= 0) or (direcao > 0 and lim >= 0):
            limite = f"{'queda' if direcao < 0 else 'alta'} de {fmt_num(abs(lim))}"
        else:
            limite = f"variação {_lim(lim, direcao, sinal=True)}"
        return f"{R} {verbo}", f"{R} {verbo} {fmt_num(abs(x))} vs {base}{de_para}; limite: {limite}"
    if tr == "tendencia":
        sentido = "queda" if x < 0 else "alta"
        return f"{R} em {sentido}", (f"{R} em {sentido} de {fmt_num(abs(x), 2)} por mês nos últimos {v.janela} meses "
                                     f"(limite {_lim(lim, direcao, sinal=True)}/mês)")
    if tr == "ultimo":
        desde = aux.get(f"{v.id}:desde")
        if desde is not None and not np.isnan(desde) and isinstance(mes, pd.Period):
            quando = f"no último registro ({fmt_mes(mes - int(desde))})"
        else:
            quando = "no último mês"
        return f"{R} {ab}", f"{R} {fmt_num(x)} {quando} (limite {L})"
    if tr == "meses_desde":
        return f"Sem {r} recente", f"{fmt_num(x, 0)} meses desde o último registro de {r} (limite {L})"
    if tr == "nunca":
        return (f"Nenhum registro de {r}", f"Nenhum registro de {r} até agora") if x >= 0.5 else \
            (f"Tem registro de {r}", f"Já houve registro de {r}")
    if tr == "ausente":
        return f"{R} ausente", f"{R} ausente em {fmt_pct(x)} dos meses recentes (limite {L})"
    if tr == "freq":
        freq = "frequente" if direcao > 0 else "raro"
        return f"{R}: {v.categoria} {freq}", (f"{R} = {v.categoria} em {fmt_pct(x)} dos registros dos últimos "
                                               f"{v.janela} meses (limite {L})")
    if tr == "ultima":
        if x >= 0.5:
            return f"{R}: {v.categoria}", f"Último registro de {r}: {v.categoria}"
        return f"{R}: não {v.categoria}", f"Último registro de {r} diferente de {v.categoria}"
    if tr == "categoria":
        return (f"{R} = {v.categoria}", f"{R} = {v.categoria}") if x >= 0.5 else \
            (f"{R} ≠ {v.categoria}", f"{R} diferente de {v.categoria}")
    if tr == "estatico":
        if v.indicador:
            return f"{R}: {'sim' if x >= 0.5 else 'não'}", f"{R}: {'sim' if x >= 0.5 else 'não'}"
        return f"{R} {ab}", f"{R}: {fmt_num(x)} (limite {L})"
    if tr == "tempo_desde":
        return f"Tempo desde {r}", f"{fmt_num(x, 0)} meses desde {r} (limite {L})"
    if tr == "contagem":
        return v.rotulo, f"{fmt_num(x)} registros (limite {L})"
    if tr == "media_eventos":
        return f"{R} {ab}", f"{R}: {fmt_num(x)} em média (limite {L})"
    if tr == "freq_eventos":
        return f"{R}: {v.categoria}", f"{R} = {v.categoria} em {fmt_pct(x)} dos registros (limite {L})"
    if tr == "persistencia":
        b = meta[v.extra["base"]]
        n = int(round(x))
        lb = _lim(v.extra["limiar_base"], v.extra["direcao_base"], fracao=b.fracao)
        ab_b = "alto" if v.extra["direcao_base"] > 0 else "baixo"
        titulo = f"{b.rotulo_coluna} {ab_b} há {n} {'mês' if n == 1 else 'meses'}" if n > 0 else f"{b.rotulo_coluna} ainda sem persistência"
        return titulo, \
            (f"{n} {'mês' if n == 1 else 'meses seguidos'} com {minusc(b.rotulo)} {lb} "
             f"(alerta a partir de {fmt_num(lim, 0)})")
    return v.rotulo, f"{v.rotulo}: {fmt_num(x)}"


def acao_para(v: Variavel, acoes: dict[str, str]) -> str | None:
    for chave in (v.id, f"{v.tabela}.{v.coluna}", v.coluna, v.extra.get("base")):
        if chave and chave in acoes:
            return acoes[chave]
    return None


def explicar(modelo, X: pd.DataFrame, aux: pd.DataFrame, linhas: pd.Index, meses: pd.Series, clientes: pd.Series,
             meta: dict[str, Variavel]) -> pd.DataFrame:
    """Uma linha por cliente × variável do modelo: valor, contribuição, limiar, passa, dispara, secundario,
    peso, titulo, detalhe. `linhas` = índices do painel (uma por cliente)."""
    C = modelo.contribuicoes(X.loc[linhas])
    out = []
    for i in linhas:
        pos = float(C.loc[i].clip(lower=0).sum())
        axd = {k: aux.at[i, k] for k in aux.columns} if len(aux.columns) else {}
        for f in modelo.features:
            v = meta[f]
            x = float(X.at[i, f]) if pd.notna(X.at[i, f]) else np.nan
            c = float(C.at[i, f])
            lim, d = modelo.limiares.get(f, np.nan), modelo.direcoes[f]
            ps = passa(x, lim, d)
            disp = c > 0 and ps
            if np.isnan(x):
                t, det = v.rotulo, f"{v.rotulo}: {getattr(modelo, 'texto_nulo', 'sem dado (usada a mediana)')}"
            else:
                t, det = texto(v, x, axd, lim, d, disp, meses[i], meta)
            out.append({"cliente": clientes[i], "linha": i, "variavel": f, "valor": x, "contribuicao": c, "limiar": lim,
                        "passa_limiar": ps, "dispara": disp, "secundario": c > 0 and not disp,
                        "peso": c / pos if c > 0 and pos > 0 else 0.0, "titulo": t, "detalhe": det})
    E = pd.DataFrame(out)
    if E.empty:
        return pd.DataFrame(columns=["cliente", "linha", "variavel", "valor", "contribuicao", "limiar", "passa_limiar",
                                     "dispara", "secundario", "peso", "titulo", "detalhe"])
    return E.sort_values(["cliente", "contribuicao", "variavel"], ascending=[True, False, True]).reset_index(drop=True)


def acao_sugerida(E_cli: pd.DataFrame, faixa: str, meta: dict[str, Variavel], acoes: dict[str, str]) -> tuple[str, str | None]:
    """(ação, id da variável dominante)."""
    for col in ("dispara", "secundario"):
        if col == "secundario" and faixa == "baixo":
            break
        cand = E_cli[E_cli[col]]
        if not len(cand):
            continue
        for f in cand["variavel"]:
            a = acao_para(meta[f], acoes)
            if a:
                return a, f
        f = cand.iloc[0]["variavel"]
        return f"Investigar {minusc(meta[f].rotulo_coluna)}", f
    return ACAO_ROTINA, None
