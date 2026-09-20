"""Preparação genérica da base e do painel cliente × mês (sem vazamento).

Tempo
-----
* Toda coluna de data é convertida para mês (AAAA-MM, datas completas e datetime → mês).
* `mes_ref` = último mês com dados nas tabelas temporais densas (o "hoje" da fila).
* Cancelados: linhas com mês ≥ mês de saída são descartadas (se a base traz o mês da saída, ele sai —
  evita o sinal que "só aparece no mês da saída"). O `deslocamento` (mediana de saída − último mês com
  dados) é reportado: 1 = o histórico termina no mês anterior à saída.
* Sem data de saída, a saída é o mês seguinte ao último registro do cancelado.

Painel
------
Uma linha por cliente × mês, do primeiro mês com dados até o fim (cancelados: mês anterior à saída;
ativos: `mes_ref`). `k` = meses até a saída (ativos: saída = `mes_ref` + 1), logo k = 1 é o último mês
observado nos dois grupos. Rótulo `y` = cancelado com k ≤ horizonte ("cancela nos próximos H meses").
Treino: linha com dados, ≥ `min_hist` meses de histórico e rótulo conhecido — linhas de ativos nos
últimos H meses ficam fora do treino (censura: ainda não se sabe se cancelam), mas recebem score.
Sem nenhuma tabela temporal, o painel é uma fotografia (uma linha por cliente, y = cancelou).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Config
from .esquema import Estrutura, coluna_chave, estruturar
from .mapeamento import Mapeamento
from .util import meses_entre, norm_id, norm_valor, para_mes, rotulo_coluna

ORDEM_PAPEL = {"clientes": 0, "estatica": 1, "temporal": 2, "eventos": 3, "ignorada": 9}
MES_FOTO = pd.Period("2000-01", "M")      # mês fictício do modo fotografia


@dataclass
class TabelaTemporal:
    nome: str
    df: pd.DataFrame                  # colunas __cli, __mes + colunas de dados
    colunas: dict[str, str]           # coluna → tipo (só as utilizáveis)
    densa: bool
    transacional: bool                # > 1,2 linhas por cliente-mês → vira contagem mensal
    densidade: float


@dataclass
class TabelaCliente:                  # estática (uma linha por cliente) ou eventos (várias, sem data)
    nome: str
    df: pd.DataFrame                  # estática: índice = cliente · eventos: coluna __cli
    colunas: dict[str, str]
    papel: str


@dataclass
class Base:
    ids: list[str]
    cancelado: pd.Series
    mes_saida: pd.Series              # saída efetiva (Period) dos cancelados; NaT nos ativos
    valor: pd.Series | None
    atributos: pd.DataFrame
    temporais: list[TabelaTemporal]
    estaticas: list[TabelaCliente]
    mes_ref: pd.Period | None
    fim_obs: pd.Period | None
    deslocamento: float | None
    estrutura: Estrutura
    rotulos: dict[str, str]
    avisos: list[str] = field(default_factory=list)
    excluidos: list[str] = field(default_factory=list)   # cancelados sem dados antes da saída
    datas: pd.DataFrame | None = None  # colunas de data da tabela de clientes (AAAA-MM), uma linha por cliente

    @property
    def fotografia(self) -> bool:
        return not self.temporais


def _colunas_utilizaveis(nome: str, df: pd.DataFrame, tipos: dict[str, str], excluir: set, m: Mapeamento) -> dict:
    return {c: tipos[c] for c in df.columns
            if c not in excluir and not m.ignorada(nome, c) and tipos.get(c) in ("numerica", "binaria", "categorica", "data")}


def preparar(tabelas: dict[str, pd.DataFrame], m: Mapeamento, cfg: Config) -> Base:
    est = estruturar(tabelas, m.tabela_clientes, m.coluna_id)
    avisos = list(est.avisos)
    cli = tabelas[m.tabela_clientes]
    ids_serie = norm_id(cli[m.coluna_id])
    ids = sorted(ids_serie.dropna().unique())
    idx = pd.Index(ids, name="cliente")

    # alvo ----------------------------------------------------------------
    alvo = tabelas[m.alvo_tabela]
    chave_alvo = m.coluna_id if m.alvo_tabela == m.tabela_clientes else coluna_chave(alvo, set(ids), m.coluna_id)
    a = pd.DataFrame({"cli": norm_id(alvo[chave_alvo])})
    if m.coluna_situacao:
        a["canc"] = alvo[m.coluna_situacao].map(norm_valor).eq(norm_valor(m.valor_cancelado)).to_numpy()
    if m.coluna_data_saida:
        a["saida"] = para_mes(alvo[m.coluna_data_saida]).to_numpy()
        if not m.coluna_situacao:
            a["canc"] = a["saida"].notna()
    else:
        a["saida"] = pd.Series(pd.NaT, index=a.index, dtype="period[M]")
    a = a.dropna(subset=["cli"]).drop_duplicates("cli", keep="last").set_index("cli")
    cancelado = a["canc"].reindex(idx).fillna(False).astype(bool)
    saida = a["saida"].reindex(idx)
    saida = saida.where(cancelado)

    # valor ----------------------------------------------------------------
    valor = None
    if m.coluna_valor:
        vt = m.valor_tabela or m.tabela_clientes
        dfv = tabelas[vt]
        iv = est.tabelas[vt]
        v = pd.DataFrame({"cli": norm_id(dfv[iv.chave]), "v": pd.to_numeric(dfv[m.coluna_valor], errors="coerce")})
        if iv.papel == "temporal" and iv.coluna_data:
            v["mes"] = para_mes(dfv[iv.coluna_data]).to_numpy()
            v = v.dropna(subset=["v"]).sort_values(["cli", "mes"])
        valor = v.dropna(subset=["cli"]).groupby("cli")["v"].last().reindex(idx)
        if valor.isna().any():
            avisos.append(f"{int(valor.isna().sum())} cliente(s) sem valor de contrato: a perda esperada deles fica vazia "
                          "e eles vão para o fim da fila de mesma probabilidade.")

    # atributos descritivos (categóricas da tabela de clientes) --------------
    icli = est.tabelas[m.tabela_clientes]
    attr_cols = [c for c in cli.columns if c != m.coluna_id and icli.tipos[c] in ("categorica", "binaria")
                 and (m.tabela_clientes, c) not in m.colunas_alvo()][:8]
    atributos = cli.assign(__cli=ids_serie).dropna(subset=["__cli"]).drop_duplicates("__cli").set_index("__cli")[attr_cols]
    atributos = atributos.reindex(idx)
    # datas da tabela de clientes (ex.: início do contrato), normalizadas em AAAA-MM: viram filtros na fila
    data_cols = [c for c in cli.columns if c != m.coluna_id and icli.tipos[c] == "data"
                 and (m.tabela_clientes, c) not in m.colunas_alvo() and not m.ignorada(m.tabela_clientes, c)][:4]
    datas = pd.DataFrame(index=idx)
    if data_cols:
        d = cli.assign(__cli=ids_serie).dropna(subset=["__cli"]).drop_duplicates("__cli").set_index("__cli")[data_cols]
        datas = d.apply(lambda s: _mes_txt(para_mes(s))).reindex(idx)

    # tabelas de variáveis (ordem canônica: papel, linhas, colunas — independe de nomes e da ordem das abas)
    infos = sorted(est.tabelas.values(), key=lambda i: (ORDEM_PAPEL[i.papel], -i.linhas, -len(i.tipos), i.nome))
    temporais: list[TabelaTemporal] = []
    estaticas: list[TabelaCliente] = []
    for it in infos:
        if it.papel == "ignorada":
            continue
        if it.nome == m.alvo_tabela and m.alvo_tabela != m.tabela_clientes:
            continue                                   # tabela do alvo inteira fica de fora (vazamento)
        df = tabelas[it.nome]
        excluir = {it.chave, it.coluna_data}
        cols = _colunas_utilizaveis(it.nome, df, it.tipos, excluir, m)
        c_ids = norm_id(df[it.chave])
        if it.papel in ("clientes", "estatica"):
            d = df.assign(__cli=c_ids).dropna(subset=["__cli"]).drop_duplicates("__cli").set_index("__cli")
            estaticas.append(TabelaCliente(it.nome, d[list(cols)].reindex(idx), cols, it.papel))
        elif it.papel == "eventos":
            cols = {c: t for c, t in cols.items() if t != "data"}
            d = df[list(cols)].assign(__cli=c_ids.to_numpy())
            d = d[d["__cli"].isin(ids)]
            estaticas.append(TabelaCliente(it.nome, d, cols, "eventos"))
            avisos.append(f"Tabela '{it.nome}' não tem data: as variáveis dela usam todos os registros do cliente "
                          "(podem incluir registros posteriores à saída — confira o alerta de vazamento).")
        else:
            cols = {c: t for c, t in cols.items() if t != "data"}
            d = df[list(cols)].assign(__cli=c_ids.to_numpy(), __mes=para_mes(df[it.coluna_data]).to_numpy())
            fora = int((~d["__cli"].isin(ids)).sum())
            if fora:
                avisos.append(f"Tabela '{it.nome}': {fora} linha(s) com ID fora da tabela de clientes foram ignoradas.")
            d = d[d["__cli"].isin(ids) & d["__mes"].notna()].reset_index(drop=True)
            if d.empty:
                continue
            temporais.append(TabelaTemporal(it.nome, d, cols, True, False, 1.0))

    # densidade das tabelas temporais --------------------------------------------
    for t in temporais:
        g = t.df.groupby("__cli")["__mes"]
        span = (meses_entre(g.max(), g.min()) + 1).astype(float)
        n_meses = t.df.drop_duplicates(["__cli", "__mes"]).groupby("__cli").size()
        t.densidade = float(n_meses.sum() / span.sum()) if span.sum() else 0.0
        t.densa = t.densidade >= cfg.densidade_minima
        t.transacional = len(t.df) / max(len(t.df.drop_duplicates(["__cli", "__mes"])), 1) > 1.2
    densas = [t for t in temporais if t.densa] or temporais

    mes_ref = fim_obs = None
    deslocamento = None
    if temporais:
        pres = pd.concat([t.df[["__cli", "__mes"]] for t in densas]).drop_duplicates()
        ultimo = pres.groupby("__cli")["__mes"].max()
        # convenção do fim do histórico dos cancelados
        sc = saida.dropna()
        comum = sc.index.intersection(ultimo.index)
        if len(comum):
            gaps = meses_entre(sc[comum], ultimo[comum])
            deslocamento = float(np.median(gaps))
            if deslocamento >= 1:
                avisos.append(f"Histórico dos cancelados termina, na mediana, {deslocamento:g} mês(es) antes do mês de saída.")
            else:
                avisos.append("O histórico dos cancelados vai até o mês da saída (ou depois): os meses a partir da saída "
                              "foram descartados para evitar vazamento.")
        # corta linhas a partir da saída
        for t in temporais:
            s = t.df["__cli"].map(saida)
            corta = s.notna() & (meses_entre(t.df["__mes"], s.astype("period[M]")) >= 0)
            if corta.any():
                t.df = t.df[~corta].reset_index(drop=True)
        pres = pd.concat([t.df[["__cli", "__mes"]] for t in densas]).drop_duplicates()
        ultimo = pres.groupby("__cli")["__mes"].max()
        mes_ref = pres["__mes"].max()
        # cancelados sem data de saída: mês seguinte ao último registro
        sem = cancelado & saida.isna()
        if sem.any():
            saida = saida.astype("period[M]")
            for c in saida.index[sem]:
                if c in ultimo.index:
                    saida[c] = ultimo[c] + 1
            if m.coluna_data_saida:
                avisos.append(f"{int(sem.sum())} cancelado(s) sem data de saída: saída = mês seguinte ao último registro.")
        maior_saida = saida.dropna().max() if saida.notna().any() else mes_ref
        fim_obs = max(mes_ref, maior_saida)
        ativos_atrasados = (~cancelado) & (ultimo.reindex(idx) < mes_ref)
        if ativos_atrasados.sum():
            avisos.append(f"{int(ativos_atrasados.sum())} cliente(s) ativo(s) sem dados em {mes_ref}: o score deles usa "
                          "os últimos dados disponíveis.")
    else:
        avisos.append("Nenhuma tabela temporal: o modelo é uma fotografia (cancelou × ativo) sem antecedência.")

    rotulos = dict(m.rotulos)
    return Base(ids, cancelado, saida, valor, atributos, temporais, estaticas, mes_ref, fim_obs, deslocamento, est,
                rotulos, avisos, datas=datas)


def _mes_txt(p: pd.Series) -> pd.Series:
    """Period mensal → 'AAAA-MM' (NaT continua vazio)."""
    return p.astype(str).where(p.notna())


def rotulo(base: Base, coluna: str) -> str:
    return base.rotulos.get(coluna) or rotulo_coluna(coluna)


# --------------------------------------------------------------------------- painel
def montar_grade(base: Base, m: Mapeamento, cfg: Config) -> pd.DataFrame:
    """Painel cliente × mês. Colunas: cliente, mes, k, cancelado, y, tem_dados, n_hist, treino, pontuavel, referencia."""
    H = int(m.horizonte_meses)
    ids = pd.Index(base.ids)
    if base.fotografia:
        g = pd.DataFrame({"cliente": ids, "mes": MES_FOTO, "k": 1, "cancelado": base.cancelado.to_numpy(),
                          "tem_dados": True, "n_hist": cfg.min_hist})
        g["y"] = g["cancelado"].astype(int)
        g["treino"] = True
        g["referencia"] = ~g["cancelado"]
        g["pontuavel"] = True
        return g

    densas = [t for t in base.temporais if t.densa] or base.temporais
    pres = pd.concat([t.df[["__cli", "__mes"]] for t in densas]).drop_duplicates()
    primeiro = pres.groupby("__cli")["__mes"].min()
    linhas = []
    for c in ids:
        canc = bool(base.cancelado[c])
        if canc:
            s = base.mes_saida[c]
            if pd.isna(s):
                base.excluidos.append(c)
                continue
            fim = min(s - 1, base.mes_ref)
            saida = s
        else:
            fim = base.mes_ref
            saida = base.mes_ref + 1
        ini = primeiro.get(c, fim)
        if pd.isna(ini) or ini > fim:
            if canc:
                base.excluidos.append(c)
                continue
            ini = fim
        n = meses_entre(fim, ini) + 1
        meses = pd.period_range(ini, periods=n, freq="M")
        linhas.append(pd.DataFrame({"cliente": c, "mes": meses, "k": [meses_entre(saida, x) for x in meses],
                                    "cancelado": canc}))
    if base.excluidos:
        base.avisos.append(f"{len(base.excluidos)} cancelado(s) sem nenhum dado antes da saída ficaram fora do painel.")
    g = pd.concat(linhas, ignore_index=True)
    chave = pd.MultiIndex.from_frame(pres.rename(columns={"__cli": "cliente", "__mes": "mes"}))
    g["tem_dados"] = pd.MultiIndex.from_frame(g[["cliente", "mes"]]).isin(chave)
    g["n_hist"] = g.groupby("cliente")["tem_dados"].cumsum()
    g["y"] = (g["cancelado"] & (g["k"] <= H)).astype(int)
    conhecido = g["cancelado"] | (g["mes"].map(lambda x: meses_entre(base.fim_obs, x)) >= H)
    g["treino"] = g["tem_dados"] & (g["n_hist"] >= cfg.min_hist) & conhecido
    g["referencia"] = (~g["cancelado"]) & (g["mes"] == base.mes_ref)
    g["pontuavel"] = g["tem_dados"] | g["referencia"]
    return g.reset_index(drop=True)
