"""Inferência de esquema: tipo de cada coluna, papel de cada tabela e sugestão de mapeamento.

Tipos de coluna: "id" | "numerica" | "categorica" | "data" | "texto" | "binaria".
Papéis de tabela:
* "clientes"  — a tabela de clientes (uma linha por cliente);
* "estatica"  — uma linha por cliente (ex.: situação, cadastro complementar);
* "temporal"  — várias linhas por cliente com uma coluna de data/mês (ex.: atendimento mensal, pesquisas);
* "eventos"   — várias linhas por cliente sem data (ex.: lista de produtos contratados);
* "ignorada"  — sem coluna ligada ao ID do cliente (ou documentação descartada na leitura).

`inspecionar(tabelas)` devolve um dict JSON-serializável (contrato da API):
    tabelas[{nome, linhas, papel_sugerido, chave, coluna_data, colunas[{nome, tipo, unicos, nulos_pct, exemplos}]}],
    sugestao {tabela_clientes, coluna_id, alvo{tabela, coluna_situacao, valor_cancelado, coluna_data_saida},
              coluna_valor{tabela, coluna} | None, horizonte_meses},
    avisos [str]
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .util import eh_datetime, eh_numerica, fracao_datas, norm_id, norm_valor, para_mes

TIPOS = ("id", "numerica", "categorica", "data", "texto", "binaria")
MAX_CATEGORIAS = 15

_NOME_ID = re.compile(r"(^id$|^id_|_id$|^cod|codigo|c[oó]digo|chave|^key$|_key$|uuid|^cpf|^cnpj)", re.I)
_NOME_CLIENTE = re.compile(r"client|customer|conta|account|assinante|usu[aá]rio|user|empresa|associad|cadastro", re.I)
_NOME_DATA = re.compile(r"m[eê]s|data|date|dt_|_dt|per[ií]odo|ref|compet|ano_?mes|dia|semana|trimestre", re.I)
_NOME_ALVO = re.compile(r"situa|status|churn|cancel|ativo|evad|saiu|sa[ií]da|estado", re.I)
_VALOR_CANCEL = re.compile(r"cancel|churn|inativ|saiu|sa[ií]da|encerr|perdid|desist|rescis|evad|exit|lost|"
                           r"inactive|closed|terminat|desligad|baixad", re.I)
_NOME_SAIDA = re.compile(r"cancel|sa[ií]da|churn|fim|encerr|t[eé]rmino|rescis|desligam|evas", re.I)
_NOME_VALOR = re.compile(r"valor|receita|mrr|mensalidade|fee|arr$|ticket|faturamento|pre[cç]o|price|revenue|amount|"
                         r"contrato_?valor|mensal", re.I)
_NOME_VALOR_NAO = re.compile(r"atraso|dias|pct|percent|nota|qtd|quant|num_|n_", re.I)


# --------------------------------------------------------------------------- tipos
def tipo_coluna(s: pd.Series, nome: str = "") -> str:
    """Tipo inferido sem conhecer o ID do cliente (o 'id' de chave é marcado depois)."""
    x = s.dropna()
    n = len(x)
    if n == 0:
        return "texto"
    if eh_datetime(s):
        return "data"
    if pd.api.types.is_bool_dtype(s):
        return "binaria"
    unicos = x.nunique()
    if eh_numerica(s):
        if unicos <= 2:
            return "binaria"
        if _NOME_DATA.search(nome) and fracao_datas(x) >= 0.95:
            return "data"
        inteiro = np.allclose(x, np.round(x))
        if inteiro and _NOME_ID.search(nome) and unicos >= 0.9 * n:
            return "id"
        return "numerica"
    if fracao_datas(x) >= 0.9:
        return "data"
    if unicos <= 2:
        return "binaria"
    if _NOME_ID.search(nome) and unicos >= 0.5 * n:
        return "id"
    if unicos <= MAX_CATEGORIAS or unicos <= 0.05 * n:
        return "categorica" if unicos <= max(MAX_CATEGORIAS, 30) else "texto"
    return "texto"


@dataclass
class InfoTabela:
    nome: str
    linhas: int
    tipos: dict[str, str]
    chave: str | None = None             # coluna com o ID do cliente
    coluna_data: str | None = None       # coluna de tempo (tabelas temporais)
    papel: str = "ignorada"
    linhas_por_cliente: float = 0.0
    motivo: str | None = None


@dataclass
class Estrutura:
    tabelas: dict[str, InfoTabela]
    tabela_clientes: str | None
    coluna_id: str | None
    ids: set = field(default_factory=set)
    avisos: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- ID do cliente
def _valores_id(s: pd.Series) -> set:
    return set(norm_id(s).dropna())


def _candidatos_id(tabelas: dict[str, pd.DataFrame], tipos: dict[str, dict[str, str]]):
    """(pontuação, tabela, coluna) de colunas únicas que podem ser o ID do cliente."""
    conjuntos = {}
    for t, df in tabelas.items():
        for c in df.columns:
            if tipos[t][c] in ("data", "binaria"):
                continue
            if eh_numerica(df[c]) and not np.allclose(df[c].dropna(), np.round(df[c].dropna())):
                continue
            conjuntos[(t, c)] = _valores_id(df[c])
    cands = []
    for t, df in tabelas.items():
        for c in df.columns:
            if (t, c) not in conjuntos:
                continue
            s = df[c]
            if s.isna().mean() > 0.05 or s.nunique() != s.notna().sum() or s.nunique() < 2:
                continue
            ids = conjuntos[(t, c)]
            # nº de outras tabelas com uma coluna cujos valores estão (≥ 90%) entre estes IDs
            ligacoes = len({t2 for (t2, c2), v2 in conjuntos.items() if t2 != t and v2
                            and len(v2 & ids) / len(v2) >= 0.9 and len(v2 & ids) >= 0.3 * len(ids)})
            pont = 10 * ligacoes + 0.1 * df.shape[1] + (2 if _NOME_ID.search(c) else 0) \
                + (3 if _NOME_CLIENTE.search(t) else 0) + (1 if _NOME_CLIENTE.search(c) else 0) \
                - (2 if eh_numerica(s) and not _NOME_ID.search(c) else 0)
            cands.append((pont, ligacoes, t, c))
    cands.sort(key=lambda x: (-x[0], x[2], x[3]))
    return cands


def coluna_chave(df: pd.DataFrame, ids: set, preferida: str | None = None) -> str | None:
    """Coluna de `df` que contém o ID do cliente (≥ 90% dos valores entre os IDs)."""
    melhor, melhor_pont = None, 0.0
    for c in df.columns:
        s = df[c]
        if eh_datetime(s):
            continue
        v = norm_id(s).dropna()
        if len(v) == 0:
            continue
        dentro = v.isin(ids).mean()
        if dentro < 0.9:
            continue
        pont = dentro + (1 if c == preferida else 0) + 0.01 * min(v.nunique(), len(ids)) / max(len(ids), 1)
        if pont > melhor_pont:
            melhor, melhor_pont = c, pont
    return melhor


def _coluna_data(df: pd.DataFrame, tipos: dict[str, str], excluir: set) -> str | None:
    cands = [c for c in df.columns if tipos.get(c) == "data" and c not in excluir]
    if not cands:
        return None
    return sorted(cands, key=lambda c: (df[c].isna().mean(), 0 if _NOME_DATA.search(c) else 1,
                                        -df[c].nunique(), list(df.columns).index(c)))[0]


# --------------------------------------------------------------------------- estrutura
def estruturar(tabelas: dict[str, pd.DataFrame], tabela_clientes: str | None = None,
               coluna_id: str | None = None) -> Estrutura:
    """Tipos, chave, coluna de data e papel de cada tabela. Sem `tabela_clientes`, sugere."""
    tipos = {t: {c: tipo_coluna(df[c], c) for c in df.columns} for t, df in tabelas.items()}
    avisos = []
    if tabela_clientes is None or coluna_id is None:
        cands = _candidatos_id(tabelas, tipos)
        if cands:
            _, _, tabela_clientes, coluna_id = cands[0]
        else:
            avisos.append("Nenhuma coluna com um valor único por linha foi encontrada para identificar o cliente.")
    ids = _valores_id(tabelas[tabela_clientes][coluna_id]) if tabela_clientes else set()

    info: dict[str, InfoTabela] = {}
    for t, df in tabelas.items():
        it = InfoTabela(t, len(df), dict(tipos[t]))
        if t == tabela_clientes:
            it.chave = coluna_id
        elif ids:
            it.chave = coluna_chave(df, ids, preferida=coluna_id)
        if it.chave is None:
            it.papel, it.motivo = "ignorada", "não tem coluna com o ID do cliente"
            if t != tabela_clientes and tabela_clientes:
                avisos.append(f"Tabela '{t}' não tem coluna ligada ao ID do cliente e será ignorada.")
            info[t] = it
            continue
        it.tipos[it.chave] = "id"
        chave_norm = norm_id(df[it.chave])
        por_cli = chave_norm.dropna().value_counts()
        it.linhas_por_cliente = float(por_cli.mean()) if len(por_cli) else 0.0
        # outras colunas que são IDs (valores quase todos únicos em texto)
        if t == tabela_clientes:
            it.papel = "clientes"
        elif por_cli.max() <= 1:
            it.papel = "estatica"
        else:
            it.coluna_data = _coluna_data(df, it.tipos, {it.chave})
            it.papel = "temporal" if it.coluna_data else "eventos"
        info[t] = it
    return Estrutura(info, tabela_clientes, coluna_id, ids, avisos)


# --------------------------------------------------------------------------- sugestões
def _sugerir_alvo(tabelas, est: Estrutura) -> dict | None:
    cands = []
    ordem = sorted(est.tabelas.values(), key=lambda i: (i.papel not in ("clientes", "estatica"), i.nome))
    for it in ordem:
        if it.papel == "ignorada":
            continue
        df = tabelas[it.nome]
        for c in df.columns:
            if c == it.chave or it.tipos[c] not in ("binaria", "categorica"):
                continue
            vals = df[c].dropna()
            if vals.nunique() > 3 or vals.nunique() < 2:
                continue
            valor = None
            for v in sorted(vals.unique(), key=lambda a: str(a)):
                if isinstance(v, str) and _VALOR_CANCEL.search(v) and not re.search(r"^n[aã]o|^not", v, re.I):
                    valor = v
                    break
            if valor is None and _NOME_ALVO.search(c) and set(norm_valor(v) for v in vals.unique()) <= {"0", "1"}:
                valor = 1 if eh_numerica(df[c]) else next(v for v in vals.unique() if norm_valor(v) == "1")
            if valor is None:
                continue
            cancel = vals.map(norm_valor) == norm_valor(valor)
            frac = cancel.mean()
            if not 0.01 <= frac <= 0.9:
                continue
            pont = (3 if _NOME_ALVO.search(c) else 0) + (2 if it.papel in ("clientes", "estatica") else 0)
            cands.append((pont, it.nome, c, valor))
    # coluna de data de saída
    def data_saida(tabela: str, mascara_cancel: pd.Series | None):
        it = est.tabelas[tabela]
        df = tabelas[tabela]
        melhor, melhor_p = None, -1.0
        for c in df.columns:
            if it.tipos.get(c) != "data" or c == it.coluna_data:
                continue
            preenchida = para_mes(df[c]).notna()
            if mascara_cancel is not None:
                concord = float((preenchida == mascara_cancel).mean())
                if concord < 0.9:
                    continue
            else:
                concord = 1.0 if 0.01 <= preenchida.mean() <= 0.9 else 0.0
                if not concord:
                    continue
            p = concord + (1 if _NOME_SAIDA.search(c) else 0)
            if mascara_cancel is None and not _NOME_SAIDA.search(c):
                continue
            if p > melhor_p:
                melhor, melhor_p = c, p
        return melhor

    if cands:
        cands.sort(key=lambda x: (-x[0], x[1], x[2]))
        _, t, c, valor = cands[0]
        df = tabelas[t]
        mascara = df[c].map(norm_valor) == norm_valor(valor)
        valor_out = valor.item() if hasattr(valor, "item") else valor
        return {"tabela": t, "coluna_situacao": c, "valor_cancelado": valor_out,
                "coluna_data_saida": data_saida(t, mascara)}
    # sem coluna de situação: data de saída preenchida só para quem saiu
    for it in ordem:
        if it.papel in ("clientes", "estatica"):
            c = data_saida(it.nome, None)
            if c:
                return {"tabela": it.nome, "coluna_situacao": None, "valor_cancelado": None, "coluna_data_saida": c}
    return None


def _sugerir_valor(tabelas, est: Estrutura, alvo: dict | None) -> dict | None:
    excl = set()
    if alvo:
        excl = {(alvo["tabela"], alvo["coluna_situacao"]), (alvo["tabela"], alvo["coluna_data_saida"])}
    cands = []
    for it in est.tabelas.values():
        if it.papel not in ("clientes", "estatica", "temporal"):
            continue
        df = tabelas[it.nome]
        for c in df.columns:
            if (it.nome, c) in excl or it.tipos.get(c) != "numerica" or not _NOME_VALOR.search(c):
                continue
            if _NOME_VALOR_NAO.search(c) and not re.search(r"valor|receita|mrr|mensalidade", c, re.I):
                continue
            x = df[c].dropna()
            if len(x) == 0 or (x < 0).mean() > 0.05:
                continue
            pont = (2 if it.papel == "clientes" else 1 if it.papel == "estatica" else 0) \
                + (2 if re.search(r"valor|mrr|mensalidade|receita", c, re.I) else 0)
            cands.append((pont, it.nome, c))
    if not cands:
        return None
    cands.sort(key=lambda x: (-x[0], x[1], x[2]))
    return {"tabela": cands[0][1], "coluna": cands[0][2]}


def _exemplos(s: pd.Series, n: int = 3) -> list:
    out = []
    for v in s.dropna().drop_duplicates().head(n):
        if isinstance(v, (pd.Timestamp, pd.Period)):
            out.append(str(v)[:10])
        elif isinstance(v, (np.integer,)):
            out.append(int(v))
        elif isinstance(v, (float, np.floating)):
            out.append(round(float(v), 4))
        elif isinstance(v, (bool, np.bool_)):
            out.append(bool(v))
        else:
            out.append(str(v))
    return out


def inspecionar(tabelas: dict[str, pd.DataFrame]) -> dict:
    """Resumo das tabelas + sugestão de mapeamento. JSON-serializável."""
    est = estruturar(tabelas)
    alvo = _sugerir_alvo(tabelas, est) if est.tabela_clientes else None
    valor = _sugerir_valor(tabelas, est, alvo) if est.tabela_clientes else None
    avisos = list(getattr(tabelas, "avisos", [])) + est.avisos
    if est.tabela_clientes and alvo is None:
        avisos.append("Não encontrei a coluna de cancelamento automaticamente: indique a coluna de situação "
                      "(e o valor que significa cancelado) ou a coluna com a data de saída.")
    elif alvo and alvo["coluna_data_saida"] is None:
        avisos.append("Não encontrei a data de saída dos cancelados: sem ela, a saída é considerada o mês seguinte "
                      "ao último registro de cada cancelado.")
    if est.tabela_clientes and valor is None:
        avisos.append("Nenhuma coluna de valor do contrato sugerida: sem ela a fila é ordenada só pela probabilidade.")
    if not any(i.papel == "temporal" for i in est.tabelas.values()):
        avisos.append("Nenhuma tabela com data/mês por cliente: o modelo fica sem antecedência (fotografia única).")

    lista = []
    for t, df in tabelas.items():
        it = est.tabelas[t]
        cols = [{"nome": c, "tipo": it.tipos[c], "unicos": int(df[c].nunique()),
                 "nulos_pct": round(100 * float(df[c].isna().mean()), 1), "exemplos": _exemplos(df[c])}
                for c in df.columns]
        lista.append({"nome": t, "linhas": int(len(df)), "papel_sugerido": it.papel, "chave": it.chave,
                      "coluna_data": it.coluna_data, "colunas": cols, "motivo": it.motivo})
    for t, motivo in getattr(tabelas, "ignoradas", {}).items():
        df = getattr(tabelas, "descartadas", {}).get(t, pd.DataFrame())
        lista.append({"nome": t, "linhas": int(len(df)), "papel_sugerido": "ignorada", "chave": None,
                      "coluna_data": None, "motivo": motivo,
                      "colunas": [{"nome": str(c), "tipo": "texto", "unicos": int(df[c].nunique()),
                                   "nulos_pct": round(100 * float(df[c].isna().mean()), 1),
                                   "exemplos": _exemplos(df[c])} for c in df.columns]})
    sug = {"tabela_clientes": est.tabela_clientes, "coluna_id": est.coluna_id, "alvo": alvo,
           "coluna_valor": valor, "horizonte_meses": 3}
    return {"tabelas": lista, "sugestao": sug, "avisos": avisos}
