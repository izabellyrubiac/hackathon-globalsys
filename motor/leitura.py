"""Leitura de bases arbitrárias: um .xlsx (uma tabela por aba) e/ou vários .csv (uma tabela por arquivo).

`ler_arquivos(caminhos)` devolve um `Tabelas` (dict nome → DataFrame) com dois atributos extras:
* `ignoradas`: {nome: motivo} — abas/arquivos descartados por parecerem documentação
  (ex.: "Leia-me", "dicionario") ou por estarem vazios;
* `avisos`: lista de textos pt-BR sobre o que foi ajustado na leitura.

Limpeza aplicada a toda tabela: nomes de coluna sem espaços nas pontas, linhas/colunas 100% vazias
removidas, colunas "Unnamed: n" vazias removidas, nomes repetidos desambiguados (`_2`) e colunas de
texto que são números (com vírgula ou ponto decimal) convertidas para número.
"""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path

import pandas as pd

from .util import eh_numerica, eh_texto, parece_numero_texto, texto_para_numero

EXT_EXCEL = {".xlsx", ".xlsm", ".xls"}
EXT_CSV = {".csv", ".tsv", ".txt"}

_NOME_DOC = re.compile(r"leia.?me|read.?me|dicion|dictionary|instru[cç]|^sobre|about|^notas?$|^notes?$|legenda|"
                       r"gloss[aá]r|metadad|metadata|^descri[cç]", re.I)
_COL_CAMPO = re.compile(r"^(campo|coluna|column|field|vari[aá]vel|atributo)s?$", re.I)
_COL_DESC = re.compile(r"^(descri[cç][aã]o|description|significado|defini[cç][aã]o)$", re.I)


class Tabelas(dict):
    """dict nome → DataFrame com `.ignoradas` ({nome: motivo}), `.descartadas` ({nome: DataFrame}) e `.avisos`."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ignoradas: dict[str, str] = {}
        self.descartadas: dict[str, pd.DataFrame] = {}
        self.avisos: list[str] = []


# --------------------------------------------------------------------------- API
def ler_arquivos(caminhos) -> Tabelas:
    """Lê .xlsx/.xls (todas as abas) e .csv/.tsv/.txt. Aceita um caminho ou uma lista."""
    if isinstance(caminhos, (str, Path)):
        caminhos = [caminhos]
    out = Tabelas()
    for c in caminhos:
        p = Path(c)
        if not p.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {p}")
        ext = p.suffix.lower()
        if ext in EXT_EXCEL:
            abas = pd.read_excel(p, sheet_name=None)
            brutas = list(abas.items())
        elif ext in EXT_CSV:
            brutas = [(p.stem, ler_csv(p))]
        else:
            raise ValueError(f"Formato não suportado: '{p.name}'. Envie .xlsx ou .csv.")
        for nome, df in brutas:
            _adicionar(out, str(nome).strip() or p.stem, df)
    if not out:
        raise ValueError("Nenhuma tabela de dados encontrada nos arquivos enviados.")
    return out


def de_dataframes(tabelas: dict[str, pd.DataFrame]) -> Tabelas:
    """Aplica a mesma limpeza/heurística de `ler_arquivos` a DataFrames já carregados."""
    out = Tabelas()
    for nome, df in tabelas.items():
        _adicionar(out, str(nome), df)
    return out


# --------------------------------------------------------------------------- internos
def _adicionar(out: Tabelas, nome: str, df: pd.DataFrame) -> None:
    base, i = nome, 2
    while nome in out or nome in out.ignoradas:
        nome = f"{base}_{i}"
        i += 1
    df = limpar(df)
    motivo = motivo_documentacao(nome, df)
    if motivo:
        out.ignoradas[nome] = motivo
        out.descartadas[nome] = df
        out.avisos.append(f"Tabela '{nome}' ignorada: {motivo}.")
        return
    out[nome] = df


def ler_csv(caminho: Path) -> pd.DataFrame:
    bruto = Path(caminho).read_bytes()
    for enc in ("utf-8-sig", "latin-1"):
        try:
            txt = bruto.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    amostra = txt[:20000]
    try:
        sep = csv.Sniffer().sniff(amostra, delimiters=",;\t|").delimiter
    except csv.Error:
        sep = max([",", ";", "\t", "|"], key=lambda s: amostra.split("\n")[0].count(s))
    df = pd.read_csv(io.StringIO(txt), sep=sep, dtype=str, keep_default_na=True)
    return df


def limpar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    df = df.loc[:, [not (c.startswith("Unnamed:") and df[c].isna().all()) for c in df.columns]]
    # nomes repetidos
    vistos: dict[str, int] = {}
    cols = []
    for c in df.columns:
        if c in vistos:
            vistos[c] += 1
            cols.append(f"{c}_{vistos[c]}")
        else:
            vistos[c] = 1
            cols.append(c)
    df.columns = cols
    # texto: espaços nas pontas; texto numérico → número
    for c in df.columns:
        s = df[c]
        if eh_texto(s):
            s = s.astype("string").str.strip().replace("", pd.NA)
            if parece_numero_texto(s):
                df[c] = texto_para_numero(s)
            else:
                df[c] = s.astype(object).where(s.notna(), None)
    return df.reset_index(drop=True)


def motivo_documentacao(nome: str, df: pd.DataFrame) -> str | None:
    """Heurística: a tabela parece documentação (texto livre) e não dados?"""
    if df.empty or df.shape[1] == 0:
        return "tabela vazia"
    texto_cols = [c for c in df.columns if not eh_numerica(df[c])]
    so_texto = len(texto_cols) == df.shape[1]
    if _NOME_DOC.search(nome) and so_texto:
        return "parece documentação (nome da aba e só texto)"
    cols = [str(c) for c in df.columns]
    if any(_COL_CAMPO.match(c) for c in cols) and any(_COL_DESC.match(c) for c in cols):
        return "parece um dicionário de dados (colunas de campo e descrição)"
    if so_texto and df.shape[1] <= 2:
        comp = pd.concat([df[c].dropna().astype(str).str.len() for c in df.columns])
        if len(comp) and comp.mean() > 30:
            return "parece documentação (1–2 colunas de texto longo)"
    return None
