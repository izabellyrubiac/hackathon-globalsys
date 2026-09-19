"""Funções auxiliares do motor: datas → mês, IDs, formatação pt-BR, AUC e rótulos legíveis."""

from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

MESES_PT = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]

_RE_DATA = [
    re.compile(r"^\d{4}-\d{1,2}(-\d{1,2})?([ T].*)?$"),       # 2025-01 · 2025-01-31 · 2025-01-31 10:00
    re.compile(r"^\d{4}/\d{1,2}(/\d{1,2})?$"),                 # 2025/01 · 2025/01/31
    re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4}([ T].*)?$"),         # 31/01/2025
    re.compile(r"^\d{1,2}/\d{4}$"),                            # 01/2025
    re.compile(r"^\d{1,2}-\d{1,2}-\d{4}$"),                    # 31-01-2025
]
_RE_NUM = re.compile(r"^[+-]?(\d{1,3}(\.\d{3})+|\d+)(,\d+)?$|^[+-]?(\d{1,3}(,\d{3})+|\d+)(\.\d+)?$")


# --------------------------------------------------------------------------- tipos básicos
def eh_numerica(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def eh_datetime(s: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(s) or isinstance(s.dtype, pd.PeriodDtype)


def eh_texto(s: pd.Series) -> bool:
    return not (eh_numerica(s) or eh_datetime(s) or pd.api.types.is_bool_dtype(s))


def fracao_datas(s: pd.Series, amostra: int = 300) -> float:
    """Fração dos valores não nulos (amostra) com cara de data (regex + conversão)."""
    if eh_datetime(s):
        return 1.0
    x = s.dropna()
    if eh_numerica(x):           # AAAAMM inteiro (202501)
        if len(x) == 0 or not np.allclose(x, np.round(x)):
            return 0.0
        v = x.astype("int64")
        ok = v.between(190001, 210012) & (v % 100).between(1, 12)
        return float(ok.mean())
    x = x.astype(str).str.strip()
    x = x[x != ""]
    if len(x) == 0:
        return 0.0
    x = x.drop_duplicates().head(amostra)
    casa = x.map(lambda v: any(r.match(v) for r in _RE_DATA))
    if casa.mean() < 0.9:
        return float(casa.mean())
    conv = para_mes(x)
    return float(conv.notna().mean())


def para_mes(s: pd.Series) -> pd.Series:
    """Converte qualquer coluna de data (Period, datetime, 'AAAA-MM', 'AAAA-MM-DD', 'DD/MM/AAAA',
    'MM/AAAA', AAAAMM inteiro) em Period mensal. Valores não reconhecidos viram NaT."""
    if isinstance(s.dtype, pd.PeriodDtype):
        return s.dt.asfreq("M")
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.dt.to_period("M")
    idx = s.index
    if eh_numerica(s):
        v = pd.to_numeric(s, errors="coerce")
        txt = v.map(lambda a: f"{int(a):06d}" if pd.notna(a) else None)
        dt = pd.to_datetime(txt, format="%Y%m", errors="coerce")
        return pd.Series(dt, index=idx).dt.to_period("M")
    x = s.astype("string").str.strip()
    x = x.where(x != "")
    dt = pd.to_datetime(x, format="ISO8601", errors="coerce")
    falta = dt.isna() & x.notna()
    if falta.any():
        y = x[falta]
        mm = y.str.match(r"^\d{1,2}/\d{4}$").fillna(False).astype(bool)
        alt = pd.Series(pd.NaT, index=y.index, dtype="datetime64[ns]")
        if mm.any():
            alt[mm] = pd.to_datetime(y[mm], format="%m/%Y", errors="coerce")
        if (~mm).any():
            alt[~mm] = pd.to_datetime(y[~mm].str.replace("-", "/", regex=False), dayfirst=True,
                                      format="mixed", errors="coerce")
        dt = dt.astype("datetime64[ns]")
        dt[falta] = alt
    return pd.Series(dt, index=idx).dt.to_period("M")


def meses_entre(a: pd.Series | pd.Period, b: pd.Series | pd.Period):
    """a − b em meses (Periods mensais)."""
    if isinstance(a, pd.Series) or isinstance(b, pd.Series):
        ay, am = (a.dt.year, a.dt.month) if isinstance(a, pd.Series) else (a.year, a.month)
        by, bm = (b.dt.year, b.dt.month) if isinstance(b, pd.Series) else (b.year, b.month)
        return (ay - by) * 12 + (am - bm)
    return (a.year - b.year) * 12 + (a.month - b.month)


def norm_id(s: pd.Series) -> pd.Series:
    """IDs como texto comparável entre tabelas: 7.0 → '7', ' C007 ' → 'C007'."""
    if eh_numerica(s):
        v = pd.to_numeric(s, errors="coerce")
        inteiro = v.notna() & np.isclose(v.fillna(0), np.round(v.fillna(0)))
        out = v.astype(object)
        out[inteiro] = v[inteiro].round().astype("int64").astype(str)
        out[~inteiro & v.notna()] = v[~inteiro & v.notna()].astype(str)
        out[v.isna()] = None
        return pd.Series(out, index=s.index, dtype="object")
    x = s.astype("string").str.strip()
    return x.where(x.notna() & (x != ""), None).astype("object")


def norm_valor(v) -> str:
    """Valor de categoria normalizado para comparação ('Cancelado ' → 'cancelado', 1.0 → '1')."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "1" if v else "0"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return str(int(v)) if float(v).is_integer() else repr(float(v))
    t = str(v).strip().lower()
    if t in ("true", "verdadeiro", "sim", "yes"):
        return "1" if t in ("true", "verdadeiro") else t
    if t in ("false", "falso"):
        return "0"
    try:
        f = float(t.replace(",", "."))
        return str(int(f)) if f.is_integer() else repr(f)
    except ValueError:
        return t


def parece_numero_texto(s: pd.Series) -> bool:
    """Coluna de texto cujos valores são números (com , ou . decimal) — sem zeros à esquerda (IDs)."""
    x = s.dropna().astype(str).str.strip()
    x = x[x != ""]
    if len(x) == 0:
        return False
    if x.str.match(r"^0\d").any():
        return False
    return bool(x.map(lambda v: bool(_RE_NUM.match(v))).mean() >= 0.98)


def texto_para_numero(s: pd.Series) -> pd.Series:
    x = s.astype("string").str.strip()
    virg = x.str.contains(",", regex=False).fillna(False)
    ponto_milhar = x.str.match(r"^[+-]?\d{1,3}(\.\d{3})+(,\d+)?$").fillna(False)
    y = x.copy()
    y[virg & ~ponto_milhar] = x[virg & ~ponto_milhar].str.replace(",", ".", regex=False)
    y[ponto_milhar] = x[ponto_milhar].str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    # vírgula como separador de milhar (1,234.5)
    milhar_virg = x.str.match(r"^[+-]?\d{1,3}(,\d{3})+(\.\d+)?$").fillna(False)
    y[milhar_virg] = x[milhar_virg].str.replace(",", "", regex=False)
    return pd.to_numeric(y, errors="coerce")


# --------------------------------------------------------------------------- AUC
def auc(y: np.ndarray, s: np.ndarray) -> float:
    """AUC (Mann–Whitney, empates = 0,5). NaN em s é ignorado. NaN se só há uma classe."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=float)
    m = ~np.isnan(s)
    y, s = y[m], s[m]
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = pd.Series(s).rank(method="average").to_numpy()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# --------------------------------------------------------------------------- textos
def rotulo_coluna(nome: str) -> str:
    """snake_case → 'Uso plataforma pct'."""
    t = re.sub(r"[_\s]+", " ", str(nome)).strip()
    t = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", t)
    return (t[:1].upper() + t[1:].lower()) if t else str(nome)


def minusc(t: str) -> str:
    return t[:1].lower() + t[1:] if t and not (len(t) > 1 and t[1].isupper()) else t


def fmt_mes(p) -> str:
    """Period → 'jun/2026'."""
    if p is None or (not isinstance(p, pd.Period) and pd.isna(p)):
        return ""
    return f"{MESES_PT[p.month - 1]}/{p.year}"


def fmt_num(x, casas: int | None = None) -> str:
    """Número em pt-BR: 1234.5 → '1.234,5'; inteiros sem casas; sinal de menos tipográfico."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    x = float(x)
    if casas is None:
        if abs(x - round(x)) < 1e-9:
            casas = 0
        elif abs(x) >= 100:
            casas = 0
        else:
            casas = 1
    t = f"{x:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    if t.startswith("-"):
        t = "−" + t[1:]
    if t in ("−0", "−0,0", "−0,00"):
        t = t[1:]
    return t


def fmt_pct(fracao: float, casas: int = 0) -> str:
    return fmt_num(100 * float(fracao), casas) + "%"


def numero_redondo(x: float) -> float:
    """Passo 'redondo' (1, 2, 2,5 ou 5 × 10^n) mais próximo de x > 0."""
    if not x or not math.isfinite(x) or x <= 0:
        return 1.0
    e = math.floor(math.log10(x))
    base = x / 10 ** e
    passo = min((1, 2, 2.5, 5, 10), key=lambda b: abs(b - base))
    return passo * 10 ** e
