"""Geração automática de variáveis candidatas, avaliadas em cada linha do painel (cliente × mês t),
usando só dados até t.

Tabelas temporais (colunas agregadas por cliente × mês; média nas numéricas):
* numérica densa   → nivel (média 3m), variacao (nível − média anterior do próprio cliente),
                     tendencia (inclinação por mês nos últimos 6 meses, ≥ 3 pontos), ultimo;
                     + ausente (fração de meses sem valor na janela) quando a coluna tem nulos.
* numérica esparsa (dados em < 60% dos meses, ex. pesquisa trimestral)
                   → ultimo (último registro), variacao (último − média dos registros anteriores),
                     media (média dos registros na janela de 2 intervalos), meses_desde (meses desde o
                     último registro) e nunca (1 = nenhum registro até t). Ausência é comportamento.
* categórica       → freq (fração dos registros da janela em cada categoria; nulo vira "(ausente)")
                     e ultima (1 = último registro nesta categoria).
* tabela com várias linhas por cliente-mês (transacional) → também a contagem mensal de registros,
  tratada como numérica densa.
Tabelas estáticas (uma linha por cliente): numéricas como estão, one-hot das categóricas de baixa
cardinalidade, colunas de data → meses desde a data (ex.: tempo de contrato).
Tabelas de eventos (sem data): contagem de registros, média das numéricas e fração de cada categoria.
Persistência (meses seguidos na zona de risco) é criada na seleção, com limiar aprendido (`selecao.py`).

Cada variável tem id estável `<coluna>__<transformacao>` (com prefixo `<tabela>.` se o nome da coluna se
repete entre tabelas) e rótulo legível automático (snake_case → "Uso plataforma pct (média 3m)").
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Config
from .painel import Base, rotulo
from .util import meses_entre, para_mes, rotulo_coluna


@dataclass
class Variavel:
    id: str
    rotulo: str
    tabela: str
    coluna: str
    transformacao: str
    rotulo_coluna: str
    categoria: str | None = None
    janela: int | None = None
    temporal: bool = True
    indicador: bool = False
    fracao: bool = False
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"id": self.id, "rotulo": self.rotulo, "tabela": self.tabela, "coluna": self.coluna,
                "transformacao": self.transformacao, "categoria": self.categoria, "janela": self.janela}


SUFIXO = {
    "nivel": "(média {j}m)", "variacao": "(variação vs média anterior)", "tendencia": "(tendência {j}m)",
    "ultimo": "(último)", "media": "(média {j}m)", "meses_desde": "(meses desde o último registro)",
    "nunca": "(nunca registrado)", "ausente": "(ausente, fração {j}m)",
}


class _Gerador:
    def __init__(self, base: Base, grade: pd.DataFrame, cfg: Config):
        self.base, self.grade, self.cfg = base, grade, cfg
        self.cols: dict[str, np.ndarray] = {}
        self.aux: dict[str, np.ndarray] = {}
        self.meta: dict[str, Variavel] = {}
        nomes = Counter()
        for t in base.temporais:
            nomes.update(list(t.colunas))
        for t in base.estaticas:
            nomes.update(list(t.colunas))
        self.repetidas = {c for c, n in nomes.items() if n > 1}
        self.ids = pd.Index(base.ids)
        if not base.fotografia:
            self.m0 = grade["mes"].min()
            self.meses = pd.period_range(self.m0, base.mes_ref, freq="M")
            self.r = meses_entre(grade["mes"], self.m0).to_numpy()
        else:
            self.meses = pd.period_range(grade["mes"].min(), periods=1, freq="M")
            self.r = np.zeros(len(grade), dtype=int)
        self.c = self.ids.get_indexer(grade["cliente"])
        self.tem = grade["tem_dados"].to_numpy()

    # ---------------------------------------------------------------- utilidades
    def vid(self, tabela: str, coluna: str, transf: str, categoria: str | None = None) -> str:
        nome = coluna if coluna not in self.repetidas else f"{tabela}.{coluna}"
        if categoria is not None:
            nome = f"{nome}={categoria}"
        return f"{nome}__{transf}"

    def add(self, v: Variavel, valores: np.ndarray) -> None:
        valores = np.asarray(valores, dtype=float)
        if v.id in self.cols:
            return
        self.cols[v.id] = valores
        self.meta[v.id] = v

    def larga(self, serie: pd.Series) -> pd.DataFrame:
        """Série indexada por (__cli, __mes) → matriz meses × clientes."""
        w = serie.unstack("__cli")
        w = w.reindex(index=self.meses, columns=self.ids)
        return w

    def no_painel(self, w: pd.DataFrame | np.ndarray) -> np.ndarray:
        a = w.to_numpy(dtype=float) if isinstance(w, pd.DataFrame) else w
        return a[self.r, self.c]

    # ---------------------------------------------------------------- transformações
    def _tendencia(self, W: pd.DataFrame, j: int) -> np.ndarray:
        t = np.arange(len(W), dtype=float)[:, None] * np.ones((1, W.shape[1]))
        x = W.to_numpy(dtype=float)
        ok = ~np.isnan(x)
        tt = np.where(ok, t, 0.0)
        xx = np.where(ok, x, 0.0)
        roll = lambda a: pd.DataFrame(a).rolling(j, min_periods=1).sum().to_numpy()  # noqa: E731
        n, st, sx = roll(ok.astype(float)), roll(tt), roll(xx)
        stt, stx = roll(tt * tt), roll(tt * xx)
        den = n * stt - st * st
        with np.errstate(invalid="ignore", divide="ignore"):
            slope = (n * stx - st * sx) / den
        slope[(n < 3) | (den <= 0)] = np.nan
        return slope

    def numerica_densa(self, tab: str, col: str, W: pd.DataFrame, presente: pd.DataFrame | None, rot: str):
        J, JT = self.cfg.janela, self.cfg.janela_tendencia
        nivel = W.rolling(J, min_periods=1).mean()
        ref = W.shift(J).expanding(min_periods=1).mean()
        var = nivel - ref
        mk = lambda tr, j=None: Variavel(self.vid(tab, col, tr), f"{rot} " + SUFIXO[tr].format(j=j), tab, col, tr, rot,  # noqa: E731
                                         janela=j)
        v = mk("nivel", J)
        self.add(v, self.no_painel(nivel))
        v = mk("variacao")
        self.add(v, self.no_painel(var))
        self.aux[v.id + ":ref"] = self.no_painel(ref)
        self.aux[v.id + ":nivel"] = self.no_painel(nivel)
        v = mk("tendencia", JT)
        self.add(v, self.no_painel(self._tendencia(W, JT)))
        self.add(mk("ultimo"), self.no_painel(W.ffill()))
        if presente is not None:
            aus = (presente.fillna(0).astype(bool) & W.isna()).astype(float)
            den = presente.fillna(0).rolling(J, min_periods=1).sum()
            if float(aus.to_numpy().sum()) >= 0.05 * float(presente.fillna(0).to_numpy().sum()):
                fr = aus.rolling(J, min_periods=1).sum() / den.where(den > 0)
                v = mk("ausente", J)
                v.fracao = True
                self.add(v, self.no_painel(fr))

    def numerica_esparsa(self, tab: str, col: str, W: pd.DataFrame, rot: str, js: int):
        x = W.to_numpy(dtype=float)
        ok = ~np.isnan(x)
        ult = W.ffill().to_numpy(dtype=float)
        cnt = np.cumsum(ok, axis=0)
        soma = np.cumsum(np.where(ok, x, 0.0), axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            ant = (soma - np.where(cnt > 0, ult, 0.0)) / (cnt - 1)
        ant[cnt < 2] = np.nan
        linhas = np.arange(len(W))[:, None] * np.ones((1, W.shape[1]))
        pos = pd.DataFrame(np.where(ok, linhas, np.nan)).ffill().to_numpy()
        desde = linhas - pos
        nunca = (cnt == 0).astype(float)
        media = W.rolling(js, min_periods=1).mean()
        mk = lambda tr, j=None: Variavel(self.vid(tab, col, tr), f"{rot} " + SUFIXO[tr].format(j=j), tab, col, tr, rot,  # noqa: E731
                                         janela=j)
        v = mk("ultimo")
        self.add(v, self.no_painel(ult))
        self.aux[v.id + ":desde"] = self.no_painel(desde)
        v = mk("variacao")
        self.add(v, self.no_painel(ult - ant))
        self.aux[v.id + ":ref"] = self.no_painel(ant)
        self.aux[v.id + ":nivel"] = self.no_painel(ult)
        self.add(mk("media", js), self.no_painel(media))
        self.add(mk("meses_desde"), self.no_painel(desde))
        v = mk("nunca")
        v.indicador = True
        self.add(v, self.no_painel(nunca))

    def categorica(self, tab: str, col: str, df: pd.DataFrame, presente: pd.DataFrame, rot: str, j: int, dens: bool):
        s = df[col].astype(object).where(df[col].notna(), "(ausente)").astype(str)
        freq = s.value_counts(normalize=True)
        niveis = [n for n in freq.index if freq[n] >= self.cfg.min_frac_categoria][: self.cfg.max_categorias]
        if len(niveis) < 2 and len(freq) <= 1:
            return
        for n in sorted(niveis):
            ind = (s == n).astype(float)
            W = self.larga(ind.groupby([df["__cli"], df["__mes"]]).mean())
            fr = W.rolling(j, min_periods=1).mean()
            v = Variavel(self.vid(tab, col, "freq", n), f"{rot} = {n} (fração {j}m)", tab, col, "freq", rot,
                         categoria=n, janela=j, fracao=True)
            self.add(v, self.no_painel(fr))
            v = Variavel(self.vid(tab, col, "ultima", n), f"{rot} = {n} (último)", tab, col, "ultima", rot,
                         categoria=n, indicador=True)
            self.add(v, self.no_painel(W.ffill().round()))

    # ---------------------------------------------------------------- tabelas
    def temporal(self, t) -> None:
        df = t.df
        chave = [df["__cli"], df["__mes"]]
        presente = self.larga(df.groupby(chave).size().astype(float)).notna().astype(float)
        presente = presente.where(presente > 0)
        tem_painel = self.tem
        # intervalo típico entre registros (para janelas das colunas esparsas)
        if t.transacional:
            cont = self.larga(df.groupby(chave).size().astype(float)).fillna(0.0)
            nome = "registros"
            rot = f"Registros em {rotulo_coluna(t.nome).lower()}"
            self.repetidas.add(nome)
            self.numerica_densa(t.nome, nome, cont, None, rot)
        for col, tipo in t.colunas.items():
            rot = rotulo(self.base, col)
            if tipo in ("numerica", "binaria") and pd.api.types.is_numeric_dtype(df[col]):
                W = self.larga(df.groupby(chave)[col].mean())
                ok = self.no_painel(W.notna().astype(float))
                dens = float(ok[tem_painel].mean()) if tem_painel.any() else 0.0
                if dens >= self.cfg.densidade_minima:
                    self.numerica_densa(t.nome, col, W, presente, rot)
                else:
                    js = self._janela_esparsa(W)
                    self.numerica_esparsa(t.nome, col, W, rot, js)
            elif tipo in ("categorica", "binaria"):
                dens = t.densidade >= self.cfg.densidade_minima
                j = self.cfg.janela if dens else self._janela_esparsa(presente)
                self.categorica(t.nome, col, df, presente, rot, j, dens)

    def _janela_esparsa(self, W: pd.DataFrame) -> int:
        ok = W.notna().to_numpy()
        gaps = []
        for j in range(ok.shape[1]):
            p = np.flatnonzero(ok[:, j])
            if len(p) > 1:
                gaps.extend(np.diff(p).tolist())
        inter = int(np.median(gaps)) if gaps else self.cfg.janela
        return int(min(12, max(self.cfg.janela, 2 * inter)))

    def estatica(self, t) -> None:
        cli = self.grade["cliente"].to_numpy()
        if t.papel == "eventos":
            df = t.df
            cont = df.groupby("__cli").size().reindex(self.ids).fillna(0.0)
            nome = f"{t.nome}.registros"
            v = Variavel(f"{nome}__contagem", f"Registros em {rotulo_coluna(t.nome).lower()}", t.nome, "registros",
                         "contagem", f"Registros em {rotulo_coluna(t.nome).lower()}", temporal=False)
            self.add(v, cont.reindex(cli).to_numpy())
            for col, tipo in t.colunas.items():
                rot = rotulo(self.base, col)
                if tipo in ("numerica", "binaria") and pd.api.types.is_numeric_dtype(df[col]):
                    mv = df.groupby("__cli")[col].mean().reindex(self.ids)
                    v = Variavel(self.vid(t.nome, col, "media_eventos"), f"{rot} (média em {rotulo_coluna(t.nome).lower()})",
                                 t.nome, col, "media_eventos", rot, temporal=False)
                    self.add(v, mv.reindex(cli).to_numpy())
                elif tipo in ("categorica", "binaria"):
                    s = df[col].astype(object).where(df[col].notna(), "(ausente)").astype(str)
                    fq = s.value_counts(normalize=True)
                    for n in sorted([n for n in fq.index if fq[n] >= self.cfg.min_frac_categoria][: self.cfg.max_categorias]):
                        fr = (s == n).groupby(df["__cli"]).mean().reindex(self.ids)
                        v = Variavel(self.vid(t.nome, col, "freq_eventos", n), f"{rot} = {n} (fração em "
                                     f"{rotulo_coluna(t.nome).lower()})", t.nome, col, "freq_eventos", rot, categoria=n,
                                     temporal=False, fracao=True)
                        self.add(v, fr.reindex(cli).to_numpy())
            return
        df = t.df
        for col, tipo in t.colunas.items():
            rot = rotulo(self.base, col)
            s = df[col]
            if tipo == "data":
                d = para_mes(s)
                if d.notna().mean() < 0.5 or self.base.fotografia:
                    continue
                dm = pd.Series(d.reindex(cli).array, index=self.grade.index)
                meses = meses_entre(self.grade["mes"], dm).to_numpy(dtype=float)
                v = Variavel(self.vid(t.nome, col, "tempo_desde"), f"Meses desde {rot.lower()}", t.nome, col,
                             "tempo_desde", rot)
                self.add(v, meses)
            elif tipo in ("numerica",) or (tipo == "binaria" and pd.api.types.is_numeric_dtype(s)):
                v = Variavel(self.vid(t.nome, col, "estatico"), rot, t.nome, col, "estatico", rot, temporal=False,
                             indicador=tipo == "binaria")
                self.add(v, pd.to_numeric(s, errors="coerce").reindex(cli).to_numpy())
            elif tipo in ("categorica", "binaria"):
                x = s.astype(object).where(s.notna(), "(ausente)").astype(str)
                fq = x.value_counts()
                minimo = max(3, self.cfg.min_frac_categoria * len(x))
                niveis = sorted([n for n in fq.index if fq[n] >= minimo][: self.cfg.max_categorias])
                if tipo == "binaria":
                    niveis = niveis[:1]
                for n in niveis:
                    v = Variavel(self.vid(t.nome, col, "categoria", n), f"{rot} = {n}", t.nome, col, "categoria", rot,
                                 categoria=n, temporal=False, indicador=True)
                    self.add(v, (x == n).astype(float).reindex(cli).to_numpy())


def gerar(base: Base, grade: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Variavel]]:
    """(X candidatas alinhadas à grade, auxiliares para textos, metadados por id)."""
    g = _Gerador(base, grade, cfg)
    for t in base.estaticas:
        g.estatica(t)
    for t in base.temporais:
        g.temporal(t)
    X = pd.DataFrame(g.cols, index=grade.index)
    aux = pd.DataFrame(g.aux, index=grade.index)
    X = X.replace([np.inf, -np.inf], np.nan)
    return X, aux, g.meta
