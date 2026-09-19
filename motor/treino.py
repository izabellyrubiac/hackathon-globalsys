"""Orquestração: `treinar(tabelas, mapeamento)` → `Resultado` com a fila explicada, a validação e os pesos.

Etapas: validar mapeamento → preparar base e painel → gerar candidatas → validação aninhada (out-of-fold
por cliente) → cortes das faixas → seleção + modelo final em todos os clientes → fila e explicações →
JSON (`saida.py`).
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from . import saida
from .config import Config
from .explicar import acao_sugerida, explicar, meta_persistencias
from .mapeamento import Mapeamento
from .modelo import Modelo, ajustar
from .painel import Base, montar_grade, preparar
from .validar import ValidacaoOOF, curva, desempenho, escolher_cortes, faixa, validar
from .variaveis import Variavel, gerar


@dataclass
class Resultado:
    clientes: dict                       # contrato clientes.json
    validacao: dict                      # contrato validacao.json
    pesos: dict                          # contrato pesos.json
    mapeamento: Mapeamento
    config: Config
    painel: pd.DataFrame                 # grade cliente × mês + p_final + p_oof
    X: pd.DataFrame                      # candidatas (+ persistências do modelo final)
    aux: pd.DataFrame
    meta: dict[str, Variavel]
    modelo: Modelo
    fila: pd.DataFrame                   # ativos no mês de referência, na ordem da fila
    explicacao: pd.DataFrame             # cliente × variável
    curva: pd.DataFrame
    cortes: dict
    oof: ValidacaoOOF | None
    base: Base
    avisos: list[str] = field(default_factory=list)

    def salvar(self, pasta: str | Path) -> dict[str, Path]:
        """Grava clientes.json, validacao.json e pesos.json (determinísticos, chaves ordenadas)."""
        return {n: saida.gravar(obj, Path(pasta) / f"{n}.json")
                for n, obj in (("clientes", self.clientes), ("validacao", self.validacao), ("pesos", self.pesos))}


def _meses_seguidos_alerta(painel: pd.DataFrame, corte: float) -> pd.Series:
    """Meses seguidos, até o último mês de cada cliente, com p_final ≥ corte."""
    q = painel[painel["pontuavel"]].sort_values(["cliente", "mes"])
    out = {}
    for c, g in q.groupby("cliente", sort=False):
        a = (g["p_final"] >= corte).to_numpy()[::-1]
        out[c] = int(np.argmin(a)) if not a.all() else len(a)
    return pd.Series(out)


def treinar(tabelas: dict[str, pd.DataFrame], m: Mapeamento, progresso=None, config: Config | None = None,
            gerado_em: str | None = None) -> Resultado:
    """Treina o motor numa base qualquer. `progresso(etapa: str, fracao: float)` é chamado a cada etapa."""
    cfg = config or Config()
    p = progresso or (lambda *_: None)
    H = int(m.horizonte_meses)
    gerado = gerado_em or os.environ.get("GERADO_EM") or date.today().isoformat()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.filterwarnings("ignore", message=".*Mean of empty slice.*")
        warnings.filterwarnings("ignore", message=".*Degrees of freedom.*")

        p("validando mapeamento", 0.0)
        avisos = list(getattr(tabelas, "avisos", [])) + m.validar(tabelas)
        p("montando painel cliente × mês", 0.03)
        base = preparar(tabelas, m, cfg)
        grade = montar_grade(base, m, cfg)
        avisos += base.avisos
        p("gerando variáveis candidatas", 0.06)
        X, aux, meta = gerar(base, grade, cfg)
        if grade.loc[grade["treino"], "y"].sum() < 3:
            from .mapeamento import ErroMapeamento
            raise ErroMapeamento(["Menos de 3 meses-cliente de cancelamento com histórico suficiente para treinar: "
                                  "confira a coluna de data de saída e as tabelas mensais."])

        oof = None
        if cfg.validar:
            p("validação cruzada (seleção aninhada)", 0.1)
            oof = validar(X, grade, meta, cfg, H, progresso=p, inicio=0.1, fim=0.8)

        p("treinando modelo final", 0.82)
        modelo, X2 = ajustar(X, grade, meta, grade["treino"], cfg, H)
        meta = {**meta, **meta_persistencias(meta, modelo.selecao.persistencias)}

        painel = grade.copy()
        painel["p_final"] = np.nan
        pts = painel["pontuavel"]
        painel.loc[pts, "p_final"] = modelo.prob(X2.loc[pts])
        painel["p_oof"] = oof.p if oof is not None else np.nan
        p_corte = painel["p_oof"] if oof is not None else painel["p_final"]
        cv = curva(painel, p_corte)
        cortes = escolher_cortes(cv)
        if oof is None:
            avisos.append("Validação desligada: cortes das faixas escolhidos no próprio treino (otimistas).")

        p("montando a fila", 0.92)
        ref = painel[painel["referencia"]]
        fila = pd.DataFrame({"cliente": ref["cliente"].to_numpy(), "linha": ref.index,
                             "probabilidade": ref["p_final"].to_numpy()})
        fila["valor_mensal"] = base.valor.reindex(fila["cliente"]).to_numpy() if base.valor is not None else np.nan
        fila["perda_anual_esperada"] = fila["probabilidade"] * fila["valor_mensal"] * 12
        fila["faixa_risco"] = faixa(fila["probabilidade"], cortes)
        chave = fila["perda_anual_esperada"] if base.valor is not None else fila["probabilidade"]
        fila = fila.assign(_ch=chave.fillna(-1.0)).sort_values(["_ch", "probabilidade", "cliente"],
                                                              ascending=[False, False, True]).drop(columns="_ch")
        fila["prioridade"] = np.arange(1, len(fila) + 1)
        fila = fila.set_index("cliente")
        fila["meses_em_alerta"] = _meses_seguidos_alerta(painel, cortes["atencao"]).reindex(fila.index).fillna(0).astype(int)
        E = explicar(modelo, X2, aux, pd.Index(fila["linha"]), painel["mes"], painel["cliente"], meta)
        acoes = {c: acao_sugerida(E[E["cliente"] == c], fila.at[c, "faixa_risco"], meta, m.acoes) for c in fila.index}
        fila["acao_sugerida"] = [acoes[c][0] for c in fila.index]
        fila["variavel_dominante"] = [acoes[c][1] for c in fila.index]

        p("gerando resultados", 0.97)
        r = Resultado({}, {}, {}, m, cfg, painel, X2, aux, meta, modelo, fila, E, cv, cortes, oof, base, avisos)
        r.clientes = saida.clientes_json(r, gerado)
        r.validacao = saida.validacao_json(r, gerado)
        r.pesos = saida.pesos_json(r, gerado)
        p("concluído", 1.0)
        return r


def desempenho_metodos(r: Resultado) -> dict[str, dict]:
    """Desempenho (out-of-fold) do motor nas faixas e da melhor variável sozinha."""
    P = r.painel
    if r.oof is None:
        return {}
    return {
        "melhor_variavel": desempenho(P, r.oof.base_alarme),
        "motor_alto": desempenho(P, (P["p_oof"] >= r.cortes["alto"]) & P["pontuavel"]),
        "motor_alto_ou_atencao": desempenho(P, (P["p_oof"] >= r.cortes["atencao"]) & P["pontuavel"]),
    }
