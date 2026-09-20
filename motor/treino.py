"""Orquestração: `treinar(tabelas, mapeamento)` → `Resultado` com a fila explicada, a validação e os pesos.

Etapas: validar mapeamento → preparar base e painel → gerar candidatas → perfil de escala → modelos
candidatos (elegibilidade) → validação aninhada de todos nas mesmas dobras (out-of-fold por cliente) →
escolha do modelo (`modelos_classificacao/escolha.py`) → cortes das faixas → seleção + modelo final em todos
os clientes (só variáveis consistentes entre as dobras) → fila e explicações → JSON (`saida.py`).
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
from .delta import ajustar_modelo_com_delta, delta_mes_anterior, risco_ajustado
from .explicar import acao_sugerida, explicar, meta_persistencias
from .mapeamento import Mapeamento
from .modelos_classificacao import escolha as esc
from .modelos_classificacao.metricas import curva, desempenho, escolher_cortes, faixa
from .modelos_classificacao.validacao_cruzada import ValidacaoOOF, validar
from .modelos_previsao import ModeloPrevisao
from .painel import Base, montar_grade, preparar
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
    modelo: ModeloPrevisao
    fila: pd.DataFrame                   # ativos no mês de referência, na ordem da fila
    explicacao: pd.DataFrame             # cliente × variável
    curva: pd.DataFrame
    cortes: dict
    oof: ValidacaoOOF | None
    base: Base
    avisos: list[str] = field(default_factory=list)
    escolha: esc.Escolha | None = None   # modelo escolhido, candidatos e comparação (validacao/pesos.json)

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

        cfg, aviso_perfil = cfg.perfil_automatico(len(base.ids), len(grade))
        if aviso_perfil:
            avisos.append(aviso_perfil)
        T = grade[grade["treino"]]
        n_pos, n_canc = int(T["y"].sum()), int(T.groupby("cliente")["y"].max().sum())
        cands = esc.candidatos(cfg, n_pos, n_canc)
        nomes = esc.a_validar(cands, cfg)

        oof = None
        if cfg.validar:
            p(f"validação cruzada (seleção aninhada; modelos: {', '.join(nomes)})", 0.1)
            oof = validar(X, grade, meta, cfg, H, progresso=p, inicio=0.1, fim=0.8, modelos=nomes)
        escolha = esc.escolher(cands, oof, grade, cfg, n_pos, n_canc)
        avisos += escolha.avisos
        freq_dobras = None
        if oof is not None:
            oof.usar(escolha.nome)
            if cfg.min_frac_dobras > 0:
                freq_dobras = oof.modelos[escolha.nome].frequencia_dobras()

        p(f"treinando modelo final ({escolha.rotulo})", 0.82)
        modelo, X2, meta = ajustar_modelo_com_delta(escolha.nome, X, grade, meta, grade["treino"], cfg, H,
                                                    freq_dobras=freq_dobras, n_jobs=cfg.n_jobs)
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
        # delta de risco (mês de referência × mês anterior): sempre calculado e enviado; ordena a fila se ligado
        fila["delta_risco"] = delta_mes_anterior(painel, pd.Index(fila["linha"])).to_numpy()
        fila["risco_ajustado"] = (risco_ajustado(fila["probabilidade"], fila["delta_risco"])
                                  if cfg.delta_na_fila else fila["probabilidade"])
        fila["perda_anual_ajustada"] = fila["risco_ajustado"] * fila["valor_mensal"] * 12
        chave = fila["perda_anual_ajustada"] if base.valor is not None else fila["risco_ajustado"]
        fila = fila.assign(_ch=chave.fillna(-1.0)).sort_values(["_ch", "probabilidade", "cliente"],
                                                              ascending=[False, False, True]).drop(columns="_ch")
        fila["prioridade"] = np.arange(1, len(fila) + 1)
        fila = fila.set_index("cliente")
        fila["meses_em_alerta"] = _meses_seguidos_alerta(painel, cortes["atencao"]).reindex(fila.index).fillna(0).astype(int)
        E = explicar(modelo, X2, aux, pd.Index(fila["linha"]), painel["mes"], painel["cliente"], meta)
        E_cli = {c: g for c, g in E.groupby("cliente", sort=False)}
        acoes = {c: acao_sugerida(E_cli.get(c, E.iloc[0:0]), fila.at[c, "faixa_risco"], meta, m.acoes) for c in fila.index}
        fila["acao_sugerida"] = [acoes[c][0] for c in fila.index]
        fila["variavel_dominante"] = [acoes[c][1] for c in fila.index]

        # checagem (reportada, não forçada): ≥ 3 sinais DISTINTOS (colunas de origem) disparando e faixa baixa
        disp = E[E["dispara"]] if len(E) else E
        n_evid = (disp.assign(_col=disp["variavel"].map(lambda v: (meta[v].tabela, meta[v].coluna)))
                  .groupby("cliente")["_col"].nunique()) if len(disp) else pd.Series(dtype=float)
        baixo = [c for c in fila.index if fila.at[c, "faixa_risco"] == "baixo" and int(n_evid.get(c, 0)) >= 3]
        if baixo:
            avisos.append(f"{len(baixo)} cliente(s) na faixa baixa com evidências de 3 ou mais colunas diferentes disparando "
                          f"({', '.join(baixo[:10])}{'…' if len(baixo) > 10 else ''}): as evidências passam do limiar, "
                          "mas o conjunto do modelo não chega ao corte de atenção — vale uma olhada.")

        p("gerando resultados", 0.97)
        r = Resultado({}, {}, {}, m, cfg, painel, X2, aux, meta, modelo, fila, E, cv, cortes, oof, base, avisos,
                      escolha)
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
