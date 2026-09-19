"""Escolha automática do modelo (com opção avançada para forçar um).

Elegibilidade (em `Config`): a logística é sempre elegível; as árvores (Random Forest e LightGBM) só com
≥ `min_pos_arvores` (200) meses-cliente positivos no treino E ≥ `min_canc_arvores` (50) clientes
cancelados — com menos que isso, elas decoram poucos casos e a comparação fica ruidosa demais.

Critério: menor log-loss FORA DA AMOSTRA (média das dobras externas — as mesmas para todos os modelos),
porque a fila usa p × valor e o log-loss mede calibração e separação ao mesmo tempo.

Desempate pela simplicidade (logística < Random Forest < LightGBM): fica o modelo MAIS SIMPLES cuja perda
não passa da do melhor por mais de 1 erro-padrão da diferença pareada por dobra — um modelo mais
complexo só vence se a melhora for maior que 1 EP.

Opção avançada `Config.modelo = "logistica" | "random_forest" | "lightgbm"`: usa esse modelo. Se ele não
for elegível, treina mesmo assim e gera aviso. `Config.avaliar_todos = True` valida também os não
elegíveis, só para comparação (no modo automático eles nunca são escolhidos).

A escolha é feita depois de ver a validação: entre 2–3 modelos o otimismo é pequeno, mas existe (as
métricas do escolhido são as da mesma validação que o escolheu).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from ..config import MODELOS_VALIDOS, Config
from ..modelos_previsao.base import MODELOS
from .metricas import comparar_modelos, diferenca_pareada

CRITERIO = ("menor log-loss fora da amostra (média das dobras externas agrupadas por cliente, as mesmas para "
            "todos os modelos); um modelo mais complexo só vence se a melhora pareada por dobra for maior que "
            "1 erro-padrão — senão fica o mais simples (logística < Random Forest < LightGBM)")


@dataclass
class Candidato:
    nome: str
    rotulo: str
    complexidade: int
    elegivel: bool
    motivo_elegibilidade: str
    avaliado: bool = False


@dataclass
class Escolha:
    nome: str
    rotulo: str
    forcado: bool
    motivo: str
    candidatos: list[Candidato]
    comparacao: pd.DataFrame = field(default_factory=pd.DataFrame)   # por modelo (ver metricas.comparar_modelos)
    avisos: list[str] = field(default_factory=list)
    criterio: str = CRITERIO
    elegibilidade: dict = field(default_factory=dict)

    def candidatos_dict(self) -> list[dict]:
        out = []
        for c in self.candidatos:
            d = asdict(c)
            d["escolhido"] = c.nome == self.nome
            if c.nome in self.comparacao.index:
                d.update({k: float(v) for k, v in self.comparacao.loc[c.nome].items()})
            out.append(d)
        return out


def candidatos(cfg: Config, n_pos: int, n_canc: int) -> list[Candidato]:
    """Os três modelos, com a elegibilidade para esta base."""
    arv_ok = n_pos >= cfg.min_pos_arvores and n_canc >= cfg.min_canc_arvores
    out = []
    for info in sorted(MODELOS.values(), key=lambda i: i.complexidade):
        if not info.arvore:
            ok, mot = True, "sempre elegível"
        elif arv_ok:
            ok, mot = True, (f"{n_pos} meses-cliente positivos (mín. {cfg.min_pos_arvores}) e {n_canc} cancelados "
                             f"(mín. {cfg.min_canc_arvores})")
        else:
            ok, mot = False, (f"não elegível: {n_pos} meses-cliente positivos (mín. {cfg.min_pos_arvores}) e "
                              f"{n_canc} cancelados (mín. {cfg.min_canc_arvores})")
        out.append(Candidato(info.nome, info.rotulo, info.complexidade, ok, mot))
    return out


def a_validar(cands: list[Candidato], cfg: Config) -> list[str]:
    """Modelos que entram na validação cruzada (na ordem de simplicidade)."""
    if cfg.modelo not in MODELOS_VALIDOS:
        raise ValueError(f"Config.modelo inválido: {cfg.modelo!r} (opções: {', '.join(MODELOS_VALIDOS)})")
    if cfg.modelo == "auto":
        return [c.nome for c in cands if c.elegivel or cfg.avaliar_todos]
    return [c.nome for c in cands if c.nome == cfg.modelo or cfg.avaliar_todos]


def _fmt(x: float, casas: int = 4) -> str:
    return f"{x:.{casas}f}".replace(".", ",")


def escolher(cands: list[Candidato], oof, grade: pd.DataFrame, cfg: Config, n_pos: int, n_canc: int) -> Escolha:
    """Decide o modelo. `oof` = ValidacaoOOF (ou None com a validação desligada)."""
    por_nome = {c.nome: c for c in cands}
    comp = comparar_modelos(grade, oof) if oof is not None else pd.DataFrame()
    for c in cands:
        c.avaliado = c.nome in comp.index
    eleg = {"min_pos_arvores": cfg.min_pos_arvores, "min_canc_arvores": cfg.min_canc_arvores,
            "positivos_treino": n_pos, "cancelados": n_canc}
    avisos: list[str] = []

    if cfg.modelo != "auto":
        c = por_nome[cfg.modelo]
        motivo = f"forçado pela opção avançada Config.modelo = '{cfg.modelo}'"
        if not c.elegivel:
            avisos.append(f"Modelo '{c.rotulo}' forçado sem atingir a elegibilidade ({c.motivo_elegibilidade}): "
                          "com poucos positivos ele tende a sobreajustar — compare o log-loss na validação.")
        return Escolha(c.nome, c.rotulo, True, motivo, cands, comp, avisos, elegibilidade=eleg)

    validos = [c for c in cands if c.elegivel and c.nome in comp.index]
    if not validos:
        c = next(c for c in cands if c.elegivel)          # a logística é sempre elegível
        motivo = ("validação desligada: fica o modelo elegível mais simples" if oof is None
                  else "sem validação dos modelos elegíveis: fica o mais simples")
        return Escolha(c.nome, c.rotulo, False, motivo, cands, comp, avisos, elegibilidade=eleg)
    if len(validos) == 1:
        c = validos[0]
        nao = [x.rotulo for x in cands if not x.elegivel]
        motivo = (f"único modelo elegível (log-loss fora da amostra {_fmt(comp.at[c.nome, 'log_loss'])} ± "
                  f"{_fmt(comp.at[c.nome, 'log_loss_ep'])})")
        if nao:
            motivo += (f"; {', '.join(nao)} exigem ≥ {cfg.min_pos_arvores} meses-cliente positivos e ≥ "
                       f"{cfg.min_canc_arvores} cancelados (esta base: {n_pos} e {n_canc})")
        return Escolha(c.nome, c.rotulo, False, motivo, cands, comp, avisos, elegibilidade=eleg)

    melhor = min(validos, key=lambda c: (comp.at[c.nome, "log_loss"], c.complexidade))
    escolhido = melhor
    dif = {}
    for c in sorted(validos, key=lambda c: c.complexidade):
        if c.nome == melhor.nome:
            break
        d, ep = diferenca_pareada(oof, c.nome, melhor.nome)
        dif[c.nome] = (d, ep)
        if not np.isnan(ep) and d <= ep:
            escolhido = c
            break
    ll = lambda n: f"{_fmt(comp.at[n, 'log_loss'])} ± {_fmt(comp.at[n, 'log_loss_ep'])}"  # noqa: E731
    if escolhido.nome == melhor.nome:
        outros = [f"{c.rotulo} {ll(c.nome)}" for c in validos if c.nome != melhor.nome]
        motivo = f"menor log-loss fora da amostra ({ll(melhor.nome)}; " + "; ".join(outros) + ")"
        mais_simples = [c for c in validos if c.complexidade < melhor.complexidade]
        if mais_simples:
            d, ep = dif.get(mais_simples[0].nome, (np.nan, np.nan))
            motivo += (f" e melhora pareada sobre {mais_simples[0].rotulo} de {_fmt(d)} > 1 erro-padrão "
                       f"({_fmt(ep)})")
    else:
        d, ep = dif[escolhido.nome]
        motivo = (f"{melhor.rotulo} teve o menor log-loss ({ll(melhor.nome)}), mas a vantagem sobre "
                  f"{escolhido.rotulo} ({ll(escolhido.nome)}) é de {_fmt(d)} por dobra, dentro de 1 erro-padrão "
                  f"({_fmt(ep)}): fica o mais simples")
    return Escolha(escolhido.nome, escolhido.rotulo, False, motivo, cands, comp, avisos, elegibilidade=eleg)
