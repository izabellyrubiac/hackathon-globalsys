"""Ponte com o `motor/`: leitura das tabelas, inspeção, conferência do mapeamento e treino em thread.

Nada de regra de negócio aqui — só orquestração. O motor é a única fonte de verdade sobre dados e modelo.
"""

from __future__ import annotations

import os
import threading
import time
import traceback
from dataclasses import fields
from pathlib import Path

from motor import ErroMapeamento, Mapeamento, Tabelas, inspecionar, ler_arquivos, treinar
from motor.config import Config

from . import armazenamento as arm
from .ajustes import limite_validacao_sincrona_bytes, max_treinos_simultaneos
from .modelos import Opcoes
from .resumo import montar_resumo

# cache de uma base por vez (bases grandes não entram): evita reler o arquivo entre confirmar e treinar
_cache: dict[str, tuple[tuple, Tabelas]] = {}
_cache_lock = threading.Lock()

# fila de treinos do processo: no máximo `API_TREINOS_SIMULTANEOS` rodando, o resto espera a vez
_fila: list[tuple[str, str]] = []
_rodando: set[tuple[str, str]] = set()
_threads: dict[tuple[str, str], threading.Thread] = {}
_fila_lock = threading.Lock()


def _assinatura(caminhos: list[Path]) -> tuple:
    return tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in caminhos)


def ler_tabelas(base_id: str, usar_cache: bool = True) -> Tabelas:
    """Lê os arquivos da base pelo motor, reaproveitando o cache quando a base é pequena."""
    caminhos = arm.caminhos_arquivos(base_id)
    faltando = [p.name for p in caminhos if not p.exists()]
    if faltando:
        raise arm.ErroArmazenamento(f"Arquivo(s) da base sumiram do disco: {', '.join(faltando)}. Envie a base de novo.",
                                    410)
    assinatura = _assinatura(caminhos)
    cabe = arm.bytes_total(base_id) <= limite_validacao_sincrona_bytes()
    if usar_cache and cabe:
        with _cache_lock:
            item = _cache.get(base_id)
            if item and item[0] == assinatura:
                return item[1]
    tabelas = ler_arquivos(caminhos)
    if usar_cache and cabe:
        with _cache_lock:
            _cache.clear()
            _cache[base_id] = (assinatura, tabelas)
    return tabelas


def esquecer(base_id: str) -> None:
    with _cache_lock:
        _cache.pop(base_id, None)


def inspecionar_base(base_id: str) -> dict:
    """Inspeciona e guarda em `inspecao.json`."""
    insp = inspecionar(ler_tabelas(base_id))
    arm.gravar_json(arm.pasta(base_id) / "inspecao.json", insp)
    return insp


# --------------------------------------------------------------------------- mapeamento
def _colunas(insp: dict) -> dict[str, list[str]]:
    return {t["nome"]: [c["nome"] for c in t.get("colunas", [])]
            for t in insp.get("tabelas", []) if t.get("papel_sugerido") != "ignorada"}


def conferir_esquema(insp: dict, m: Mapeamento) -> list[str]:
    """Conferência barata (só nomes de tabela e de coluna), a partir de `inspecao.json`."""
    tab = _colunas(insp)
    if not tab:
        return []
    nomes = ", ".join(f"'{t}'" for t in tab)
    problemas: list[str] = []

    def checar(tabela: str | None, coluna: str | None, papel: str) -> None:
        if not tabela:
            return
        if tabela not in tab:
            problemas.append(f"A tabela {papel} '{tabela}' não existe na base (tabelas: {nomes}).")
        elif coluna and coluna not in tab[tabela]:
            lista = ", ".join(f"'{c}'" for c in tab[tabela])
            problemas.append(f"A coluna {papel} '{coluna}' não existe na tabela '{tabela}' (colunas: {lista}).")

    checar(m.tabela_clientes, m.coluna_id, "de clientes")
    checar(m.alvo_tabela, m.coluna_situacao, "do cancelamento")
    checar(m.alvo_tabela, m.coluna_data_saida, "da data de saída")
    if m.coluna_valor:
        checar(m.valor_tabela or m.tabela_clientes, m.coluna_valor, "do valor do contrato")
    if not m.coluna_situacao and not m.coluna_data_saida:
        problemas.append("Informe a coluna de situação (com o valor que significa cancelado) ou a coluna de data "
                         "de saída: sem cancelamento não há como aprender.")
    return problemas


def conferir_mapeamento(base_id: str, m: Mapeamento) -> tuple[list[str], list[str]]:
    """(problemas, avisos). Base pequena: conferência completa com os dados; base grande: só o esquema
    (a completa roda no treino, em segundo plano)."""
    insp = arm.ler_json(arm.pasta(base_id) / "inspecao.json") or {}
    problemas = conferir_esquema(insp, m)
    if problemas:
        return problemas, []
    if arm.bytes_total(base_id) > limite_validacao_sincrona_bytes():
        return [], ["Base grande: o mapeamento passou pela conferência de nomes; a conferência completa dos dados "
                    "roda no treino e aparece no status se falhar."]
    try:
        tabelas = ler_tabelas(base_id)
        return [], list(tabelas.avisos) + m.validar(tabelas)
    except ErroMapeamento as e:
        return list(e.problemas), []


# --------------------------------------------------------------------------- config
def montar_config(op: Opcoes) -> Config:
    """Monta o `Config` lendo os campos por `dataclasses.fields` (campos novos do motor não quebram a API)."""
    campos = {f.name for f in fields(Config)}
    padrao = Config()
    diretos = {"modelo": op.modelo, "min_hist": op.min_hist, "delta_risco": op.delta_risco,
               "validar": op.validar, "dobras": op.dobras, "semente": op.semente}
    valores = {k: v for k, v in diretos.items() if v is not None and k in campos}
    problemas: list[str] = []
    for k, v in (op.avancado or {}).items():
        alvo = getattr(padrao, k, None)
        if alvo is not None and not isinstance(v, type(alvo)):
            if isinstance(alvo, bool) or isinstance(v, bool):
                problemas.append(f"A opção avançada '{k}' espera {type(alvo).__name__}.")
                continue
            try:
                v = type(alvo)(v)
            except (TypeError, ValueError):
                problemas.append(f"A opção avançada '{k}' espera {type(alvo).__name__} (recebido: {v!r}).")
                continue
        valores[k] = v
    if problemas:
        raise ErroMapeamento(problemas)
    return Config(**valores)


# --------------------------------------------------------------------------- treino
def _progresso(base_id: str, eid: str, inicio: float):
    ultimo = {"t": 0.0, "etapa": ""}

    def cb(etapa: str, fracao: float) -> None:
        agora = time.monotonic()
        if etapa == ultimo["etapa"] and agora - ultimo["t"] < 0.5 and fracao < 1.0:
            return
        ultimo.update(t=agora, etapa=etapa)
        arm.escrever_status_execucao(base_id, eid, etapa=etapa, fracao=round(float(fracao), 3),
                                     segundos=round(agora - inicio, 1))

    return cb


def _treinar(base_id: str, eid: str) -> None:
    """Roda uma execução do começo ao fim e deixa o status/resumo no disco (nunca levanta)."""
    destino = arm.pasta_execucao(base_id, eid)
    arm.escrever_status_execucao(base_id, eid, estado="treinando", etapa="preparando", fracao=0.0,
                                 iniciado_em=arm.agora(), pid=os.getpid(), mono=time.monotonic())
    inicio = time.monotonic()
    try:
        m = Mapeamento.from_dict(arm.ler_json(destino / "mapeamento.json") or {})
        cfg = Config(**(arm.ler_json(destino / "config.json") or {}))
        tabelas = ler_tabelas(base_id)
        r = treinar(tabelas, m, progresso=_progresso(base_id, eid, inicio), config=cfg)
        r.salvar(destino)
        st = arm.escrever_status_execucao(
            base_id, eid, estado="pronta", etapa="concluído", fracao=1.0, problemas=[], mensagem=None,
            avisos=list(r.avisos), n_clientes=len(r.clientes.get("clientes", [])),
            concluido_em=arm.agora(), segundos=round(time.monotonic() - inicio, 1), pid=None, mono=None)
        arm.gravar_json(destino / "resumo.json", montar_resumo(r.clientes, r.validacao, r.pesos, st))
        arm.ativar_se_primeira(base_id)
    except ErroMapeamento as e:
        arm.escrever_status_execucao(base_id, eid, estado="erro", etapa="mapeamento inválido",
                                     problemas=list(e.problemas), concluido_em=arm.agora(),
                                     mensagem="Mapeamento inválido: " + " | ".join(e.problemas),
                                     segundos=round(time.monotonic() - inicio, 1), pid=None, mono=None)
    except Exception as e:  # noqa: BLE001 — qualquer falha do motor vira status legível
        traceback.print_exc()
        arm.escrever_status_execucao(base_id, eid, estado="erro", etapa="falha no treino",
                                     mensagem=f"O treino falhou: {type(e).__name__}: {e}",
                                     concluido_em=arm.agora(),
                                     segundos=round(time.monotonic() - inicio, 1), pid=None, mono=None)


# --------------------------------------------------------------------------- fila
def _rodar(base_id: str, eid: str) -> None:
    try:
        _treinar(base_id, eid)
    finally:
        with _fila_lock:
            _rodando.discard((base_id, eid))
        _bombear()


def _bombear() -> None:
    """Começa quantas execuções da fila couberem no limite de treinos simultâneos."""
    limite = max_treinos_simultaneos()
    with _fila_lock:
        novas = []
        while _fila and len(_rodando) < limite:
            item = _fila.pop(0)
            _rodando.add(item)
            novas.append(item)
    for item in novas:
        t = threading.Thread(target=_rodar, args=item, daemon=True, name=f"treino-{item[0]}-{item[1]}")
        _threads[item] = t
        t.start()


def disparar_treino(base_id: str, m: Mapeamento, cfg: Config, opcoes: dict,
                    rotulo: str | None = None) -> dict:
    """Cria a execução (estado "na_fila") e a coloca na fila. Devolve o status inicial."""
    eid = arm.nova_execucao_id(arm.pasta_execucoes(base_id), cfg.modelo, rotulo)
    st = arm.criar_execucao(base_id, eid, m.to_dict(), cfg.to_dict(), opcoes, rotulo)
    with _fila_lock:
        _fila.append((base_id, eid))
        posicao = len(_fila)
    arm.escrever_status_execucao(base_id, eid, etapa=f"na fila (posição {posicao})")
    _bombear()
    return arm.item_execucao(base_id, eid)          # status público (sem pid/mono) + resumo + ativa


def esperar_treino(base_id: str, execucao_id: str | None = None, segundos: float = 600.0) -> None:
    """Só para os testes: espera as threads de treino da base (ou de uma execução) terminarem."""
    fim = time.monotonic() + segundos
    while time.monotonic() < fim:
        alvos = [k for k in list(_threads) if k[0] == base_id and execucao_id in (None, k[1])]
        vivas = [_threads[k] for k in alvos if _threads[k].is_alive()]
        pendente = any(k in _fila or k in _rodando for k in alvos)
        if not vivas and not pendente:
            return
        for t in vivas:
            t.join(max(0.0, fim - time.monotonic()))
        time.sleep(0.02)
