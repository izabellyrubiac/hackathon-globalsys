"""Armazenamento em arquivos (sem banco): uma pasta por base em `dados/bases/<base_id>/`.

    arquivos/           os arquivos enviados, com o nome saneado
    base.json           {base_id, nome, criada_em, arquivos:[{nome, bytes}], bytes_total}
    inspecao.json       saída de `motor.inspecionar` (tabelas, colunas, sugestão de mapeamento)
    mapeamento.json     o mapeamento do último treino pedido (o de cada execução fica com ela)
    ativo.json          {"execucao_id": "..."} — qual execução as rotas de compatibilidade servem
    execucoes/<execucao_id>/
        config.json     `motor.Config` usado
        mapeamento.json mapeamento confirmado para esta execução
        status.json     {estado, etapa, fracao, mensagem, problemas, avisos, rotulo, ...}
        clientes.json / validacao.json / pesos.json   resultados do motor
        resumo.json     números da tela de comparação (sem abrir os arquivos grandes)

Cada pedido de treino cria uma **execução** (versão do modelo). Estados de uma execução:
"na_fila" → "treinando" → "pronta" | "erro". A base não tem estado próprio: `estado_base` deriva
do que está rodando, da execução ativa ou da mais recente.

O status fica no disco para sobreviver a um reinício; um `threading.Lock` por execução serializa as
escritas dentro do processo e a gravação é atômica (arquivo temporário + `os.replace`).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
import unicodedata
import uuid
from datetime import datetime
from pathlib import Path

from .ajustes import EXTENSOES_OK, raiz_bases

ESTADOS = ("na_fila", "treinando", "pronta", "erro", "cancelada")
ATIVAS = ("na_fila", "treinando")           # execuções que ainda vão mexer no disco
RESULTADOS = ("clientes", "validacao", "pesos")
DEMONSTRACOES = ("inovaapps", "redes")

_RE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,80}$")
_RE_EXEC = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$")   # o id tem o "T" do carimbo de tempo
_locks: dict[str, threading.Lock] = {}
_lock_geral = threading.Lock()


class ErroArmazenamento(ValueError):
    """Problema de entrada tratável (vira 400/404/409/413 no `main.py`)."""

    def __init__(self, mensagem: str, codigo: int = 400):
        self.codigo = codigo
        super().__init__(mensagem)


# --------------------------------------------------------------------------- nomes e caminhos
def agora() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _sem_acento(texto: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", texto) if not unicodedata.combining(c))


def nome_seguro(nome: str) -> str:
    """Nome de arquivo sem caminho, sem acento e sem surpresa (bloqueia `..`, `/` e `\\`)."""
    bruto = str(nome or "").replace("\\", "/").split("/")[-1]
    limpo = re.sub(r"[^A-Za-z0-9._ -]+", "_", _sem_acento(bruto)).strip(" .")
    limpo = limpo[:150]
    if not limpo or limpo in (".", ".."):
        raise ErroArmazenamento(f"Nome de arquivo inválido: '{nome}'. Use letras, números, '.', '-' ou '_'.")
    if Path(limpo).suffix.lower() not in EXTENSOES_OK:
        extensoes = ", ".join(sorted(EXTENSOES_OK))
        raise ErroArmazenamento(f"Formato não suportado: '{bruto}'. Envie um .xlsx (uma tabela por aba) ou "
                                f"vários .csv (extensões aceitas: {extensoes}).")
    return limpo


def novo_base_id(nome: str) -> str:
    """Identificador legível a partir do nome do arquivo + sufixo aleatório (sempre único)."""
    slug = re.sub(r"[^a-z0-9]+", "-", _sem_acento(str(nome or "")).lower()).strip("-")[:40]
    return f"{slug or 'base'}-{uuid.uuid4().hex[:6]}"


def valido(base_id: str) -> bool:
    return bool(_RE_ID.match(str(base_id or "")))


def pasta(base_id: str) -> Path:
    """Pasta da base. Levanta 404 se o id for inválido (nunca sai de `dados/bases`)."""
    if not valido(base_id):
        raise ErroArmazenamento(f"Base '{base_id}' não encontrada.", 404)
    raiz = raiz_bases()
    p = (raiz / base_id).resolve()
    if p.parent != raiz:
        raise ErroArmazenamento(f"Base '{base_id}' não encontrada.", 404)
    return p


def existe(base_id: str) -> bool:
    try:
        return (pasta(base_id) / "base.json").exists()
    except ErroArmazenamento:
        return False


def exigir(base_id: str) -> Path:
    p = pasta(base_id)
    if not (p / "base.json").exists():
        raise ErroArmazenamento(f"Base '{base_id}' não encontrada. Envie a base em POST /api/bases.", 404)
    return p


def caminhos_arquivos(base_id: str) -> list[Path]:
    meta = ler_json(exigir(base_id) / "base.json") or {}
    return [pasta(base_id) / "arquivos" / a["nome"] for a in meta.get("arquivos", [])]


def lock(chave: str) -> threading.Lock:
    with _lock_geral:
        return _locks.setdefault(chave, threading.Lock())


# --------------------------------------------------------------------------- JSON
def ler_json(caminho: Path):
    try:
        return json.loads(Path(caminho).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def gravar_json(caminho: Path, obj) -> Path:
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    tmp = caminho.with_name(f".{caminho.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, caminho)
    return caminho


# --------------------------------------------------------------------------- execuções: caminhos
def _rotulo_curto(texto: str, limite: int = 24) -> str:
    return re.sub(r"[^a-z0-9]+", "-", _sem_acento(str(texto or "")).lower()).strip("-")[:limite]


def nova_execucao_id(pasta_execucoes: Path, modelo: str = "auto", rotulo: str | None = None) -> str:
    """Curto e ordenável por tempo: `20260920T011305-lightgbm` (sufixo numérico se houver colisão)."""
    carimbo = datetime.now().strftime("%Y%m%dT%H%M%S")
    sufixo = _rotulo_curto(rotulo) or _rotulo_curto(modelo) or "auto"
    base = f"{carimbo}-{sufixo}"[:80]
    eid, n = base, 1
    while (pasta_execucoes / eid).exists():
        n += 1
        eid = f"{base}-{n}"[:80]
    return eid


def pasta_execucoes(base_id: str) -> Path:
    return pasta(base_id) / "execucoes"


def valido_execucao(execucao_id: str) -> bool:
    return bool(_RE_EXEC.match(str(execucao_id or "")))


def pasta_execucao(base_id: str, execucao_id: str) -> Path:
    """Pasta da execução, saneada (nunca sai de `execucoes/`)."""
    raiz = pasta_execucoes(base_id)
    if not valido_execucao(execucao_id):
        raise ErroArmazenamento(f"Execução '{execucao_id}' não encontrada nesta base.", 404)
    p = (raiz / execucao_id).resolve()
    if p.parent != raiz.resolve():
        raise ErroArmazenamento(f"Execução '{execucao_id}' não encontrada nesta base.", 404)
    return p


def exigir_execucao(base_id: str, execucao_id: str) -> Path:
    exigir(base_id)
    p = pasta_execucao(base_id, execucao_id)
    if not (p / "status.json").exists():
        raise ErroArmazenamento(f"Execução '{execucao_id}' não encontrada na base '{base_id}'. "
                                f"Veja GET /api/bases/{base_id}/execucoes.", 404)
    return p


def ids_execucoes(base_id: str) -> list[str]:
    """Ids existentes, da mais recente para a mais antiga (o id começa pelo carimbo de tempo)."""
    raiz = pasta_execucoes(base_id)
    if not raiz.exists():
        return []
    return sorted((p.name for p in raiz.iterdir()
                   if p.is_dir() and valido_execucao(p.name) and (p / "status.json").exists()), reverse=True)


# --------------------------------------------------------------------------- execuções: status
def _status_inicial(base_id: str, execucao_id: str, **campos) -> dict:
    return {"base_id": base_id, "execucao_id": execucao_id, "rotulo": None, "estado": "na_fila",
            "etapa": "na fila", "fracao": 0.0, "mensagem": None, "problemas": [], "avisos": [],
            "n_clientes": None, "criada_em": agora(), "iniciado_em": None, "concluido_em": None,
            "atualizado_em": agora(), "segundos": 0.0, "pid": os.getpid(), "opcoes": None, **campos}


def _publico(st: dict) -> dict:
    out = dict(st)
    out["segundos"] = segundos_decorridos(st)
    out.pop("pid", None)
    out.pop("mono", None)
    return out


def ler_status_execucao(base_id: str, execucao_id: str) -> dict:
    """Status do disco. Execução órfã (a API reiniciou e o processo morreu) vira erro legível."""
    p = exigir_execucao(base_id, execucao_id)
    st = ler_json(p / "status.json") or _status_inicial(base_id, execucao_id)
    if st.get("estado") in ATIVAS and st.get("pid") not in (None, os.getpid()):
        st = escrever_status_execucao(
            base_id, execucao_id, estado="erro", etapa="interrompido", pid=None,
            mensagem="Esta execução foi interrompida porque a API reiniciou. Peça o treino de novo.",
            fracao=st.get("fracao") or 0.0)
    return st


def escrever_status_execucao(base_id: str, execucao_id: str, **campos) -> dict:
    """Mescla `campos` no status.json da execução (atômico, com lock por execução)."""
    p = pasta_execucao(base_id, execucao_id)
    with lock(f"{base_id}/{execucao_id}"):
        st = ler_json(p / "status.json") or _status_inicial(base_id, execucao_id)
        st.update(campos)
        st["base_id"], st["execucao_id"] = base_id, execucao_id
        st["atualizado_em"] = agora()
        gravar_json(p / "status.json", st)
        return st


def segundos_decorridos(st: dict) -> float:
    inicio = st.get("mono")
    if st.get("estado") == "treinando" and isinstance(inicio, (int, float)):
        return round(time.monotonic() - float(inicio), 1)
    return round(float(st.get("segundos") or 0.0), 1)


def criar_execucao(base_id: str, execucao_id: str, mapeamento: dict, config: dict, opcoes: dict,
                   rotulo: str | None = None) -> dict:
    """Cria a pasta da execução com estado "na_fila" e grava mapeamento e config."""
    p = pasta_execucao(base_id, execucao_id)
    p.mkdir(parents=True, exist_ok=True)
    gravar_json(p / "mapeamento.json", mapeamento)
    gravar_json(p / "config.json", config)
    gravar_json(pasta(base_id) / "mapeamento.json", mapeamento)     # último mapeamento pedido (compat.)
    gravar_json(pasta(base_id) / "config.json", config)
    st = _status_inicial(base_id, execucao_id, rotulo=(rotulo or None), opcoes=opcoes)
    gravar_json(p / "status.json", st)
    return st


# --------------------------------------------------------------------------- execução ativa
def ativa(base_id: str) -> str | None:
    """Id da execução ativa (a que as rotas de compatibilidade servem), ou None."""
    doc = ler_json(pasta(base_id) / "ativo.json") or {}
    eid = doc.get("execucao_id")
    if not eid or not valido_execucao(eid) or not (pasta_execucao(base_id, eid) / "status.json").exists():
        return None
    return eid


def ativar(base_id: str, execucao_id: str) -> str:
    """Marca a execução como ativa. Só execução "pronta" pode ser ativada."""
    st = ler_status_execucao(base_id, execucao_id)
    if st.get("estado") != "pronta":
        raise ErroArmazenamento(f"A execução '{execucao_id}' está em '{st.get('estado')}' — só dá para ativar "
                                "uma execução pronta.", 409)
    gravar_json(pasta(base_id) / "ativo.json", {"execucao_id": execucao_id, "ativada_em": agora()})
    return execucao_id


def ativar_se_primeira(base_id: str) -> str | None:
    """A primeira execução pronta de uma base vira ativa sozinha."""
    if ativa(base_id) is None:
        for eid in reversed(ids_execucoes(base_id)):
            if (ler_json(pasta_execucao(base_id, eid) / "status.json") or {}).get("estado") == "pronta":
                return ativar(base_id, eid)
    return ativa(base_id)


def exigir_ativa(base_id: str) -> str:
    eid = ativa(base_id)
    if eid is not None:
        return eid
    prontas = [e for e in ids_execucoes(base_id)
               if (ler_json(pasta_execucao(base_id, e) / "status.json") or {}).get("estado") == "pronta"]
    if prontas:
        return ativar(base_id, prontas[0])
    rodando = [e for e in ids_execucoes(base_id)
               if (ler_json(pasta_execucao(base_id, e) / "status.json") or {}).get("estado") in ATIVAS]
    if rodando:
        raise ErroArmazenamento("Esta base ainda não tem nenhuma execução pronta — o treino está rodando. "
                                f"Acompanhe em GET /api/bases/{base_id}/execucoes.", 404)
    raise ErroArmazenamento("Esta base ainda não tem nenhuma execução pronta. Confirme o mapeamento em "
                            f"POST /api/bases/{base_id}/treinar.", 404)


# --------------------------------------------------------------------------- execuções: consulta
def resumo_execucao(base_id: str, execucao_id: str) -> dict | None:
    return ler_json(pasta_execucao(base_id, execucao_id) / "resumo.json")


def item_execucao(base_id: str, execucao_id: str, ativo: str | None = None) -> dict:
    """Status público + resumo + se é a ativa (uma linha da lista de versões)."""
    st = ler_status_execucao(base_id, execucao_id)
    return {**_publico(st), "ativa": execucao_id == (ativa(base_id) if ativo is None else ativo),
            "resumo": resumo_execucao(base_id, execucao_id)}


def listar_execucoes(base_id: str) -> list[dict]:
    """Todas as execuções da base, da mais recente para a mais antiga."""
    exigir(base_id)
    ativo = ativa(base_id)
    return [item_execucao(base_id, eid, ativo) for eid in ids_execucoes(base_id)]


def ha_treino_em_andamento(base_id: str) -> bool:
    """Alguma execução deste processo ainda na fila ou treinando."""
    for e in ids_execucoes(base_id):
        st = ler_json(pasta_execucao(base_id, e) / "status.json") or {}
        if st.get("estado") in ATIVAS and st.get("pid") in (None, os.getpid()):
            return True
    return False


def estado_base(base_id: str) -> dict:
    """Status derivado da base: o que está rodando, senão a ativa, senão a mais recente."""
    exigir(base_id)
    ids = ids_execucoes(base_id)
    estados = {e: ler_status_execucao(base_id, e) for e in ids}
    escolhida = None
    for estado in ("treinando", "na_fila"):
        candidatas = [e for e in ids if estados[e].get("estado") == estado]
        if candidatas:
            escolhida = candidatas[0]
            break
    if escolhida is None:
        escolhida = ativa(base_id) or (ids[0] if ids else None)
    if escolhida is None:
        return {"base_id": base_id, "execucao_id": None, "rotulo": None, "estado": "inspecionada",
                "etapa": "base enviada e inspecionada", "fracao": 0.0, "mensagem": None, "problemas": [],
                "avisos": list((ler_json(pasta(base_id) / "inspecao.json") or {}).get("avisos", [])),
                "n_clientes": None, "iniciado_em": None, "atualizado_em": None, "segundos": 0.0,
                "opcoes": None, "n_execucoes": 0, "execucao_ativa": None}
    st = _publico(estados[escolhida])
    return {**st, "n_execucoes": len(ids), "execucao_ativa": ativa(base_id)}


def apagar_execucao(base_id: str, execucao_id: str) -> None:
    p = exigir_execucao(base_id, execucao_id)
    st = ler_status_execucao(base_id, execucao_id)
    if execucao_id == ativa(base_id):
        raise ErroArmazenamento("Esta é a execução ativa. Ative outra versão antes de apagar esta "
                                f"(POST /api/bases/{base_id}/execucoes/<outra>/ativar).", 409)
    if st.get("estado") in ATIVAS:
        rotulo = "na fila" if st.get("estado") == "na_fila" else "em treino"
        raise ErroArmazenamento(f"A execução '{execucao_id}' está {rotulo}. Espere terminar para apagar.", 409)
    shutil.rmtree(p, ignore_errors=True)
    with _lock_geral:
        _locks.pop(f"{base_id}/{execucao_id}", None)


def caminho_resultado(base_id: str, execucao_id: str, nome: str) -> Path:
    if nome not in RESULTADOS:
        raise ErroArmazenamento(f"Resultado desconhecido: '{nome}'.", 404)
    p = exigir_execucao(base_id, execucao_id) / f"{nome}.json"
    if not p.exists():
        st = ler_status_execucao(base_id, execucao_id)
        estado = st.get("estado")
        if estado in ATIVAS:
            raise ErroArmazenamento(f"A execução '{execucao_id}' ainda não terminou ({st.get('etapa')}). Acompanhe "
                                    f"em GET /api/bases/{base_id}/execucoes/{execucao_id}.", 404)
        if estado == "erro":
            raise ErroArmazenamento(f"Esta execução falhou: {st.get('mensagem') or 'erro desconhecido'}", 404)
        raise ErroArmazenamento(f"O arquivo '{nome}.json' desta execução sumiu do disco. Treine de novo.", 404)
    return p


# --------------------------------------------------------------------------- bases
def gravar_meta(base_id: str, nome: str, arquivos: list[dict]) -> dict:
    meta = {"base_id": base_id, "nome": nome, "criada_em": agora(), "arquivos": arquivos,
            "bytes_total": sum(int(a["bytes"]) for a in arquivos)}
    gravar_json(pasta(base_id) / "base.json", meta)
    return meta


def bytes_total(base_id: str) -> int:
    meta = ler_json(pasta(base_id) / "base.json") or {}
    return int(meta.get("bytes_total") or 0)


def listar() -> list[dict]:
    """Todas as bases, da mais recente para a mais antiga."""
    raiz = raiz_bases()
    if not raiz.exists():
        return []
    out = []
    for p in sorted(raiz.iterdir()):
        if not p.is_dir() or not (p / "base.json").exists() or not valido(p.name):
            continue
        meta = ler_json(p / "base.json") or {}
        st = estado_base(p.name)
        eid = st.get("execucao_ativa")
        resumo = resumo_execucao(p.name, eid) if eid else None
        out.append({"base_id": p.name, "nome": meta.get("nome") or p.name,
                    "criada_em": meta.get("criada_em"), "bytes_total": meta.get("bytes_total"),
                    "arquivos": [a.get("nome") for a in meta.get("arquivos", [])],
                    "estado": st.get("estado"), "etapa": st.get("etapa"),
                    "n_clientes": (resumo or {}).get("n_clientes") or st.get("n_clientes"),
                    "n_execucoes": st.get("n_execucoes", 0), "execucao_ativa": eid,
                    "demonstracao": p.name in DEMONSTRACOES})
    return sorted(out, key=lambda b: (b.get("criada_em") or ""), reverse=True)


def apagar(base_id: str) -> None:
    p = exigir(base_id)
    if ha_treino_em_andamento(base_id):
        raise ErroArmazenamento("Não dá para apagar uma base com treino em andamento ou na fila. "
                                "Espere terminar.", 409)
    shutil.rmtree(p, ignore_errors=True)
    with _lock_geral:
        for chave in [k for k in _locks if k == base_id or k.startswith(f"{base_id}/")]:
            _locks.pop(chave, None)
