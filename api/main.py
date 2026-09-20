"""Endpoints da API. Camada fina: recebe arquivos, chama o `motor/` e devolve JSON.

    uv run uvicorn api.main:app --reload      # docs em http://localhost:8000/docs
"""

from __future__ import annotations

import json
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile, status as http
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from motor import ErroMapeamento, Mapeamento
from motor.leitura import EXT_EXCEL

from . import VERSAO, armazenamento as arm, demo, servico
from .ajustes import ORIGENS_PERMITIDAS, PEDACO_BYTES, max_arquivos, max_upload_bytes
from .modelos import (FAIXAS, BaseCriada, BaseDetalhe, BaseResumo, Comparacao, Execucao, Inspecao,
                      ListaExecucoes, PaginaClientes, PedidoTreino, Saude, Status, TreinoAceito)
from .resumo import comparar


@asynccontextmanager
async def ciclo(app: FastAPI):
    try:
        criadas = await run_in_threadpool(demo.preparar)
        if criadas:
            print(f"[api] bases de demonstração: {', '.join(criadas)}")
    except Exception as e:  # noqa: BLE001 — a demonstração nunca impede a API de subir
        print(f"[api] bases de demonstração não preparadas: {type(e).__name__}: {e}")
    yield


app = FastAPI(
    title="INOVAAPPS 2026 — API de risco de cancelamento",
    version=VERSAO,
    description=__doc__,
    lifespan=ciclo,
)

app.add_middleware(CORSMiddleware, allow_origin_regex=ORIGENS_PERMITIDAS, allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])


# --------------------------------------------------------------------------- erros em pt-BR
COD_422 = getattr(http, "HTTP_422_UNPROCESSABLE_CONTENT", None) or http.HTTP_422_UNPROCESSABLE_ENTITY
COD_413 = getattr(http, "HTTP_413_CONTENT_TOO_LARGE", None) or http.HTTP_413_REQUEST_ENTITY_TOO_LARGE


@app.exception_handler(arm.ErroArmazenamento)
async def _erro_armazenamento(_: Request, e: arm.ErroArmazenamento):
    return JSONResponse(status_code=e.codigo, content={"detalhe": str(e), "problemas": []})


@app.exception_handler(ErroMapeamento)
async def _erro_mapeamento(_: Request, e: ErroMapeamento):
    return JSONResponse(status_code=COD_422,
                        content={"detalhe": "Mapeamento inválido.", "problemas": list(e.problemas)})


@app.exception_handler(HTTPException)
async def _erro_http(_: Request, e: HTTPException):
    """Toda resposta de erro tem o mesmo formato: {detalhe, problemas}."""
    d = e.detail
    corpo = d if isinstance(d, dict) and "detalhe" in d else {"detalhe": str(d), "problemas": []}
    return JSONResponse(status_code=e.status_code, content=corpo, headers=getattr(e, "headers", None))


@app.exception_handler(RequestValidationError)
async def _erro_corpo(_: Request, e: RequestValidationError):
    problemas = [f"{'.'.join(str(x) for x in p.get('loc', ())[1:]) or 'corpo'}: {p.get('msg')}" for p in e.errors()]
    return JSONResponse(status_code=COD_422,
                        content={"detalhe": "Não entendi os dados enviados.", "problemas": problemas})


def _erro(codigo: int, mensagem: str, problemas: list[str] | None = None):
    return HTTPException(status_code=codigo, detail={"detalhe": mensagem, "problemas": problemas or []})


# --------------------------------------------------------------------------- utilidades
def _status(base_id: str) -> dict:
    return arm.estado_base(base_id)


def _inspecao(base_id: str) -> dict:
    insp = arm.ler_json(arm.pasta(base_id) / "inspecao.json")
    if insp is None:
        insp = servico.inspecionar_base(base_id)
    return insp


# --------------------------------------------------------------------------- saúde
@app.get("/api/saude", response_model=Saude, tags=["saúde"], summary="A API está de pé?")
def saude():
    return {"ok": True, "versao": VERSAO, "bases": len(arm.listar())}


# --------------------------------------------------------------------------- bases
@app.post("/api/bases", response_model=BaseCriada, status_code=http.HTTP_201_CREATED, tags=["bases"],
          summary="Enviar uma base (.xlsx com várias abas ou vários .csv)")
async def enviar_base(arquivos: list[UploadFile] = File(..., description="Um .xlsx ou vários .csv.")):
    """Grava os arquivos em disco por streaming, lê as tabelas pelo motor e devolve a inspeção
    (tabelas, colunas, tipos, exemplos) com a **sugestão de mapeamento** para o usuário confirmar."""
    if not arquivos:
        raise _erro(400, "Nenhum arquivo enviado. Envie um .xlsx (uma tabela por aba) ou vários .csv.")
    if len(arquivos) > max_arquivos():
        raise _erro(400, f"Envie no máximo {max_arquivos()} arquivos de uma vez.")

    nomes = [arm.nome_seguro(a.filename) for a in arquivos]
    if len(set(nomes)) != len(nomes):
        raise _erro(400, "Dois arquivos com o mesmo nome no envio. Renomeie um deles.")
    excel = [n for n in nomes if Path(n).suffix.lower() in EXT_EXCEL]
    if excel and len(nomes) > 1:
        raise _erro(400, "Envie um único .xlsx (uma tabela por aba) ou vários .csv — não os dois juntos.")

    base_id = arm.novo_base_id(Path(nomes[0]).stem)
    destino = arm.pasta(base_id) / "arquivos"
    destino.mkdir(parents=True, exist_ok=True)
    limite, total, meta = max_upload_bytes(), 0, []
    try:
        for envio, nome in zip(arquivos, nomes):
            caminho = destino / nome
            tamanho = 0
            with caminho.open("wb") as saida:
                while True:
                    pedaco = await envio.read(PEDACO_BYTES)
                    if not pedaco:
                        break
                    total += len(pedaco)
                    tamanho += len(pedaco)
                    if total > limite:
                        raise _erro(COD_413,
                                    f"Envio maior que o limite de {limite // (1024 * 1024)} MB.")
                    await run_in_threadpool(saida.write, pedaco)
            if tamanho == 0:
                raise _erro(400, f"O arquivo '{nome}' está vazio.")
            meta.append({"nome": nome, "bytes": tamanho})
        arm.gravar_meta(base_id, Path(nomes[0]).stem if len(nomes) == 1 else f"{len(nomes)} arquivos", meta)
        insp = await run_in_threadpool(servico.inspecionar_base, base_id)
    except (HTTPException, arm.ErroArmazenamento):
        servico.esquecer(base_id)
        shutil.rmtree(arm.pasta(base_id), ignore_errors=True)
        raise
    except (ValueError, OSError) as e:
        servico.esquecer(base_id)
        shutil.rmtree(arm.pasta(base_id), ignore_errors=True)
        raise _erro(400, f"Não consegui ler a base: {e}") from e

    m = arm.ler_json(arm.pasta(base_id) / "base.json")
    return {**m, "arquivos": [a["nome"] for a in meta], "inspecao": insp, "status": _status(base_id)}


@app.get("/api/bases", response_model=list[BaseResumo], tags=["bases"], summary="Listar as bases")
def listar_bases():
    """Mais recentes primeiro. `n_clientes` só existe nas bases já treinadas."""
    return arm.listar()


@app.get("/api/bases/{base_id}", response_model=BaseDetalhe, tags=["bases"],
         summary="Inspeção, mapeamento salvo, execuções e status de uma base")
def detalhe_base(base_id: str):
    p = arm.exigir(base_id)
    meta = arm.ler_json(p / "base.json") or {}
    return {**meta, "arquivos": [a["nome"] for a in meta.get("arquivos", [])],
            "demonstracao": base_id in arm.DEMONSTRACOES, "inspecao": _inspecao(base_id),
            "mapeamento": arm.ler_json(p / "mapeamento.json"), "opcoes": arm.ler_json(p / "config.json"),
            "status": _status(base_id), "execucao_ativa": arm.ativa(base_id),
            "execucoes": arm.listar_execucoes(base_id)}


@app.get("/api/bases/{base_id}/inspecao", response_model=Inspecao, tags=["bases"],
         summary="Só a inspeção (tabelas, colunas e sugestão de mapeamento)")
def inspecao_base(base_id: str):
    arm.exigir(base_id)
    return _inspecao(base_id)


@app.delete("/api/bases/{base_id}", status_code=http.HTTP_204_NO_CONTENT, tags=["bases"],
            summary="Apagar a base e os resultados")
def apagar_base(base_id: str):
    arm.apagar(base_id)
    servico.esquecer(base_id)
    return None


# --------------------------------------------------------------------------- treino
@app.post("/api/bases/{base_id}/treinar", response_model=TreinoAceito, status_code=http.HTTP_202_ACCEPTED,
          tags=["treino"], summary="Confirmar o mapeamento e criar uma execução (treino em segundo plano)")
def treinar_base(base_id: str, pedido: PedidoTreino):
    """Confere o mapeamento (422 com a lista de problemas em pt-BR) e **cria uma execução** — uma versão
    do modelo com seus próprios resultados. Responde 202 na hora com o `execucao_id`; acompanhe em
    `GET /api/bases/{base_id}/execucoes/{execucao_id}`.

    Vários treinos podem ser pedidos em sequência: rodam no máximo `API_TREINOS_SIMULTANEOS` (padrão 1)
    por vez e os demais ficam `na_fila` até chegar a vez (nunca mais 409 aqui).

    `rotulo` (opcional) é um nome curto livre para a versão. A primeira execução pronta da base vira a
    ativa automaticamente.

    Opções avançadas (`opcoes`): `modelo` ("auto" | "logistica" | "random_forest" | "lightgbm"),
    `horizonte_meses`, `min_hist`, `validar`, `dobras`, `semente` e `avancado` (qualquer outro campo de
    `motor.Config`). `delta_risco` (padrão `false`) liga o motor em dois estágios: testado nas bases
    INOVAAPPS e redes, **não melhora o acerto** (um braço placebo com o delta embaralhado entrega o mesmo
    ganho) e custa 3,2× o tempo de treino; mantido como opção avançada."""
    arm.exigir(base_id)
    dados = pedido.mapeamento.model_dump()
    if pedido.opcoes.horizonte_meses is not None:
        dados["horizonte_meses"] = pedido.opcoes.horizonte_meses
    m = Mapeamento.from_dict(dados)
    cfg = servico.montar_config(pedido.opcoes)

    problemas, avisos = servico.conferir_mapeamento(base_id, m)
    if problemas:
        raise _erro(COD_422, "Mapeamento inválido.", problemas)
    opcoes = {"modelo": cfg.modelo, "horizonte_meses": int(m.horizonte_meses), "min_hist": cfg.min_hist,
              "delta_risco": bool(cfg.delta_risco), "validar": cfg.validar, "dobras": cfg.dobras,
              "semente": cfg.semente}
    st = servico.disparar_treino(base_id, m, cfg, opcoes, (pedido.rotulo or "").strip() or None)
    return {"base_id": base_id, "execucao_id": st["execucao_id"], "execucao": st,
            "status": _status(base_id), "avisos": avisos}


@app.get("/api/bases/{base_id}/status", response_model=Status, tags=["treino"],
         summary="Etapa, progresso e estado do treino (visão da base)")
def status_base(base_id: str):
    """Deriva das execuções: a que está treinando, senão a que está na fila, senão a ativa, senão a mais
    recente. `estado`: "inspecionada" | "na_fila" | "treinando" | "pronta" | "erro"."""
    arm.exigir(base_id)
    return _status(base_id)


# --------------------------------------------------------------------------- execuções (versões)
@app.get("/api/bases/{base_id}/execucoes", response_model=ListaExecucoes, tags=["execuções"],
         summary="Listar as versões do modelo desta base (mais recente primeiro)")
def listar_execucoes(base_id: str):
    """Cada item traz o status e o `resumo.json` (null enquanto a execução não fica pronta), além de
    `ativa` — a versão que as rotas `clientes`/`validacao`/`pesos` sem `execucoes/` servem."""
    arm.exigir(base_id)
    return {"base_id": base_id, "execucao_ativa": arm.ativa(base_id),
            "execucoes": arm.listar_execucoes(base_id)}


@app.get("/api/bases/{base_id}/execucoes/{execucao_id}", response_model=Execucao, tags=["execuções"],
         summary="Status e resumo de uma execução")
def detalhe_execucao(base_id: str, execucao_id: str):
    arm.exigir_execucao(base_id, execucao_id)
    return arm.item_execucao(base_id, execucao_id)


@app.post("/api/bases/{base_id}/execucoes/{execucao_id}/ativar", response_model=Execucao, tags=["execuções"],
          summary="Tornar esta a versão ativa")
def ativar_execucao(base_id: str, execucao_id: str):
    """Só uma execução `pronta` pode ser ativada (409 nas demais). A ativa é a que as rotas de
    compatibilidade (`/clientes`, `/validacao`, `/pesos`) servem."""
    arm.exigir_execucao(base_id, execucao_id)
    arm.ativar(base_id, execucao_id)
    return arm.item_execucao(base_id, execucao_id)


@app.delete("/api/bases/{base_id}/execucoes/{execucao_id}", status_code=http.HTTP_204_NO_CONTENT,
            tags=["execuções"], summary="Apagar uma versão")
def apagar_execucao(base_id: str, execucao_id: str):
    """409 se for a execução ativa (ative outra antes) ou se ela estiver na fila/treinando."""
    arm.apagar_execucao(base_id, execucao_id)
    return None


@app.get("/api/bases/{base_id}/comparar", response_model=Comparacao, tags=["execuções"],
         summary="Tabela comparativa entre versões")
def comparar_execucoes(base_id: str,
                       execucoes: list[str] | None = Query(None,
                                                           description="Ids a comparar (repita o parâmetro). "
                                                                       "Sem nenhum, compara todas as prontas.")):
    """Monta a tabela só com os `resumo.json` — não abre `clientes.json` nem `validacao.json`.
    Precisa de pelo menos duas execuções prontas."""
    arm.exigir(base_id)
    ativa = arm.ativa(base_id)
    if execucoes:
        vistos, ids = set(), []
        for eid in execucoes:
            if eid not in vistos:
                vistos.add(eid)
                arm.exigir_execucao(base_id, eid)
                ids.append(eid)
    else:
        ids = [e for e in arm.ids_execucoes(base_id)
               if arm.ler_status_execucao(base_id, e).get("estado") == "pronta"]
    resumos, sem_resumo = [], []
    for eid in ids:
        r = arm.resumo_execucao(base_id, eid)
        if r:
            resumos.append(r)
        else:
            sem_resumo.append(eid)
    if sem_resumo:
        raise _erro(409, "Só dá para comparar execuções prontas. Ainda sem resultado: "
                         f"{', '.join(sem_resumo)}.")
    if len(resumos) < 2:
        raise _erro(400, "Informe pelo menos duas execuções prontas para comparar "
                         f"(esta base tem {len(resumos)}).")
    return {"base_id": base_id, **comparar(resumos, ativa)}


# --------------------------------------------------------------------------- resultados
def _clientes(base_id: str, execucao_id: str, limite: int, desde: int,
              faixa: list[str] | None, incluir_historico: bool) -> dict:
    if faixa:
        ruins = [f for f in faixa if f not in FAIXAS]
        if ruins:
            raise _erro(400, f"Faixa inválida: {', '.join(ruins)}. Use {', '.join(FAIXAS)}.")
    doc = json.loads(arm.caminho_resultado(base_id, execucao_id, "clientes").read_text(encoding="utf-8"))
    fila = doc.get("clientes", [])
    filtrada = [c for c in fila if c.get("faixa_risco") in faixa] if faixa else fila
    pagina = filtrada[desde:desde + limite]
    if not incluir_historico:
        pagina = [{k: v for k, v in c.items() if k != "historico"} for c in pagina]
    return {"base_id": base_id, "execucao_id": execucao_id, "gerado_em": doc.get("gerado_em"),
            "mes_referencia": doc.get("mes_referencia"), "modelo": doc.get("modelo"),
            "resumo": doc.get("resumo"), "total": len(fila), "total_filtrado": len(filtrada),
            "desde": desde, "limite": limite, "clientes": pagina}


LIMITE = Query(100, ge=1, le=5000, description="Quantos clientes devolver.")
DESDE = Query(0, ge=0, description="Quantos pular (a partir do topo da fila).")
FAIXA = Query(None, description='Filtro: "alto", "atencao" e/ou "baixo".')
HIST = Query(True, description="False deixa a resposta bem menor.")


@app.get("/api/bases/{base_id}/execucoes/{execucao_id}/clientes", response_model=PaginaClientes,
         tags=["resultados"], summary="Fila de clientes de uma execução (paginada, na ordem do motor)")
def clientes_execucao(base_id: str, execucao_id: str, limite: int = LIMITE, desde: int = DESDE,
                      faixa: list[str] | None = FAIXA, incluir_historico: bool = HIST):
    """A ordem é a do motor (perda anual esperada) e **nunca** é refeita aqui: filtrar não reordena."""
    arm.exigir_execucao(base_id, execucao_id)
    return _clientes(base_id, execucao_id, limite, desde, faixa, incluir_historico)


@app.get("/api/bases/{base_id}/execucoes/{execucao_id}/validacao", tags=["resultados"],
         summary="validacao.json de uma execução")
def validacao_execucao(base_id: str, execucao_id: str):
    """Contrato em `motor/saida.py`. Devolvido direto do disco, sem reserializar."""
    arm.exigir_execucao(base_id, execucao_id)
    return FileResponse(arm.caminho_resultado(base_id, execucao_id, "validacao"),
                        media_type="application/json")


@app.get("/api/bases/{base_id}/execucoes/{execucao_id}/pesos", tags=["resultados"],
         summary="pesos.json de uma execução")
def pesos_execucao(base_id: str, execucao_id: str):
    """Contrato em `motor/saida.py`. Devolvido direto do disco, sem reserializar."""
    arm.exigir_execucao(base_id, execucao_id)
    return FileResponse(arm.caminho_resultado(base_id, execucao_id, "pesos"), media_type="application/json")


@app.get("/api/bases/{base_id}/clientes", response_model=PaginaClientes, tags=["resultados"],
         summary="Fila de clientes da execução ativa (paginada, na ordem do motor)")
def clientes(base_id: str, limite: int = LIMITE, desde: int = DESDE,
             faixa: list[str] | None = FAIXA, incluir_historico: bool = HIST):
    """Atalho para a **execução ativa** (404 com mensagem clara se a base ainda não tem nenhuma pronta)."""
    arm.exigir(base_id)
    return _clientes(base_id, arm.exigir_ativa(base_id), limite, desde, faixa, incluir_historico)


@app.get("/api/bases/{base_id}/validacao", tags=["resultados"],
         summary="validacao.json da execução ativa (honestidade, dobras, métodos, coeficientes)")
def validacao(base_id: str):
    """Contrato em `motor/saida.py`. Devolvido direto do disco, sem reserializar."""
    arm.exigir(base_id)
    return FileResponse(arm.caminho_resultado(base_id, arm.exigir_ativa(base_id), "validacao"),
                        media_type="application/json")


@app.get("/api/bases/{base_id}/pesos", tags=["resultados"],
         summary="pesos.json da execução ativa (regras, candidatas, seleção e modelo escolhido)")
def pesos(base_id: str):
    """Contrato em `motor/saida.py`. Devolvido direto do disco, sem reserializar."""
    arm.exigir(base_id)
    return FileResponse(arm.caminho_resultado(base_id, arm.exigir_ativa(base_id), "pesos"),
                        media_type="application/json")
