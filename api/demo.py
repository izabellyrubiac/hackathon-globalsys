"""Bases de demonstração, registradas na subida da API (cada uma só se os arquivos existirem).

    inovaapps   INOVAAPPS_base_de_dados.xlsx (raiz do projeto) — 80 clientes, 18 meses
    redes       amostra_redes/base_motor/*.csv — 2.910 clientes de várias redes

Aqui só se registra a base (arquivos + inspeção + mapeamento sugerido); o treino fica com
`api/preparar_demos.py` ou com o `POST /api/bases/{id}/treinar` normal. Se um arquivo faltar, a base
é ignorada sem quebrar nada. Nada aqui escreve em `web/public/data/` — aquilo é resultado versionado
do time, não da API.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from . import armazenamento as arm
from . import servico
from .ajustes import arquivo_demo, demo_ligada, limite_validacao_sincrona_bytes, pasta_demo_redes

BASE_ID = "inovaapps"          # compatibilidade: a demonstração original
REDES_ID = "redes"

# Mapeamentos conhecidos de cada demonstração (o usuário pode trocar na tela; isto é só o padrão).
MAPEAMENTOS: dict[str, dict] = {
    BASE_ID: {
        "tabela_clientes": "clientes", "coluna_id": "cliente_id",
        "alvo_tabela": "situacao_clientes", "coluna_situacao": "situacao", "valor_cancelado": "Cancelado",
        "coluna_data_saida": "mes_cancelamento",
        "valor_tabela": "clientes", "coluna_valor": "valor_mensal",
        "horizonte_meses": 3, "colunas_ignoradas": [],
    },
    REDES_ID: {
        "tabela_clientes": "clientes", "coluna_id": "cliente_id",
        "alvo_tabela": "situacao", "coluna_situacao": "situacao", "valor_cancelado": "Cancelado",
        "coluna_data_saida": "mes_saida",
        "valor_tabela": "clientes", "coluna_valor": "valor_mensal_medio",
        "horizonte_meses": 3,
        # média do gasto usa meses posteriores a cada linha do painel → só para a prioridade da fila
        "colunas_ignoradas": ["clientes.valor_mensal_medio"],
    },
}

NOMES = {BASE_ID: "INOVAAPPS (demonstração)", REDES_ID: "Redes (demonstração)"}


def _copiar(origem: Path, destino: Path) -> None:
    """Tenta um hardlink (as bases de demonstração são grandes) e cai para cópia se não der."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(origem, destino)
    except OSError:
        shutil.copyfile(origem, destino)


def registrar(base_id: str, origens: list[Path]) -> str | None:
    if arm.existe(base_id):
        return base_id
    if not origens or any(not o.exists() for o in origens):
        return None
    meta = []
    for origem in origens:
        destino = arm.pasta(base_id) / "arquivos" / arm.nome_seguro(origem.name)
        _copiar(origem, destino)
        meta.append({"nome": destino.name, "bytes": destino.stat().st_size})
    arm.gravar_meta(base_id, NOMES.get(base_id, base_id), meta)
    arm.gravar_json(arm.pasta(base_id) / "mapeamento.json", MAPEAMENTOS[base_id])
    # base grande (redes): a inspeção lê centenas de MB — fica para o primeiro acesso, não para a subida
    if arm.bytes_total(base_id) <= limite_validacao_sincrona_bytes():
        servico.inspecionar_base(base_id)
    return base_id


def arquivos_inovaapps() -> list[Path]:
    origem = arquivo_demo()
    return [origem] if origem.exists() else []


def arquivos_redes() -> list[Path]:
    pasta = pasta_demo_redes()
    return sorted(pasta.glob("*.csv")) if pasta.is_dir() else []


def preparar() -> list[str]:
    """Registra as bases de demonstração que existirem. Devolve os base_id criados/encontrados."""
    if not demo_ligada():
        return []
    out = []
    for base_id, arquivos in ((BASE_ID, arquivos_inovaapps()), (REDES_ID, arquivos_redes())):
        try:
            criada = registrar(base_id, arquivos)
        except Exception as e:  # noqa: BLE001 — uma demonstração ruim nunca impede a API de subir
            print(f"[api] base de demonstração '{base_id}' não preparada: {type(e).__name__}: {e}")
            continue
        if criada:
            out.append(criada)
    return out
