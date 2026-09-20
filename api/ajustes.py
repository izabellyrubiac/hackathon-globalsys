"""Ajustes da API (todos com padrão razoável; sobrescreva por variável de ambiente).

API_DADOS_BASES            pasta das bases enviadas (padrão `dados/bases`)
API_MAX_UPLOAD_MB          tamanho máximo do conjunto enviado (padrão 1024 MB)
API_MAX_ARQUIVOS           número máximo de arquivos por upload (padrão 30)
API_VALIDACAO_SINCRONA_MB  acima disso, o mapeamento só passa pela conferência de esquema na hora do
                           POST /treinar; a conferência completa (dados) roda no treino, em segundo plano
API_TREINOS_SIMULTANEOS    quantos treinos rodam ao mesmo tempo (padrão 1); os demais esperam na fila
API_DEMO                   "0" desliga as bases de demonstração
API_DEMO_ARQUIVO           caminho do .xlsx da demonstração INOVAAPPS (padrão na raiz do projeto)
API_DEMO_REDES             pasta com os .csv da demonstração `redes` (padrão dados/amostra_redes/base_motor)
"""

from __future__ import annotations

import os
from pathlib import Path

from motor.leitura import EXT_CSV, EXT_EXCEL

RAIZ_PROJETO = Path(__file__).resolve().parent.parent

EXTENSOES_OK = EXT_EXCEL | EXT_CSV
PEDACO_BYTES = 1024 * 1024          # upload gravado em disco de 1 MB em 1 MB


def _num(nome: str, padrao: float) -> float:
    try:
        return float(os.environ.get(nome, padrao))
    except ValueError:
        return padrao


def raiz_bases() -> Path:
    """Pasta onde ficam as bases (lida a cada chamada: os testes trocam por uma pasta temporária)."""
    bruto = os.environ.get("API_DADOS_BASES") or (RAIZ_PROJETO / "dados" / "bases")
    return Path(bruto).expanduser().resolve()


def max_upload_bytes() -> int:
    return int(_num("API_MAX_UPLOAD_MB", 1024) * 1024 * 1024)


def max_arquivos() -> int:
    return int(_num("API_MAX_ARQUIVOS", 30))


def limite_validacao_sincrona_bytes() -> int:
    return int(_num("API_VALIDACAO_SINCRONA_MB", 200) * 1024 * 1024)


def max_treinos_simultaneos() -> int:
    return max(1, int(_num("API_TREINOS_SIMULTANEOS", 1)))


def demo_ligada() -> bool:
    return os.environ.get("API_DEMO", "1") not in ("0", "false", "False", "nao", "não")


def arquivo_demo() -> Path:
    bruto = os.environ.get("API_DEMO_ARQUIVO") or (RAIZ_PROJETO / "dados" / "desafio" / "INOVAAPPS_base_de_dados.xlsx")
    return Path(bruto).expanduser()


def pasta_demo_redes() -> Path:
    bruto = os.environ.get("API_DEMO_REDES") or (RAIZ_PROJETO / "dados" / "amostra_redes" / "base_motor")
    return Path(bruto).expanduser()


ORIGENS_PERMITIDAS = r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$"
