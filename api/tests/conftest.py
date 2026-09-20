"""Ambiente dos testes da API: as bases vão para uma pasta temporária e as demonstrações ficam
desligadas (dois testes religam). Nada aqui toca em `dados/` de verdade nem na suíte do motor
(`tests/`).

    uv run pytest api/tests -q
"""

from __future__ import annotations

import os
import time

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="session", autouse=True)
def ambiente(tmp_path_factory):
    os.environ["API_DADOS_BASES"] = str(tmp_path_factory.mktemp("bases"))
    os.environ["API_DEMO"] = "0"
    yield


@pytest.fixture(scope="session")
def cliente(ambiente):
    from api.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture
def base(cliente):
    """Uma base sintética já enviada e inspecionada. Devolve o base_id."""
    from api.tests import sintetica

    r = cliente.post("/api/bases", files=sintetica.arquivos())
    assert r.status_code == 201, r.text
    base_id = r.json()["base_id"]
    yield base_id
    cliente.delete(f"/api/bases/{base_id}")


ATIVAS = ("na_fila", "treinando")


def esperar(cliente, base_id: str, limite: float = 300.0) -> dict:
    """Acompanha GET /status até nenhuma execução da base estar na fila ou treinando."""
    fim = time.monotonic() + limite
    st = cliente.get(f"/api/bases/{base_id}/status").json()
    while st.get("estado") in ATIVAS and time.monotonic() < fim:
        time.sleep(0.05)
        st = cliente.get(f"/api/bases/{base_id}/status").json()
    return st


def esperar_execucao(cliente, base_id: str, execucao_id: str, limite: float = 300.0) -> dict:
    """Acompanha uma execução até sair de 'na_fila'/'treinando'."""
    fim = time.monotonic() + limite
    rota = f"/api/bases/{base_id}/execucoes/{execucao_id}"
    e = cliente.get(rota).json()
    while e.get("estado") in ATIVAS and time.monotonic() < fim:
        time.sleep(0.05)
        e = cliente.get(rota).json()
    return e
