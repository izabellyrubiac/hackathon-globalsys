"""Fixtures compartilhadas: a base INOVAAPPS treinada uma vez por sessão (motor genérico, sem nada específico)."""

from __future__ import annotations

from pathlib import Path

import pytest

from motor import Mapeamento, inspecionar, ler_arquivos, treinar

RAIZ = Path(__file__).resolve().parent.parent
XLSX = RAIZ / "dados" / "desafio" / "INOVAAPPS_base_de_dados.xlsx"
GERADO = "2026-01-01"


@pytest.fixture(scope="session")
def tabelas_inova():
    return ler_arquivos([XLSX])


@pytest.fixture(scope="session")
def inova(tabelas_inova):
    """(inspeção, mapeamento sugerido, resultado) da INOVAAPPS pelo caminho genérico."""
    insp = inspecionar(tabelas_inova)
    m = Mapeamento.de_sugestao(insp)
    r = treinar(tabelas_inova, m, gerado_em=GERADO)
    return insp, m, r
