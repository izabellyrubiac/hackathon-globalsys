"""Escala: persistência vetorizada, perfil automático e uma base larga (300 clientes × 400 colunas de ruído
+ 3 de sinal) que precisa achar o sinal em tempo limitado."""

from __future__ import annotations

import time

import numpy as np
import pytest

from motor import Config, Mapeamento, de_dataframes, inspecionar, treinar
from motor.selecao import meses_seguidos

from .conftest import GERADO
from .sinteticas import base_larga

LIMITE_S = 300


def _meses_seguidos_laco(valor, cliente, ordem, limiar, direcao):
    v, c = valor[ordem], cliente[ordem]
    lado = np.where(np.isnan(v), False, (v >= limiar) if direcao > 0 else (v <= limiar))
    out, run = np.zeros(len(v)), 0
    for i in range(len(v)):
        if i > 0 and c[i] != c[i - 1]:
            run = 0
        run = run + 1 if lado[i] else 0
        out[i] = run
    res = np.empty(len(v))
    res[ordem] = out
    return res


def test_meses_seguidos_vetorizado_igual_ao_laco():
    rng = np.random.default_rng(3)
    n = 5000
    cliente = rng.integers(0, 200, n).astype(str)
    mes = rng.permutation(n)
    valor = rng.normal(0, 1, n)
    valor[rng.random(n) < 0.1] = np.nan
    ordem = np.lexsort((mes, cliente))
    for lim, d in ((0.3, 1), (-0.2, -1)):
        np.testing.assert_array_equal(meses_seguidos(valor, cliente, ordem, lim, d),
                                      _meses_seguidos_laco(valor, cliente, ordem, lim, d))


def test_perfil_automatico():
    cfg, aviso = Config().perfil_automatico(80, 1300)
    assert aviso is None and cfg == Config()
    cfg, aviso = Config().perfil_automatico(3000, 60_000)
    assert aviso and cfg.dobras == 5 and cfg.rodadas == 20 and cfg.razao_ativos_rodada == 4.0
    cfg, aviso = Config(perfil="fixo").perfil_automatico(3000, 300_000)
    assert aviso is None and cfg.dobras == 10


@pytest.mark.lento
def test_base_larga_acha_sinal():
    t0 = time.time()
    t = de_dataframes(base_larga())
    m = Mapeamento.de_sugestao(inspecionar(t))
    r = treinar(t, m, gerado_em=GERADO)
    dt = time.time() - t0
    print(f"\nbase larga: {dt:.0f}s · modelo {r.pesos['modelo']['escolhido']} · {r.pesos['modelo']['motivo']}")
    assert dt < LIMITE_S, f"{dt:.0f}s"
    assert r.pesos["resumo"]["n_candidatas"] > 1500
    assert any("pré-filtro de escala" in str(v["motivo_descarte"]) for v in r.pesos["variaveis"])
    sel = [v for v in r.pesos["variaveis"] if v["selecionada"]]
    colunas = {v["coluna"] for v in sel}
    assert {"sinal_uso", "sinal_chamados", "sinal_nota"} <= colunas
    ruido = [v for v in sel if not v["coluna"].startswith("sinal_")]
    assert len(ruido) <= 1, [v["id"] for v in ruido]
    auc = {a["k"]: a["auc"] for a in r.validacao["auc_por_mes"] if a["score"] == "motor"}
    assert auc[1] > 0.85
