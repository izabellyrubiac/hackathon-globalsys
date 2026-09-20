"""Vários modelos no motor: import do LightGBM, escolha automática, modelo forçado, árvores numa base
sintética grande não linear (interação uso baixo E chamados altos) e determinismo."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from motor import Config, Mapeamento, de_dataframes, inspecionar, treinar

from .conftest import GERADO
from .sinteticas import base_nao_linear

# validação da INOVAAPPS antes das correções (alto + atenção, out-of-fold): as correções não podem piorar
ANTES = {"canc_k1": 21, "canc_k2": 19, "canc_k3": 13, "ativos_hoje": 6, "ativos_algum_mes": 28}
AUC_ANTES = {1: 0.979, 2: 0.951, 3: 0.893}


def test_import_lightgbm_e_a_biblioteca():
    """`motor/modelos_previsao/lightgbm.py` não pode esconder a biblioteca `lightgbm`."""
    import lightgbm as lib

    from motor.modelos_previsao import _arvores
    from motor.modelos_previsao import lightgbm as mod
    assert mod.lgb is lib and _arvores.lgb is lib
    assert mod.lgb is not mod and hasattr(mod.lgb, "train") and hasattr(mod.lgb, "Booster")
    assert Path(mod.lgb.__file__).parent.name == "lightgbm" and "modelos_previsao" not in mod.lgb.__file__
    assert mod.VERSAO_LIGHTGBM == lib.__version__


# --------------------------------------------------------------------------- INOVAAPPS
def test_inova_escolhe_logistica(inova):
    _, _, r = inova
    pm = r.pesos["modelo"]
    assert pm["escolhido"] == "logistica" and not pm["forcado"]
    cand = {c["nome"]: c for c in pm["candidatos"]}
    assert set(cand) == {"logistica", "random_forest", "lightgbm"}
    assert cand["logistica"]["elegivel"] and cand["logistica"]["avaliado"]
    for n in ("random_forest", "lightgbm"):
        assert not cand[n]["elegivel"] and not cand[n]["avaliado"] and "não elegível" in cand[n]["motivo_elegibilidade"]
    assert r.clientes["modelo"]["nome"] == "logistica"
    assert r.validacao["escolha"]["modelo"] == "logistica"


def test_inova_correcoes_nao_pioram(inova):
    _, _, r = inova
    met = {m["id"]: m for m in r.validacao["metodos"]}["motor_alto_ou_atencao"]
    for k in ("canc_k1", "canc_k2", "canc_k3"):
        assert met[k] >= ANTES[k], (k, met[k])
    for k in ("ativos_hoje", "ativos_algum_mes"):
        assert met[k] <= ANTES[k], (k, met[k])
    auc = {a["k"]: a["auc"] for a in r.validacao["auc_por_mes"] if a["score"] == "motor"}
    for k, v in AUC_ANTES.items():
        assert auc[k] >= v - 5e-4, (k, auc[k])
    # consistência entre dobras: as variáveis do modelo final aparecem em ≥ 50% das dobras
    sel = [v for v in r.pesos["variaveis"] if v["selecionada"]]
    assert all(v["frequencia_dobras"] >= 0.5 for v in sel)


@pytest.fixture(scope="module")
def forcado(tabelas_inova, inova):
    _, m, _ = inova
    return treinar(tabelas_inova, m, gerado_em=GERADO, config=Config(modelo="lightgbm"))


def test_inova_lightgbm_forcado(forcado):
    r = forcado
    assert r.pesos["modelo"]["escolhido"] == "lightgbm" and r.pesos["modelo"]["forcado"]
    assert any("forçado" in a for a in r.avisos)
    for obj in (r.clientes, r.validacao, r.pesos):
        json.dumps(obj, allow_nan=False)
    cl = r.clientes["clientes"]
    assert len(cl) == 58 and [x["prioridade"] for x in cl] == list(range(1, 59))
    ajustadas = [x["perda_anual_ajustada"] for x in cl]
    assert ajustadas == sorted(ajustadas, reverse=True)
    assert {x["faixa_risco"] for x in cl} <= {"alto", "atencao", "baixo"}
    assert all(v["coef"] is None and v["importancia"] is not None for v in r.clientes["modelo"]["variaveis"])
    assert r.validacao["C"] is None and r.pesos["resumo"]["calibracao"]["a"] > 0
    # contribuições aditivas: intercepto + soma = logito
    mod, F = r.modelo, r.fila
    Xq = r.X.loc[F["linha"]]
    np.testing.assert_allclose(mod.intercepto + mod.contribuicoes(Xq).sum(axis=1).to_numpy(), mod.logito(Xq), atol=1e-6)


def test_arvore_deterministica(tabelas_inova, inova):
    _, m, _ = inova
    cfg = Config(modelo="random_forest", validar=False)
    a = treinar(tabelas_inova, m, gerado_em=GERADO, config=cfg)
    b = treinar(tabelas_inova, m, gerado_em=GERADO, config=cfg)
    for x, y in ((a.clientes, b.clientes), (a.pesos, b.pesos), (a.validacao, b.validacao)):
        assert json.dumps(x, sort_keys=True) == json.dumps(y, sort_keys=True)


# --------------------------------------------------------------------------- base grande não linear
@pytest.fixture(scope="module")
def grande():
    t = de_dataframes(base_nao_linear())
    m = Mapeamento.de_sugestao(inspecionar(t))
    assert (m.tabela_clientes, m.coluna_id, m.coluna_situacao, m.coluna_data_saida) == \
        ("clientes", "codigo", "churn", "data_saida")
    return treinar(t, m, gerado_em=GERADO)


def test_grande_escolhe_arvore(grande):
    r = grande
    pm = r.pesos["modelo"]
    cand = {c["nome"]: c for c in pm["candidatos"]}
    assert all(c["elegivel"] and c["avaliado"] for c in cand.values())
    assert pm["escolhido"] in ("lightgbm", "random_forest"), pm["motivo"]
    esc = cand[pm["escolhido"]]
    assert esc["log_loss"] + esc["log_loss_ep"] < cand["logistica"]["log_loss"]
    assert any("perfil rápido" in a for a in r.avisos) and r.validacao["n_dobras"] == 5


def test_grande_acha_variaveis_plantadas(grande):
    sel = [v for v in grande.pesos["variaveis"] if v["selecionada"]]
    colunas = {v["coluna"] for v in sel}
    assert {"uso", "chamados"} <= colunas
    imp = sorted(sel, key=lambda v: -v["importancia"])
    assert {imp[0]["coluna"], imp[1]["coluna"]} <= {"uso", "chamados"}   # as duas mais importantes são as plantadas
    ruido = sum(v["importancia"] for v in sel if v["coluna"] not in ("uso", "chamados"))
    assert ruido < 0.1 * sum(v["importancia"] for v in sel)


def test_grande_monotonia_e_aditividade(grande):
    mod = grande.modelo
    X = grande.X.loc[grande.fila["linha"][:40]]
    C = mod.contribuicoes(X)
    np.testing.assert_allclose(mod.intercepto + C.sum(axis=1).to_numpy(), mod.logito(X), atol=1e-6)
    for f in mod.features:
        d = mod.direcoes[f]
        grade_v = np.linspace(np.nanpercentile(grande.X[f], 1), np.nanpercentile(grande.X[f], 99), 25)
        for i in range(0, len(X), 8):
            Xi = X.iloc[[i] * len(grade_v)].copy()
            Xi[f] = grade_v
            assert np.all(np.diff(d * mod.logito(Xi)) >= -1e-9), f        # risco nunca contra a direção
            assert np.all(np.diff(d * mod.contribuicoes(Xi)[f].to_numpy()) >= -1e-9), f   # nem a contribuição
