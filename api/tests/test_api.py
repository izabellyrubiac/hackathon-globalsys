"""Testes da API (TestClient). Verificação é executar: nada de navegador.

    uv run pytest api/tests -q
"""

from __future__ import annotations

import pytest

from api.tests import sintetica
from api.tests.conftest import esperar, esperar_execucao

RAPIDO = {"validar": False, "semente": 0}       # sem validação cruzada: o treino leva poucos segundos


def treinar(cliente, base_id, mapeamento=None, rotulo=None, **opcoes):
    corpo = {"mapeamento": mapeamento or sintetica.MAPEAMENTO, "opcoes": {**RAPIDO, **opcoes}}
    if rotulo is not None:
        corpo["rotulo"] = rotulo
    return cliente.post(f"/api/bases/{base_id}/treinar", json=corpo)


# --------------------------------------------------------------------------- saúde e upload
def test_saude(cliente):
    r = cliente.get("/api/saude")
    assert r.status_code == 200
    assert r.json()["ok"] is True and r.json()["versao"]


def test_docs_abrem(cliente):
    assert cliente.get("/openapi.json").status_code == 200
    assert cliente.get("/docs").status_code == 200


def test_upload_inspeciona_e_sugere(cliente, base):
    r = cliente.get(f"/api/bases/{base}")
    assert r.status_code == 200
    insp = r.json()["inspecao"]
    tabelas = {t["nome"]: t for t in insp["tabelas"]}
    assert {"clientes", "situacao", "uso"} <= set(tabelas)
    assert tabelas["uso"]["papel_sugerido"] == "temporal"
    assert tabelas["clientes"]["linhas"] == 40
    colunas = {c["nome"]: c for c in tabelas["uso"]["colunas"]}
    assert colunas["uso"]["tipo"] == "numerica" and colunas["uso"]["exemplos"]

    s = insp["sugestao"]
    assert s["tabela_clientes"] == "clientes" and s["coluna_id"] == "codigo"
    assert s["alvo"]["tabela"] == "situacao" and s["alvo"]["valor_cancelado"] == "Cancelado"
    assert s["coluna_valor"]["coluna"] == "mrr"
    assert r.json()["status"]["estado"] == "inspecionada"


def test_upload_recusa_extensao_desconhecida(cliente):
    r = cliente.post("/api/bases", files=[("arquivos", ("malicioso.exe", b"MZ", "application/octet-stream"))])
    assert r.status_code == 400
    assert "xlsx" in r.json()["detalhe"]


def test_upload_recusa_xlsx_junto_com_csv(cliente):
    r = cliente.post("/api/bases", files=[("arquivos", ("a.xlsx", b"x", "application/vnd.ms-excel")),
                                          ("arquivos", ("b.csv", b"a,b\n1,2\n", "text/csv"))])
    assert r.status_code == 400


def test_upload_recusa_arquivo_vazio(cliente):
    r = cliente.post("/api/bases", files=[("arquivos", ("vazio.csv", b"", "text/csv"))])
    assert r.status_code == 400


def test_upload_recusa_arquivo_grande_demais(cliente, monkeypatch):
    monkeypatch.setenv("API_MAX_UPLOAD_MB", "0.001")
    r = cliente.post("/api/bases", files=[("arquivos", ("grande.csv", b"a,b\n" + b"1,2\n" * 5000, "text/csv"))])
    assert r.status_code == 413
    assert "limite" in r.json()["detalhe"]


def test_upload_neutraliza_path_traversal(cliente):
    dados = sintetica.csvs()
    arquivos = [("arquivos", ("../../../etc/clientes.csv", dados["clientes.csv"], "text/csv")),
                ("arquivos", ("situacao.csv", dados["situacao.csv"], "text/csv")),
                ("arquivos", ("uso.csv", dados["uso.csv"], "text/csv"))]
    r = cliente.post("/api/bases", files=arquivos)
    assert r.status_code == 201, r.text
    corpo = r.json()
    assert corpo["arquivos"] == ["clientes.csv", "situacao.csv", "uso.csv"]

    from api import armazenamento as arm

    pasta = arm.pasta(corpo["base_id"]) / "arquivos"
    assert {p.name for p in pasta.iterdir()} == {"clientes.csv", "situacao.csv", "uso.csv"}
    assert pasta.resolve().is_relative_to(arm.raiz_bases())
    cliente.delete(f"/api/bases/{corpo['base_id']}")


def test_base_inexistente_404(cliente):
    assert cliente.get("/api/bases/nao-existe/status").status_code == 404
    assert cliente.get("/api/bases/..%2F..%2Fetc/status").status_code == 404


def test_listar_bases(cliente, base):
    r = cliente.get("/api/bases")
    assert r.status_code == 200
    linha = next(b for b in r.json() if b["base_id"] == base)
    assert linha["estado"] == "inspecionada" and linha["n_clientes"] is None
    assert linha["arquivos"] == ["clientes.csv", "situacao.csv", "uso.csv"]


# --------------------------------------------------------------------------- mapeamento inválido
def test_mapeamento_coluna_inexistente_422(cliente, base):
    m = {**sintetica.MAPEAMENTO, "coluna_id": "nao_existe"}
    r = treinar(cliente, base, m)
    assert r.status_code == 422
    assert any("nao_existe" in p for p in r.json()["problemas"])


def test_mapeamento_tabela_inexistente_422(cliente, base):
    m = {**sintetica.MAPEAMENTO, "alvo_tabela": "fantasma"}
    r = treinar(cliente, base, m)
    assert r.status_code == 422
    assert any("fantasma" in p for p in r.json()["problemas"])


def test_mapeamento_valor_cancelado_errado_422(cliente, base):
    m = {**sintetica.MAPEAMENTO, "valor_cancelado": "Churn"}
    r = treinar(cliente, base, m)
    assert r.status_code == 422
    assert any("Churn" in p for p in r.json()["problemas"])


def test_mapeamento_sem_cancelamento_422(cliente, base):
    m = {**sintetica.MAPEAMENTO, "coluna_situacao": None, "coluna_data_saida": None}
    r = treinar(cliente, base, m)
    assert r.status_code == 422
    assert any("cancelamento" in p or "situação" in p for p in r.json()["problemas"])


def test_horizonte_fora_da_faixa_422(cliente, base):
    r = treinar(cliente, base, {**sintetica.MAPEAMENTO, "horizonte_meses": 99})
    assert r.status_code == 422


def test_opcao_avancada_desconhecida_422(cliente, base):
    r = treinar(cliente, base, avancado={"nao_existe_no_motor": 1})
    assert r.status_code == 422
    assert any("nao_existe_no_motor" in p for p in r.json()["problemas"])


def test_modelo_invalido_422(cliente, base):
    r = treinar(cliente, base, modelo="chute")
    assert r.status_code == 422


# --------------------------------------------------------------------------- resultados antes da hora
def test_resultados_404_antes_de_pronta(cliente, base):
    for rota in ("clientes", "validacao", "pesos"):
        r = cliente.get(f"/api/bases/{base}/{rota}")
        assert r.status_code == 404
        assert "execução pronta" in r.json()["detalhe"]


def test_execucoes_vazias_e_404_coerentes(cliente, base):
    lista = cliente.get(f"/api/bases/{base}/execucoes")
    assert lista.status_code == 200
    assert lista.json() == {"base_id": base, "execucao_ativa": None, "execucoes": []}
    assert cliente.get(f"/api/bases/{base}/status").json()["estado"] == "inspecionada"

    r = cliente.get(f"/api/bases/{base}/execucoes/nao-existe")
    assert r.status_code == 404 and "não encontrada" in r.json()["detalhe"]
    assert cliente.get(f"/api/bases/{base}/execucoes/..%2F..%2Fetc").status_code == 404
    assert cliente.post(f"/api/bases/{base}/execucoes/nao-existe/ativar").status_code == 404
    assert cliente.delete(f"/api/bases/{base}/execucoes/nao-existe").status_code == 404
    assert cliente.get(f"/api/bases/{base}/execucoes/nao-existe/clientes").status_code == 404
    assert cliente.get("/api/bases/nao-existe/execucoes").status_code == 404


def test_apagar_base_com_treino_em_andamento_409(cliente, base):
    """O 409 sumiu do POST /treinar (agora há fila), mas continua no DELETE da base."""
    assert treinar(cliente, base).status_code == 202
    assert treinar(cliente, base).status_code == 202       # segunda seguida: entra na fila, não dá 409
    assert cliente.delete(f"/api/bases/{base}").status_code == 409
    esperar(cliente, base)


# --------------------------------------------------------------------------- treino de ponta a ponta
@pytest.fixture(scope="module")
def treinada(cliente):
    """Uma base sintética treinada de verdade (sem validação cruzada, para ser rápida)."""
    r = cliente.post("/api/bases", files=sintetica.arquivos())
    base_id = r.json()["base_id"]
    assert treinar(cliente, base_id, rotulo="primeira").status_code == 202
    st = esperar(cliente, base_id)
    assert st["estado"] == "pronta", st
    yield base_id
    cliente.delete(f"/api/bases/{base_id}")


def test_status_de_ponta_a_ponta(cliente, treinada):
    st = cliente.get(f"/api/bases/{treinada}/status").json()
    assert st["estado"] == "pronta" and st["fracao"] == 1.0
    assert st["etapa"] == "concluído" and st["n_clientes"] and st["segundos"] >= 0
    assert st["opcoes"]["modelo"] == "auto" and st["opcoes"]["delta_risco"] is False
    assert st["execucao_id"] and st["execucao_ativa"] == st["execucao_id"] and st["n_execucoes"] == 1
    assert cliente.get("/api/bases").json()
    linha = next(b for b in cliente.get("/api/bases").json() if b["base_id"] == treinada)
    assert linha["estado"] == "pronta" and linha["n_clientes"] == st["n_clientes"]
    assert linha["n_execucoes"] == 1 and linha["execucao_ativa"] == st["execucao_id"]


def test_primeira_execucao_vira_ativa_e_serve_as_rotas_de_compatibilidade(cliente, treinada):
    lista = cliente.get(f"/api/bases/{treinada}/execucoes").json()
    assert len(lista["execucoes"]) == 1
    e = lista["execucoes"][0]
    assert e["ativa"] is True and e["rotulo"] == "primeira" and e["estado"] == "pronta"
    assert lista["execucao_ativa"] == e["execucao_id"]

    ativa = cliente.get(f"/api/bases/{treinada}/clientes?limite=3").json()
    por_execucao = cliente.get(
        f"/api/bases/{treinada}/execucoes/{e['execucao_id']}/clientes?limite=3").json()
    assert ativa == por_execucao and ativa["execucao_id"] == e["execucao_id"]


def test_clientes_paginado_e_em_ordem(cliente, treinada):
    tudo = cliente.get(f"/api/bases/{treinada}/clientes?limite=5000").json()
    assert tudo["total"] == tudo["total_filtrado"] == len(tudo["clientes"]) == 28
    assert tudo["modelo"]["horizonte_meses"] == 3 and tudo["resumo"]["clientes_ativos"] == 28
    prioridades = [c["prioridade"] for c in tudo["clientes"]]
    assert prioridades == sorted(prioridades) == list(range(1, 29))

    pagina = cliente.get(f"/api/bases/{treinada}/clientes?limite=5&desde=5").json()
    assert [c["cliente_id"] for c in pagina["clientes"]] == [c["cliente_id"] for c in tudo["clientes"][5:10]]
    assert pagina["desde"] == 5 and pagina["total"] == 28

    sem_hist = cliente.get(f"/api/bases/{treinada}/clientes?limite=2&incluir_historico=false").json()
    assert all("historico" not in c for c in sem_hist["clientes"])
    assert "historico" in tudo["clientes"][0]


def test_evidencias_trazem_comparacao_e_serie(cliente, treinada):
    """Cada evidência aponta a série mensal e traz "últimos 3 meses × 6 anteriores", mesmo sem o histórico."""
    doc = cliente.get(f"/api/bases/{treinada}/clientes?limite=5000&incluir_historico=false").json()
    chaves = {s["chave"] for s in doc["modelo"]["series_historico"]}
    itens = [e for c in doc["clientes"] for e in c["evidencias"] + c["fatores_secundarios"]]
    assert itens and all("chave_serie" in e and "comparacao" in e for e in itens)
    comparados = [e for e in itens if e["comparacao"]]
    assert comparados
    for e in comparados:
        cmp = e["comparacao"]
        assert cmp["chave"] == e["chave_serie"] in chaves
        assert 1 <= cmp["meses_recente"] <= 3 and 1 <= cmp["meses_anterior"] <= 6
        assert cmp["recente"] is not None and cmp["anterior"] is not None


def test_clientes_filtro_por_faixa_nao_reordena(cliente, treinada):
    tudo = cliente.get(f"/api/bases/{treinada}/clientes?limite=5000").json()["clientes"]
    r = cliente.get(f"/api/bases/{treinada}/clientes?limite=5000&faixa=alto&faixa=atencao").json()
    esperado = [c["cliente_id"] for c in tudo if c["faixa_risco"] in ("alto", "atencao")]
    assert [c["cliente_id"] for c in r["clientes"]] == esperado
    assert r["total"] == 28 and r["total_filtrado"] == len(esperado)
    assert cliente.get(f"/api/bases/{treinada}/clientes?faixa=inventada").status_code == 400


def test_validacao_e_pesos(cliente, treinada):
    v = cliente.get(f"/api/bases/{treinada}/validacao").json()
    assert v["horizonte_meses"] == 3 and v["painel"]["cancelados"] == 12
    assert v["config"]["validar"] is False and v["config"]["delta_risco"] is False

    p = cliente.get(f"/api/bases/{treinada}/pesos").json()
    assert p["variaveis"] and p["resumo"]["n_selecionadas"] >= 1
    assert p["modelo"]["escolhido"] in ("logistica", "random_forest", "lightgbm")
    assert any(v["selecionada"] for v in p["variaveis"])


def test_detalhe_guarda_mapeamento_e_opcoes(cliente, treinada):
    d = cliente.get(f"/api/bases/{treinada}").json()
    assert d["mapeamento"]["coluna_id"] == "codigo" and d["mapeamento"]["valor_cancelado"] == "Cancelado"
    assert d["opcoes"]["validar"] is False and d["opcoes"]["modelo"] == "auto"


# --------------------------------------------------------------------------- fila de treinos
def test_fila_aceita_treinos_seguidos_sem_409(cliente, base, monkeypatch):
    """O 409 de "já há treino" sumiu do POST /treinar: os pedidos extras ficam `na_fila`."""
    from api import servico

    monkeypatch.setattr(servico, "max_treinos_simultaneos", lambda: 0)   # ninguém começa: fila congelada
    pedidos = [treinar(cliente, base, rotulo=r) for r in ("um", "dois", "tres")]
    assert [p.status_code for p in pedidos] == [202, 202, 202]
    ids = [p.json()["execucao_id"] for p in pedidos]
    assert len(set(ids)) == 3
    assert all(p.json()["execucao"]["estado"] == "na_fila" for p in pedidos)
    assert all(p.json()["status"]["estado"] == "na_fila" for p in pedidos)

    lista = cliente.get(f"/api/bases/{base}/execucoes").json()
    assert {e["execucao_id"] for e in lista["execucoes"]} == set(ids)
    assert all(e["estado"] == "na_fila" and e["resumo"] is None for e in lista["execucoes"])
    assert lista["execucao_ativa"] is None

    r = cliente.delete(f"/api/bases/{base}/execucoes/{ids[0]}")
    assert r.status_code == 409 and "na fila" in r.json()["detalhe"]
    assert cliente.delete(f"/api/bases/{base}").status_code == 409

    monkeypatch.undo()
    servico._bombear()
    assert esperar(cliente, base)["estado"] == "pronta"
    lista = cliente.get(f"/api/bases/{base}/execucoes").json()["execucoes"]
    assert [e["estado"] for e in lista] == ["pronta"] * 3
    assert sum(1 for e in lista if e["ativa"]) == 1
    assert all(e["resumo"]["n_clientes"] == 28 for e in lista)


# --------------------------------------------------------------------------- versões do modelo
@pytest.fixture(scope="module")
def versionada(cliente):
    """Uma base com três execuções prontas (modelos/horizontes diferentes)."""
    base_id = cliente.post("/api/bases", files=sintetica.arquivos()).json()["base_id"]
    pedidos = [("logistica h3", {"modelo": "logistica"}),
               ("logistica h2", {"modelo": "logistica", "horizonte_meses": 2}),
               ("min_hist 5", {"modelo": "logistica", "min_hist": 5})]
    ids = []
    for rotulo, opcoes in pedidos:
        r = treinar(cliente, base_id, rotulo=rotulo, **opcoes)
        assert r.status_code == 202, r.text
        ids.append(r.json()["execucao_id"])
    for eid in ids:
        assert esperar_execucao(cliente, base_id, eid)["estado"] == "pronta"
    yield base_id, ids
    cliente.delete(f"/api/bases/{base_id}")


def test_varias_execucoes_na_mesma_base(cliente, versionada):
    base_id, ids = versionada
    lista = cliente.get(f"/api/bases/{base_id}/execucoes").json()
    assert [e["execucao_id"] for e in lista["execucoes"]] == sorted(ids, reverse=True)   # recente primeiro
    assert {e["rotulo"] for e in lista["execucoes"]} == {"logistica h3", "logistica h2", "min_hist 5"}
    assert lista["execucao_ativa"] == ids[0]                                            # a primeira pronta
    assert sum(1 for e in lista["execucoes"] if e["ativa"]) == 1
    assert cliente.get(f"/api/bases/{base_id}").json()["execucao_ativa"] == ids[0]

    # cada execução guarda o seu resultado, separado das outras
    h = {eid: cliente.get(f"/api/bases/{base_id}/execucoes/{eid}/clientes?limite=1").json()["modelo"]
         ["horizonte_meses"] for eid in ids}
    assert h[ids[0]] == 3 and h[ids[1]] == 2


def test_resumo_traz_os_campos_da_comparacao(cliente, versionada):
    base_id, ids = versionada
    r = cliente.get(f"/api/bases/{base_id}/execucoes/{ids[0]}").json()
    assert r["estado"] == "pronta" and r["ativa"] is True
    s = r["resumo"]
    assert s["execucao_id"] == ids[0] and s["rotulo"] == "logistica h3"
    assert s["modelo"]["nome"] == "logistica" and s["modelo"]["forcado"] is True
    assert set(s["auc"]) == {"k1", "k2", "k3"}
    assert set(s["deteccao"]["alto"]) == {"k1", "k2", "k3"}
    assert set(s["alarme_falso"]["alto"]) == {"hoje", "doze_meses", "pct_meses"}
    assert s["faixas"]["alto"] + s["faixas"]["atencao"] + s["faixas"]["baixo"] == s["n_clientes"] == 28
    assert s["perda_anual_esperada_total"] > 0 and s["receita_em_risco_mensal"] is not None
    assert s["horizonte_meses"] == 3 and s["cortes"]["alto"] >= s["cortes"]["atencao"]
    assert s["opcoes"]["delta_risco"] is False and s["opcoes"]["validar"] is False
    assert s["duracao_s"] >= 0 and s["variaveis"]["n_selecionadas"] >= 1
    assert 1 <= len(s["variaveis"]["principais"]) <= 5
    assert all({"id", "rotulo", "peso"} <= set(v) for v in s["variaveis"]["principais"])
    assert s["painel"]["cancelados"] == 12


def test_resumo_com_validacao_ligada(cliente):
    """Com validação cruzada, o resumo traz AUC, detecção e alarme falso de verdade."""
    base_id = cliente.post("/api/bases", files=sintetica.arquivos()).json()["base_id"]
    try:
        r = cliente.post(f"/api/bases/{base_id}/treinar",
                         json={"mapeamento": sintetica.MAPEAMENTO, "rotulo": "validada",
                               "opcoes": {"validar": True, "semente": 0, "dobras": 4}})
        assert r.status_code == 202, r.text
        e = esperar_execucao(cliente, base_id, r.json()["execucao_id"])
        assert e["estado"] == "pronta", e
        s = e["resumo"]
        assert s["validacao_ligada"] is True
        assert all(0.0 <= s["auc"][k] <= 1.0 for k in ("k1", "k2", "k3"))
        assert s["modelo"]["log_loss"] > 0 and s["modelo"]["log_loss_ep"] >= 0
        assert [c["nome"] for c in s["candidatos"]] == ["logistica", "random_forest", "lightgbm"]
        assert s["deteccao"]["n_cancelados"] == 12
        assert all(0 <= s["deteccao"]["alto"][k] <= 12 for k in ("k1", "k2", "k3"))
        assert s["alarme_falso"]["n_ativos"] == 28
        assert s["alarme_falso"]["alto"]["hoje"] <= 28 and s["alarme_falso"]["alto"]["doze_meses"] <= 28
        assert s["alarme_falso"]["alto"]["pct_meses"] is not None
    finally:
        cliente.delete(f"/api/bases/{base_id}")


def test_ativar_troca_a_execucao_servida(cliente, versionada):
    base_id, ids = versionada
    try:
        r = cliente.post(f"/api/bases/{base_id}/execucoes/{ids[1]}/ativar")
        assert r.status_code == 200 and r.json()["ativa"] is True
        assert cliente.get(f"/api/bases/{base_id}/execucoes").json()["execucao_ativa"] == ids[1]
        ativa = cliente.get(f"/api/bases/{base_id}/clientes?limite=1").json()
        assert ativa["execucao_id"] == ids[1] and ativa["modelo"]["horizonte_meses"] == 2
        assert cliente.get(f"/api/bases/{base_id}/validacao").json()["horizonte_meses"] == 2
        assert cliente.get(f"/api/bases/{base_id}/pesos").json()["horizonte_meses"] == 2
    finally:
        cliente.post(f"/api/bases/{base_id}/execucoes/{ids[0]}/ativar")


def test_comparar_execucoes(cliente, versionada):
    base_id, ids = versionada
    r = cliente.get(f"/api/bases/{base_id}/comparar", params={"execucoes": [ids[0], ids[1]]})
    assert r.status_code == 200, r.text
    c = r.json()
    assert [e["execucao_id"] for e in c["execucoes"]] == [ids[0], ids[1]]
    assert c["execucoes"][0]["ativa"] is True and c["execucoes"][1]["ativa"] is False
    campos = {l["campo"]: l["valores"] for l in c["linhas"]}
    assert all(len(v) == 2 for v in campos.values())
    assert all(m and "logística" in m for m in campos["modelo"])
    assert campos["faixa_alto"][0] is not None and campos["delta_risco"] == [False, False]
    assert len(c["resumos"]) == 2

    todas = cliente.get(f"/api/bases/{base_id}/comparar").json()          # sem filtro: todas as prontas
    assert len(todas["execucoes"]) == 3


def test_comparar_precisa_de_duas_prontas(cliente, versionada, base):
    base_id, ids = versionada
    r = cliente.get(f"/api/bases/{base_id}/comparar", params={"execucoes": [ids[0]]})
    assert r.status_code == 400 and "pelo menos duas" in r.json()["detalhe"]
    assert cliente.get(f"/api/bases/{base}/comparar").status_code == 400
    assert cliente.get(f"/api/bases/{base_id}/comparar",
                       params={"execucoes": [ids[0], "nao-existe"]}).status_code == 404


def test_apagar_execucao_recusa_a_ativa_e_aceita_as_outras(cliente):
    base_id = cliente.post("/api/bases", files=sintetica.arquivos()).json()["base_id"]
    try:
        ids = [treinar(cliente, base_id, rotulo=r).json()["execucao_id"] for r in ("a", "b")]
        for eid in ids:
            assert esperar_execucao(cliente, base_id, eid)["estado"] == "pronta"
        ativa = cliente.get(f"/api/bases/{base_id}/execucoes").json()["execucao_ativa"]
        outra = next(e for e in ids if e != ativa)

        r = cliente.delete(f"/api/bases/{base_id}/execucoes/{ativa}")
        assert r.status_code == 409 and "ativa" in r.json()["detalhe"]
        assert cliente.delete(f"/api/bases/{base_id}/execucoes/{outra}").status_code == 204
        assert cliente.get(f"/api/bases/{base_id}/execucoes/{outra}").status_code == 404
        restantes = cliente.get(f"/api/bases/{base_id}/execucoes").json()["execucoes"]
        assert [e["execucao_id"] for e in restantes] == [ativa]
        assert cliente.get(f"/api/bases/{base_id}/clientes?limite=1").status_code == 200
    finally:
        cliente.delete(f"/api/bases/{base_id}")


def test_execucao_com_erro_nao_vira_ativa(cliente, base):
    """Mapeamento que passa na conferência de esquema mas quebra no motor."""
    m = {**sintetica.MAPEAMENTO, "coluna_valor": "regiao"}                # valor de contrato não numérico
    r = treinar(cliente, base, m, modelo="logistica")
    if r.status_code == 202:
        eid = r.json()["execucao_id"]
        e = esperar_execucao(cliente, base, eid)
        assert e["estado"] in ("pronta", "erro")
        if e["estado"] == "erro":
            assert e["mensagem"] and e["resumo"] is None
            assert cliente.get(f"/api/bases/{base}/execucoes/{eid}/clientes").status_code == 404
            assert cliente.delete(f"/api/bases/{base}/execucoes/{eid}").status_code == 204
    else:
        assert r.status_code == 422


def test_migracao_do_layout_antigo(cliente, tmp_path):
    """Base gravada no layout velho (resultados na raiz) vira a execução `legado`, já ativa."""
    import shutil

    from api import armazenamento as arm

    origem = cliente.post("/api/bases", files=sintetica.arquivos()).json()["base_id"]
    assert treinar(cliente, origem).status_code == 202
    assert esperar(cliente, origem)["estado"] == "pronta"
    eid = cliente.get(f"/api/bases/{origem}/execucoes").json()["execucao_ativa"]

    antiga = arm.novo_base_id("antiga")
    p = arm.pasta(antiga)
    shutil.copytree(arm.pasta(origem) / "arquivos", p / "arquivos")
    for nome in ("base.json", "inspecao.json", "mapeamento.json", "config.json"):
        shutil.copyfile(arm.pasta(origem) / nome, p / nome)
    arm.gravar_json(p / "base.json", {**arm.ler_json(p / "base.json"), "base_id": antiga})
    for nome in arm.RESULTADOS:
        shutil.copyfile(arm.pasta_execucao(origem, eid) / f"{nome}.json", p / f"{nome}.json")
    arm.gravar_json(p / "status.json", {"base_id": antiga, "estado": "pronta", "etapa": "concluído",
                                        "segundos": 12.5, "avisos": [], "opcoes": {"modelo": "auto"}})
    cliente.delete(f"/api/bases/{origem}")

    try:
        lista = cliente.get(f"/api/bases/{antiga}/execucoes").json()
        assert lista["execucao_ativa"] == "legado"
        assert [e["execucao_id"] for e in lista["execucoes"]] == ["legado"]
        legado = lista["execucoes"][0]
        assert legado["estado"] == "pronta" and legado["ativa"] is True
        assert legado["resumo"]["n_clientes"] == 28 and legado["resumo"]["modelo"]["nome"]
        assert cliente.get(f"/api/bases/{antiga}/clientes?limite=1").json()["execucao_id"] == "legado"
        assert not (p / "clientes.json").exists() and not (p / "status.json").exists()
        assert cliente.get(f"/api/bases/{antiga}/status").json()["estado"] == "pronta"
    finally:
        cliente.delete(f"/api/bases/{antiga}")


# --------------------------------------------------------------------------- delta de risco
def test_treino_com_delta_risco(cliente):
    r = cliente.post("/api/bases", files=sintetica.arquivos())
    base_id = r.json()["base_id"]
    try:
        aceito = treinar(cliente, base_id, delta_risco=True)
        assert aceito.status_code == 202
        assert aceito.json()["status"]["opcoes"]["delta_risco"] is True
        st = esperar(cliente, base_id)
        assert st["estado"] == "pronta", st
        assert st["opcoes"]["delta_risco"] is True

        p = cliente.get(f"/api/bases/{base_id}/pesos").json()
        ids = [v["id"] for v in p["variaveis"]]
        assert any(i.startswith("risco_previsto__delta") for i in ids), ids
        v = cliente.get(f"/api/bases/{base_id}/validacao").json()
        assert v["config"]["delta_risco"] is True
    finally:
        cliente.delete(f"/api/bases/{base_id}")


def test_delta_risco_desligado_por_padrao_e_documentado(cliente):
    campo = cliente.get("/openapi.json").json()["components"]["schemas"]["Opcoes"]["properties"]["delta_risco"]
    assert campo["default"] is False
    assert "não melhora o acerto" in campo["description"] and "3,2×" in campo["description"]
    rota = cliente.get("/openapi.json").json()["paths"]["/api/bases/{base_id}/treinar"]["post"]
    assert "não melhora o acerto" in rota["description"]


def test_opcoes_avancadas_chegam_no_motor(cliente):
    r = cliente.post("/api/bases", files=sintetica.arquivos())
    base_id = r.json()["base_id"]
    try:
        aceito = treinar(cliente, base_id, modelo="logistica", horizonte_meses=2, min_hist=5,
                         avancado={"n_persistencia": 3})
        assert aceito.status_code == 202
        st = esperar(cliente, base_id)
        assert st["estado"] == "pronta", st
        v = cliente.get(f"/api/bases/{base_id}/validacao").json()
        assert v["horizonte_meses"] == 2
        assert v["config"]["modelo"] == "logistica" and v["config"]["min_hist"] == 5
        assert v["config"]["n_persistencia"] == 3
        assert v["escolha"]["forcado"] is True and v["escolha"]["modelo"] == "logistica"
    finally:
        cliente.delete(f"/api/bases/{base_id}")


# --------------------------------------------------------------------------- apagar
def test_apagar_base(cliente):
    r = cliente.post("/api/bases", files=sintetica.arquivos())
    base_id = r.json()["base_id"]
    assert cliente.delete(f"/api/bases/{base_id}").status_code == 204
    assert cliente.get(f"/api/bases/{base_id}").status_code == 404
    assert base_id not in [b["base_id"] for b in cliente.get("/api/bases").json()]
    assert cliente.delete(f"/api/bases/{base_id}").status_code == 404


# --------------------------------------------------------------------------- bases de demonstração
def test_base_de_demonstracao(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from api.ajustes import arquivo_demo

    if not arquivo_demo().exists():
        pytest.skip("INOVAAPPS_base_de_dados.xlsx não está na raiz do projeto.")
    monkeypatch.setenv("API_DADOS_BASES", str(tmp_path / "bases"))
    monkeypatch.setenv("API_DEMO", "1")
    monkeypatch.setenv("API_DEMO_REDES", str(tmp_path / "sem-redes"))     # a de redes fica de fora
    from api.main import app

    with TestClient(app) as c:
        d = c.get("/api/bases/inovaapps")
        assert d.status_code == 200, d.text
        corpo = d.json()
        assert corpo["demonstracao"] is True and corpo["status"]["estado"] == "inspecionada"
        assert corpo["execucoes"] == [] and corpo["execucao_ativa"] is None
        tabelas = {t["nome"] for t in corpo["inspecao"]["tabelas"]}
        assert {"clientes", "atendimento_mensal", "pesquisas_nps", "situacao_clientes"} <= tabelas
        s = corpo["inspecao"]["sugestao"]
        assert s["tabela_clientes"] == "clientes" and s["coluna_id"] == "cliente_id"
        assert s["alvo"]["tabela"] == "situacao_clientes" and s["alvo"]["valor_cancelado"] == "Cancelado"
        assert s["coluna_valor"]["coluna"] == "valor_mensal"
        assert any(b["base_id"] == "inovaapps" and b["demonstracao"] for b in c.get("/api/bases").json())
        # o mapeamento conhecido já fica salvo, pronto para o POST /treinar do front
        assert corpo["mapeamento"]["coluna_data_saida"] == "mes_cancelamento"
        assert c.get("/api/bases/redes").status_code == 404


def test_demonstracao_redes_registrada_quando_os_csv_existem(tmp_path, monkeypatch):
    """A base `redes` só é registrada se a pasta de CSV existir — aqui, uma cópia minúscula."""
    from fastapi.testclient import TestClient

    from api import demo

    pasta = tmp_path / "redes"
    pasta.mkdir()
    dados = sintetica.csvs()
    (pasta / "clientes.csv").write_bytes(dados["clientes.csv"].replace(b"codigo,", b"cliente_id,")
                                         .replace(b",mrr", b",valor_mensal_medio"))
    (pasta / "situacao.csv").write_bytes(dados["situacao.csv"].replace(b"codigo,", b"cliente_id,"))
    (pasta / "uso.csv").write_bytes(dados["uso.csv"].replace(b"codigo,", b"cliente_id,"))
    monkeypatch.setenv("API_DADOS_BASES", str(tmp_path / "bases"))
    monkeypatch.setenv("API_DEMO", "1")
    monkeypatch.setenv("API_DEMO_ARQUIVO", str(tmp_path / "sem-inovaapps.xlsx"))
    monkeypatch.setenv("API_DEMO_REDES", str(pasta))
    from api.main import app

    with TestClient(app) as c:
        assert c.get("/api/bases/inovaapps").status_code == 404      # arquivo ausente: ignorada, sem quebrar
        d = c.get("/api/bases/redes")
        assert d.status_code == 200, d.text
        corpo = d.json()
        assert corpo["demonstracao"] is True and corpo["status"]["estado"] == "inspecionada"
        assert corpo["arquivos"] == ["clientes.csv", "situacao.csv", "uso.csv"]
        m = corpo["mapeamento"]
        assert m["alvo_tabela"] == "situacao" and m["coluna_data_saida"] == "mes_saida"
        assert m["coluna_valor"] == "valor_mensal_medio"
        assert m["colunas_ignoradas"] == ["clientes.valor_mensal_medio"]
        assert demo.MAPEAMENTOS["redes"]["tabela_clientes"] == "clientes"

        r = c.post("/api/bases/redes/treinar", json={"mapeamento": m, "opcoes": RAPIDO, "rotulo": "demo"})
        assert r.status_code == 202, r.text
        e = esperar_execucao(c, "redes", r.json()["execucao_id"])
        assert e["estado"] == "pronta", e
        assert e["ativa"] is True and e["resumo"]["rotulo"] == "demo"


def test_demonstracao_nao_escreve_em_web_public(tmp_path, monkeypatch):
    """A API nunca mexe em web/public/data/ (resultado versionado do time)."""
    from api.ajustes import RAIZ_PROJETO

    alvo = RAIZ_PROJETO / "web" / "public" / "data"
    antes = sorted(p.name for p in alvo.iterdir()) if alvo.exists() else None
    monkeypatch.setenv("API_DADOS_BASES", str(tmp_path / "bases"))
    from api import demo

    monkeypatch.setenv("API_DEMO", "1")
    monkeypatch.setenv("API_DEMO_REDES", str(tmp_path / "sem-redes"))
    criadas = demo.preparar()
    depois = sorted(p.name for p in alvo.iterdir()) if alvo.exists() else None
    assert antes == depois
    for base_id in criadas:
        assert (tmp_path / "bases" / base_id / "base.json").exists()
