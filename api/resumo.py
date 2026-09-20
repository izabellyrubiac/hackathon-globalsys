"""`resumo.json`: tudo que a tela de comparação de versões precisa sem abrir os arquivos grandes.

Extraído de `clientes.json`, `validacao.json` e `pesos.json` (contratos em `motor/saida.py`) mais o
status da execução. Nada é recalculado aqui: só leitura e recorte.

Campos (todos podem vir `null` quando a base não tem o dado — sem coluna de valor, validação desligada):

    execucao_id, rotulo, criada_em, concluido_em, duracao_s (tempo do treino), estado
    gerado_em, mes_referencia, validacao_ligada, tem_valor
    modelo      {nome, rotulo, forcado, motivo, criterio, log_loss, log_loss_ep, complexidade}
    candidatos  [{nome, rotulo, escolhido, elegivel, avaliado, log_loss, log_loss_ep}]
    horizonte_meses, cortes {alto, atencao}
    auc         {k1, k2, k3}                  AUC out-of-fold em −1/−2/−3 (nos cancelados vs. ativos)
    deteccao    {n_cancelados, alto: {k1,k2,k3}, alto_ou_atencao: {k1,k2,k3},
                 antecedencia_mediana_alto, antecedencia_mediana_alto_ou_atencao}
    alarme_falso{n_ativos, alto: {hoje, doze_meses, pct_meses},
                 alto_ou_atencao: {hoje, doze_meses, pct_meses}}
    faixas      {alto, atencao, baixo}         nº de clientes ativos por faixa
    n_clientes, em_risco, em_risco_alto,
    receita_em_risco_mensal, receita_risco_alto_mensal, perda_anual_esperada_total  (R$; null sem valor)
    variaveis   {n_candidatas, n_selecionadas, principais: [{id, rotulo, coluna, coef, importancia,
                                                             direcao, peso}] (até 5)}
    opcoes      {modelo, horizonte_meses, min_hist, delta_risco, validar, dobras, semente}
    painel      {clientes, cancelados, ativos, linhas_treino, positivos_treino}
    avisos
"""

from __future__ import annotations

KS = (1, 2, 3)


def _num(x) -> float | None:
    try:
        return None if x is None else float(x)
    except (TypeError, ValueError):
        return None


def _metodo(validacao: dict, ident: str) -> dict:
    return next((m for m in validacao.get("metodos") or [] if m.get("id") == ident), {})


def _auc_motor(validacao: dict) -> dict:
    """AUC out-of-fold do motor em −1/−2/−3 (`auc_por_mes`, score = "motor")."""
    por_k = {int(l["k"]): l for l in validacao.get("auc_por_mes") or []
             if l.get("score") == "motor" and l.get("k") is not None}
    return {f"k{k}": _num(por_k.get(k, {}).get("auc")) for k in KS}


def _deteccao(validacao: dict) -> dict:
    alto, ambas = _metodo(validacao, "motor_alto"), _metodo(validacao, "motor_alto_ou_atencao")
    painel = validacao.get("painel") or {}
    return {"n_cancelados": painel.get("cancelados"),
            "alto": {f"k{k}": alto.get(f"canc_k{k}") for k in KS},
            "alto_ou_atencao": {f"k{k}": ambas.get(f"canc_k{k}") for k in KS},
            "antecedencia_mediana_alto": _num(alto.get("antecedencia_mediana")),
            "antecedencia_mediana_alto_ou_atencao": _num(ambas.get("antecedencia_mediana"))}


def _alarme_falso(validacao: dict) -> dict:
    painel = validacao.get("painel") or {}

    def bloco(m: dict) -> dict:
        return {"hoje": m.get("ativos_hoje"), "doze_meses": m.get("ativos_algum_mes"),
                "pct_meses": _num(m.get("meses_alarme_ativos_pct"))}

    return {"n_ativos": painel.get("ativos"),
            "alto": bloco(_metodo(validacao, "motor_alto")),
            "alto_ou_atencao": bloco(_metodo(validacao, "motor_alto_ou_atencao"))}


def _modelo(validacao: dict, pesos: dict) -> tuple[dict, list[dict]]:
    escolha = validacao.get("escolha") or (pesos.get("modelo") or {})
    nome = escolha.get("modelo") or escolha.get("escolhido")
    candidatos = validacao.get("modelos") or (pesos.get("modelo") or {}).get("candidatos") or []
    meu = next((c for c in candidatos if c.get("nome") == nome), {})
    modelo = {"nome": nome, "rotulo": escolha.get("rotulo"), "forcado": escolha.get("forcado"),
              "motivo": escolha.get("motivo"), "criterio": escolha.get("criterio"),
              "complexidade": meu.get("complexidade"),
              "log_loss": _num(meu.get("log_loss")), "log_loss_ep": _num(meu.get("log_loss_ep"))}
    curtos = [{"nome": c.get("nome"), "rotulo": c.get("rotulo"), "escolhido": bool(c.get("escolhido")),
               "elegivel": c.get("elegivel"), "avaliado": c.get("avaliado"),
               "log_loss": _num(c.get("log_loss")), "log_loss_ep": _num(c.get("log_loss_ep"))}
              for c in candidatos]
    return modelo, curtos


def _variaveis(clientes: dict, pesos: dict) -> dict:
    resumo = pesos.get("resumo") or {}
    selecionadas = [v for v in pesos.get("variaveis") or [] if v.get("selecionada")]
    if not selecionadas:
        selecionadas = list((clientes.get("modelo") or {}).get("variaveis") or [])

    def peso(v: dict) -> float:
        for chave in ("importancia", "coef"):
            x = _num(v.get(chave))
            if x is not None:
                return abs(x)
        return 0.0

    principais = sorted(selecionadas, key=lambda v: (-peso(v), str(v.get("id"))))[:5]
    return {"n_candidatas": resumo.get("n_candidatas") or len(pesos.get("variaveis") or []),
            "n_selecionadas": resumo.get("n_selecionadas") if resumo.get("n_selecionadas") is not None
            else len(selecionadas),
            "principais": [{"id": v.get("id"), "rotulo": v.get("rotulo"), "coluna": v.get("coluna"),
                            "coef": _num(v.get("coef")), "importancia": _num(v.get("importancia")),
                            "direcao": v.get("direcao"), "peso": round(peso(v), 4)} for v in principais]}


def montar_resumo(clientes: dict, validacao: dict, pesos: dict, status: dict) -> dict:
    """Junta os três JSON do motor e o status numa ficha pequena, pronta para a comparação."""
    m_cli = clientes.get("modelo") or {}
    r_cli = clientes.get("resumo") or {}
    fila = clientes.get("clientes") or []
    faixas = {f: sum(1 for c in fila if c.get("faixa_risco") == f) for f in ("alto", "atencao", "baixo")}
    modelo, candidatos = _modelo(validacao, pesos)
    cfg = validacao.get("config") or {}
    opcoes = dict(status.get("opcoes") or {})
    for chave in ("modelo", "min_hist", "delta_risco", "validar", "dobras", "semente"):
        if chave in cfg and opcoes.get(chave) is None:      # o Config usado manda no que ficou em branco
            opcoes[chave] = cfg.get(chave)
    opcoes["delta_risco"] = bool(opcoes.get("delta_risco"))
    opcoes["horizonte_meses"] = m_cli.get("horizonte_meses") or validacao.get("horizonte_meses")
    painel = validacao.get("painel") or {}
    return {
        "execucao_id": status.get("execucao_id"), "rotulo": status.get("rotulo"),
        "criada_em": status.get("criada_em"), "concluido_em": status.get("concluido_em"),
        "duracao_s": _num(status.get("segundos")), "estado": status.get("estado"),
        "gerado_em": clientes.get("gerado_em"), "mes_referencia": clientes.get("mes_referencia"),
        "modelo": modelo, "candidatos": candidatos,
        "horizonte_meses": m_cli.get("horizonte_meses") or validacao.get("horizonte_meses"),
        "cortes": m_cli.get("cortes") or validacao.get("cortes"),
        "validacao_ligada": bool(validacao.get("metodos")),
        "auc": _auc_motor(validacao), "deteccao": _deteccao(validacao), "alarme_falso": _alarme_falso(validacao),
        "faixas": faixas, "n_clientes": r_cli.get("clientes_ativos") or len(fila),
        "em_risco": r_cli.get("em_risco"), "em_risco_alto": r_cli.get("em_risco_alto"),
        "receita_em_risco_mensal": _num(r_cli.get("receita_em_risco_mensal")),
        "receita_risco_alto_mensal": _num(r_cli.get("receita_risco_alto_mensal")),
        "perda_anual_esperada_total": _num(r_cli.get("perda_anual_esperada_total")),
        "tem_valor": m_cli.get("coluna_valor") is not None,
        "variaveis": _variaveis(clientes, pesos), "opcoes": opcoes,
        "painel": {k: painel.get(k) for k in ("clientes", "cancelados", "ativos", "linhas_treino",
                                              "positivos_treino")},
        "avisos": list(status.get("avisos") or validacao.get("avisos") or []),
    }


# --------------------------------------------------------------------------- comparação
LINHAS_COMPARACAO = [
    ("modelo", "Modelo escolhido", lambda r: (r.get("modelo") or {}).get("rotulo")),
    ("log_loss", "Log-loss fora da amostra (± EP)", lambda r: (r.get("modelo") or {}).get("log_loss")),
    ("log_loss_ep", "Erro-padrão do log-loss", lambda r: (r.get("modelo") or {}).get("log_loss_ep")),
    ("auc_k1", "AUC em −1", lambda r: (r.get("auc") or {}).get("k1")),
    ("auc_k2", "AUC em −2", lambda r: (r.get("auc") or {}).get("k2")),
    ("auc_k3", "AUC em −3", lambda r: (r.get("auc") or {}).get("k3")),
    ("deteccao_alto_k1", "Cancelados pegos na faixa alto em −1",
     lambda r: ((r.get("deteccao") or {}).get("alto") or {}).get("k1")),
    ("deteccao_alto_k2", "Cancelados pegos na faixa alto em −2",
     lambda r: ((r.get("deteccao") or {}).get("alto") or {}).get("k2")),
    ("deteccao_alto_k3", "Cancelados pegos na faixa alto em −3",
     lambda r: ((r.get("deteccao") or {}).get("alto") or {}).get("k3")),
    ("deteccao_ambas_k1", "Cancelados pegos em alto ou atenção em −1",
     lambda r: ((r.get("deteccao") or {}).get("alto_ou_atencao") or {}).get("k1")),
    ("alarme_falso_hoje", "Ativos com alarme hoje (faixa alto)",
     lambda r: ((r.get("alarme_falso") or {}).get("alto") or {}).get("hoje")),
    ("alarme_falso_12m", "Ativos com alarme em algum dos 12 meses (faixa alto)",
     lambda r: ((r.get("alarme_falso") or {}).get("alto") or {}).get("doze_meses")),
    ("faixa_alto", "Clientes na faixa alto", lambda r: (r.get("faixas") or {}).get("alto")),
    ("faixa_atencao", "Clientes na faixa atenção", lambda r: (r.get("faixas") or {}).get("atencao")),
    ("faixa_baixo", "Clientes na faixa baixo", lambda r: (r.get("faixas") or {}).get("baixo")),
    ("receita_em_risco_mensal", "Receita mensal em risco (R$)", lambda r: r.get("receita_em_risco_mensal")),
    ("perda_anual_esperada_total", "Perda anual esperada (R$)", lambda r: r.get("perda_anual_esperada_total")),
    ("n_selecionadas", "Variáveis selecionadas", lambda r: (r.get("variaveis") or {}).get("n_selecionadas")),
    ("delta_risco", "Delta de risco ligado", lambda r: (r.get("opcoes") or {}).get("delta_risco")),
    ("duracao_s", "Duração do treino (s)", lambda r: r.get("duracao_s")),
]


def comparar(resumos: list[dict], ativa: str | None = None) -> dict:
    """Tabela comparativa: uma coluna por execução, uma linha por métrica."""
    colunas = [{"execucao_id": r.get("execucao_id"), "rotulo": r.get("rotulo"),
                "criada_em": r.get("criada_em"), "ativa": r.get("execucao_id") == ativa,
                "modelo": (r.get("modelo") or {}).get("nome")} for r in resumos]
    linhas = [{"campo": campo, "rotulo": rotulo, "valores": [f(r) for r in resumos]}
              for campo, rotulo, f in LINHAS_COMPARACAO]
    return {"execucoes": colunas, "linhas": linhas, "resumos": resumos}
