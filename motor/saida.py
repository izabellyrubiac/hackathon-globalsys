"""Contrato dos JSON de resultado (usado pela API e por `analise/gerar_json.py`).

Todos: determinísticos (mesma entrada → mesma saída; chaves ordenadas ao gravar), sem NaN (→ null),
floats arredondados. Só `gerado_em` muda de um dia para o outro (fixável com GERADO_EM=AAAA-MM-DD).

clientes.json
-------------
gerado_em, mes_referencia ("AAAA-MM"; null no modo fotografia)
modelo      {tipo, horizonte_meses, cortes{alto, atencao}, formula_prioridade,
             coluna_valor {tabela, coluna} | null,
             variaveis [{id, rotulo, tabela, coluna, transformacao, coef, direcao, limiar}],
             series_historico [{chave, tabela, coluna, rotulo, tipo ("numerica"|"categorica")}]}
resumo      {clientes_ativos, em_risco (alto+atenção), em_risco_alto, em_atencao,
             receita_em_risco_mensal, receita_risco_alto_mensal, perda_anual_esperada_total}
             (os três valores em R$ ficam null sem coluna de valor)
clientes[]  (todos os ativos, na ordem da fila)
  cliente_id, atributos {coluna categórica da tabela de clientes: valor}, valor_mensal (null sem valor),
  situacao "Ativo", risco (probabilidade de cancelar nos próximos H meses), faixa_risco ("alto"|"atencao"|"baixo"),
  prioridade (1..N, única e contínua; ordem = perda_anual_esperada desc, sem valor = risco desc),
  perda_anual_esperada (risco × valor_mensal × 12; null sem valor), meses_em_alerta (meses seguidos até o mês
  de referência com risco ≥ corte de atenção),
  evidencias[] / fatores_secundarios[]: {sinal (id da variável), coluna, tabela, titulo, detalhe, peso,
      contribuicao, valor, limiar}
  acao_sugerida, sinal_dominante (id da variável ou null), coluna_dominante
  historico[] {mes_ref, risco (modelo final, só com dados até o mês), valores {chave da série: valor}}

validacao.json
--------------
gerado_em, metodo, honestidade, horizonte_meses, n_dobras, painel{…}, cortes{alto, atencao, youden, criterios},
metodos[{id, nome, canc_k1..3, ativos_hoje, ativos_algum_mes, meses_alarme_ativos_pct, antecedencia_mediana}],
auc_por_mes[{k, score, auc, n, n_canc}], precisao_topo[{k, top, cancelados, metodo}], curva[…],
coeficientes[{variavel, rotulo, coluna, coef, odds_ratio, direcao, frequencia_dobras, min_dobras, max_dobras}],
intercepto, C, cancelados[{cliente_id, valor_mensal, mes_saida, risco_k1..3, faixa_k1..3}], avisos, config

pesos.json
----------
gerado_em, horizonte_meses, regras{…}, resumo{n_candidatas, n_selecionadas, limite_variaveis, C_l1, C_l2, …},
variaveis[{id, rotulo, tabela, coluna, transformacao, categoria, selecionada, coef, odds_ratio, auc_univariada,
           auc_treino, direcao, direcao_texto, frequencia_selecao, consistencia_sinal, frequencia_dobras,
           limiar, nulos_pct, vazamento, motivo_descarte}]
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .util import fmt_num

NOME_METODO = {
    "melhor_variavel": "Melhor variável sozinha (limiar de Youden)",
    "motor_alto": "Motor — faixa alto",
    "motor_alto_ou_atencao": "Motor — faixa alto ou atenção",
}


def limpo(x, casas: int = 4):
    """Tipos JSON: NaN/None → None, numpy → nativo, floats arredondados, Period → 'AAAA-MM'."""
    if isinstance(x, dict):
        return {str(k): limpo(v, casas) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [limpo(v, casas) for v in x]
    if x is None or x is pd.NA or x is pd.NaT:
        return None
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        if math.isnan(x) or math.isinf(x):
            return None
        v = round(float(x), casas)
        return 0.0 if v == 0 else v
    if isinstance(x, pd.Period):
        return str(x)
    if isinstance(x, pd.Timestamp):
        return x.date().isoformat()
    return x


def gravar(obj: dict, caminho: str | Path) -> Path:
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    txt = json.dumps(limpo(obj), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    caminho.write_text(txt + "\n", encoding="utf-8")
    return caminho


def _mes(p) -> str | None:
    return None if p is None or (not isinstance(p, pd.Period) and pd.isna(p)) else str(p)


# --------------------------------------------------------------------------- clientes.json
def _series(r) -> list[dict]:
    """Colunas de origem (temporais) das variáveis do modelo, para o histórico de cada cliente."""
    vistos, out = set(), []
    temporais = {t.nome: t for t in r.base.temporais}
    for f in r.modelo.features:
        v = r.meta[f]
        if v.tabela not in temporais or (v.tabela, v.coluna) in vistos:
            continue
        vistos.add((v.tabela, v.coluna))
        t = temporais[v.tabela]
        tipo = "numerica" if v.coluna == "registros" or pd.api.types.is_numeric_dtype(t.df[v.coluna]) else "categorica"
        repetida = sum(1 for x in temporais.values() if v.coluna in x.colunas) > 1 or v.coluna == "registros"
        out.append({"chave": f"{v.tabela}.{v.coluna}" if repetida else v.coluna, "tabela": v.tabela, "coluna": v.coluna,
                    "rotulo": v.rotulo_coluna, "tipo": tipo})
    return out


def _valores_mensais(r, series: list[dict]) -> dict[str, pd.Series]:
    temporais = {t.nome: t for t in r.base.temporais}
    out = {}
    for s in series:
        df = temporais[s["tabela"]].df
        g = df.groupby(["__cli", "__mes"])
        if s["coluna"] == "registros":
            out[s["chave"]] = g.size()
        elif s["tipo"] == "numerica":
            out[s["chave"]] = g[s["coluna"]].mean()
        else:
            out[s["chave"]] = g[s["coluna"]].last()
    return out


def _item(e, meta) -> dict:
    v = meta[e.variavel]
    return {"sinal": e.variavel, "coluna": v.coluna, "tabela": v.tabela, "titulo": e.titulo, "detalhe": e.detalhe,
            "peso": round(float(e.peso), 3), "contribuicao": round(float(e.contribuicao), 3),
            "valor": limpo(e.valor, 2), "limiar": limpo(e.limiar, 4)}


def clientes_json(r, gerado: str) -> dict:
    F, E, P, meta = r.fila, r.explicacao, r.painel, r.meta
    series = _series(r)
    vals = _valores_mensais(r, series)
    tem_valor = r.base.valor is not None
    hist = P[P["tem_dados"]].sort_values(["cliente", "mes"])
    lista = []
    for cid, f in F.iterrows():
        e = E[E["cliente"] == cid]
        dom = f["variavel_dominante"]
        h = []
        if not r.base.fotografia:
            for _, row in hist[hist["cliente"] == cid].iterrows():
                valores = {}
                for s in series:
                    x = vals[s["chave"]].get((cid, row["mes"]), None)
                    valores[s["chave"]] = (limpo(x, 2) if s["tipo"] == "numerica" else (None if x is None or pd.isna(x) else str(x)))
                h.append({"mes_ref": str(row["mes"]), "risco": limpo(row["p_final"], 4), "valores": valores})
        attrs = {c: (None if pd.isna(x) else str(x)) for c, x in r.base.atributos.loc[cid].items()}
        lista.append({
            "cliente_id": cid, "atributos": attrs,
            "valor_mensal": limpo(f["valor_mensal"], 2) if tem_valor else None,
            "situacao": "Ativo", "risco": round(float(f["probabilidade"]), 4), "faixa_risco": str(f["faixa_risco"]),
            "prioridade": int(f["prioridade"]),
            "perda_anual_esperada": limpo(f["perda_anual_esperada"], 2) if tem_valor else None,
            "meses_em_alerta": int(f["meses_em_alerta"]),
            "evidencias": [_item(x, meta) for x in e[e["dispara"]].itertuples()],
            "fatores_secundarios": [_item(x, meta) for x in e[e["secundario"]].itertuples()],
            "acao_sugerida": f["acao_sugerida"],
            "sinal_dominante": dom if isinstance(dom, str) else None,
            "coluna_dominante": meta[dom].coluna if isinstance(dom, str) else None,
            "historico": h,
        })
    em = F[F["faixa_risco"] != "baixo"]
    alto = F[F["faixa_risco"] == "alto"]
    soma = (lambda s: round(float(s.sum()), 2)) if tem_valor else (lambda s: None)
    m = r.mapeamento
    variaveis = [{"id": f, "rotulo": meta[f].rotulo, "tabela": meta[f].tabela, "coluna": meta[f].coluna,
                  "transformacao": meta[f].transformacao, "coef": round(float(c), 4),
                  "direcao": r.modelo.direcoes[f], "limiar": limpo(r.modelo.limiares.get(f), 4)}
                 for f, c in zip(r.modelo.features, r.modelo.coef)]
    return limpo({
        "gerado_em": gerado,
        "mes_referencia": _mes(r.base.mes_ref),
        "modelo": {
            "tipo": "Regressão logística L2 (variáveis padronizadas) com seleção automática por stability selection; "
                    "painel cliente × mês sem vazamento",
            "horizonte_meses": int(m.horizonte_meses),
            "cortes": {"alto": r.cortes["alto"], "atencao": r.cortes["atencao"]},
            "formula_prioridade": ("risco × valor_mensal × 12 (perda anual esperada)" if tem_valor
                                   else "risco (sem coluna de valor do contrato)"),
            "coluna_valor": {"tabela": m.valor_tabela or m.tabela_clientes, "coluna": m.coluna_valor} if tem_valor else None,
            "variaveis": variaveis,
            "series_historico": series,
        },
        "resumo": {
            "clientes_ativos": len(F), "em_risco": len(em), "em_risco_alto": len(alto),
            "em_atencao": int((F["faixa_risco"] == "atencao").sum()),
            "receita_em_risco_mensal": soma(em["valor_mensal"]), "receita_risco_alto_mensal": soma(alto["valor_mensal"]),
            "perda_anual_esperada_total": soma(F["perda_anual_esperada"]),
        },
        "clientes": lista,
    })


# --------------------------------------------------------------------------- validacao.json
def validacao_json(r, gerado: str) -> dict:
    from .treino import desempenho_metodos
    from .validar import auc_por_mes, precisao_topo

    P, cfg = r.painel, r.config
    T = P[P["treino"]]
    cli = P.groupby("cliente")["cancelado"].first()
    base = {
        "gerado_em": gerado, "horizonte_meses": int(r.mapeamento.horizonte_meses),
        "painel": {"linhas": len(P), "linhas_com_dados": int(P["tem_dados"].sum()), "linhas_treino": len(T),
                   "positivos_treino": int(T["y"].sum()), "clientes": int(len(cli)), "cancelados": int(cli.sum()),
                   "ativos": int((~cli).sum()),
                   "linhas_censuradas": int((P["tem_dados"] & ~P["cancelado"] & ~P["treino"]
                                             & (P["n_hist"] >= cfg.min_hist)).sum()),
                   "cancelados_excluidos": len(r.base.excluidos),
                   "deslocamento_saida_meses": r.base.deslocamento},
        "cortes": {**r.cortes,
                   "criterio_alto": "menor corte com precisão ≥ 50% nos meses-cliente marcados (validação out-of-fold)",
                   "criterio_atencao": "corte de Youden nos meses-cliente de treino (sensibilidade − alarme falso, out-of-fold)"},
        "avisos": r.avisos, "config": cfg.to_dict(),
        "intercepto": r.modelo.intercepto, "C": r.modelo.C,
    }
    if r.oof is None:
        return limpo({**base, "metodo": "Validação desligada", "metodos": [], "auc_por_mes": [], "precisao_topo": [],
                      "curva": [], "coeficientes": [], "cancelados": []})
    oof = r.oof
    metodos = [{"id": k, "nome": NOME_METODO[k], **v} for k, v in desempenho_metodos(r).items()]
    melhor_vars = pd.Series(oof.base_vars).value_counts()
    metodos[0]["variavel"] = melhor_vars.index[0]
    metodos[0]["rotulo"] = r.meta[melhor_vars.index[0]].rotulo if melhor_vars.index[0] in r.meta else melhor_vars.index[0]
    auc = auc_por_mes(P, {"motor": P["p_oof"], "melhor_variavel": oof.base_score})
    topo = pd.concat([precisao_topo(P, P["p_oof"]).assign(metodo="motor"),
                      precisao_topo(P, oof.base_score).assign(metodo="melhor_variavel")])
    dob = oof.dobras
    co = []
    for f, c in zip(r.modelo.features, r.modelo.coef):
        d = dob[f] if f in dob else pd.Series(dtype=float)
        co.append({"variavel": f, "rotulo": r.meta[f].rotulo, "coluna": r.meta[f].coluna, "coef": float(c),
                   "odds_ratio": float(np.exp(c)), "direcao": r.modelo.direcoes[f],
                   "frequencia_dobras": float(d.notna().sum() / oof.n_dobras),
                   "min_dobras": float(d.min()) if d.notna().any() else None,
                   "max_dobras": float(d.max()) if d.notna().any() else None})
    canc = []
    from .validar import faixa
    for cid in cli.index[cli.to_numpy()]:
        q = P[(P["cliente"] == cid)].set_index("k")
        item = {"cliente_id": cid, "mes_saida": _mes(r.base.mes_saida.get(cid)),
                "valor_mensal": limpo(r.base.valor.get(cid), 2) if r.base.valor is not None else None}
        for k in (1, 2, 3):
            pk = q.at[k, "p_oof"] if k in q.index else np.nan
            item[f"risco_k{k}"] = pk
            item[f"faixa_k{k}"] = str(faixa([pk], r.cortes)[0]) if pd.notna(pk) else None
        canc.append(item)
    cols = ["corte", "canc_k1", "canc_k2", "canc_k3", "ativos_hoje", "ativos_algum_mes", "meses_alarme_ativos_pct",
            "precisao_linhas", "recall_linhas", "fpr_linhas"]
    return limpo({
        **base,
        "metodo": (f"Validação out-of-fold agrupada por cliente: {oof.n_dobras} dobras estratificadas por cancelou; "
                   "em cada uma, o motor inteiro (filtro, AUC, persistência, redundância, stability selection, C e "
                   "limiares) é refeito só com os clientes de treino e os de teste recebem probabilidade em todos os meses"),
        "honestidade": ("A seleção de variáveis é aninhada na validação, então as métricas não são otimistas pela escolha "
                        "das variáveis. Só os dois cortes das faixas são escolhidos olhando as probabilidades out-of-fold "
                        "(leve otimismo nas contagens por faixa; a AUC e a precisão no topo não dependem dos cortes)."),
        "n_dobras": oof.n_dobras,
        "metodos": metodos,
        "auc_por_mes": auc.to_dict("records"),
        "precisao_topo": topo.to_dict("records"),
        "curva": r.curva[cols].to_dict("records"),
        "coeficientes": co,
        "melhor_variavel_por_dobra": {str(k): int(v) for k, v in melhor_vars.items()},
        "cancelados": canc,
    })


# --------------------------------------------------------------------------- pesos.json
def pesos_json(r, gerado: str) -> dict:
    rel = r.modelo.selecao.relatorio.copy()
    cfg = r.config
    coef = dict(zip(r.modelo.features, r.modelo.coef))
    dob = r.oof.dobras if r.oof is not None else pd.DataFrame()
    linhas = []
    for vid, x in rel.iterrows():
        v = r.meta.get(vid)
        if v is None:
            continue
        d = x.get("direcao")
        sel = vid in coef
        linhas.append({
            "id": vid, "rotulo": v.rotulo, "tabela": v.tabela, "coluna": v.coluna, "transformacao": v.transformacao,
            "categoria": v.categoria, "selecionada": sel,
            "coef": coef.get(vid), "odds_ratio": float(np.exp(coef[vid])) if sel else None,
            "auc_univariada": x.get("auc_oof"), "auc_treino": x.get("auc_treino"),
            "direcao": None if pd.isna(d) else int(d),
            "direcao_texto": None if pd.isna(d) else ("maior = mais risco" if d > 0 else "menor = mais risco"),
            "frequencia_selecao": x.get("frequencia_selecao"), "consistencia_sinal": x.get("consistencia_sinal"),
            "frequencia_dobras": float(dob[vid].notna().sum() / r.oof.n_dobras) if r.oof is not None and vid in dob else
            (0.0 if r.oof is not None else None),
            "limiar": r.modelo.limiares.get(vid), "nulos_pct": x.get("nulos_pct"),
            "vazamento": bool(x.get("vazamento")), "motivo_descarte": None if sel else (x.get("motivo") or "retirada"),
        })
    def num(v):
        return 0.0 if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)

    linhas.sort(key=lambda l: (not l["selecionada"], -num(l["frequencia_selecao"]), -num(l["auc_univariada"]), l["id"]))
    sel = r.modelo.selecao
    n_pos = int(r.painel.loc[r.painel["treino"], "y"].sum())
    n_canc = int(r.painel.groupby("cliente")["cancelado"].first().sum())
    return limpo({
        "gerado_em": gerado, "horizonte_meses": int(r.mapeamento.horizonte_meses),
        "regras": {
            "filtro": f"descarta > {fmt_num(100 * cfg.max_nulos, 0)}% de nulos e ≥ {fmt_num(100 * cfg.max_moda, 0)}% "
                      "das linhas no mesmo valor",
            "auc_univariada": f"out-of-fold agrupada por cliente ({cfg.dobras_internas} dobras), direção de risco "
                              "aprendida no treino de cada dobra",
            "vazamento": f"AUC univariada > {cfg.auc_vazamento} ou AUC ≥ 0,90 só no último mês antes da saída "
                         "(≤ 0,55 nos meses anteriores do horizonte) → excluída",
            "persistencia": f"meses seguidos na zona de risco para as {cfg.n_persistencia} colunas de maior AUC "
                            "(limiar de Youden arredondado)",
            "redundancia": f"|Spearman| > {cfg.max_spearman} → fica a de maior AUC",
            "estabilidade": (f"{cfg.rodadas} subamostras de {fmt_num(100 * cfg.frac_subamostra, 0)}% dos clientes, "
                             f"regressão logística {'L1' if cfg.l1_ratio >= 1 else f'elastic-net (l1_ratio {cfg.l1_ratio})'} "
                             "com C por validação agrupada; selecionada se escolhida em ≥ "
                             f"{fmt_num(100 * cfg.limiar_frequencia, 0)}% das rodadas com o mesmo sinal em ≥ "
                             f"{fmt_num(100 * cfg.consistencia_sinal, 0)}% delas e sinal igual à direção univariada"),
            "limite": (f"no máximo mín(positivos/{cfg.linhas_por_variavel}, cancelados/{cfg.clientes_por_variavel}) "
                       f"variáveis, mínimo 2 = mín({n_pos}/{cfg.linhas_por_variavel}, {n_canc}/"
                       f"{cfg.clientes_por_variavel}) = {sel.limite}"),
            "modelo_final": "regressão logística L2 nas selecionadas, C por GroupKFold (log-loss), sem class_weight; "
                            "variável com sinal invertido sai",
            "limiar_evidencia": "corte de Youden no treino, arredondado para um número redondo (mantendo ≥ 90% do J)",
        },
        "resumo": {"n_candidatas": len(linhas), "n_selecionadas": len(r.modelo.features), "limite_variaveis": sel.limite,
                   "C_l1": sel.C_l1, "C_l2": r.modelo.C, "intercepto": r.modelo.intercepto,
                   "linhas_treino": int(r.painel["treino"].sum()), "positivos_treino": n_pos, "cancelados": n_canc,
                   "retiradas_sinal_invertido": r.modelo.removidas,
                   "suspeitas_vazamento": [l["id"] for l in linhas if l["vazamento"]]},
        "variaveis": linhas,
    })
