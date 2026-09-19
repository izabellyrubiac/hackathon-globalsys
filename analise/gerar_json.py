"""Gera os JSON estáticos do app a partir da base e do score (`score.rodar`).

Uso:  uv run python analise/gerar_json.py

Saídas (em `web/public/data/`, JSON determinístico: chaves ordenadas, sem NaN → null; só `gerado_em`
muda de um dia para o outro — pode ser fixado com a variável de ambiente GERADO_EM=AAAA-MM-DD):

clientes.json
-------------
gerado_em                 data da geração (AAAA-MM-DD)
mes_referencia            último mês observado ("2026-06")
modelo                    {tipo, horizonte_meses, cortes{alto, atencao}, formula_prioridade}          [extra]
resumo
  clientes_ativos         58
  em_risco                nº de ativos nas faixas "alto" ou "atencao"
  receita_em_risco_mensal soma do valor_mensal desses clientes (R$)
  em_risco_alto           nº na faixa "alto"                                                           [extra]
  em_atencao              nº na faixa "atencao"                                                        [extra]
  receita_risco_alto_mensal  soma do valor_mensal da faixa "alto"                                     [extra]
  perda_anual_esperada_total soma de perda_anual_esperada dos 58                                      [extra]
clientes[]                (ordenados pela prioridade)
  cliente_id, segmento, porte, plano (com acentos, como em dados.carregar), valor_mensal, situacao ("Ativo")
  risco                   probabilidade (0–1) de cancelar nos próximos 3 meses (regressão logística)
  faixa_risco             "alto" | "atencao" | "baixo" (cortes de probabilidade em `modelo.cortes`)
  prioridade              1..58, única e contínua, ordem decrescente de perda_anual_esperada
  perda_anual_esperada    risco × valor_mensal × 12 (R$)                                               [extra]
  meses_em_alerta         meses seguidos (até jun/2026) com queda de uso ≥ 10 p.p. ou ≥ 2 outros sinais [extra]
  evidencias[]            variáveis que DISPARAM: contribuição > 0 no modelo E valor além do limiar do
                          notebook 02; ordem decrescente de contribuição
      sinal               campo de origem (uso_plataforma_pct, tempo_medio_resolucao_h, pct_sla_cumprido,
                          chamados_criticos, reclamacoes_formais, dias_atraso_pagamento, nota_nps,
                          nps_sem_resposta, meses_em_alerta)
      titulo, detalhe     textos pt-BR (ex.: "Uso da plataforma caiu" / "Caiu 11,7 p.p. vs a média ...")
      peso                fração (0–1) do aumento de risco do cliente atribuída à variável = contribuição /
                          soma das contribuições positivas (evidências + fatores secundários somam 1)
      contribuicao        coef × valor padronizado, em log-odds relativos ao cliente médio          [extra]
      valor               valor da variável em jun/2026 (unidade da variável)                        [extra]
  fatores_secundarios[]   mesmo formato: contribuição > 0 mas sem passar do limiar                      [extra]
  acao_sugerida           ação mapeada da variável dominante (maior evidência acionável)
  sinal_dominante         `sinal` da variável que definiu a ação (null = acompanhamento de rotina)      [extra]
  historico[]             18 meses (jan/2025–jun/2026): mes_ref, uso_plataforma_pct, pct_sla_cumprido,
                          tempo_medio_resolucao_h, chamados_criticos, reclamacoes_formais,
                          dias_atraso_pagamento, nota_nps (null fora dos meses de pesquisa ou sem resposta),
                          classificacao_nps (null fora dos meses de pesquisa; "Sem resposta" incluído),
                          risco (score do modelo final naquele mês, só com dados até ele; null sem dados) [extra: classificacao_nps, risco]

validacao.json
--------------
Validação out-of-fold deixando um cliente de fora (80 modelos): painel, cortes e critérios,
comparação com os baselines (só queda de uso; regra combinada do notebook 02) — detecção dos 22
cancelados em −1/−2/−3, alarmes falsos nos 58 ativos, antecedência —, AUC por mês, cancelados entre
os primeiros da fila, curva corte × detecção × alarme, coeficientes e o detalhe de cada cancelado.
"""

from __future__ import annotations

import json
import math
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dados import MES_FINAL, carregar  # noqa: E402
from score import HORIZONTE, VARIAVEIS, auc_por_mes, faixa, precisao_topo, rodar  # noqa: E402

SAIDA = Path(__file__).resolve().parent.parent / "web" / "public" / "data"
METRICAS_HIST = ["uso_plataforma_pct", "pct_sla_cumprido", "tempo_medio_resolucao_h", "chamados_criticos",
                 "reclamacoes_formais", "dias_atraso_pagamento"]
NOME_METODO = {"so_uso": "Só queda de uso (≥ 10 p.p. vs média anterior)",
               "regra": "Regra combinada do notebook 02 (uso ou ≥ 2 outros sinais, 2 meses seguidos)",
               "logistica_alto": "Score — faixa alto", "logistica_alto_ou_atencao": "Score — faixa alto ou atenção"}


def limpo(x, casas: int = 2):
    """Converte para tipos JSON: NaN/None → None, numpy → nativo, floats arredondados."""
    if isinstance(x, dict):
        return {str(k): limpo(v, casas) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [limpo(v, casas) for v in x]
    if x is None or (isinstance(x, float) and math.isnan(x)) or x is pd.NA or x is pd.NaT:
        return None
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        v = round(float(x), casas)
        return 0.0 if v == 0 else v
    if isinstance(x, pd.Period):
        return str(x)
    return x


def gravar(obj, nome: str) -> Path:
    SAIDA.mkdir(parents=True, exist_ok=True)
    txt = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    caminho = SAIDA / nome
    caminho.write_text(txt + "\n", encoding="utf-8")
    return caminho


def _itens(E: pd.DataFrame, cid: str, coluna: str, A: pd.Series) -> list[dict]:
    e = E[(E["cliente_id"] == cid) & E[coluna]]
    return [{"sinal": VARIAVEIS[r.variavel].coluna, "titulo": r.titulo, "detalhe": r.detalhe,
             "peso": round(float(r.peso), 3), "contribuicao": round(float(r.contribuicao), 3),
             "valor": limpo(A[r.variavel], 2)} for r in e.itertuples()]


def historico_cliente(d, P: pd.DataFrame, cid: str) -> list[dict]:
    at = d.atendimento[d.atendimento["cliente_id"] == cid].set_index("mes_ref")
    nps = d.nps[d.nps["cliente_id"] == cid].set_index("mes_ref")
    risco = P[(P["cliente_id"] == cid)].set_index("mes_ref")["p_final"]
    out = []
    for mes in at.index:
        linha = {"mes_ref": str(mes), **{m: limpo(at.at[mes, m], 1) for m in METRICAS_HIST}}
        tem = mes in nps.index
        linha["nota_nps"] = limpo(nps.at[mes, "nota_nps"], 1) if tem and nps.at[mes, "respondeu"] else None
        linha["classificacao_nps"] = str(nps.at[mes, "classificacao_nps"]) if tem else None
        linha["risco"] = limpo(risco.get(mes, np.nan), 4)
        out.append(linha)
    return out


def main() -> None:
    d = carregar()
    R = rodar(d)
    P, F, E, CUT, M = R.painel, R.fila, R.explicacao, R.cortes, R.modelo
    cli = d.clientes.set_index("cliente_id")
    gerado = os.environ.get("GERADO_EM", date.today().isoformat())

    # ---------------------------------------------------------------- clientes.json
    clientes = []
    for cid, r in F.iterrows():
        dom = r["variavel_dominante"]
        clientes.append({
            "cliente_id": cid, "segmento": str(cli.at[cid, "segmento"]), "porte": str(cli.at[cid, "porte"]),
            "plano": str(cli.at[cid, "plano"]), "valor_mensal": int(r["valor_mensal"]), "situacao": "Ativo",
            "risco": round(float(r["probabilidade"]), 4), "faixa_risco": str(r["faixa_risco"]),
            "prioridade": int(r["prioridade"]), "perda_anual_esperada": round(float(r["perda_anual_esperada"]), 2),
            "meses_em_alerta": int(r["persistencia"]),
            "evidencias": _itens(E, cid, "dispara", r), "fatores_secundarios": _itens(E, cid, "secundario", r),
            "acao_sugerida": r["acao_sugerida"],
            "sinal_dominante": VARIAVEIS[dom].coluna if isinstance(dom, str) else None,
            "historico": historico_cliente(d, P, cid),
        })
    em = F[F["faixa_risco"] != "baixo"]
    alto = F[F["faixa_risco"] == "alto"]
    doc = {
        "gerado_em": gerado,
        "mes_referencia": str(MES_FINAL),
        "modelo": {
            "tipo": "Regressão logística L2 (variáveis padronizadas), painel cliente × mês sem vazamento",
            "horizonte_meses": HORIZONTE,
            "cortes": {"alto": CUT["alto"], "atencao": CUT["atencao"]},
            "formula_prioridade": "risco × valor_mensal × 12 (perda anual esperada)",
        },
        "resumo": {
            "clientes_ativos": len(F), "em_risco": len(em), "receita_em_risco_mensal": int(em["valor_mensal"].sum()),
            "em_risco_alto": len(alto), "em_atencao": int((F["faixa_risco"] == "atencao").sum()),
            "receita_risco_alto_mensal": int(alto["valor_mensal"].sum()),
            "perda_anual_esperada_total": round(float(F["perda_anual_esperada"].sum()), 2),
        },
        "clientes": clientes,
    }
    doc = limpo(doc, 4)
    verificar(doc, len(cli))
    c1 = gravar(doc, "clientes.json")

    # ---------------------------------------------------------------- validacao.json
    T = P[P["treino"]]
    metodos = [{"id": m, "nome": NOME_METODO[m], **{k: v for k, v in r.items()}} for m, r in R.baselines.items()]
    auc = auc_por_mes(P, {"score": (P["p_oof"], 1), "so_uso": (P["uso"], -1), "n_sinais_regra": (P["n_sinais"], 1)})
    topo = pd.concat([precisao_topo(P, P["p_oof"]).assign(metodo="score"),
                      precisao_topo(P, P["uso"], -1).assign(metodo="so_uso")])
    co = M.coeficientes()
    canc = []
    for cid in cli.index[cli["status"] == "Cancelou"]:
        q = P[(P["cliente_id"] == cid) & P["k"].le(3)].set_index("k")
        canc.append({"cliente_id": cid, "valor_mensal": int(cli.at[cid, "valor_mensal"]),
                     "mes_cancelamento": str(cli.at[cid, "mes_cancelamento"]),
                     **{f"risco_k{k}": q.at[k, "p_oof"] for k in (1, 2, 3)},
                     **{f"faixa_k{k}": (str(faixa([q.at[k, "p_oof"]], CUT)[0]) if pd.notna(q.at[k, "p_oof"]) else None) for k in (1, 2, 3)},
                     **{f"regra_k{k}": bool(q.at[k, "alerta_regra"]) for k in (1, 2, 3)}})
    val = {
        "gerado_em": gerado,
        "metodo": "Leave-one-client-out: 80 modelos, cada cliente avaliado por um modelo treinado sem ele "
                  "(C e checagem de sinal refeitos em cada dobra)",
        "horizonte_meses": HORIZONTE,
        "painel": {"linhas": len(P), "linhas_com_dados": int(P["tem_dados"].sum()), "linhas_treino": len(T),
                   "positivos_treino": int(T["y"].sum()), "clientes": int(P["cliente_id"].nunique())},
        "cortes": {**CUT,
                   "criterio_alto": "menor corte com precisão ≥ 50% nos meses-cliente marcados (validação out-of-fold)",
                   "criterio_atencao": "maior corte que detecta em −2 e −3 ao menos tantos cancelados quanto a regra do notebook 02"},
        "metodos": metodos,
        "auc_por_mes": auc.to_dict("records"),
        "precisao_topo": topo.to_dict("records"),
        "curva": R.curva[["corte", "canc_k1", "canc_k2", "canc_k3", "ativos_hoje", "ativos_algum_mes",
                          "meses_alarme_ativos_pct", "precisao_linhas", "recall_linhas", "fpr_linhas"]].to_dict("records"),
        "coeficientes": [{"variavel": f, "sinal": VARIAVEIS[f].coluna, "rotulo": VARIAVEIS[f].rotulo, "coef": co.at[f, "coef"],
                          "odds_ratio": co.at[f, "odds_ratio"], "direcao_esperada": VARIAVEIS[f].direcao,
                          "min_dobras": R.dobras[f].min(), "max_dobras": R.dobras[f].max()} for f in M.features],
        "intercepto": M.intercepto(),
        "C": M.C,
        "variaveis_removidas": M.removidas,
        "cancelados": canc,
    }
    c2 = gravar(limpo(val, 4), "validacao.json")

    print(f"{c1}: {len(clientes)} clientes · {doc['resumo']['em_risco_alto']} alto · {doc['resumo']['em_atencao']} atenção · "
          f"receita em risco R$ {doc['resumo']['receita_em_risco_mensal']:,}/mês".replace(",", "."))
    print(f"{c2}: cortes alto ≥ {CUT['alto']:.2f}, atenção ≥ {CUT['atencao']:.2f}")


def verificar(doc: dict, n_base: int) -> None:
    cl = doc["clientes"]
    assert n_base == 80, n_base
    assert len(cl) == 58 == doc["resumo"]["clientes_ativos"]
    assert sorted(c["prioridade"] for c in cl) == list(range(1, 59))
    assert all(c["faixa_risco"] in ("alto", "atencao", "baixo") for c in cl)
    assert all(0 <= c["risco"] <= 1 for c in cl)
    assert all(c["evidencias"] for c in cl if c["faixa_risco"] == "alto"), "cliente alto sem evidência"
    json.dumps(doc, allow_nan=False)          # falha se sobrou NaN/inf


if __name__ == "__main__":
    main()
