"""Experimento: o **delta de risco** aumenta o acerto preditivo?

Dois braços, MESMAS dobras (mesma semente, mesma base, mesma grade), só a flag muda:

* **A — sem delta** (`Config.delta_risco=False`): o motor como está hoje, delta em lugar nenhum;
* **B — com delta** (`Config.delta_risco=True`): motor em dois estágios (`motor/delta.py`).

Um braço extra, de graça, sobre o resultado de A:

* **C — delta só na ordenação**: o modelo é o de A, mas a fila é ordenada por
  `min(p + máx(Δp, 0), 1) × valor × 12` em vez de `p × valor × 12` — quem subiu de risco sobe na fila,
  sem nenhum ajuste de modelo. Compara só a qualidade da ordenação com A.

Tudo out-of-fold (`p_oof`, probabilidade de um modelo que não viu aquele cliente). A comparação usa
diferença **pareada por dobra** ± erro-padrão: é o que separa ganho real de ruído.

Uso:
    uv run python analise/experimentos/delta_risco.py --braco a        # roda e grava o braço A
    uv run python analise/experimentos/delta_risco.py --braco b        # … e o braço B
    uv run python analise/experimentos/delta_risco.py --braco comparar  # A × B × C
    uv run python analise/experimentos/delta_risco.py --braco inovaapps # referência pequena (A × B)
    uv run python analise/experimentos/delta_risco.py                   # a, b e comparar em sequência

Saída: `amostra_redes/resultado/delta/*.json` (+ `*_painel.csv.gz`, dados brutos out-of-fold).
Com `--min-hist` diferente do padrão (4), tudo vai para a subpasta `hist<N>/`.
Nada em `web/public/data/` é tocado: o padrão de produção continua sem delta.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from analise.redes.rodar import carregar, mapeamento  # noqa: E402
from motor.config import Config  # noqa: E402
from motor.modelos_classificacao.metricas import (  # noqa: E402
    KS, auc_por_mes, desempenho, erro_padrao, precisao_topo,
)
from motor.modelos_previsao.base import log_loss  # noqa: E402
from motor.util import auc  # noqa: E402

DESTINO = RAIZ / "amostra_redes" / "resultado" / "delta"
BRACOS = {"a": False, "b": True}
# braço P (controle): o delta embaralhado — mesma distribuição, informação zero. Só o LightGBM (o modelo
# escolhido nos dois braços) é validado, nas MESMAS dobras, então o log-loss por dobra é comparável ao de A.
CONFIGS = {
    "a": dict(delta_risco=False, avaliar_todos=True),
    "b": dict(delta_risco=True, avaliar_todos=True),
    "p": dict(delta_risco=True, delta_placebo=True, avaliar_todos=False, modelo="lightgbm"),
}
ROTULO = {"a": "A — sem delta", "b": "B — com delta", "p": "P — placebo (delta embaralhado, só LightGBM)"}
TOPOS = (10, 20, 50, 100, 200)


def limpar(o):
    """numpy → tipos do Python e NaN → null, para o JSON ficar válido e legível."""
    if isinstance(o, dict):
        return {str(k): limpar(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [limpar(v) for v in o]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else round(float(o), 6)
    if o is None or isinstance(o, str):
        return o
    return str(o)


def gravar(caminho: Path, obj) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(limpar(obj), ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")


# --------------------------------------------------------------------------- rodar um braço
def painel_de(r) -> pd.DataFrame:
    """Painel out-of-fold enxuto: o que as métricas precisam, com a dobra externa de cada cliente."""
    P = r.painel[["cliente", "mes", "k", "cancelado", "y", "treino", "pontuavel", "referencia"]].copy()
    P["mes"] = P["mes"].astype(str)
    P["p_oof"] = r.oof.p.to_numpy() if r.oof is not None else np.nan
    P["p_final"] = r.painel["p_final"].to_numpy()
    P["valor"] = P["cliente"].map(r.base.valor) if r.base.valor is not None else np.nan
    dobra = {c: i for i, cli in enumerate(r.oof.clientes_dobra) for c in cli}
    P["dobra"] = P["cliente"].map(dobra)
    return P


def rodar_braco(braco: str, horizonte: int, min_hist: int, delta_dobras: int) -> dict:
    from motor import treinar

    t0 = time.perf_counter()
    tabelas = carregar()
    cfg = Config(min_hist=min_hist, delta_dobras=delta_dobras, **CONFIGS[braco])
    marcos: list[tuple[float, str]] = []

    def progresso(etapa: str, f: float) -> None:
        marcos.append((round(time.perf_counter() - t0, 1), etapa))
        print(f"  [{time.perf_counter() - t0:7.1f}s] {100 * f:5.1f}% {etapa}", flush=True)

    r = treinar(tabelas, mapeamento(horizonte), progresso=progresso, config=cfg)
    dt = time.perf_counter() - t0

    DESTINO.mkdir(parents=True, exist_ok=True)
    P = painel_de(r)
    P.to_csv(DESTINO / f"{braco}_painel.csv.gz", index=False)

    deltas = [v for v in r.pesos["variaveis"] if v["tabela"] == "modelo"]
    info = {
        "braco": braco, "rotulo": ROTULO[braco], "delta_risco": cfg.delta_risco,
        "config": {k: v for k, v in cfg.to_dict().items()
                   if k in ("min_hist", "dobras", "semente", "modelo", "delta_risco", "delta_risco_3m",
                            "delta_dobras", "delta_placebo", "avaliar_todos", "perfil")},
        "horizonte_meses": horizonte,
        "segundos": round(dt, 1), "marcos": marcos,
        "clientes": int(r.painel["cliente"].nunique()),
        "cancelados": int(r.painel.groupby("cliente")["cancelado"].first().sum()),
        "linhas_painel": int(len(r.painel)), "linhas_treino": int(r.painel["treino"].sum()),
        "positivos_treino": int(r.painel.loc[r.painel["treino"], "y"].sum()),
        "modelo_escolhido": r.escolha.nome, "motivo_escolha": r.escolha.motivo,
        "elegibilidade": r.escolha.elegibilidade,
        "candidatos": r.escolha.candidatos_dict(),
        "cortes": r.cortes,
        "variaveis_selecionadas": [{k: v.get(k) for k in ("id", "rotulo", "coef", "importancia",
                                                          "frequencia_selecao", "frequencia_dobras", "limiar")}
                                   for v in r.pesos["variaveis"] if v["selecionada"]],
        "candidatas_delta": [{k: v.get(k) for k in ("id", "rotulo", "selecionada", "coef", "odds_ratio",
                                                    "importancia", "auc_univariada", "frequencia_selecao",
                                                    "frequencia_dobras", "limiar", "motivo_descarte")}
                             for v in deltas],
        "log_loss_dobras": {n: [None if np.isnan(x) else float(x) for x in m.log_loss_dobras]
                            for n, m in r.oof.modelos.items()},
        "avisos": r.validacao.get("avisos", []),
    }
    gravar(DESTINO / f"{braco}.json", info)
    print(f"\nbraço {braco}: {dt:.1f}s · modelo {r.escolha.nome} · "
          f"delta selecionado: {[v['id'] for v in deltas if v['selecionada']] or 'nenhum'}")
    return info


# --------------------------------------------------------------------------- métricas
def _fila(P: pd.DataFrame, chave: pd.Series) -> pd.DataFrame:
    """Um cliente por linha no mês −1 (cancelados: último mês; ativos: referência), na ordem da fila."""
    m = (P["k"] == 1) & P["pontuavel"] & chave.notna()
    q = P.loc[m, ["cliente", "cancelado", "valor"]].assign(chave=chave[m])
    return q.sort_values(["chave", "cliente"], ascending=[False, True]).reset_index(drop=True)


def qualidade_fila(P: pd.DataFrame, p: pd.Series, topos=TOPOS) -> dict:
    """Ordenação pela perda anual esperada (p × valor × 12): cancelados no topo e valor em jogo capturado."""
    perda = p * P["valor"] * 12
    q = _fila(P, perda.where(P["valor"].notna()))
    canc_total = int(q["cancelado"].sum())
    valor_canc = float(q.loc[q["cancelado"], "valor"].sum())
    out = {"clientes_na_fila": int(len(q)), "cancelados_na_fila": canc_total,
           "valor_mensal_cancelados": round(valor_canc, 2)}
    for n in topos:
        cab = q.head(n)
        out[f"cancelados_top{n}"] = int(cab["cancelado"].sum())
        out[f"valor_capturado_top{n}_pct"] = (round(100 * float(cab.loc[cab["cancelado"], "valor"].sum()) / valor_canc, 1)
                                              if valor_canc else None)
    ordem = np.arange(1, len(q) + 1)
    out["posicao_mediana_cancelados"] = float(np.median(ordem[q["cancelado"].to_numpy()])) if canc_total else None
    return out


def metricas(P: pd.DataFrame, p: pd.Series, cortes: dict) -> dict:
    """Todas as métricas out-of-fold de um braço."""
    t = P["treino"] & p.notna()
    a = auc_por_mes(P, {"p": p}, ks=KS).set_index("k")["auc"]
    alarme = {f: (p >= cortes[f]) & P["pontuavel"] for f in ("alto", "atencao")}
    topo = precisao_topo(P, p).set_index(["k", "top"])["cancelados"]
    return {
        "log_loss": log_loss(P.loc[t, "y"].to_numpy(), p[t].to_numpy()),
        "auc_linhas": auc(P.loc[t, "y"].to_numpy(), p[t].to_numpy()),
        **{f"auc_k{k}": float(a.get(k, np.nan)) for k in KS},
        "faixa_alto": desempenho(P, alarme["alto"]),
        "faixa_alto_ou_atencao": desempenho(P, alarme["atencao"]),
        "precisao_topo": {f"k{k}_top{n}": int(topo.loc[(k, n)]) for k in KS for n in (10, 20)},
        "fila_perda_esperada": qualidade_fila(P, p),
    }


def por_dobra(P: pd.DataFrame, p: pd.Series) -> dict[str, np.ndarray]:
    """Métricas pareáveis, uma por dobra externa (linhas/clientes de teste da dobra)."""
    saidas: dict[str, list] = {k: [] for k in ("log_loss", "auc_linhas", "auc_k1", "auc_k2", "auc_k3",
                                               "recall_top10pct_fila")}
    for d in sorted(P["dobra"].dropna().unique()):
        g = P[P["dobra"] == d]
        pg = p[g.index]
        t = g["treino"] & pg.notna()
        saidas["log_loss"].append(log_loss(g.loc[t, "y"].to_numpy(), pg[t].to_numpy()) if t.any() else np.nan)
        saidas["auc_linhas"].append(auc(g.loc[t, "y"].to_numpy(), pg[t].to_numpy()) if t.any() else np.nan)
        for k in KS:
            m = (g["k"] == k) & g["pontuavel"] & pg.notna()
            saidas[f"auc_k{k}"].append(auc(g.loc[m, "cancelado"].to_numpy(), pg[m].to_numpy()) if m.any() else np.nan)
        q = _fila(g, (pg * g["valor"] * 12).where(g["valor"].notna()))
        n = max(1, int(round(0.1 * len(q))))
        saidas["recall_top10pct_fila"].append(float(q.head(n)["cancelado"].sum() / max(1, q["cancelado"].sum())))
    return {k: np.asarray(v, dtype=float) for k, v in saidas.items()}


def pareada(a: np.ndarray, b: np.ndarray) -> dict:
    """b − a por dobra (positivo = b maior). `sinal_claro` = |média| > 1 erro-padrão."""
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    md, ep = float(np.nanmean(d)), erro_padrao(d)
    return {"media_a": float(np.nanmean(a)), "media_b": float(np.nanmean(b)), "diferenca": md, "erro_padrao": ep,
            "sinal_claro": bool(not np.isnan(ep) and abs(md) > ep), "por_dobra": [float(x) for x in d]}


# --------------------------------------------------------------------------- braço C e comparação
def braco_c(P: pd.DataFrame, p: pd.Series) -> dict:
    """Delta só na ordenação: p_c = min(p + máx(Δp, 0), 1); a fila continua por p_c × valor × 12."""
    d = P[["cliente", "mes"]].assign(p=p.to_numpy()).sort_values(["cliente", "mes"], kind="stable")
    delta = (d["p"] - d.groupby("cliente", sort=False)["p"].shift(1)).reindex(P.index)
    pc = np.minimum(p + delta.clip(lower=0).fillna(0.0), 1.0)
    return {"regra": "p_c = min(p + máx(Δp, 0), 1); fila por p_c × valor × 12",
            "fila_perda_esperada": qualidade_fila(P, pc),
            "linhas_com_delta": int(delta.notna().sum()), "linhas_no_painel": int(len(P))}


def ler(braco: str) -> tuple[dict, pd.DataFrame]:
    info = json.loads((DESTINO / f"{braco}.json").read_text(encoding="utf-8"))
    P = pd.read_csv(DESTINO / f"{braco}_painel.csv.gz")
    for c in ("cancelado", "treino", "pontuavel", "referencia"):
        P[c] = P[c].astype(bool)
    return info, P


def comparar() -> dict:
    ia, Pa = ler("a")
    ib, Pb = ler("b")
    if not Pa[["cliente", "mes"]].equals(Pb[["cliente", "mes"]]):
        raise SystemExit("Os dois braços não têm o mesmo painel — refaça os dois com a mesma base.")
    if not Pa["dobra"].equals(Pb["dobra"]):
        raise SystemExit("Os dois braços não usaram as mesmas dobras — confira a semente.")
    ma, mb = metricas(Pa, Pa["p_oof"], ia["cortes"]), metricas(Pb, Pb["p_oof"], ib["cortes"])
    da, db = por_dobra(Pa, Pa["p_oof"]), por_dobra(Pb, Pb["p_oof"])
    dif = {k: pareada(da[k], db[k]) for k in da}
    # log-loss por dobra do modelo ESCOLHIDO de cada braço (mesmas dobras nos dois)
    lla = np.asarray(ia["log_loss_dobras"][ia["modelo_escolhido"]], dtype=float)
    llb = np.asarray(ib["log_loss_dobras"][ib["modelo_escolhido"]], dtype=float)
    dif["log_loss_modelo_escolhido"] = pareada(lla, llb)

    # braço P (placebo): delta embaralhado. Mede o ruído de refazer a seleção com 2 candidatas a mais.
    placebo = None
    if (DESTINO / "p.json").exists():
        ip, Pp = ler("p")
        dp = por_dobra(Pp, Pp["p_oof"])
        placebo = {"rotulo": ip["rotulo"], "modelo": ip["modelo_escolhido"], "segundos": ip["segundos"],
                   "cortes": ip["cortes"], "metricas": metricas(Pp, Pp["p_oof"], ip["cortes"]),
                   "candidatas_delta": ip["candidatas_delta"],
                   "diferenca_pareada_por_dobra": {k: pareada(da[k], dp[k]) for k in da}}

    melhora = -dif["log_loss"]["diferenca"]                      # log-loss: menor é melhor
    ep = dif["log_loss"]["erro_padrao"]
    claro = bool(not np.isnan(ep) and abs(melhora) > ep)
    entrou = [v["id"] for v in ib["candidatas_delta"] if v["selecionada"]]
    dobras_delta = {v["id"]: v.get("frequencia_dobras") for v in ib["candidatas_delta"]}
    if not entrou and not any(dobras_delta.values()):
        veredito = ("NÃO: o delta foi descartado pela seleção em todas as dobras e no modelo final — não entra "
                    "em nenhuma previsão. A diferença entre A e B é só o ruído de refazer a seleção com duas "
                    "candidatas a mais (ver o braço P, placebo).")
    else:
        veredito = ("o delta MELHORA o log-loss out-of-fold além do ruído" if melhora > 0 and claro else
                    "o delta PIORA o log-loss out-of-fold além do ruído" if melhora < 0 and claro else
                    "o delta NÃO muda o log-loss out-of-fold além do ruído (diferença dentro de 1 erro-padrão)")
    out = {
        "base": "amostra_redes", "gerado_em": time.strftime("%Y-%m-%d"),
        "a": {"rotulo": ia["rotulo"], "modelo": ia["modelo_escolhido"], "segundos": ia["segundos"],
              "cortes": ia["cortes"], "metricas": ma},
        "b": {"rotulo": ib["rotulo"], "modelo": ib["modelo_escolhido"], "segundos": ib["segundos"],
              "cortes": ib["cortes"], "metricas": mb, "candidatas_delta": ib["candidatas_delta"]},
        "c": {"rotulo": "C — delta só na ordenação (modelo do braço A)", **braco_c(Pa, Pa["p_oof"])},
        "p": placebo,
        "diferenca_pareada_por_dobra": dif,
        "veredito": veredito,
        "veredito_detalhe": {
            "log_loss_a": dif["log_loss"]["media_a"], "log_loss_b": dif["log_loss"]["media_b"],
            "melhora_log_loss": melhora, "erro_padrao": ep, "maior_que_o_ruido": claro,
            "delta_entrou_no_modelo": entrou,
            "delta_fracao_das_dobras": dobras_delta,
            "melhora_do_placebo": (None if placebo is None else
                                   -placebo["diferenca_pareada_por_dobra"]["log_loss"]["diferenca"]),
        },
        "modelos_por_braco": {b: {c["nome"]: {k: c.get(k) for k in ("elegivel", "avaliado", "escolhido",
                                                                    "log_loss", "log_loss_ep", "auc_linhas",
                                                                    "auc_k1", "auc_k2", "auc_k3")}
                                  for c in i["candidatos"]} for b, i in (("a", ia), ("b", ib))},
    }
    gravar(DESTINO / "comparacao.json", out)
    imprimir(out)
    return out


def imprimir(o: dict) -> None:
    f = lambda x, c=4: "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{c}f}".replace(".", ",")  # noqa: E731
    ma, mb = o["a"]["metricas"], o["b"]["metricas"]
    print(f"\n{'métrica':38s} {'A (sem delta)':>15s} {'B (com delta)':>15s}")
    print("-" * 70)
    print(f"{'modelo escolhido':38s} {o['a']['modelo']:>15s} {o['b']['modelo']:>15s}")
    for nome, ch in (("log-loss (linhas de treino)", "log_loss"), ("AUC nas linhas", "auc_linhas"),
                     ("AUC em −1", "auc_k1"), ("AUC em −2", "auc_k2"), ("AUC em −3", "auc_k3")):
        print(f"{nome:38s} {f(ma[ch]):>15s} {f(mb[ch]):>15s}")
    for faixa_nome, ch in (("alto", "faixa_alto"), ("alto+atenção", "faixa_alto_ou_atencao")):
        for k in (1, 2, 3):
            print(f"{f'detecção {faixa_nome} em −{k}':38s} {ma[ch][f'canc_k{k}']:>15d} {mb[ch][f'canc_k{k}']:>15d}")
        print(f"{f'alarme falso {faixa_nome} hoje':38s} {ma[ch]['ativos_hoje']:>15d} {mb[ch]['ativos_hoje']:>15d}")
        print(f"{f'alarme falso {faixa_nome} em 12 meses':38s} {ma[ch]['ativos_algum_mes']:>15d} "
              f"{mb[ch]['ativos_algum_mes']:>15d}")
    for ch in ("k1_top10", "k1_top20"):
        print(f"{'cancelados no ' + ch:38s} {ma['precisao_topo'][ch]:>15d} {mb['precisao_topo'][ch]:>15d}")
    for n in TOPOS:
        print(f"{f'fila (perda esperada): canc. top {n}':38s} "
              f"{ma['fila_perda_esperada'][f'cancelados_top{n}']:>15d} "
              f"{mb['fila_perda_esperada'][f'cancelados_top{n}']:>15d}")
    print("\ndiferença pareada por dobra (B − A) ± erro-padrão:")
    for k, d in o["diferenca_pareada_por_dobra"].items():
        print(f"  {k:32s} {f(d['diferenca']):>10s} ± {f(d['erro_padrao']):<10s} "
              f"{'maior que o ruído' if d['sinal_claro'] else 'dentro do ruído'}")
    print("\nbraço C (delta só na ordenação):")
    for n in TOPOS:
        print(f"  canc. top {n:<4d} A={o['a']['metricas']['fila_perda_esperada'][f'cancelados_top{n}']:<5d} "
              f"C={o['c']['fila_perda_esperada'][f'cancelados_top{n}']}")
    if o.get("p"):
        print("\nbraço P (placebo: delta embaralhado, informação zero) — P − A por dobra:")
        for k, d in o["p"]["diferenca_pareada_por_dobra"].items():
            print(f"  {k:32s} {f(d['diferenca']):>10s} ± {f(d['erro_padrao']):<10s} "
                  f"{'maior que o ruído' if d['sinal_claro'] else 'dentro do ruído'}")
    print(f"\nVEREDITO: {o['veredito']}")
    print(f"  delta no modelo final: {o['veredito_detalhe']['delta_entrou_no_modelo'] or 'não entrou'}")
    print(f"  fração das dobras em que entrou: {o['veredito_detalhe']['delta_fracao_das_dobras']}")
    if o["veredito_detalhe"]["melhora_do_placebo"] is not None:
        print(f"  melhora do log-loss: delta {o['veredito_detalhe']['melhora_log_loss']:+.4f} × "
              f"placebo {o['veredito_detalhe']['melhora_do_placebo']:+.4f}")


# --------------------------------------------------------------------------- referência pequena
def rodar_inova(horizonte: int = 3) -> dict:
    """Mesmo A × B na INOVAAPPS (80 clientes, 22 cancelados) — referência pequena, roda em segundos."""
    from motor import Mapeamento, inspecionar, ler_arquivos, treinar

    tab = ler_arquivos([RAIZ / "INOVAAPPS_base_de_dados.xlsx"])
    m = Mapeamento.de_sugestao(inspecionar(tab))
    m.horizonte_meses = horizonte
    bracos = {}
    for b, flag in BRACOS.items():
        t0 = time.perf_counter()
        r = treinar(tab, m, config=Config(avaliar_todos=True, delta_risco=flag), gerado_em="2026-01-01")
        P = painel_de(r)
        bracos[b] = {
            "rotulo": ROTULO[b], "segundos": round(time.perf_counter() - t0, 1), "modelo": r.escolha.nome,
            "cortes": r.cortes, "metricas": metricas(P, P["p_oof"], r.cortes),
            "candidatas_delta": [{k: v.get(k) for k in ("id", "selecionada", "coef", "auc_univariada",
                                                        "frequencia_selecao", "frequencia_dobras", "motivo_descarte")}
                                 for v in r.pesos["variaveis"] if v["tabela"] == "modelo"],
            "_painel": P,
        }
        print(f"  INOVAAPPS braço {b}: {bracos[b]['segundos']}s · modelo {r.escolha.nome}")
    da, db = por_dobra(bracos["a"]["_painel"], bracos["a"]["_painel"]["p_oof"]), \
        por_dobra(bracos["b"]["_painel"], bracos["b"]["_painel"]["p_oof"])
    out = {"base": "INOVAAPPS", "horizonte_meses": horizonte,
           "a": {k: v for k, v in bracos["a"].items() if k != "_painel"},
           "b": {k: v for k, v in bracos["b"].items() if k != "_painel"},
           "diferenca_pareada_por_dobra": {k: pareada(da[k], db[k]) for k in da}}
    gravar(DESTINO / "inovaapps.json", out)
    d = out["diferenca_pareada_por_dobra"]["log_loss"]
    print(f"  INOVAAPPS log-loss A={d['media_a']:.4f} B={d['media_b']:.4f} "
          f"(B−A {d['diferenca']:+.4f} ± {d['erro_padrao']:.4f})")
    return out


# --------------------------------------------------------------------------- CLI
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--braco", choices=["a", "b", "p", "comparar", "inovaapps", "tudo"], default="tudo")
    ap.add_argument("--horizonte", type=int, default=3)
    ap.add_argument("--min-hist", type=int, default=4)
    ap.add_argument("--delta-dobras", type=int, default=Config.delta_dobras)
    a = ap.parse_args()
    if a.min_hist != Config.min_hist:            # cada min_hist em sua própria pasta (não sobrescreve)
        global DESTINO
        DESTINO = DESTINO / f"hist{a.min_hist}"
    if a.braco == "comparar":
        comparar()
    elif a.braco == "inovaapps":
        rodar_inova(a.horizonte)
    elif a.braco in CONFIGS:
        rodar_braco(a.braco, a.horizonte, a.min_hist, a.delta_dobras)
    else:
        # cada braço em um processo: a base é grande e o pico de memória não se soma
        for b in ("a", "b", "p"):
            print(f"\n===== braço {b} =====", flush=True)
            subprocess.run([sys.executable, __file__, "--braco", b, "--horizonte", str(a.horizonte),
                            "--min-hist", str(a.min_hist), "--delta-dobras", str(a.delta_dobras)], check=True)
        comparar()


if __name__ == "__main__":
    main()
