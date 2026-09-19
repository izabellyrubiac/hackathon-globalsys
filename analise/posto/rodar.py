"""Roda o motor na base real do posto (gerada por `preparar.py`) e compara os modelos nas mesmas dobras.

Uso: uv run python analise/posto/rodar.py [--horizonte 3] [--min-hist 4]
     →  amostra_1243d7ed/resultado/h<H>_hist<N>/*.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from motor import Mapeamento, inspecionar, ler_arquivos, treinar  # noqa: E402
from motor.config import Config  # noqa: E402
BASE = RAIZ / "amostra_1243d7ed" / "base_motor"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizonte", type=int, default=3)
    ap.add_argument("--min-hist", type=int, default=Config.min_hist, help="meses de histórico para a linha treinar")
    a = ap.parse_args()
    saida = RAIZ / "amostra_1243d7ed" / "resultado" / f"h{a.horizonte}_hist{a.min_hist}"

    t0 = time.perf_counter()
    tabelas = ler_arquivos(sorted(BASE.glob("*.csv")))
    insp = inspecionar(tabelas)
    print("sugestão do motor:", json.dumps(insp["sugestao"], ensure_ascii=False))
    for t in insp["tabelas"]:
        print(f"  {t['nome']:15s} {t['linhas']:>7,} linhas · papel sugerido: {t['papel_sugerido']}")

    m = Mapeamento(tabela_clientes="clientes", coluna_id="cliente_codigo",
                   alvo_tabela="situacao", coluna_situacao="situacao", valor_cancelado="Cancelado",
                   coluna_data_saida="mes_saida", valor_tabela="clientes", coluna_valor="valor_mensal_medio",
                   horizonte_meses=a.horizonte,
                   # média do gasto usa meses posteriores a cada linha do painel → só para a prioridade
                   colunas_ignoradas=["clientes.valor_mensal_medio"])

    def progresso(etapa: str, f: float) -> None:
        print(f"  [{time.perf_counter() - t0:6.1f}s] {100 * f:5.1f}% {etapa}", flush=True)

    r = treinar(tabelas, m, progresso=progresso, config=Config(avaliar_todos=True, min_hist=a.min_hist))
    r.salvar(saida)
    dt = time.perf_counter() - t0

    print(f"\ntempo total: {dt:.1f}s · resultados em {saida.relative_to(RAIZ)}")
    for av in r.clientes.get("avisos", []) or r.validacao.get("avisos", []):
        print("aviso:", av)
    print("\nescolha:", json.dumps(r.validacao["escolha"], ensure_ascii=False, indent=1))
    print("\nmodelos:", json.dumps(r.validacao["modelos"], ensure_ascii=False, indent=1))
    print("\nvariáveis do modelo final:")
    for v in r.pesos.get("variaveis", []):
        if v.get("selecionada"):
            print(f"  {v.get('rotulo', v.get('id'))}: {v.get('importancia', v.get('coeficiente'))}")
    print("\nfila (top 10):")
    for c in r.clientes["clientes"][:10]:
        ev = "; ".join(e["titulo"] for e in c["evidencias"][:3])
        print(f"  {c['prioridade']:>3} {c['cliente_id']:>14} p={c['risco']:.3f} {c['faixa_risco']:8s} {ev}")


if __name__ == "__main__":
    main()
