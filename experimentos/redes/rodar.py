"""Roda o motor na amostra multi-rede (gerada por `preparar.py`) e compara os modelos nas mesmas dobras.

Uso: uv run python experimentos/redes/rodar.py [--horizonte 3] [--min-hist 4] [--delta]
     →  dados/amostra_redes/resultado/h<H>_hist<N>[_delta]/*.json
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

from motor import Mapeamento, ler_arquivos, treinar  # noqa: E402
from motor.config import Config  # noqa: E402

BASE = RAIZ / "dados" / "amostra_redes" / "base_motor"
SAIDA = RAIZ / "dados" / "amostra_redes" / "resultado"


def carregar():
    """Tabelas da base preparada (`preparar.py`)."""
    if not BASE.exists():
        raise SystemExit(f"Base não preparada: rode `uv run python experimentos/redes/preparar.py` ({BASE})")
    return ler_arquivos(sorted(BASE.glob("*.csv")))


def mapeamento(horizonte: int = 3) -> Mapeamento:
    return Mapeamento(
        tabela_clientes="clientes", coluna_id="cliente_id",
        alvo_tabela="situacao", coluna_situacao="situacao", valor_cancelado="Cancelado",
        coluna_data_saida="mes_saida", valor_tabela="clientes", coluna_valor="valor_mensal_medio",
        horizonte_meses=horizonte,
        # média do gasto usa meses posteriores a cada linha do painel → só para a prioridade da fila
        colunas_ignoradas=["clientes.valor_mensal_medio"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizonte", type=int, default=3)
    ap.add_argument("--min-hist", type=int, default=Config.min_hist)
    ap.add_argument("--delta", action="store_true", help="liga a candidata delta de risco (dois estágios)")
    a = ap.parse_args()
    destino = SAIDA / f"h{a.horizonte}_hist{a.min_hist}{'_delta' if a.delta else ''}"

    t0 = time.perf_counter()
    tabelas = carregar()
    for nome, df in tabelas.items():
        print(f"  {nome:15s} {len(df):>9,} linhas")

    def progresso(etapa: str, f: float) -> None:
        print(f"  [{time.perf_counter() - t0:6.1f}s] {100 * f:5.1f}% {etapa}", flush=True)

    cfg = Config(avaliar_todos=True, min_hist=a.min_hist, delta_risco=a.delta)
    r = treinar(tabelas, mapeamento(a.horizonte), progresso=progresso, config=cfg)
    r.salvar(destino)
    print(f"\ntempo total: {time.perf_counter() - t0:.1f}s · resultados em {destino.relative_to(RAIZ)}")
    for av in r.validacao.get("avisos", []):
        print("aviso:", av)
    print("\nescolha:", json.dumps(r.validacao["escolha"], ensure_ascii=False, indent=1))
    print("\nmodelos:", json.dumps(r.validacao["modelos"], ensure_ascii=False, indent=1))
    print("\nvariáveis do modelo final:")
    for v in r.pesos.get("variaveis", []):
        if v.get("selecionada"):
            print(f"  {v.get('rotulo', v.get('id'))}: {v.get('importancia', v.get('coeficiente'))}")


if __name__ == "__main__":
    main()
