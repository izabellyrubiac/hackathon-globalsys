"""Deixa as bases de demonstração prontas para o front: registra, treina uma execução e a ativa.

    uv run python -m api.preparar_demos                 # as duas bases (pula as que já têm execução pronta)
    uv run python -m api.preparar_demos --base redes    # só uma
    uv run python -m api.preparar_demos --forcar        # treina de novo mesmo já tendo execução pronta

Idempotente: sem `--forcar`, uma base que já tem execução pronta é pulada. Tempos observados:
INOVAAPPS ~30 s; redes ~8 min (LightGBM, 2.910 clientes). Não escreve em `web/public/data/`.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from motor import Mapeamento                                            # noqa: E402
from motor.config import Config                                         # noqa: E402

from api import armazenamento as arm                                    # noqa: E402
from api import demo, servico                                           # noqa: E402

BASES = {demo.BASE_ID: demo.arquivos_inovaapps, demo.REDES_ID: demo.arquivos_redes}


def _pronta(base_id: str) -> str | None:
    """Id da primeira execução pronta da base, se houver."""
    for eid in arm.ids_execucoes(base_id):
        if arm.ler_status_execucao(base_id, eid).get("estado") == "pronta":
            return eid
    return None


def preparar_base(base_id: str, forcar: bool = False, rotulo: str | None = None) -> str | None:
    """Registra, treina uma execução e a deixa ativa. Devolve o `execucao_id` (ou None se pulou)."""
    arquivos = BASES[base_id]()
    if not arquivos:
        print(f"[{base_id}] arquivos não encontrados — pulando.")
        return None
    if not arm.existe(base_id):
        print(f"[{base_id}] registrando {len(arquivos)} arquivo(s)…")
    demo.registrar(base_id, arquivos)

    ja = _pronta(base_id)
    if ja and not forcar:
        if arm.ativa(base_id) is None:
            arm.ativar(base_id, ja)
        print(f"[{base_id}] já tem a execução pronta '{ja}' (ativa: {arm.ativa(base_id)}) — pulando. "
              "Use --forcar para treinar de novo.")
        return None

    if arm.ler_json(arm.pasta(base_id) / "inspecao.json") is None:
        t = time.perf_counter()
        print(f"[{base_id}] inspecionando (pode demorar em base grande)…", flush=True)
        servico.inspecionar_base(base_id)
        print(f"[{base_id}] inspeção em {time.perf_counter() - t:.1f}s")

    m = Mapeamento.from_dict(demo.MAPEAMENTOS[base_id])
    cfg = Config()
    opcoes = {"modelo": cfg.modelo, "horizonte_meses": int(m.horizonte_meses), "min_hist": cfg.min_hist,
              "delta_risco": bool(cfg.delta_risco), "validar": cfg.validar, "dobras": cfg.dobras,
              "semente": cfg.semente}
    t0 = time.perf_counter()
    st = servico.disparar_treino(base_id, m, cfg, opcoes, rotulo or "demonstração")
    eid = st["execucao_id"]
    print(f"[{base_id}] execução '{eid}' na fila…", flush=True)

    ultimo = ""
    while True:
        st = arm.ler_status_execucao(base_id, eid)
        marca = f"{st.get('estado')}/{st.get('etapa')}"
        if marca != ultimo:
            ultimo = marca
            print(f"[{base_id}] [{time.perf_counter() - t0:6.1f}s] {100 * float(st.get('fracao') or 0):5.1f}% "
                  f"{st.get('etapa')}", flush=True)
        if st.get("estado") in ("pronta", "erro"):
            break
        time.sleep(1.0)

    if st.get("estado") == "erro":
        print(f"[{base_id}] FALHOU: {st.get('mensagem')}")
        for p in st.get("problemas") or []:
            print(f"    - {p}")
        return None

    arm.ativar(base_id, eid)
    r = arm.resumo_execucao(base_id, eid) or {}
    modelo = (r.get("modelo") or {}).get("rotulo") or (r.get("modelo") or {}).get("nome")
    print(f"[{base_id}] pronta em {time.perf_counter() - t0:.1f}s · execução '{eid}' ativa · modelo {modelo} · "
          f"{r.get('n_clientes')} clientes · {(r.get('variaveis') or {}).get('n_selecionadas')} variáveis")
    for aviso in st.get("avisos") or []:
        print(f"    aviso: {aviso}")
    return eid


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", choices=sorted(BASES), action="append",
                    help="qual demonstração preparar (pode repetir; padrão: todas)")
    ap.add_argument("--forcar", action="store_true", help="treina de novo mesmo já tendo execução pronta")
    ap.add_argument("--rotulo", default=None, help="rótulo da execução criada (padrão: 'demonstração')")
    a = ap.parse_args()

    inicio = time.perf_counter()
    print(f"bases em {arm.raiz_bases()}")
    for base_id in (a.base or sorted(BASES)):
        preparar_base(base_id, forcar=a.forcar, rotulo=a.rotulo)
    print(f"tempo total: {time.perf_counter() - inicio:.1f}s")


if __name__ == "__main__":
    main()
