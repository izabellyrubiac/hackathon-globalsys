"""Gera os JSON estáticos do app a partir do motor genérico com o mapeamento da INOVAAPPS (`score.rodar`).

Uso:  uv run python analise/gerar_json.py

Saídas em `web/public/data/` — contrato documentado em `motor/saida.py`:
* clientes.json   fila dos ativos (ordem = perda anual esperada), evidências, ação e histórico mensal;
* validacao.json  validação out-of-fold agrupada por cliente com seleção aninhada;
* pesos.json      todas as variáveis candidatas: selecionadas (coeficiente, frequência) e descartadas (motivo).
JSON determinístico (chaves ordenadas, sem NaN). Só `gerado_em` muda de um dia para o outro
(fixe com GERADO_EM=AAAA-MM-DD).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score import rodar  # noqa: E402

SAIDA = Path(__file__).resolve().parent.parent / "web" / "public" / "data"


def verificar(r) -> None:
    doc = r.clientes
    cl = doc["clientes"]
    n_base = r.validacao["painel"]["clientes"]
    assert n_base == 80, n_base
    assert len(cl) == 58 == doc["resumo"]["clientes_ativos"]
    assert sorted(c["prioridade"] for c in cl) == list(range(1, 59))
    assert all(c["faixa_risco"] in ("alto", "atencao", "baixo") for c in cl)
    assert all(0 <= c["risco"] <= 1 for c in cl)
    assert all(c["evidencias"] for c in cl if c["faixa_risco"] == "alto"), "cliente alto sem evidência"
    perdas = [c["perda_anual_esperada"] for c in cl]
    assert perdas == sorted(perdas, reverse=True), "fila fora da ordem de perda anual esperada"
    for obj in (r.clientes, r.validacao, r.pesos):
        json.dumps(obj, allow_nan=False)          # falha se sobrou NaN/inf


def main() -> None:
    r = rodar(progresso=lambda etapa, f: print(f"  [{100 * f:3.0f}%] {etapa}", flush=True))
    verificar(r)
    caminhos = r.salvar(SAIDA)
    res = r.clientes["resumo"]
    sel = [v["id"] for v in r.pesos["variaveis"] if v["selecionada"]]
    print(f"{caminhos['clientes']}: {res['clientes_ativos']} clientes · {res['em_risco_alto']} alto · "
          f"{res['em_atencao']} atenção · receita em risco R$ {res['receita_em_risco_mensal']:,.0f}/mês".replace(",", "."))
    print(f"{caminhos['validacao']}: cortes alto ≥ {r.cortes['alto']:.2f}, atenção ≥ {r.cortes['atencao']:.2f}")
    print(f"{caminhos['pesos']}: {len(sel)} de {len(r.pesos['variaveis'])} variáveis selecionadas: {', '.join(sel)}")


if __name__ == "__main__":
    main()
