"""Base sintética minúscula (semente fixa) gerada no próprio teste: 3 CSV em memória.

Nada da INOVAAPPS aqui. 40 clientes × 12 meses; 12 cancelam e o uso cai nos meses antes da saída, então o
motor tem o que aprender sem demorar.
"""

from __future__ import annotations

import io
import random

MESES = [f"2025-{m:02d}" for m in range(1, 13)]


def csvs(n: int = 40, n_canc: int = 12, semente: int = 7) -> dict[str, bytes]:
    """{nome do arquivo: conteúdo}. Tabelas: clientes, situacao, uso."""
    rng = random.Random(semente)
    ids = [f"C{i:03d}" for i in range(n)]
    canc = sorted(rng.sample(ids, n_canc))
    saida = {c: rng.randint(7, len(MESES) - 1) for c in canc}   # índice do mês de saída (histórico vai até o anterior)

    cli = io.StringIO()
    cli.write("codigo,regiao,mrr\n")
    for c in ids:
        cli.write(f"{c},{rng.choice(['Norte', 'Sul', 'Leste'])},{rng.randrange(500, 5000, 50)}\n")

    sit = io.StringIO()
    sit.write("codigo,situacao,mes_saida\n")
    for c in ids:
        sit.write(f"{c},Cancelado,{MESES[saida[c]]}\n" if c in saida else f"{c},Ativo,\n")

    uso = io.StringIO()
    uso.write("codigo,mes,uso,chamados,ruido\n")
    for c in ids:
        base_uso, base_ch = rng.uniform(55, 90), rng.uniform(0.5, 3.0)
        fim = saida.get(c, len(MESES))
        for j, mes in enumerate(MESES[:fim]):
            perto = max(0, 4 - (fim - j))                       # 3, 2, 1 nos meses antes da saída
            u = max(0.0, base_uso - 16 * perto + rng.gauss(0, 4))
            ch = max(0.0, base_ch + 2.2 * perto + rng.gauss(0, 0.6))
            uso.write(f"{c},{mes},{u:.1f},{ch:.1f},{rng.gauss(50, 10):.1f}\n")

    return {"clientes.csv": cli.getvalue().encode(), "situacao.csv": sit.getvalue().encode(),
            "uso.csv": uso.getvalue().encode()}


def arquivos(conteudo: dict[str, bytes] | None = None) -> list[tuple[str, tuple[str, bytes, str]]]:
    """No formato do `files=` do TestClient."""
    return [("arquivos", (nome, dados, "text/csv")) for nome, dados in (conteudo or csvs()).items()]


MAPEAMENTO = {
    "tabela_clientes": "clientes",
    "coluna_id": "codigo",
    "alvo_tabela": "situacao",
    "coluna_situacao": "situacao",
    "valor_cancelado": "Cancelado",
    "coluna_data_saida": "mes_saida",
    "valor_tabela": "clientes",
    "coluna_valor": "mrr",
    "horizonte_meses": 3,
}
