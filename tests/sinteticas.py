"""Bases sintéticas (semente fixa) para os testes de modelos e de escala. Nada aqui é da INOVAAPPS."""

from __future__ import annotations

import numpy as np
import pandas as pd

MESES = pd.period_range("2024-01", "2025-12", freq="M")      # 24 meses


def _tabelas(ids, saida, linhas, rng) -> dict[str, pd.DataFrame]:
    clientes = pd.DataFrame({"codigo": ids, "regiao": rng.choice(["Norte", "Sul", "Leste"], len(ids)),
                             "mrr": rng.integers(500, 5000, len(ids)).astype(float)})
    situ = pd.DataFrame({"codigo": ids, "churn": [int(c in saida) for c in ids],
                         "data_saida": [str(saida[c]) if c in saida else None for c in ids]})
    return {"clientes": clientes, "situacao": situ, "uso_mensal": pd.DataFrame(linhas)}


def base_nao_linear(n: int = 3000, semente: int = 11) -> dict[str, pd.DataFrame]:
    """Risco = uso baixo E chamados altos (interação que a logística não captura; monotônica nos dois).
    Uso e chamados andam em passeio aleatório independente por cliente; a saída é sorteada todo mês com
    probabilidade σ(−6 + 5,5·[uso 3m < 35 e chamados 3m > 7]); o histórico acaba no mês anterior à saída."""
    rng = np.random.default_rng(semente)
    ids = [f"N{i:05d}" for i in range(n)]
    linhas, saida = [], {}
    for c in ids:
        uso, cham = rng.uniform(15, 90), rng.uniform(0, 13)
        hu, hc = [], []
        for j, m in enumerate(MESES):
            uso = float(np.clip(uso + rng.normal(0, 4), 0, 100))
            cham = float(np.clip(cham + rng.normal(0, 0.8), 0, 25))
            u, ch = uso + rng.normal(0, 3), float(rng.poisson(cham))
            hu.append(u)
            hc.append(ch)
            linhas.append({"codigo": c, "mes": str(m), "uso": round(u, 1), "chamados": ch,
                           "ruido_a": round(rng.normal(50, 10), 1), "ruido_b": float(rng.poisson(4))})
            if j >= 5 and j < len(MESES) - 1:
                perigo = np.mean(hu[-3:]) < 35 and np.mean(hc[-3:]) > 7
                if rng.random() < 1 / (1 + np.exp(-(-6 + 5.5 * perigo))):
                    saida[c] = MESES[j + 1]
                    break
    return _tabelas(ids, saida, linhas, rng)


def base_larga(n: int = 300, n_ruido: int = 400, semente: int = 5) -> dict[str, pd.DataFrame]:
    """300 clientes × 24 meses, 400 colunas de ruído e 3 de sinal (uma cai, uma sobe e uma fica baixa antes
    da saída). ~30% cancelam."""
    rng = np.random.default_rng(semente)
    ids = [f"L{i:04d}" for i in range(n)]
    canc = sorted(rng.choice(ids, int(0.3 * n), replace=False))     # ordenado: não depende do hash do Python
    saida = {c: MESES[int(rng.integers(8, len(MESES)))] for c in canc}
    linhas = []
    ruido_base = rng.normal(0, 1, (n, n_ruido))
    for i, c in enumerate(ids):
        b1, b2, b3 = rng.uniform(40, 80), rng.uniform(2, 6), rng.uniform(60, 90)
        for m in MESES:
            if c in saida and m >= saida[c]:
                break
            k = (saida[c] - m).n if c in saida else 99
            perto = max(0, 5 - k)                      # 4, 3, 2, 1 nos 4 meses antes da saída
            reg = {"codigo": c, "mes": str(m),
                   "sinal_uso": round(b1 - 7 * perto + rng.normal(0, 5), 2),
                   "sinal_chamados": round(b2 + 1.5 * perto + rng.normal(0, 1), 2),
                   "sinal_nota": round(b3 - 8 * perto + rng.normal(0, 6), 2)}
            ruido = ruido_base[i] + rng.normal(0, 1, n_ruido)
            reg.update({f"r{j:03d}": round(float(v), 2) for j, v in enumerate(ruido)})
            linhas.append(reg)
    return _tabelas(ids, saida, linhas, rng)
