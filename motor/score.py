"""Score da fila (opcional): média ponderada de risco, valor do contrato e variação do risco.

    score = 100 · (p1·risco + p2·valor + p3·delta) ÷ (p1 + p2 + p3)

Cada item vai de 0 a 1 dentro da fila:
* risco  = probabilidade de cancelar;
* valor  = valor mensal ÷ maior valor da fila (sem coluna de valor, o peso do valor é ignorado);
* delta  = `delta_risco` reescalado entre a maior queda (0) e a maior alta (1), com o "sem variação" (0) sempre dentro da
           escala; sem mês anterior conta como "sem variação".

Sem `Config.pesos_score` o motor mantém a ordem de sempre (perda anual ajustada). Com pesos, a fila é ordenada por este
score (empate: maior risco). Tudo aqui é função pura sobre os números que já estão no `clientes.json`, então dá para
refazer a ordem de uma versão pronta (`reordenar`) sem treinar de novo.
"""

from __future__ import annotations

import numpy as np

CHAVES = ("risco", "valor", "delta")
ROTULOS = {"risco": "risco de cancelar", "valor": "valor mensal", "delta": "variação do risco"}


def validar_pesos(pesos) -> dict[str, float] | None:
    """Pesos normalizados em {risco, valor, delta} (chave ausente = 0); None/vazio = sem score. ValueError em pt-BR."""
    if not pesos:
        return None
    if not isinstance(pesos, dict):
        raise ValueError("Os pesos do score devem ser um objeto {risco, valor, delta}.")
    extras = sorted(set(pesos) - set(CHAVES))
    if extras:
        raise ValueError(f"Peso desconhecido: {', '.join(extras)}. Use {', '.join(CHAVES)}.")
    out = {}
    for k in CHAVES:
        try:
            v = float(pesos.get(k, 0.0) or 0.0)
        except (TypeError, ValueError):
            raise ValueError(f"O peso '{k}' precisa ser um número.") from None
        if not np.isfinite(v) or v < 0:
            raise ValueError(f"O peso '{k}' precisa ser um número de 0 em diante.")
        out[k] = v
    if sum(out.values()) <= 0:
        raise ValueError("Ao menos um peso do score precisa ser maior que 0.")
    return out


def pontuar(risco, valor, delta, pesos: dict[str, float]) -> np.ndarray:
    """Score de 0 a 100 por cliente. `valor` None/NaN = sem coluna de valor; `delta` NaN = sem variação."""
    r = np.asarray(risco, dtype=float)
    p = validar_pesos(pesos)
    tem_valor = valor is not None and np.isfinite(np.asarray(valor, dtype=float)).any()
    peso_valor = p["valor"] if tem_valor else 0.0
    soma = p["risco"] + peso_valor + p["delta"]
    if soma <= 0:                       # só pesava o valor e ele não existe
        return 100.0 * r
    v = np.zeros_like(r)
    if tem_valor:
        va = np.nan_to_num(np.asarray(valor, dtype=float), nan=0.0)
        v = va / va.max() if va.max() > 0 else v
    d = np.nan_to_num(np.asarray(delta, dtype=float), nan=0.0)
    dmin, dmax = min(d.min(), 0.0) if len(d) else 0.0, max(d.max(), 0.0) if len(d) else 0.0
    dn = (d - dmin) / (dmax - dmin) if dmax > dmin else np.zeros_like(d)
    return 100.0 * (p["risco"] * r + peso_valor * v + p["delta"] * dn) / soma


def formula(pesos: dict[str, float], tem_valor: bool) -> str:
    p = dict(pesos)
    if not tem_valor:
        p["valor"] = 0.0
    soma = sum(p.values()) or 1.0
    partes = [f"{100 * p[k] / soma:.0f}% {ROTULOS[k]}" for k in CHAVES if p[k] > 0]
    return "score de 0 a 100 = " + " + ".join(partes) + " (cada item em escala de 0 a 100% dentro da fila)"


def reordenar(doc: dict, pesos) -> dict:
    """Refaz `score`, `prioridade` e `modelo.formula_prioridade` de um `clientes.json` já pronto.

    `pesos` None/vazio volta à ordem padrão do treino (perda anual ajustada; sem valor, risco ajustado)."""
    p = validar_pesos(pesos)
    cl = doc.get("clientes", [])
    modelo = doc.setdefault("modelo", {})
    tem_valor = any(c.get("valor_mensal") is not None for c in cl)
    nan = lambda x: np.nan if x is None else x  # noqa: E731
    if p:
        sc = pontuar([c["risco"] for c in cl], [nan(c.get("valor_mensal")) for c in cl] if tem_valor else None,
                     [nan(c.get("delta_risco")) for c in cl], p) if cl else []
        for c, s in zip(cl, sc):
            c["score"] = round(float(s), 1)
        cl.sort(key=lambda c: (-c["score"], -c["risco"], str(c["cliente_id"])))
        modelo["formula_prioridade"] = formula(p, tem_valor)
    else:
        chave = "perda_anual_ajustada" if tem_valor else "risco_ajustado"
        for c in cl:
            c["score"] = None
        cl.sort(key=lambda c: (-(c.get(chave) if c.get(chave) is not None else -1.0), -c["risco"], str(c["cliente_id"])))
        modelo["formula_prioridade"] = modelo.get("formula_prioridade_padrao", modelo.get("formula_prioridade"))
    for i, c in enumerate(cl, 1):
        c["prioridade"] = i
    doc["pesos_score"] = p
    return doc
