"""Tira o que ainda identifica cliente e rede da base `dados/bases/redes/`, no lugar.

A base já veio sem nome, razão social, documento e emitente (ver `experimentos/redes/preparar.py`).
O que sobrou identificando é a **chave**: `cliente_id` = `<rede_id md5>-<cliente_codigo>`, e o
`cliente_codigo` é o código do cliente no ERP do posto (para parte deles, o próprio CPF/CNPJ).
Aqui `cliente_id` vira `c0001…` e `rede_id` vira `rede_01…`, sorteados com semente aleatória de
verdade e **sem gravar o mapa em disco** — é isso que torna a troca irreversível.

Nenhuma outra coluna é criada ou removida: valor, data, hora, categoria, forma, prazo, margem,
tipo_pessoa e o resto do comportamento ficam como estão, inclusive o de sinal fraco.

Os CSV da base são hardlinks para `dados/amostra_redes/base_motor/*.csv`: por isso a gravação é em
arquivo temporário + `os.replace`, que troca a entrada do diretório e deixa a origem intacta.

Uso: uv run python limpar_base.py
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent
BASE = RAIZ / "dados" / "bases" / "redes"
ARQUIVOS = BASE / "arquivos"
PEDACO = 500_000                      # linhas por pedaço na leitura dos CSV grandes
JA_ANONIMO = re.compile(r"^c\d+$")
# `<rede_id md5>-<cliente_codigo>` ou só o `rede_id`: a forma antiga da chave, dentro ou fora de texto
CHAVE_ANTIGA = re.compile(r"[0-9a-f]{8}-\d+|\b[0-9a-f]{8}\b")
SEM_MAPA = "id removido"        # resíduo de execução anterior, sem como remapear
# lista de resíduos numa frase (com ou sem o parêntese que uma passada anterior deixou)
LISTA_SEM_MAPA = re.compile(rf"\(?{re.escape(SEM_MAPA)}\)?(, \(?{re.escape(SEM_MAPA)}\)?)+")


def ler_mapa() -> tuple[dict[str, str], dict[str, str]]:
    """`cliente_id` → `c0001…` e `rede_id` → `rede_01…`, em ordem sorteada. Só em memória."""
    cad = pd.read_csv(ARQUIVOS / "clientes.csv", usecols=["cliente_id", "rede_id"], dtype="string")
    clientes = sorted(cad["cliente_id"].dropna().unique())
    redes = sorted(cad["rede_id"].dropna().unique())
    if all(JA_ANONIMO.match(c) for c in clientes):
        return {}, {}

    sorteio = random.SystemRandom()
    sorteio.shuffle(clientes)
    sorteio.shuffle(redes)
    largura_c = max(4, len(str(len(clientes))))
    largura_r = max(2, len(str(len(redes))))
    return ({c: f"c{i:0{largura_c}d}" for i, c in enumerate(clientes, 1)},
            {r: f"rede_{i:0{largura_r}d}" for i, r in enumerate(redes, 1)})


def trocar_csv(caminho: Path, mapa_cliente: dict[str, str], mapa_rede: dict[str, str]) -> int:
    """Reescreve o CSV trocando `cliente_id`/`rede_id`. `dtype=str` para não reformatar número."""
    temporario = caminho.with_suffix(".csv.novo")
    linhas = 0
    with pd.read_csv(caminho, dtype=str, chunksize=PEDACO, keep_default_na=False) as pedacos:
        for i, pedaco in enumerate(pedacos):
            if "cliente_id" in pedaco.columns:
                pedaco["cliente_id"] = pedaco["cliente_id"].map(lambda v: mapa_cliente.get(v, v))
            if "rede_id" in pedaco.columns:
                pedaco["rede_id"] = pedaco["rede_id"].map(lambda v: mapa_rede.get(v, v))
            pedaco.to_csv(temporario, index=False, header=(i == 0), mode="w" if i == 0 else "a")
            linhas += len(pedaco)
    os.replace(temporario, caminho)   # quebra o hardlink: a origem em amostra_redes/ não é tocada
    return linhas


def trocar_texto(texto: str, mapa_cliente: dict[str, str], mapa_rede: dict[str, str]) -> str:
    """Troca toda chave antiga no texto — os `avisos` citam cliente dentro de frase, não só em campo.

    Chave que não está no mapa veio de execução anterior (a base já foi trocada e o mapa não é
    guardado): some, para nenhum código de ERP sobrar. Token que não é chave fica intacto.
    """
    def troca(m: re.Match) -> str:
        bruto = m.group(0)
        if bruto in mapa_cliente:
            return mapa_cliente[bruto]
        if bruto in mapa_rede:
            return mapa_rede[bruto]
        return SEM_MAPA if "-" in bruto else bruto

    # aviso cita vários clientes em sequência: uma lista de resíduos vira uma menção só
    return LISTA_SEM_MAPA.sub("ids removidos", CHAVE_ANTIGA.sub(troca, texto))


def trocar_json(valor, mapa_cliente: dict[str, str], mapa_rede: dict[str, str]):
    """Percorre o JSON trocando as chaves nos valores string (nome de campo fica intacto)."""
    if isinstance(valor, dict):
        return {k: trocar_json(v, mapa_cliente, mapa_rede) for k, v in valor.items()}
    if isinstance(valor, list):
        return [trocar_json(v, mapa_cliente, mapa_rede) for v in valor]
    if isinstance(valor, str):
        return trocar_texto(valor, mapa_cliente, mapa_rede)
    return valor


def main() -> None:
    t0 = time.perf_counter()
    mapa_cliente, mapa_rede = ler_mapa()
    tamanhos = {}
    if mapa_cliente:
        print(f"{len(mapa_cliente):,} clientes e {len(mapa_rede)} redes a remapear\n")
        for caminho in sorted(ARQUIVOS.glob("*.csv")):
            linhas = trocar_csv(caminho, mapa_cliente, mapa_rede)
            tamanhos[caminho.name] = caminho.stat().st_size
            print(f"  {caminho.stem:15s} {linhas:>9,} linhas")
    else:
        print("CSV já anonimizados — só a varredura dos JSON.")

    jsons = [BASE / "inspecao.json"]
    jsons += sorted(p for p in BASE.glob("execucoes/*/*.json"))
    tocados = 0
    for caminho in jsons:
        if not caminho.exists():
            continue
        antes = caminho.read_text(encoding="utf-8")
        depois = json.dumps(trocar_json(json.loads(antes), mapa_cliente, mapa_rede),
                            ensure_ascii=False, indent=2, sort_keys=True)
        if json.loads(depois) != json.loads(antes):
            caminho.write_text(depois + "\n", encoding="utf-8")
            tocados += 1
    print(f"\n  {tocados} de {len(jsons)} JSON reescritos (inspeção e execuções)")

    if not tamanhos:
        print(f"\ntempo: {time.perf_counter() - t0:.1f}s")
        return

    meta_caminho = BASE / "base.json"
    meta = json.loads(meta_caminho.read_text(encoding="utf-8"))
    for item in meta.get("arquivos", []):
        item["bytes"] = tamanhos.get(item["nome"], item["bytes"])
    meta["bytes_total"] = sum(item["bytes"] for item in meta.get("arquivos", []))
    meta_caminho.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"  base.json: {meta['bytes_total'] / 1e6:.1f} MB")
    print(f"\ncliente_id trocado por chave surrogate · tempo: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
