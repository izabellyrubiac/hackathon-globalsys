"""Score de risco da INOVAAPPS = motor genérico (`motor/`) com o mapeamento desta base.

Uso:
    from score import rodar
    r = rodar()                  # motor.Resultado: r.fila, r.explicacao, r.painel, r.modelo, r.pesos, ...

Tudo o que é específico da INOVAAPPS fica aqui (e só aqui):
* tabela de clientes `clientes` (ID `cliente_id`); alvo em `situacao_clientes`: situacao == "Cancelado",
  mês da saída em `mes_cancelamento` (a tabela inteira fica fora das variáveis);
* valor do contrato `clientes.valor_mensal` → prioridade = probabilidade × valor_mensal × 12;
* nomes legíveis das colunas (`dados.ROTULOS`), acentos de segmento/porte/plano e o dicionário de ações.
Variáveis, pesos, limiares, direções de risco e cortes das faixas são aprendidos pelo motor.
"""

from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from dados import _ACENTOS, ARQUIVO, ROTULOS  # noqa: E402
from motor import Config, Mapeamento, Resultado, ler_arquivos, treinar  # noqa: E402

HORIZONTE = 3

_ESTABILIZAR = "Montar plano técnico de estabilização (tempo de resolução, SLA e chamados críticos)"
_NPS = "Entrevista de escuta sobre a experiência (NPS)"
ACOES = {
    "uso_plataforma_pct": "Agendar revisão de valor e adoção da plataforma com o decisor",
    "reunioes_realizadas": "Agendar revisão de valor e adoção da plataforma com o decisor",
    "reunioes_previstas": "Agendar revisão de valor e adoção da plataforma com o decisor",
    "tempo_medio_resolucao_h": _ESTABILIZAR,
    "pct_sla_cumprido": _ESTABILIZAR,
    "chamados_criticos": _ESTABILIZAR,
    "chamados_abertos": _ESTABILIZAR,
    "chamados_reabertos": _ESTABILIZAR,
    "chamados_dentro_sla": _ESTABILIZAR,
    "reclamacoes_formais": "Contato do gestor da conta para tratar as reclamações formais",
    "dias_atraso_pagamento": "Alinhamento financeiro sobre pagamentos em atraso",
    "nota_nps": _NPS,
    "respondeu": _NPS,
    "classificacao_nps": _NPS,
}

MAPEAMENTO = Mapeamento(
    tabela_clientes="clientes", coluna_id="cliente_id",
    alvo_tabela="situacao_clientes", coluna_situacao="situacao", valor_cancelado="Cancelado",
    coluna_data_saida="mes_cancelamento",
    valor_tabela="clientes", coluna_valor="valor_mensal",
    horizonte_meses=HORIZONTE, acoes=ACOES,
    rotulos={**ROTULOS, "respondeu": "Respondeu o NPS"},
)


def tabelas(arquivo: str | Path = ARQUIVO):
    """Abas da planilha (documentação ignorada) com os acentos de segmento/porte/plano."""
    t = ler_arquivos([arquivo])
    for c in ("segmento", "porte", "plano"):
        t["clientes"][c] = t["clientes"][c].replace(_ACENTOS)
    return t


def rodar(arquivo: str | Path = ARQUIVO, progresso=None, config: Config | None = None,
          gerado_em: str | None = None) -> Resultado:
    """Treina o motor na base INOVAAPPS (validação aninhada incluída) e devolve o `motor.Resultado`."""
    return treinar(tabelas(arquivo), MAPEAMENTO, progresso=progresso, config=config, gerado_em=gerado_em)
