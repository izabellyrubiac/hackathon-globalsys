# INOVAAPPS 2026 — Previsão de cancelamento (Globalsys)

## Desafio
Carteira de 80 clientes com contrato recorrente, 18 meses de histórico (jan/2025 a jun/2026). 22 clientes cancelaram e 58 continuam ativos.
A solução precisa apontar, **entre os ativos**, quais correm risco de cancelar e responder a três perguntas:
**com quem falar, por que cada um e em que ordem**. O resultado é uma fila de atendimento, não só uma nota por cliente.

A banca avalia três critérios:
1. **Antecedência**: com quantos meses o sinal aparece antes da saída.
2. **Separação**: evitar alarme falso em quem permaneceu.
3. **Valor em jogo**: risco alto em contrato pequeno é diferente de risco médio em contrato grande.

Documentos de origem: `Desafio - INOVAAPPS 2026.pdf` e `INOVAAPPS_base_de_dados.xlsx`.

## Dados (`INOVAAPPS_base_de_dados.xlsx`)
| Aba | Linhas | Colunas |
|---|---|---|
| `clientes` | 80 | cliente_id, segmento, porte, plano, valor_mensal, sla_contratado_h, inicio_contrato |
| `atendimento_mensal` | 1295 | cliente_id, mes_ref (AAAA-MM), chamados_abertos, chamados_criticos, chamados_reabertos, chamados_dentro_sla, pct_sla_cumprido, tempo_medio_resolucao_h, reclamacoes_formais, uso_plataforma_pct, dias_atraso_pagamento, reunioes_previstas, reunioes_realizadas |
| `pesquisas_nps` | 422 | cliente_id, mes_ref, respondeu, nota_nps, classificacao_nps (Promotor/Neutro/Detrator/Sem resposta) |
| `situacao_clientes` | 80 | cliente_id, situacao (Ativo/Cancelado), mes_cancelamento |

- O histórico de um cliente cancelado termina no **mês anterior** ao `mes_cancelamento` (conferido nos 22 casos): o mês da saída nunca tem dados.
- `mes_relativo` (em `analise/dados.py`): cancelados = `mes_ref − mes_cancelamento`; ativos alinhados a jul/2026. Assim, −1 é o último mês com dados nos dois grupos (jun/2026 para os ativos).
- `carregar()` acentua segmento, porte e plano (Logística, Médio, Avançado); a planilha crua não tem acentos.
- **Deixar de responder o NPS é comportamento, não dado faltante.**

## Stack
| Camada | Tecnologia | Pasta |
|---|---|---|
| Motor genérico (features, seleção supervisionada, explicação, orquestração) | Python, pandas, scikit-learn | `motor/` |
| Modelos de previsão (um arquivo por modelo: logística, Random Forest, LightGBM) | scikit-learn, LightGBM | `motor/modelos_previsao/` |
| Avaliação e escolha dos modelos (um arquivo por método: validação cruzada, métricas, escolha) | Python, pandas | `motor/modelos_classificacao/` |
| Análise da base INOVAAPPS | Jupyter (Plotly), usa o `motor/` | `analise/` |
| API (upload da base, mapeamento, treino, resultados) | FastAPI | `api/` |
| Frontend | Next.js + React + TypeScript | `web/` |
| Visual | Tailwind + shadcn/ui + Recharts | `web/` |

O ambiente Python usa **uv** (`uv sync`, `uv run ...`). Não existe banco de dados: as bases enviadas e os resultados ficam em arquivos (`dados/bases/<id>/`).

### Sistema genérico (qualquer base)
- O usuário envia a base pela tela (um .xlsx com várias abas ou vários .csv).
- Na tela ele indica: a **tabela de clientes** e a coluna que identifica o cliente; a **coluna de cancelamento** (situação e mês da saída — obrigatória, é o rótulo do aprendizado supervisionado); e, se houver, a **coluna de valor do contrato** (para a prioridade). O sistema sugere tudo automaticamente; o usuário confirma.
- Todas as outras tabelas/colunas viram candidatas a variável. O `motor/` gera as variáveis, **escolhe sozinho quais pesam** (seleção supervisionada com validação agrupada por cliente), treina o modelo e devolve a fila explicada.
- **Escolha automática do modelo**: logística, Random Forest e LightGBM são validados nas mesmas dobras agrupadas por cliente; fica o de menor log-loss fora da amostra (um mais complexo só vence por mais de 1 erro-padrão). As árvores só são elegíveis com ≥ 200 meses-cliente positivos e ≥ 50 cancelados (na INOVAAPPS fica a logística). Opção avançada: `Config.modelo` força um modelo.
- A tabela/colunas do cancelamento **nunca** entram como variável (vazamento).
- A INOVAAPPS é só mais uma base: `analise/score.py` usa o `motor/` com o mapeamento dela.

## Especialistas (`.claude/agents/`)
- `analista-dados`: EDA, motor genérico (variáveis, seleção supervisionada, score) e validação com histórico.
- `engenheiro-dados`: leitura de bases arbitrárias, inferência de esquema e o contrato do JSON de resultado.
- `dev-backend`: API FastAPI em `api/` (upload, mapeamento, treino, resultados).
- `dev-frontend`: app Next.js, páginas (upload/mapeamento, fila, cliente, validação), tipos e chamadas à API.
- `designer-ui`: visual, componentes shadcn, gráficos Recharts e acessibilidade.

## Convenções
- Todo o produto fica em **português (pt-BR)**: interface, nomes de colunas e de campos do JSON. Valores usam `R$ 12.345` e datas usam `jun/2026`.
- Cores de status fixas em todo o projeto: **Cancelou/risco alto = vermelho**, **atenção = âmbar**, **ativo/saudável = azul-acinzentado**.
- Notebooks têm quase nenhum texto. Os gráficos precisam se explicar sozinhos, com título descritivo, eixos nomeados e legenda.
- **Fila** (`motor/`): ordenada **só pela perda anual esperada** (probabilidade × valor do contrato × 12; sem coluna de valor, só a probabilidade) — decisão do time; o frontend não reordena por faixa. Faixas (alto/atencao/baixo) vêm de cortes de probabilidade, **nunca** de limite de capacidade da equipe; todos os ativos entram na fila. A tela deve mostrar a faixa ao lado da posição.
- **A IA não faz conferência visual**: não renderiza gráficos em PNG (kaleido etc.), não tira screenshots e não abre navegador para conferir telas. A verificação é só executar o código sem erros e checar os números; a revisão visual fica com o time.
