# Previsão de cancelamento — Desafio INOVAAPPS 2026 (Globalsys)

Quem abre a solução precisa responder a três perguntas sobre uma carteira de clientes:
**com quem falar, por que cada um e em que ordem.**

O sistema recebe **qualquer base** com uma tabela de clientes e tabelas de valores ligados a eles,
descobre sozinho quais sinais antecedem o cancelamento, treina um modelo, e devolve uma **fila de
atendimento explicada**: o risco de cada cliente, as variáveis que estão disparando e a ação sugerida.

## Como está organizado

| Pasta | O que é |
|---|---|
| `motor/` | O motor genérico: lê a base, infere o esquema, gera variáveis, **escolhe sozinho as que importam**, treina, valida e explica. Não tem nada específico de uma base. |
| `motor/modelos_previsao/` | Um arquivo por modelo: `logistica.py`, `lightgbm.py`, `random_forest.py` (mais `base.py` e `_arvores.py`). |
| `motor/modelos_classificacao/` | Um arquivo por método de avaliação: `validacao_cruzada.py`, `metricas.py`, `escolha.py` (qual modelo vence). |
| `experimentos/` | Tudo que é exploração e não faz parte do produto: `inovaapps/` (notebooks 01 exploração, 02 sinais, 03 score, mais o mapeamento da base), `redes/` (preparo e treino das bases reais) e `delta_risco.py`. |
| `dados/` | `desafio/` (xlsx e PDF do desafio), `amostra_redes/` (bases reais, fora do git) e `bases/` (uploads da API, fora do git). |
| `api/` | API FastAPI: enviar base, mapear, treinar, comparar versões e servir a fila. |
| `web/` | Frontend (a próxima etapa). Hoje só `web/public/data/` com os JSON do notebook 03. |
| `tests/` | Testes do motor. Os da API ficam em `api/tests/`. |

Ambiente Python com **uv**. Não há banco de dados: bases e resultados são arquivos.

```bash
uv sync                                   # instala tudo
uv run pytest -q                          # testes do motor + da API (o marcador `lento` fica de fora)
uv run python -m api.preparar_demos       # deixa as bases INOVAAPPS e redes treinadas e prontas
uv run uvicorn api.main:app --reload      # API em http://localhost:8000/docs
uv run jupyter lab                        # notebooks em experimentos/inovaapps/
```

## Como o motor funciona

1. **Painel cliente × mês.** Cada linha usa só dados até aquele mês; o rótulo é "cancela nos próximos
   3 meses". O histórico de quem saiu é cortado no mês da saída, para não vazar o futuro.
2. **Variáveis candidatas** de cada coluna: valor recente, variação contra a média do próprio cliente,
   tendência, persistência (meses seguidos na zona de risco) e ausência de registro, que é comportamento.
3. **Seleção supervisionada.** Descarta redundantes e suspeitas de vazamento, aprende a direção de risco
   pelos dados e fica só com as variáveis escolhidas de forma consistente em reamostragens e nas dobras.
4. **Modelo.** Logística, Random Forest ou LightGBM. A escolha é automática, pelo menor log-loss fora da
   amostra nas mesmas dobras; em empate técnico fica o mais simples. As árvores só entram com base grande
   (≥ 200 meses-cliente positivos e ≥ 50 cancelados).
5. **Fila.** Ordenada **só** pela perda anual esperada (probabilidade × valor × 12). As faixas
   alto/atenção/baixo vêm de cortes de probabilidade, nunca de capacidade da equipe.
6. **Explicação.** A contribuição de cada variável no próprio modelo; ela "dispara" quando a contribuição
   é positiva e o valor passa do limiar aprendido.

Toda validação é agrupada por cliente: o cliente avaliado nunca está no treino, e a seleção de variáveis é
refeita dentro de cada dobra.

## Bases

| Base | O que é | Como preparar |
|---|---|---|
| **INOVAAPPS** | O desafio: 80 clientes, 18 meses, 22 cancelamentos. | `dados/desafio/INOVAAPPS_base_de_dados.xlsx`. `uv run python experimentos/inovaapps/gerar_json.py` gera os JSON de `web/public/data/`. |
| **Redes (real)** | 3.000 clientes de 35 redes de postos, 1,6 M de vendas. | `uv run python experimentos/redes/preparar.py` → `dados/amostra_redes/base_motor/*.csv`. Depois `uv run python experimentos/redes/rodar.py`. |

As bases reais não têm campo de cancelamento. A regra que o time definiu (em `experimentos/redes/preparo_comum.py`):
**um mês sem nenhuma compra é um cancelamento**, e depois disso o cliente não é reavaliado. O último mês da
exportação, incompleto, fica de fora. Pela regra, a base das redes fica com 2.910 clientes, 1.840
cancelados e 1.070 ativos — e foi a primeira vez que as árvores ficaram elegíveis com dados reais: o
**LightGBM venceu** a logística por mais de um erro-padrão.

## Experimento do delta de risco (testado e descartado)

Pergunta: acrescentar o delta do próprio risco previsto (`risco do mês − risco do mês anterior`) melhora o
acerto? Resposta medida: **não**.

- Um braço **placebo**, com o delta embaralhado, entregou o mesmo "ganho" — ou seja, era ruído.
- Com o histórico mínimo padrão, o motor descartou o delta sozinho em todas as dobras.
- Ligá-lo custa **3,2× o tempo de treino**.

O código continua em `motor/delta.py`, desligado (`Config.delta_risco = False`), e a API aceita a opção
como avançada. Resultados em `dados/amostra_redes/resultado/delta/` e script em
`experimentos/delta_risco.py`.

## API

`uv run uvicorn api.main:app --reload` e `/docs` para a documentação interativa.

1. `POST /api/bases` — envia um `.xlsx` ou vários `.csv`; responde com as tabelas, as colunas e a
   **sugestão de mapeamento**.
2. `POST /api/bases/{id}/treinar` — mapeamento confirmado (tabela de clientes e coluna de id, coluna de
   cancelamento e mês de saída, coluna de valor) mais as opções; treina em segundo plano criando uma
   **versão**.
3. `GET /api/bases/{id}/execucoes` — as **versões** já treinadas da base, cada uma com um resumo
   (modelo escolhido, log-loss, AUC, detecção, alarme falso, faixas). `POST .../execucoes/{eid}/ativar`
   escolhe qual vale, e `GET .../comparar?execucoes=a&execucoes=b` devolve a tabela de comparação pronta.
   Treinos pedidos em sequência entram numa fila, um por vez.
4. `GET /api/bases/{id}/clientes|validacao|pesos` — a fila da versão ativa, os números da validação e o
   peso de cada variável.

A coluna de cancelamento **nunca** vira variável, e a API nunca reordena a fila: só filtra.

`uv run python -m api.preparar_demos` deixa as duas bases prontas: a INOVAAPPS em ~30 s e a das redes em
~9,5 min (inspeção mais treino). É idempotente: rodar de novo não refaz nada, a menos de `--forcar`.
