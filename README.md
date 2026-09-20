# Previsão de cancelamento — Desafio INOVAAPPS 2026 (Globalsys)

Quem abre a solução precisa responder a três perguntas sobre uma carteira de clientes:
**com quem falar, por que cada um e em que ordem.**

O sistema recebe **qualquer base** com uma tabela de clientes e tabelas de valores ligados a eles,
descobre sozinho quais sinais antecedem o cancelamento, treina um modelo, e devolve uma **fila de
atendimento explicada**: o risco de cada cliente, as variáveis que estão disparando e a ação sugerida.

Para instalar e subir na sua máquina: **[docs/rodar-local.md](docs/rodar-local.md)**.

## Como está organizado

| Pasta | O que é |
|---|---|
| `motor/` | O motor genérico: lê a base, infere o esquema, gera variáveis, **escolhe sozinho as que importam**, treina, valida e explica. Não tem nada específico de uma base. |
| `motor/modelos_previsao/` | Um arquivo por modelo: `logistica.py`, `lightgbm.py`, `random_forest.py` (mais `base.py` e `_arvores.py`). |
| `motor/modelos_classificacao/` | Um arquivo por método de avaliação: `validacao_cruzada.py`, `metricas.py`, `escolha.py` (qual modelo vence). |
| `experimentos/` | Tudo que é exploração e não faz parte do produto: `inovaapps/` (notebooks 01 exploração, 02 sinais, 03 score, mais o mapeamento da base), `redes/` (preparo e treino das bases reais) e `delta_risco.py`. |
| `dados/` | `desafio/` (xlsx e PDF do desafio), `amostra_redes/` (bases reais, fora do git) e `bases/` (uploads da API, fora do git). |
| `api/` | API FastAPI: enviar base, mapear, treinar, comparar versões e servir a fila. |
| `web/` | O app: React + Vite + TypeScript. Três telas — Modelos, Fila e Validação — falando com a API. |
| `tests/` | Testes do motor. Os da API ficam em `api/tests/`. |
| `docs/` | Documentação do projeto. Por ora, como instalar e rodar localmente. |

Ambiente Python com **uv**. Não há banco de dados: bases e resultados são arquivos.

```bash
uv sync                                   # instala tudo
uv run pytest -q                          # testes do motor + da API (o marcador `lento` fica de fora)
uv run python -m api.preparar_demos       # deixa as bases INOVAAPPS e redes treinadas e prontas
uv run uvicorn api.main:app --reload      # API em http://localhost:8000/docs
uv run jupyter lab                        # notebooks em experimentos/inovaapps/
```

E o app, noutro terminal:

```bash
cd web
npm install
npm run dev          # http://localhost:5173 — o proxy leva /api para a API na 8000
npm run build        # bundle de produção em web/dist/
```

A API precisa estar no ar antes do `npm run dev`; a tela diz claramente quando não está.

Como a IA não faz conferência visual neste projeto, a verificação do front é executar e checar
números — as duas suítes rodam contra a API de pé, com a base INOVAAPPS:

```bash
npm run checar       # tipos
npm run verificar    # as funções da tela contra os números que a API devolve
npm run renderizar   # renderiza cada tela com os dados reais, mais os casos-limite
```

`verificar` confere, entre outras coisas, que a ordem da fila é a do motor e que filtrar não a
refaz, e que a soma da perda anual bate com o resumo. `renderizar` monta as telas de verdade e
pega o que só aparece na montagem — campo nulo, base sem coluna de valor, cliente sem nenhum
sinal, validação desligada, modelo de árvore sem coeficientes.

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
5. **Fila.** Ordenada pela perda anual **ajustada pelo delta**: `min(p + máx(Δp, 0), 1) × valor × 12`, com Δp = risco de
   hoje − risco do mês anterior (quem piorou passa na frente). As faixas
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

## Delta de risco

O delta **ordena a fila** (padrão `Config.delta_na_fila = True`) e vai no JSON de cada cliente (`delta_risco`,
`risco_ajustado`, `perda_anual_ajustada`), sem entrar no modelo. Como **variável do modelo** ele foi testado e descartado:

Pergunta: acrescentar o delta do próprio risco previsto (`risco do mês − risco do mês anterior`) melhora o
acerto? Resposta medida: **não**.

- Um braço **placebo**, com o delta embaralhado, entregou o mesmo "ganho" — ou seja, era ruído.
- Com o histórico mínimo padrão, o motor descartou o delta sozinho em todas as dobras.
- Ligá-lo custa **3,2× o tempo de treino**.

O código do modelo em dois estágios continua em `motor/delta.py`, desligado (`Config.delta_risco = False`), e a API aceita a opção
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
5. `PATCH /api/bases/{id}/execucoes/{eid}` — renomeia a versão (`rotulo`) e/ou refaz a ordem da fila com
   outros `pesos_score` (`{risco, valor, delta}`), sem treinar de novo; `pesos_score: null` volta à ordem padrão.
6. `POST /api/bases/{id}/execucoes/{eid}/cancelar` — cancela um treino na fila (sai na hora) ou em andamento (o
   motor para na próxima etapa ou dobra); a versão fica `cancelada`.
7. `POST /api/bases/{id}/analise` — "Analisar dados": cancelados × ativos, métrica a métrica, nos 12 meses antes
   da saída (mediana e faixa P25–P75), calculado no servidor sobre a base guardada e sem precisar de modelo.

No treino (`opcoes`): `pesos_colunas` (peso 0–1 por coluna) e `pesos_score` (score que ordena a fila). Cada
cliente da fila traz também `datas` (colunas de data da tabela de clientes, em `AAAA-MM`) e, com `pesos_score`, `score`.

A coluna de cancelamento **nunca** vira variável, e a API nunca reordena a fila além do que o motor define: só filtra.

## App (`web/`)

| Tela | O que mostra |
|---|---|
| **Fila** | A fila explicada: posição, faixa, score (quando a versão tem pesos), risco, variação vs. o mês anterior, perda anual esperada e os sinais que dispararam. Clicar num cliente abre o painel com as evidências, os gráficos de 18 meses e a ação sugerida. Os filtros são gerados a partir dos atributos e das datas (ex.: início do contrato) que a base tiver. |
| **Modelos** | Enviar uma base, confirmar o que prever (a partir da leitura que o motor faz), analisar os dados, escolher quais colunas entram (com peso manual opcional), definir os pesos do score da fila e treinar. Cada treino vira uma versão, com progresso real vindo do motor e botão para cancelar; a lista permite renomear, mudar os pesos do score, trocar qual versão vale e apagar. |
| **Validação** | Os três critérios da banca: com quanta antecedência o sinal aparece, o quanto separa quem ficou, e o que acontece no topo da fila — mais os 22 que saíram, com o risco que o modelo dava a cada um antes da saída. |

O app **nunca reordena a fila**: a ordem é a `prioridade` que o motor devolve, e filtrar só esconde
linhas. O que a tela já pediu e o motor ainda não sustentava foi resolvido; o histórico está em
[`PENDENCIAS-MOTOR.md`](PENDENCIAS-MOTOR.md).

`uv run python -m api.preparar_demos` deixa as duas bases prontas: a INOVAAPPS em ~30 s e a das redes em
~9,5 min (inspeção mais treino). É idempotente: rodar de novo não refaz nada, a menos de `--forcar`.
