# Rodar o projeto na sua máquina

São duas peças: a **API** (Python, porta 8000), que lê as bases e roda o motor, e o **app**
(React + Vite, porta 5173), que é a tela. Sobem em terminais separados e conversam entre si.
Não há banco de dados — bases e resultados são arquivos em `dados/bases/`.

## O que precisa estar instalado

| Ferramenta | Versão | Para quê |
|---|---|---|
| [uv](https://docs.astral.sh/uv/) | 0.12 ou mais novo | ambiente e dependências do Python |
| Python | 3.12 | o `uv` instala a versão certa sozinho, lendo o `.python-version` |
| Node.js | 22 ou mais novo | o app |
| npm | 10 ou mais novo | vem com o Node |

Conferindo:

```bash
uv --version && node --version && npm --version
```

Se faltar o `uv`: `curl -LsSf https://astral.sh/uv/install.sh | sh`.

## Primeira vez

Da raiz do projeto, três comandos:

```bash
uv sync                                   # ambiente Python e todas as dependências
uv run python -m api.preparar_demos       # deixa as bases de demonstração treinadas e prontas
cd web && npm install && cd ..            # dependências do app
```

O `preparar_demos` treina a base do desafio (INOVAAPPS, ~30 s) e, se os arquivos das redes
estiverem em `dados/amostra_redes/base_motor/`, também aquela (~8 min). É **idempotente**: rodar
de novo não refaz nada, a menos que você passe `--forcar`. Para treinar só uma:

```bash
uv run python -m api.preparar_demos --base inovaapps
```

Esse passo é opcional — sem ele o sistema sobe igual, só que sem nenhum modelo treinado, e a tela
mostra "Nenhum modelo treinado" até você enviar uma base e treinar pela interface. Com ele, a tela
já abre com a fila do desafio pronta.

## Subir

**Terminal 1 — a API:**

```bash
uv run uvicorn api.main:app --reload
```

Fica em http://localhost:8000, com a documentação interativa em http://localhost:8000/docs.
O `--reload` reinicia sozinho quando você mexe no Python.

**Terminal 2 — o app:**

```bash
cd web
npm run dev
```

**Abra http://localhost:5173.**

A API precisa estar no ar **antes** de você usar a tela. Se não estiver, o app diz isso com todas
as letras ("Não consegui falar com a API. Ela está de pé em http://localhost:8000?") em vez de
ficar carregando para sempre.

Você não precisa configurar endereço de API nem CORS: o Vite tem um proxy que leva tudo que começa
com `/api` para a porta 8000. Para apontar para outro lugar, use a variável `API_URL`:

```bash
API_URL=http://192.168.0.10:8000 npm run dev
```

## O que fazer na tela

1. **Modelos** — a base do desafio já aparece na lista se você rodou o `preparar_demos`. Para usar
   outra, clique em *Enviar base de dados* e mande um `.xlsx` com uma tabela por aba, ou vários
   `.csv`. O sistema lê os arquivos e preenche sozinho o que ele entendeu: qual é a tabela de
   clientes, qual coluna identifica o cliente, onde está o cancelamento e qual é o valor do
   contrato. Confira, desmarque as colunas que não devem virar variável, dê um nome à versão e
   clique em *Treinar modelo*. O progresso na tela é o do motor de verdade.
2. **Fila** — quem atender primeiro, por quê e o que fazer. Clique numa linha para abrir o painel
   do cliente, com as evidências e os gráficos dos 18 meses.
3. **Validação** — com quanta antecedência o sinal apareceu, o quanto separa quem ficou, e o que
   houve no topo da fila; mais os 22 que saíram, com o risco que o modelo dava a cada um antes da
   saída.

Alguns controles aparecem apagados, com o motivo escrito embaixo: são coisas que a tela ofereceria
mas que o motor não sustenta hoje. A lista e a explicação de cada um estão em
[`PENDENCIAS-MOTOR.md`](../PENDENCIAS-MOTOR.md).

## Conferir que está tudo certo

Neste projeto a verificação é executar o código e olhar os números — não há conferência visual
automática. Com a API de pé e a base do desafio treinada:

```bash
uv run pytest -q                 # motor e API (89 testes; ~4 min)

cd web
npm run checar                   # tipos
npm run verificar                # as funções da tela contra os números que a API devolve
npm run renderizar               # monta cada tela com os dados reais, mais os casos-limite
npm run build                    # bundle de produção em web/dist/
```

`verificar` confere, entre outras coisas, que a fila mantém a ordem do motor, que filtrar não a
refaz e que a soma da perda anual bate com o resumo. `renderizar` monta as telas de verdade e pega
o que só aparece na montagem: campo nulo, base sem coluna de valor, cliente sem nenhum sinal,
validação desligada.

## Quando dá errado

| Sintoma | O que é |
|---|---|
| A tela diz que não consegue falar com a API | A API não está no ar, ou está em outra porta. Suba o terminal 1 e confira http://localhost:8000/api/saude. |
| "Nenhum modelo treinado" | Nenhuma base tem versão pronta. Rode o `preparar_demos` ou treine uma pela aba Modelos. |
| A coluna "Variação vs. mês anterior" está toda em "—" | A versão ativa foi treinada antes de o motor passar a mandar esse campo. Treine de novo: `uv run python -m api.preparar_demos --base inovaapps --forcar`. |
| Porta 5173 ou 8000 ocupada | `uvicorn ... --port 8001` e, no app, `API_URL=http://localhost:8001 npm run dev`. Para o Vite, `npm run dev -- --port 5174`. |
| `npm run verificar` falha dizendo que a API não responde | Ele fala com a porta 8000 direto, sem passar pelo proxy. Aponte para outra com `API=http://localhost:8001 npm run verificar`. |
| O treino de uma base grande parece travado | Não travou: o motor valida em dobras e demora. A tela mostra a etapa e a fração reais. Um treino já começado não pode ser cancelado (veja `PENDENCIAS-MOTOR.md`). |

Quer começar do zero? `rm -rf dados/bases/` apaga todas as bases enviadas e todos os resultados —
e nada mais: os arquivos originais do desafio ficam em `dados/desafio/`.

## Ajustes da API (opcional)

Todos têm padrão razoável; são variáveis de ambiente lidas por `api/ajustes.py`.

| Variável | Padrão | O que faz |
|---|---|---|
| `API_DADOS_BASES` | `dados/bases` | onde as bases enviadas ficam |
| `API_MAX_UPLOAD_MB` | 1024 | tamanho máximo de um envio |
| `API_MAX_ARQUIVOS` | 30 | quantos arquivos por envio |
| `API_TREINOS_SIMULTANEOS` | 1 | quantos treinos rodam ao mesmo tempo; os outros esperam na fila |
| `API_DEMO` | `1` | `0` desliga as bases de demonstração |
| `API_DEMO_ARQUIVO` | `dados/desafio/INOVAAPPS_base_de_dados.xlsx` | de onde vem a base do desafio |
| `API_DEMO_REDES` | `dados/amostra_redes/base_motor` | de onde vêm os `.csv` das redes |

## Onde fica o quê

```
motor/      o motor: lê a base, monta as variáveis, escolhe as que pesam, treina, valida e explica
api/        a API FastAPI que embrulha o motor
web/        o app React + Vite — as três telas
dados/      desafio/ (o xlsx original) e bases/ (o que foi enviado e os resultados; fora do git)
experimentos/  notebooks e testes que não fazem parte do produto
tests/      testes do motor (os da API ficam em api/tests/)
docs/       este documento
```
