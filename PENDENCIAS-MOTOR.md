# Pendências do motor — o que a tela pede e o motor ainda não dá

O `motor/` não é editado por este trabalho. Onde a interface oferecia algo que o motor não
sustenta, o controle continua **visível e desativado**, com o motivo à mostra, e a explicação
longa fica aqui. Cada seção tem a âncora que o app referencia em `web/src/desativado.ts` — esse
arquivo é a fonte de verdade da interface, este aqui é a explicação para quem for mexer no motor.

Quem quiser resolver qualquer um destes itens mexe em `motor/` e/ou em `api/`, nunca no front
sozinho: o que falta é capacidade do motor, não tela.

---

## peso-manual-por-variavel

**Na tela**: aba Modelos, card 3 "Variáveis" — o par de botões *Automático / Manual* e os
controles de peso por variável.

**O que o protótipo oferecia**: um peso de 0 a 1 para cada variável marcada.

**Por que não dá hoje**: o motor escolhe as variáveis e seus pesos por conta própria, com
*stability selection* em reamostragens de clientes e validação em dobras agrupadas
(`motor/selecao.py`, `motor/treino.py`). Não existe entrada de peso por coluna em `motor.Config`
nem em `motor.Mapeamento`, e a API recusa campo desconhecido (`extra="forbid"` em
`api/modelos.py`). Um peso vindo da tela também desmontaria a garantia central do projeto: os
números da validação valem porque a seleção é refeita dentro de cada dobra, sem intervenção.

**O que continua funcionando**: a parte que importa. O *checkbox* de cada coluna está vivo e vira
`colunas_ignoradas: ["tabela.coluna"]` no mapeamento — formato que `motor/mapeamento.py` aceita.
Medido na base INOVAAPPS: desmarcar `pesquisas_nps.nota_nps` levou as candidatas de 82 para 77, a
AUC de 0,9788 para 0,9679 e os cancelados pegos em −1 de 21 para 16. Ou seja, o usuário decide
**quais** variáveis entram; só não decide **quanto** cada uma pesa.

**O que o motor precisaria ganhar**: um campo de pesos ou de restrições por variável em `Config`,
propagado até `motor/selecao.py`, e uma resposta clara sobre o que fazer com a validação depois —
um peso escolhido a olho torna as métricas otimistas.

---

## renomear-uma-versao

**Na tela**: aba Modelos, botão *Renomear versão* abaixo da tabela "Modelos treinados".

**Por que não dá hoje**: o `rotulo` de uma execução é gravado na criação
(`POST /api/bases/{id}/treinar`) e não há rota para alterá-lo depois. `api/armazenamento.py` não
expõe renomeação, e o `execucao_id` carrega o rótulo no próprio nome
(`20260920T112210-teste-do-front-sem-nps`), então renomear de verdade mexeria no identificador.

**Alternativa que já está na tela**: o campo *Nome desta versão* no card 4 define o rótulo antes
do treino. Para trocar o nome de uma versão existente, treine de novo com o mesmo mapeamento e
outro rótulo — na INOVAAPPS isso leva cerca de 15 s.

**O que a API precisaria ganhar**: um `PATCH /api/bases/{id}/execucoes/{eid}` que mude só o
`rotulo` no `status.json`, mantendo o `execucao_id` como está.

---

## pesos-do-score-na-fila

**Na tela**: não aparece — a coluna "Score" do protótipo foi substituída pela posição na fila.

**O que o protótipo fazia**: calculava no navegador um score de 0 a 100, média ponderada de risco,
valor mensal e variação do risco, e reordenava a fila por ele.

**Por que saiu**: a fila já vem ordenada do motor, pela perda anual ajustada pelo delta de risco
(`min(p + máx(Δp, 0), 1) × valor × 12`), e `prioridade` é 1..N nessa ordem. A convenção do projeto
é explícita: o frontend não reordena a fila. Um score na tela criaria uma segunda ordenação
concorrente com a do motor, e a explicação de "por que este cliente é o primeiro" deixaria de
bater com o que o motor calculou.

**Onde a informação reaparece**: a tela mostra a fórmula que o próprio motor manda pronta
(`modelo.formula_prioridade`) acima da tabela, e deixa claro que filtrar não reordena.

**Para mudar o critério**: é opção do motor, não da tela — `Config.delta_na_fila=False` volta a
ordenar por `p × valor × 12`. A API aceita a opção; a tela não a expõe.

---

## filtro-por-inicio-de-contrato

**Na tela**: aba Fila, painel de filtros — campo *Contrato iniciado desde*.

**Por que não dá hoje**: de cada cliente a fila traz `atributos`, que só recebe as colunas
**categóricas** da tabela de clientes (`motor/saida.py`), mais `valor_mensal`. Datas ficam de
fora. Na INOVAAPPS os atributos são exatamente `plano`, `porte` e `segmento` — `inicio_contrato`
existe na planilha e não chega na fila.

**Alternativa descartada**: daria para ler `inicio_contrato` do .xlsx que o usuário acabou de
enviar e cruzar por `cliente_id` no navegador. Ficaria um filtro que só funciona na sessão em que
a base foi enviada, some num F5 e nunca funciona nas bases de demonstração — pior que não ter.

**O que o motor precisaria ganhar**: incluir em `atributos` (ou num bloco novo) as colunas de data
da tabela de clientes, já normalizadas em `AAAA-MM`.

**Observação**: os outros filtros por atributo são gerados a partir das chaves que chegam, então
funcionam em qualquer base sem alteração de código.

---

## analisar-dados-sem-arquivo-local

**Na tela**: aba Modelos, card *Análise dos dados* — desativado nas bases que vieram do servidor.

**Como funciona quando está ativo**: o mesmo arquivo que vai para a API é lido também no
navegador, com SheetJS, e a análise (média móvel de 3 meses por cliente, alinhada em meses até a
saída, com mediana e faixa P25–P75 por grupo) roda inteira do lado do cliente. É exploratória,
para ajudar a decidir o mapeamento antes de treinar.

**Por que desativa**: uma base de demonstração — ou qualquer base aberta depois de recarregar a
página — nunca passou por este navegador. Não há arquivo em memória para ler, e a API não devolve
os dados crus: ela entrega a inspeção (tabelas, colunas, tipos, exemplos), os agregados e a fila.

**Onde está a resposta melhor**: a aba **Validação** responde à mesma pergunta — o sinal separa
quem saiu de quem ficou, e com quanta antecedência? — com números validados por cliente
(`metodos`, `auc_por_mes`, `precisao_topo`, `cancelados`), em vez de uma olhada nos dados crus.

**O que a API precisaria ganhar**: uma rota de análise exploratória por base, devolvendo as séries
já agregadas por grupo e mês relativo. Seria também a chance de tirar a duplicação: hoje
`web/src/analise/exploratoria.ts` repete no navegador uma leitura que o notebook de sinais já faz
em Python.

---

## cancelar-um-treino

**Na tela**: aba Modelos, card 4 — botão *Cancelar treino*.

**Por que não dá hoje**: o treino roda numa thread do processo da API
(`api/servico.py`), e não há rota nem mecanismo de interrupção. O callback de progresso só
reporta; não há ponto de parada cooperativo dentro de `motor/treino.py`.

**O que existe no lugar**: o acompanhamento é honesto — etapa e fração vêm do próprio motor, e
uma execução que falha vira `estado: "erro"` com a mensagem e os problemas em pt-BR. Treinos
pedidos em sequência entram numa fila e rodam um por vez (`API_TREINOS_SIMULTANEOS`, padrão 1).

**O que precisaria ganhar**: um sinal de cancelamento conferido dentro do laço de dobras de
`motor/validar.py` e uma rota `POST /api/bases/{id}/execucoes/{eid}/cancelar`.

---

## Nota sobre a versão do SheetJS

`web/package.json` usa `xlsx@0.18.5` do npm, a mesma versão do CDN que o protótipo carregava. É a
última publicada no registro público; as correções posteriores (entre elas a de *prototype
pollution*) a SheetJS publica só no CDN próprio. O risco aqui é contido: a biblioteca só abre
arquivos que o próprio usuário escolheu na máquina dele, e o resultado não sai do navegador — a
leitura que importa é a do motor, no servidor. Trocar pelo tarball do CDN
(`https://cdn.sheetjs.com/xlsx-0.20.3/xlsx-0.20.3.tgz`) exige que a máquina de build alcance esse
host.

---

## Divergência de stack registrada

O `CLAUDE.md` declarava `Next.js` e `Tailwind + shadcn/ui + Recharts` para o frontend. O app foi
feito em **React + Vite**, com o CSS do protótipo portado e gráficos em SVG próprio — decisão do
time, para preservar o visual já aprovado e não reconstruir a aparência do zero. A tabela de stack
do `CLAUDE.md` foi corrigida para refletir o que existe.
