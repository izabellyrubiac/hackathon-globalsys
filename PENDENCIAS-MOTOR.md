# Pendências do motor

**Nenhuma pendência aberta.** Os seis itens que a tela pedia e o motor não sustentava foram resolvidos; o que
mudou em cada um fica registrado aqui para quem for mexer no motor.

| Antes (controle desativado) | Agora | Onde está |
|---|---|---|
| **Peso manual por variável** | Peso 0–1 por coluna escala a penalização da seleção da logística (0 tira a coluna; nas árvores só o 0 vale). A seleção e a validação são refeitas com os mesmos pesos, então as métricas continuam honestas. | `Config.pesos_colunas` · `motor/selecao.py` (`peso_da_variavel`) · `opcoes.pesos_colunas` na API · card 3 de Modelos |
| **Renomear uma versão** | Só o `rotulo` muda; o `execucao_id` continua o mesmo. | `PATCH /api/bases/{id}/execucoes/{eid}` · `api/servico.py` (`atualizar_execucao`) |
| **Pesos do score na fila** | Score 0–100 = média ponderada de risco, valor e variação do risco; ordena a fila quando há pesos, e dá para refazer a ordem de uma versão pronta sem retreinar. | `Config.pesos_score` · `motor/score.py` · `PATCH` acima · coluna Score da Fila |
| **Filtro por início de contrato** | As colunas de data da tabela de clientes chegam na fila em `AAAA-MM` (`datas`, `modelo.colunas_data`); a tela cria um filtro por coluna. | `motor/painel.py` (`Base.datas`) · `motor/saida.py` |
| **Analisar dados sem arquivo local** | A análise roda no servidor sobre a base guardada (não depende do navegador nem de modelo treinado). Sem a coluna do mês da saída, o motor usa o fim do histórico. | `motor/exploratoria.py` · `POST /api/bases/{id}/analise` |
| **Cancelar um treino** | Parada cooperativa: o motor confere o sinal a cada etapa e a cada dobra da validação. A versão fica `cancelada`. | `motor.TreinoCancelado` · `treinar(..., cancelar=)` · `POST .../execucoes/{eid}/cancelar` |

Consequência: o navegador não lê mais planilhas (a dependência `xlsx` saiu do `web/`); só a API lê os dados.

---

## Divergência de stack registrada

O `CLAUDE.md` declarava `Next.js` e `Tailwind + shadcn/ui + Recharts` para o frontend. O app foi
feito em **React + Vite**, com o CSS do protótipo portado e gráficos em SVG próprio — decisão do
time, para preservar o visual já aprovado e não reconstruir a aparência do zero. A tabela de stack
do `CLAUDE.md` foi corrigida para refletir o que existe.
