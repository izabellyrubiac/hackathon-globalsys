/** Catálogo único do que a tela oferece e o motor não sustenta.
 *
 * Uma entrada por controle: o texto que aparece na tela e o motivo técnico. O mesmo conteúdo
 * está em PENDENCIAS-MOTOR.md, na raiz do projeto — este arquivo é a fonte de verdade da
 * interface, aquele é a explicação longa para quem for mexer no motor.
 *
 * Regra do projeto: nada aqui é contornado editando `motor/`. O controle fica visível,
 * desativado e com o motivo à mostra.
 */

export interface Pendencia {
  /** Texto curto que vai no `title` e na nota sob o controle. */
  motivo: string
  /** Âncora da seção correspondente em PENDENCIAS-MOTOR.md. */
  ancora: string
}

export const PENDENCIAS = {
  pesoManual: {
    motivo: 'O motor escolhe sozinho quais variáveis pesam, por seleção supervisionada validada '
      + 'em dobras agrupadas por cliente. Definir peso à mão exigiria mudar o motor. Desmarcar '
      + 'uma coluna, logo acima, continua funcionando.',
    ancora: 'peso-manual-por-variavel',
  },
  renomearVersao: {
    motivo: 'O rótulo da versão é definido no treino. A API não tem rota para renomear uma '
      + 'execução depois de criada.',
    ancora: 'renomear-uma-versao',
  },
  pesosDoScore: {
    motivo: 'A ordem da fila vem do motor (perda anual ajustada pelo delta de risco) e o '
      + 'frontend nunca a refaz. Pesos de ordenação na tela criariam uma segunda fila.',
    ancora: 'pesos-do-score-na-fila',
  },
  inicioContrato: {
    motivo: 'A data de início do contrato não chega na fila: de cada cliente o motor manda os '
      + 'atributos categóricos e o valor mensal, e datas ficam de fora.',
    ancora: 'filtro-por-inicio-de-contrato',
  },
  analiseSemArquivo: {
    motivo: 'A análise roda no seu navegador, sobre o arquivo enviado nesta sessão. Esta base '
      + 'veio do servidor, então não há arquivo local para analisar. A aba Validação responde à '
      + 'mesma pergunta com os números do motor.',
    ancora: 'analisar-dados-sem-arquivo-local',
  },
  cancelarTreino: {
    motivo: 'Um treino já iniciado não pode ser interrompido: a API não tem rota de cancelamento.',
    ancora: 'cancelar-um-treino',
  },
} as const satisfies Record<string, Pendencia>
