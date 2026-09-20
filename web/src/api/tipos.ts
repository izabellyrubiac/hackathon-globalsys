/** Contratos da API, em pt-BR como o resto do projeto.
 *
 * Espelham `motor/saida.py` (clientes.json, validacao.json, pesos.json) e `api/modelos.py`.
 * Onde a API usa `extra="allow"`, o tipo leva um índice aberto: o contrato do motor pode ganhar
 * campos novos sem quebrar o build daqui.
 */

export type Faixa = 'alto' | 'atencao' | 'baixo'
export const FAIXAS: Faixa[] = ['alto', 'atencao', 'baixo']

export type EstadoExecucao = 'na_fila' | 'treinando' | 'pronta' | 'erro'
export type EstadoBase = 'inspecionada' | EstadoExecucao

// --------------------------------------------------------------------------- inspeção
export type TipoColuna = 'id' | 'numerica' | 'categorica' | 'data' | 'texto' | 'binaria'
export type PapelTabela = 'clientes' | 'estatica' | 'temporal' | 'eventos' | 'ignorada'

export interface Coluna {
  nome: string
  tipo: TipoColuna
  unicos: number
  nulos_pct: number
  exemplos: unknown[]
}

export interface Tabela {
  nome: string
  linhas: number
  papel_sugerido: PapelTabela
  chave: string | null
  coluna_data: string | null
  motivo: string | null
  colunas: Coluna[]
}

export interface Sugestao {
  tabela_clientes: string | null
  coluna_id: string | null
  alvo: {
    tabela: string | null
    coluna_situacao: string | null
    valor_cancelado: unknown
    coluna_data_saida: string | null
  } | null
  coluna_valor: { tabela: string | null; coluna: string | null } | null
  horizonte_meses: number
}

export interface Inspecao {
  tabelas: Tabela[]
  sugestao: Sugestao
  avisos: string[]
}

// --------------------------------------------------------------------------- mapeamento e treino
/** Mesmos campos de `motor.Mapeamento`. A API recusa campo desconhecido (`extra="forbid"`). */
export interface Mapeamento {
  tabela_clientes: string
  coluna_id: string
  alvo_tabela: string
  coluna_situacao: string | null
  valor_cancelado: unknown
  coluna_data_saida: string | null
  valor_tabela: string | null
  coluna_valor: string | null
  horizonte_meses: number
  /** "coluna" ou "tabela.coluna" — o que não pode virar variável. */
  colunas_ignoradas: string[]
  acoes: Record<string, string>
  rotulos: Record<string, string>
}

export interface Opcoes {
  modelo: 'auto' | 'logistica' | 'random_forest' | 'lightgbm'
  horizonte_meses?: number | null
  min_hist?: number | null
  /** Opção avançada, desligada. O CLAUDE.md pede que o front não a mostre. */
  delta_risco?: boolean
  validar?: boolean | null
  dobras?: number | null
  semente?: number | null
  avancado?: Record<string, unknown>
}

export interface PedidoTreino {
  mapeamento: Mapeamento
  opcoes: Opcoes
  rotulo?: string | null
}

// --------------------------------------------------------------------------- status e execuções
export interface Status {
  base_id: string
  estado: EstadoBase
  etapa: string | null
  fracao: number
  mensagem: string | null
  problemas: string[]
  avisos: string[]
  n_clientes: number | null
  execucao_id: string | null
  execucao_ativa: string | null
  n_execucoes: number
  iniciado_em: string | null
  atualizado_em: string | null
  segundos: number
  opcoes: Record<string, unknown> | null
}

/** `resumo.json` — a ficha pequena que a API monta por execução (api/resumo.py). */
export interface ResumoExecucao {
  execucao_id: string
  rotulo: string | null
  criada_em: string | null
  concluido_em: string | null
  duracao_s: number | null
  gerado_em: string | null
  mes_referencia: string | null
  validacao_ligada: boolean
  tem_valor: boolean
  modelo: { nome: string; rotulo: string; forcado: boolean; motivo: string; criterio: string
            complexidade: number; log_loss?: number | null; log_loss_ep?: number | null }
  horizonte_meses: number
  cortes: { alto: number; atencao: number }
  auc: { k1: number | null; k2: number | null; k3: number | null }
  deteccao: {
    n_cancelados: number
    alto: { k1: number; k2: number; k3: number }
    alto_ou_atencao: { k1: number; k2: number; k3: number }
    antecedencia_mediana_alto: number | null
    antecedencia_mediana_alto_ou_atencao: number | null
  }
  alarme_falso: {
    n_ativos: number
    alto: { hoje: number; doze_meses: number; pct_meses: number }
    alto_ou_atencao: { hoje: number; doze_meses: number; pct_meses: number }
  }
  faixas: { alto: number; atencao: number; baixo: number }
  n_clientes: number
  em_risco: number
  em_risco_alto: number
  receita_em_risco_mensal: number | null
  receita_risco_alto_mensal: number | null
  perda_anual_esperada_total: number | null
  variaveis: {
    n_candidatas: number
    n_selecionadas: number
    principais: { id: string; rotulo: string; coluna: string; coef: number | null
                  importancia: number; direcao: number; peso: number }[]
  }
  opcoes: Record<string, unknown>
  painel: Record<string, number>
  avisos: string[]
  [k: string]: unknown
}

export interface Execucao {
  base_id: string
  execucao_id: string
  rotulo: string | null
  estado: EstadoExecucao
  etapa: string | null
  fracao: number
  mensagem: string | null
  problemas: string[]
  avisos: string[]
  n_clientes: number | null
  criada_em: string | null
  iniciado_em: string | null
  concluido_em: string | null
  segundos: number
  opcoes: Record<string, unknown> | null
  ativa: boolean
  resumo: ResumoExecucao | null
}

// --------------------------------------------------------------------------- bases
export interface BaseResumo {
  base_id: string
  nome: string
  criada_em: string | null
  bytes_total: number | null
  arquivos: string[]
  estado: EstadoBase
  etapa: string | null
  n_clientes: number | null
  n_execucoes: number
  execucao_ativa: string | null
  demonstracao: boolean
}

export interface BaseDetalhe {
  base_id: string
  nome: string
  criada_em: string | null
  arquivos: string[]
  bytes_total: number | null
  demonstracao: boolean
  inspecao: Inspecao
  mapeamento: Mapeamento | null
  opcoes: Record<string, unknown> | null
  status: Status
  execucao_ativa: string | null
  execucoes: Execucao[]
}

export interface BaseCriada {
  base_id: string
  nome: string
  criada_em: string
  arquivos: string[]
  bytes_total: number
  inspecao: Inspecao
  status: Status
}

export interface TreinoAceito {
  base_id: string
  execucao_id: string
  execucao: Execucao
  status: Status
  avisos: string[]
}

// --------------------------------------------------------------------------- fila (clientes.json)
export interface Comparacao {
  chave: string
  recente: number
  anterior: number
  meses_recente: number
  meses_anterior: number
}

export interface Evidencia {
  sinal: string
  coluna: string
  tabela: string
  titulo: string
  detalhe: string
  peso: number
  contribuicao: number
  valor: number | null
  limiar: number | null
  /** chave em `modelo.series_historico` e em `historico[].valores`; null fora de série mensal. */
  chave_serie: string | null
  comparacao: Comparacao | null
}

export interface PontoHistorico {
  mes_ref: string
  risco: number | null
  valores: Record<string, number | null>
}

export interface ClienteFila {
  cliente_id: string
  /** Colunas categóricas da tabela de clientes — as chaves mudam de base para base. */
  atributos: Record<string, string>
  valor_mensal: number | null
  situacao: string
  risco: number
  faixa_risco: Faixa
  /** Posição na fila, 1..N. A ordem é do motor e nunca é refeita aqui. */
  prioridade: number
  perda_anual_esperada: number | null
  /** risco de hoje − risco do mês anterior; null sem mês anterior pontuado. */
  delta_risco?: number | null
  risco_ajustado?: number
  perda_anual_ajustada?: number | null
  meses_em_alerta: number
  evidencias: Evidencia[]
  fatores_secundarios: Evidencia[]
  acao_sugerida: string
  sinal_dominante: string | null
  coluna_dominante: string | null
  historico?: PontoHistorico[]
}

export interface VariavelModelo {
  id: string
  rotulo: string
  tabela: string
  coluna: string
  transformacao: string
  coef: number | null
  importancia: number
  direcao: number
  limiar: number | null
}

export interface SerieHistorico {
  chave: string
  tabela: string
  coluna: string
  rotulo: string
  tipo: 'numerica' | 'categorica'
}

export interface ModeloFila {
  nome: string
  rotulo: string
  tipo: string
  horizonte_meses: number
  cortes: { alto: number; atencao: number }
  formula_prioridade: string
  coluna_valor: { tabela: string; coluna: string } | null
  variaveis: VariavelModelo[]
  series_historico: SerieHistorico[]
}

export interface ResumoFila {
  clientes_ativos: number
  em_risco: number
  em_risco_alto: number
  em_atencao: number
  receita_em_risco_mensal: number | null
  receita_risco_alto_mensal: number | null
  perda_anual_esperada_total: number | null
}

export interface PaginaClientes {
  base_id: string
  execucao_id: string | null
  gerado_em: string | null
  mes_referencia: string | null
  modelo: ModeloFila | null
  resumo: ResumoFila | null
  total: number
  total_filtrado: number
  desde: number
  limite: number
  clientes: ClienteFila[]
}

// --------------------------------------------------------------------------- validacao.json
export interface MetodoValidacao {
  id: 'melhor_variavel' | 'motor_alto' | 'motor_alto_ou_atencao'
  nome: string
  canc_k1: number
  canc_k2: number
  canc_k3: number
  ativos_hoje: number
  ativos_algum_mes: number
  meses_alarme_ativos_pct: number
  antecedencia_mediana: number | null
  variavel?: string
  rotulo?: string
}

export interface CanceladoValidacao {
  cliente_id: string
  valor_mensal: number | null
  mes_saida: string
  risco_k1: number | null
  risco_k2: number | null
  risco_k3: number | null
  faixa_k1: Faixa | null
  faixa_k2: Faixa | null
  faixa_k3: Faixa | null
}

export interface Coeficiente {
  variavel: string
  rotulo: string
  coluna: string
  coef: number | null
  odds_ratio: number | null
  importancia: number
  direcao: number
  frequencia_dobras: number
  min_dobras: number | null
  max_dobras: number | null
}

export interface ModeloAvaliado {
  nome: string
  rotulo: string
  complexidade: number
  elegivel: boolean
  motivo_elegibilidade: string
  avaliado: boolean
  escolhido: boolean
  /** As métricas só existem nos modelos avaliados: num não elegível a chave nem aparece. */
  log_loss?: number | null
  log_loss_ep?: number | null
  log_loss_conjunto?: number | null
  auc_linhas?: number | null
  auc_k1?: number | null
  auc_k2?: number | null
  auc_k3?: number | null
}

export interface Validacao {
  gerado_em: string
  metodo: string
  honestidade: string
  horizonte_meses: number
  n_dobras: number
  painel: Record<string, number>
  cortes: { alto: number; atencao: number; youden: number
            criterio_alto: string; criterio_atencao: string }
  escolha: { modelo: string; rotulo: string; forcado: boolean; motivo: string; criterio: string }
  modelos: ModeloAvaliado[]
  metodos: MetodoValidacao[]
  auc_por_mes: { k: number; score: 'motor' | 'melhor_variavel'; auc: number; n: number; n_canc: number }[]
  precisao_topo: { k: number; top: number; cancelados: number; metodo: string }[]
  curva: Record<string, number>[]
  coeficientes: Coeficiente[]
  intercepto: number | null
  C: number | null
  cancelados: CanceladoValidacao[]
  avisos: string[]
  config: Record<string, unknown>
  [k: string]: unknown
}
