/** Funções puras sobre o resultado do motor. Nada aqui reordena a fila: a ordem é a que a
 * API devolve (`prioridade`, do motor), e filtrar nunca a refaz.
 */

import type { ClienteFila, Evidencia, Execucao, BaseResumo, Faixa } from './api/tipos.ts'
import { norm, recuarMes } from './formato.ts'

// --------------------------------------------------------------------------- variação de risco
export type Variacao =
  | { tipo: 'sem-dado' }
  | { tipo: 'igual' | 'pior' | 'melhor'; pp: number; rel: number; origem: 'motor' | 'historico' }

/** Quanto o risco mudou do mês anterior para cá.
 *
 * Primeiro o `delta_risco` do motor. Execuções treinadas antes desse campo existir não o trazem;
 * aí vale o penúltimo ponto do histórico — que só existe quando a resposta veio com
 * `incluir_historico=true`. Sem os dois, a célula fica "—" e a tela explica o porquê.
 */
export function variacao(c: ClienteFila): Variacao {
  let pp: number | null = null
  let anterior: number | null = null
  let origem: 'motor' | 'historico' = 'motor'

  if (typeof c.delta_risco === 'number') {
    pp = c.delta_risco * 100
    anterior = c.risco - c.delta_risco
  } else {
    const h = c.historico
    const ant = h && h.length >= 2 ? h[h.length - 2]?.risco : null
    if (typeof ant === 'number') {
      pp = (c.risco - ant) * 100
      anterior = ant
      origem = 'historico'
    }
  }
  if (pp == null) return { tipo: 'sem-dado' }

  const rel = anterior ? (pp / (anterior * 100)) * 100 : 0
  const tipo = Math.abs(pp) < 1 ? 'igual' : pp > 0 ? 'pior' : 'melhor'
  return { tipo, pp, rel, origem }
}

export const SETA: Record<string, string> = { pior: '▲', melhor: '▼', igual: '●' }

// --------------------------------------------------------------------------- atributos dinâmicos
/** As chaves de `atributos` mudam de base para base (são as colunas categóricas da tabela de
 * clientes). Descarta o que tem variedade demais para virar um select — o mesmo limiar de
 * categorias que o motor usa (`Config.max_categorias = 12`). */
export function atributosDisponiveis(clientes: ClienteFila[]): { chave: string; valores: string[] }[] {
  const mapa = new Map<string, Set<string>>()
  for (const c of clientes) {
    for (const [k, v] of Object.entries(c.atributos || {})) {
      if (v == null || v === '') continue
      if (!mapa.has(k)) mapa.set(k, new Set())
      mapa.get(k)!.add(String(v))
    }
  }
  return [...mapa.entries()]
    .filter(([, s]) => s.size > 1 && s.size <= 12)
    .map(([chave, s]) => ({ chave, valores: [...s].sort((a, b) => a.localeCompare(b, 'pt-BR')) }))
    .sort((a, b) => a.chave.localeCompare(b.chave, 'pt-BR'))
}

export const rotuloAtributo = (k: string): string => {
  const t = k.replace(/_/g, ' ')
  return t.charAt(0).toUpperCase() + t.slice(1)
}

// --------------------------------------------------------------------------- filtros
export interface Filtros {
  busca: string
  faixa: Faixa | ''
  atributos: Record<string, string>
  valorMinimo: string
}

export const FILTROS_VAZIOS: Filtros = { busca: '', faixa: '', atributos: {}, valorMinimo: '' }

/** Filtra preservando a ordem de chegada — que é a ordem da fila do motor. */
export function aplicarFiltros(clientes: ClienteFila[], f: Filtros): ClienteFila[] {
  const q = norm(f.busca)
  const min = f.valorMinimo === '' ? null : Number(f.valorMinimo)
  return clientes.filter((c) => {
    if (f.faixa && c.faixa_risco !== f.faixa) return false
    if (q && !norm(c.cliente_id).includes(q)) return false
    for (const [k, v] of Object.entries(f.atributos)) {
      if (v && String(c.atributos?.[k] ?? '') !== v) return false
    }
    if (min != null && !Number.isNaN(min) && min > 0) {
      if (c.valor_mensal == null || c.valor_mensal < min) return false
    }
    return true
  })
}

export const temFiltro = (f: Filtros): boolean =>
  Boolean(f.busca || f.faixa || f.valorMinimo) || Object.values(f.atributos).some(Boolean)

// --------------------------------------------------------------------------- alerta e evidências
/** "em alerta há 3 meses" e, no título, desde quando. */
export function desdeQuando(mesReferencia: string | null, meses: number): string | null {
  if (!meses) return null
  return recuarMes(mesReferencia, meses - 1)
}

/** Texto da comparação, na mesma regra do motor: 3 meses recentes contra os 6 anteriores. */
export function textoComparacao(e: Evidencia, nf: (x: number) => string): string {
  const m = e.comparacao
  if (!m) return e.detalhe
  const r = m.meses_recente === 1 ? 'no último mês' : `nos últimos ${m.meses_recente} meses`
  const a = m.meses_anterior === 1 ? 'no mês anterior' : `nos ${m.meses_anterior} meses anteriores`
  return `${nf(m.recente)} ${r} contra ${nf(m.anterior)} ${a}`
}

// --------------------------------------------------------------------------- versões
/** Uma linha da tabela "Modelos treinados": as execuções de todas as bases, achatadas. */
export interface LinhaVersao {
  chave: string
  base_id: string
  base_nome: string
  demonstracao: boolean
  execucao_id: string
  rotulo: string
  estado: Execucao['estado']
  etapa: string | null
  fracao: number
  mensagem: string | null
  problemas: string[]
  ativa: boolean
  criada_em: string | null
  duracao_s: number | null
  algoritmo: string | null
  variaveis: string | null
  auc_k1: number | null
  antecedencia: number | null
  alarme_hoje: number | null
  cancelados_pegos: string | null
  n_clientes: number | null
}

export function achatarExecucoes(
  bases: BaseResumo[],
  porBase: Record<string, Execucao[]>,
): LinhaVersao[] {
  const linhas: LinhaVersao[] = []
  for (const b of bases) {
    for (const e of porBase[b.base_id] || []) {
      const r = e.resumo
      const d = r?.deteccao
      linhas.push({
        chave: `${b.base_id}::${e.execucao_id}`,
        base_id: b.base_id,
        base_nome: b.nome,
        demonstracao: b.demonstracao,
        execucao_id: e.execucao_id,
        rotulo: e.rotulo || e.execucao_id,
        estado: e.estado,
        etapa: e.etapa,
        fracao: e.fracao,
        mensagem: e.mensagem,
        problemas: e.problemas || [],
        ativa: e.ativa,
        criada_em: e.criada_em,
        duracao_s: r?.duracao_s ?? null,
        algoritmo: r ? r.modelo.rotulo + (r.modelo.forcado ? ' (forçado)' : '') : null,
        variaveis: r ? `${r.variaveis.n_selecionadas} de ${r.variaveis.n_candidatas}` : null,
        auc_k1: r?.auc?.k1 ?? null,
        antecedencia: d?.antecedencia_mediana_alto ?? null,
        alarme_hoje: r?.alarme_falso?.alto?.hoje ?? null,
        cancelados_pegos: d ? `${d.alto.k1} de ${d.n_cancelados}` : null,
        n_clientes: e.n_clientes,
      })
    }
  }
  return linhas.sort((a, b) => String(b.criada_em ?? '').localeCompare(String(a.criada_em ?? '')))
}

export const EM_ANDAMENTO = (e: string): boolean => e === 'na_fila' || e === 'treinando'
