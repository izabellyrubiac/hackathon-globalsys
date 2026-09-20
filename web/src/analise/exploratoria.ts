/** Análise exploratória antes do treino, calculada no navegador — porte do `analisar()` do
 * protótipo, com a mesma leitura do notebook de sinais.
 *
 * Por métrica mensal: média móvel de 3 meses por cliente, alinhada em meses até a saída
 * (cancelados: mês − mês da saída; ativos: último mês da base + 1, então −1 é o último mês com
 * dados nos dois grupos), e mediana e faixa P25–P75 por grupo e mês.
 *
 * Isto não fala com o motor: é uma olhada nos dados crus, para o usuário decidir o mapeamento.
 * A medida de verdade, validada por cliente, está na aba Validação.
 */

import { norm } from '../formato.ts'
import { colunasDe, type Planilhas } from './planilha.ts'

export const MESES_AN = 12
export const JANELA_AN = 3

/** Colunas que revelam o cancelamento e não podem virar variável — mesma ideia do motor. */
export const VAZAMENTO = /cancel|saida|situacao|status|churn/
export const TEMPO = /(^| )(mes|competencia|periodo)( |$)/

export interface Ponto {
  rel: number
  c: Quartis | null
  a: Quartis | null
}

export interface Quartis { med: number; p25: number; p75: number; n: number }

export interface Serie {
  tabela: string
  coluna: string
  rotulo: string
  auc: number
  antecedencia: number | null
  pontos: Ponto[]
}

export type Analise =
  | { erro: string }
  | { series: Serie[]; cancelados: number; ativos: number; referencia: string }

/** Índice absoluto do mês: aceita Date, "AAAA-MM", "AAAA/MM" e "MM/AAAA". */
export function mesIdx(v: unknown): number | null {
  if (v instanceof Date && !Number.isNaN(v.getTime())) {
    const d = new Date(v.getTime() + 432e5) // desloca para o fuso local antes de extrair o mês
    return d.getUTCFullYear() * 12 + d.getUTCMonth()
  }
  const t = String(v ?? '').trim()
  let m = t.match(/^(\d{4})[-/](\d{1,2})/)
  if (m) return +m[1]! * 12 + +m[2]! - 1
  m = t.match(/^(\d{1,2})[-/](\d{4})$/)
  return m ? +m[2]! * 12 + +m[1]! - 1 : null
}

export const idxParaMes = (i: number): string =>
  `${Math.floor(i / 12)}-${String((i % 12) + 1).padStart(2, '0')}`

export function quantil(ordenados: number[], p: number): number {
  const i = (ordenados.length - 1) * p
  const a = Math.floor(i), b = Math.ceil(i)
  return ordenados[a]! + (ordenados[b]! - ordenados[a]!) * (i - a)
}

export function quartis(v: number[]): Quartis | null {
  if (!v.length) return null
  const s = [...v].sort((a, b) => a - b)
  return { med: quantil(s, 0.5), p25: quantil(s, 0.25), p75: quantil(s, 0.75), n: s.length }
}

/** AUC de cancelados × ativos (Mann-Whitney, empates pela média dos postos). */
export function auc(pos: number[], neg: number[]): number {
  if (!pos.length || !neg.length) return 0.5
  const t = [...pos.map((v) => [v, 1]), ...neg.map((v) => [v, 0])].sort((a, b) => a[0]! - b[0]!)
  let soma = 0
  for (let i = 0; i < t.length;) {
    let j = i
    while (j < t.length && t[j]![0] === t[i]![0]) j++
    const posto = (i + j + 1) / 2
    for (let k = i; k < j; k++) if (t[k]![1]) soma += posto
    i = j
  }
  return (soma - (pos.length * (pos.length + 1)) / 2) / (pos.length * neg.length)
}

/** Maior k tal que, de −k até −1, a mediana de quem cancelou fica fora da faixa P25–P75 dos ativos. */
export function antecedencia(pontos: Ponto[]): number | null {
  let k = 0
  for (let r = -1; r >= -MESES_AN; r--) {
    const p = pontos[r + MESES_AN]
    if (!p?.c || !p.a) break
    if (!(p.c.med < p.a.p25 || p.c.med > p.a.p75)) break
    k = -r
  }
  return k || null
}

const PALAVRAS: Record<string, string> = {
  criticos: 'críticos', medio: 'médio', resolucao: 'resolução', reclamacoes: 'reclamações',
  reunioes: 'reuniões', nps: 'NPS', sla: 'SLA', servico: 'serviço', servicos: 'serviços',
  numero: 'número', ultimo: 'último', media: 'média', historico: 'histórico',
}

/** Nome legível de uma coluna: sem sufixo técnico (pct → %, h → horas) e com acentos de volta. */
export function rotuloSerie(c: string): string {
  let w = c.split('_').filter(Boolean)
  let un = ''
  if (w.includes('pct')) { w = w.filter((x) => x !== 'pct'); un = ' (%)' }
  else if (w.length > 1 && w[w.length - 1] === 'h') { w.pop(); un = ' (h)' }
  const t = w.map((x) => PALAVRAS[x.toLowerCase()] || x).join(' ') + un
  return t.charAt(0).toUpperCase() + t.slice(1)
}

export interface EntradaAnalise {
  planilhas: Planilhas
  colunaId: string
  alvoTabela: string
  /** Coluna com o mês da saída. Sem ela não dá para alinhar os grupos. */
  colunaDataSaida: string | null
}

export function analisar(e: EntradaAnalise): Analise {
  const { planilhas, colunaId, alvoTabela, colunaDataSaida } = e
  if (!colunaDataSaida) {
    return { erro: 'Para comparar quem saiu com quem ficou é preciso a coluna com o mês da saída. '
                 + 'Escolha-a em "O que prever".' }
  }
  const idn = norm(colunaId)
  const alvo = planilhas[alvoTabela]
  if (!alvo) return { erro: `A tabela '${alvoTabela}' não foi lida nesta sessão.` }
  const idAlvo = colunasDe(alvo).find((c) => norm(c) === idn)
  if (!idAlvo) return { erro: 'A tabela do cancelamento não tem a coluna do identificador.' }

  // saída: id → mês da saída (null = continua ativo)
  const saida = new Map<string, number | null>()
  for (const r of alvo) {
    const id = String(r[idAlvo] ?? '').trim()
    const v = r[colunaDataSaida]
    if (v === '' || v == null) saida.set(id, null)
    else {
      const i = mesIdx(v)
      if (i != null) saida.set(id, i)
    }
  }

  // colunas mensais numéricas de todas as tabelas ligadas ao cliente
  const cols: { tabela: string; coluna: string; colId: string; colMes: string }[] = []
  let maxMes = -Infinity
  for (const t of Object.keys(planilhas)) {
    const linhas = planilhas[t]!
    const nomes = colunasDe(linhas)
    const colId = nomes.find((c) => norm(c) === idn)
    const colMes = nomes.find((c) => TEMPO.test(norm(c)))
    if (!colId || !colMes) continue
    for (const r of linhas) {
      const i = mesIdx(r[colMes])
      if (i != null && i > maxMes) maxMes = i
    }
    for (const c of nomes) {
      const n = norm(c)
      if (n === idn || c === colMes || VAZAMENTO.test(n) || TEMPO.test(n)) continue
      const numericas = linhas.filter((r) => typeof r[c] === 'number').length
      if (numericas >= linhas.length * 0.5) cols.push({ tabela: t, coluna: c, colId, colMes })
    }
  }
  if (!cols.length) {
    return { erro: 'Nenhuma tabela mensal (com coluna de mês) e métrica numérica para analisar.' }
  }

  // Os ativos são alinhados a "último mês da base + 1": assim −1 é o último mês com dados nos
  // dois grupos, a mesma convenção do `mes_relativo` dos experimentos.
  const refAtivos = maxMes + 1
  const series: Serie[] = []
  const idsCancelados = new Set<string>()
  const idsAtivos = new Set<string>()

  for (const { tabela, coluna, colId, colMes } of cols) {
    // por cliente: mês relativo → média móvel de 3 meses
    const porCliente = new Map<string, Map<number, number>>()
    for (const r of planilhas[tabela]!) {
      const id = String(r[colId] ?? '').trim()
      if (!saida.has(id)) continue
      const i = mesIdx(r[colMes])
      const val = r[coluna]
      if (i == null || typeof val !== 'number') continue
      if (!porCliente.has(id)) porCliente.set(id, new Map())
      porCliente.get(id)!.set(i, val)
    }

    const porMes = new Map<number, { c: number[]; a: number[] }>()
    for (const [id, meses] of porCliente) {
      const ref = saida.get(id)
      const cancelou = ref != null
      const base = cancelou ? ref! : refAtivos
      ;(cancelou ? idsCancelados : idsAtivos).add(id)
      for (let rel = -MESES_AN; rel <= -1; rel++) {
        const abs = base + rel
        const janela: number[] = []
        for (let k = 0; k < JANELA_AN; k++) {
          const v = meses.get(abs - k)
          if (typeof v === 'number') janela.push(v)
        }
        if (!janela.length) continue
        const media = janela.reduce((s, x) => s + x, 0) / janela.length
        if (!porMes.has(rel)) porMes.set(rel, { c: [], a: [] })
        porMes.get(rel)![cancelou ? 'c' : 'a'].push(media)
      }
    }

    const pontos: Ponto[] = []
    for (let rel = -MESES_AN; rel <= -1; rel++) {
      const g = porMes.get(rel)
      pontos.push({ rel, c: g ? quartis(g.c) : null, a: g ? quartis(g.a) : null })
    }
    const ultimo = porMes.get(-1)
    series.push({
      tabela, coluna, rotulo: rotuloSerie(coluna),
      auc: ultimo ? auc(ultimo.c, ultimo.a) : 0.5,
      antecedencia: antecedencia(pontos),
      pontos,
    })
  }

  // Ordem = relevância: a separação no último mês, o mesmo critério da pré-seleção do motor.
  series.sort((a, b) => Math.abs(b.auc - 0.5) - Math.abs(a.auc - 0.5))
  return {
    series,
    cancelados: idsCancelados.size,
    ativos: idsAtivos.size,
    referencia: idxParaMes(maxMes),
  }
}
