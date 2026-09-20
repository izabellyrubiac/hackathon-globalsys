/** Formatação pt-BR do projeto: `R$ 12.345` e `jun/2026` (convenção do CLAUDE.md). */

export const MES = ['jan', 'fev', 'mar', 'abr', 'mai', 'jun',
                    'jul', 'ago', 'set', 'out', 'nov', 'dez']

export const brl = (n: number | null | undefined): string =>
  n == null ? '—' : 'R$ ' + Math.round(n).toLocaleString('pt-BR')

export const pc = (p: number | null | undefined): string =>
  p == null ? '—' : Math.round(p * 100) + '%'

/** Uma casa decimal quando o número é pequeno; inteiro quando é grande. */
export const nf = (x: number | null | undefined): string =>
  x == null ? '—' : x.toLocaleString('pt-BR', { maximumFractionDigits: 1 })

export const num = (x: number | null | undefined, casas = 2): string =>
  x == null ? '—' : x.toLocaleString('pt-BR', { minimumFractionDigits: casas, maximumFractionDigits: casas })

/** Sinal explícito, com o menos tipográfico. */
export const sg = (x: number): string =>
  (x > 0 ? '+' : x < 0 ? '−' : '') + Math.abs(x).toLocaleString('pt-BR', { maximumFractionDigits: 1 })

/** "2026-06" → "jun/2026" */
export const mesFmt = (m: string | null | undefined): string =>
  m && m.length >= 7 ? MES[+m.slice(5, 7) - 1] + '/' + m.slice(0, 4) : '—'

/** "2026-09-20T00:16:43-03:00" → "20/set/2026 00:16" */
export function dataHora(iso: string | null | undefined): string {
  if (!iso || iso.length < 16) return '—'
  return `${iso.slice(8, 10)}/${MES[+iso.slice(5, 7) - 1]}/${iso.slice(0, 4)} ${iso.slice(11, 16)}`
}

/** Recua N meses a partir de "AAAA-MM". */
export function recuarMes(mes: string | null | undefined, n: number): string | null {
  if (!mes || mes.length < 7) return null
  const total = +mes.slice(0, 4) * 12 + (+mes.slice(5, 7) - 1) - n
  if (total < 0) return null
  return `${Math.floor(total / 12)}-${String((total % 12) + 1).padStart(2, '0')}`
}

/** Sem acento, minúsculo — para busca e comparação de nomes de coluna. */
export const norm = (s: unknown): string =>
  String(s ?? '').normalize('NFD').replace(/[̀-ͯ]/g, '')
    .toLowerCase().replace(/[^a-z0-9%]+/g, ' ').trim()

export const kb = (n: number): string =>
  (n / 1024).toLocaleString('pt-BR', { maximumFractionDigits: 0 }) + ' KB'

export const ROTULO_FAIXA: Record<string, string> = {
  alto: 'Alto', atencao: 'Atenção', baixo: 'Baixo',
}
