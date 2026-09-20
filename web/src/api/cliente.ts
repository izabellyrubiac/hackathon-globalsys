/** Uma função por rota da API. Todas passam por `pedir`, que traduz o erro uniforme da API
 * (`{detalhe, problemas[]}`, garantido pelos exception handlers de `api/main.py`) em `ErroApi`.
 *
 * As rotas são relativas: o proxy do Vite leva "/api" para a API FastAPI (ver vite.config.ts).
 */

import type {
  BaseCriada, BaseDetalhe, BaseResumo, Execucao, Faixa, Inspecao, PaginaClientes, PedidoTreino,
  Status, TreinoAceito, Validacao,
} from './tipos'

export class ErroApi extends Error {
  status: number
  detalhe: string
  problemas: string[]

  constructor(status: number, detalhe: string, problemas: string[] = []) {
    super(detalhe)
    this.name = 'ErroApi'
    this.status = status
    this.detalhe = detalhe
    this.problemas = problemas
  }

  /** Mensagem pronta para a tela: o detalhe e, abaixo, cada problema em pt-BR. */
  get texto(): string {
    return this.problemas.length ? `${this.detalhe} ${this.problemas.join(' ')}` : this.detalhe
  }
}

async function pedir<T>(rota: string, init?: RequestInit): Promise<T> {
  let r: Response
  try {
    r = await fetch(rota, init)
  } catch (e) {
    if (e instanceof DOMException && e.name === 'AbortError') throw e
    throw new ErroApi(0, 'Não consegui falar com a API. Ela está no ar em http://localhost:8000?')
  }
  if (r.status === 204) return null as T
  const texto = await r.text()
  let corpo: unknown = null
  try {
    corpo = texto ? JSON.parse(texto) : null
  } catch {
    if (!r.ok) throw new ErroApi(r.status, texto.slice(0, 300) || `Erro ${r.status}.`)
    throw new ErroApi(r.status, 'A API respondeu algo que não é JSON.')
  }
  if (!r.ok) {
    const c = corpo as { detalhe?: string; problemas?: string[] } | null
    throw new ErroApi(r.status, c?.detalhe || `Erro ${r.status}.`, c?.problemas || [])
  }
  return corpo as T
}

const json = (corpo: unknown, sinal?: AbortSignal): RequestInit => ({
  method: 'POST',
  headers: { 'content-type': 'application/json' },
  body: JSON.stringify(corpo),
  signal: sinal,
})

// --------------------------------------------------------------------------- rotas
export const saude = (s?: AbortSignal) =>
  pedir<{ ok: boolean; versao: string; bases: number }>('/api/saude', { signal: s })

export const listarBases = (s?: AbortSignal) =>
  pedir<BaseResumo[]>('/api/bases', { signal: s })

export const detalheBase = (id: string, s?: AbortSignal) =>
  pedir<BaseDetalhe>(`/api/bases/${encodeURIComponent(id)}`, { signal: s })

export const inspecao = (id: string, s?: AbortSignal) =>
  pedir<Inspecao>(`/api/bases/${encodeURIComponent(id)}/inspecao`, { signal: s })

export function enviarBase(arquivos: File[], s?: AbortSignal) {
  const corpo = new FormData()
  for (const a of arquivos) corpo.append('arquivos', a, a.name)
  return pedir<BaseCriada>('/api/bases', { method: 'POST', body: corpo, signal: s })
}

export const treinar = (id: string, pedido: PedidoTreino, s?: AbortSignal) =>
  pedir<TreinoAceito>(`/api/bases/${encodeURIComponent(id)}/treinar`, json(pedido, s))

export const status = (id: string, s?: AbortSignal) =>
  pedir<Status>(`/api/bases/${encodeURIComponent(id)}/status`, { signal: s })

export const listarExecucoes = (id: string, s?: AbortSignal) =>
  pedir<{ base_id: string; execucao_ativa: string | null; execucoes: Execucao[] }>(
    `/api/bases/${encodeURIComponent(id)}/execucoes`, { signal: s })

export const execucao = (id: string, eid: string, s?: AbortSignal) =>
  pedir<Execucao>(
    `/api/bases/${encodeURIComponent(id)}/execucoes/${encodeURIComponent(eid)}`, { signal: s })

export const ativarExecucao = (id: string, eid: string) =>
  pedir<Execucao>(
    `/api/bases/${encodeURIComponent(id)}/execucoes/${encodeURIComponent(eid)}/ativar`,
    { method: 'POST' })

export const apagarExecucao = (id: string, eid: string) =>
  pedir<null>(`/api/bases/${encodeURIComponent(id)}/execucoes/${encodeURIComponent(eid)}`,
              { method: 'DELETE' })

export interface OpcoesFila {
  limite?: number
  desde?: number
  faixa?: Faixa[]
  incluir_historico?: boolean
}

export function fila(id: string, o: OpcoesFila = {}, s?: AbortSignal) {
  const q = new URLSearchParams()
  if (o.limite != null) q.set('limite', String(o.limite))
  if (o.desde != null) q.set('desde', String(o.desde))
  if (o.incluir_historico != null) q.set('incluir_historico', String(o.incluir_historico))
  for (const f of o.faixa || []) q.append('faixa', f)
  const cauda = q.toString()
  return pedir<PaginaClientes>(
    `/api/bases/${encodeURIComponent(id)}/clientes${cauda ? '?' + cauda : ''}`, { signal: s })
}

export const validacao = (id: string, s?: AbortSignal) =>
  pedir<Validacao>(`/api/bases/${encodeURIComponent(id)}/validacao`, { signal: s })

export const pesos = (id: string, s?: AbortSignal) =>
  pedir<Record<string, unknown>>(`/api/bases/${encodeURIComponent(id)}/pesos`, { signal: s })
