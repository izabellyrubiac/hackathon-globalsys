/** Roteamento por hash, como no protótipo: `#/fila`, `#/modelos`, `#/validacao`.
 *
 * São três telas sem parâmetro de caminho; um router de biblioteca não pagaria o próprio peso.
 * `useSyncExternalStore` sobre `hashchange` mantém o React em dia sem estado duplicado.
 */

import { useSyncExternalStore } from 'react'

export const ROTAS = ['fila', 'modelos', 'validacao'] as const
export type Rota = (typeof ROTAS)[number]

export const TITULO: Record<Rota, string> = {
  fila: 'Fila de atendimento',
  modelos: 'Modelos',
  validacao: 'Validação',
}

function ler(): Rota {
  const h = location.hash.replace(/^#\/?/, '').split('?')[0]
  return (ROTAS as readonly string[]).includes(h) ? (h as Rota) : 'fila'
}

function assinar(aviso: () => void) {
  window.addEventListener('hashchange', aviso)
  return () => window.removeEventListener('hashchange', aviso)
}

export const useRota = (): Rota => useSyncExternalStore(assinar, ler, () => 'fila' as Rota)

export const irPara = (r: Rota) => { location.hash = '#/' + r }
