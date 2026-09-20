/** Caixa modal sobre a tela, com o mesmo comportamento do protótipo: foco preso dentro,
 * Esc fecha e clique no fundo não fecha (para não perder um arquivo já escolhido).
 */

import { useEffect, useRef, type ReactNode } from 'react'
import { useEsc } from './Toast'

const FOCAVEIS = 'a[href],button:not([disabled]),input:not([disabled]),select:not([disabled]),textarea,[tabindex]:not([tabindex="-1"])'

export function Popup({ titulo, largura, aoFechar, children }: {
  titulo: string
  /** "med" é a caixa estreita de 560px; o padrão é a grande (80vw × 80vh). */
  largura?: 'med'
  aoFechar: () => void
  children: ReactNode
}) {
  const caixa = useRef<HTMLDivElement>(null)
  useEsc(aoFechar)

  useEffect(() => {
    const el = caixa.current
    if (!el) return
    el.querySelector<HTMLElement>(FOCAVEIS)?.focus()
    const prender = (e: KeyboardEvent) => {
      if (e.key !== 'Tab') return
      const itens = [...el.querySelectorAll<HTMLElement>(FOCAVEIS)].filter((x) => x.offsetParent !== null)
      if (!itens.length) return
      const primeiro = itens[0], ultimo = itens[itens.length - 1]
      if (e.shiftKey && document.activeElement === primeiro) { e.preventDefault(); ultimo.focus() }
      else if (!e.shiftKey && document.activeElement === ultimo) { e.preventDefault(); primeiro.focus() }
    }
    el.addEventListener('keydown', prender)
    return () => el.removeEventListener('keydown', prender)
  }, [])

  const id = 'pp-titulo'
  return (
    <div className="ov">
      <div ref={caixa} className={'pp' + (largura === 'med' ? ' med' : '')}
           role="dialog" aria-modal="true" aria-labelledby={id}>
        <h2 id={id}>{titulo}</h2>
        {children}
      </div>
    </div>
  )
}
