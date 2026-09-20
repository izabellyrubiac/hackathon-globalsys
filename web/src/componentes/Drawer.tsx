/** Painel lateral do cliente. Esc fecha e o foco vai para o botão Fechar, como no protótipo. */

import { useEffect, useRef, type ReactNode } from 'react'
import { useEsc } from './Toast'

export function Drawer({ rotulo, aoFechar, children }: {
  rotulo: string
  aoFechar: () => void
  children: ReactNode
}) {
  const fechar = useRef<HTMLButtonElement>(null)
  useEsc(aoFechar)
  useEffect(() => { fechar.current?.focus() }, [])

  return (
    <div className="drw" role="dialog" aria-label={rotulo}>
      <button ref={fechar} className="btn ghost sm" style={{ float: 'right' }} onClick={aoFechar}>
        Fechar
      </button>
      {children}
    </div>
  )
}
