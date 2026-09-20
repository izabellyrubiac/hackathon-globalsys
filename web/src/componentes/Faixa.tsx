/** Elementos pequenos e repetidos: faixa de risco e chip.
 *
 * As cores das faixas são fixas em todo o projeto (CLAUDE.md):
 * alto = vermelho, atenção = âmbar, baixo/saudável = azul-acinzentado.
 */

import type { ReactNode } from 'react'
import type { Faixa as TFaixa } from '../api/tipos'
import { ROTULO_FAIXA } from '../formato'

export function Faixa({ faixa }: { faixa: TFaixa }) {
  return <span className={'fx ' + faixa}>{ROTULO_FAIXA[faixa] ?? faixa}</span>
}

export function Chip({ children, title }: { children: ReactNode; title?: string }) {
  return <span className="chip" title={title}>{children}</span>
}
