/** Elementos pequenos e repetidos: faixa de risco, chip e o rótulo de "isto está desativado".
 *
 * As cores das faixas são fixas em todo o projeto (CLAUDE.md):
 * alto = vermelho, atenção = âmbar, baixo/saudável = azul-acinzentado.
 */

import type { ReactNode } from 'react'
import type { Faixa as TFaixa } from '../api/tipos'
import type { Pendencia } from '../desativado'
import { ROTULO_FAIXA } from '../formato'

export function Faixa({ faixa }: { faixa: TFaixa }) {
  return <span className={'fx ' + faixa}>{ROTULO_FAIXA[faixa] ?? faixa}</span>
}

export function Chip({ children, title }: { children: ReactNode; title?: string }) {
  return <span className="chip" title={title}>{children}</span>
}

/** Envolve um controle que existe na tela mas que o motor não sustenta.
 *
 * O conteúdo continua visível, apagado e sem receber clique; o motivo aparece embaixo e no
 * `title`. A lista completa está em PENDENCIAS-MOTOR.md, na raiz do projeto.
 */
export function Desativado({ pendencia, children }: { pendencia: Pendencia; children: ReactNode }) {
  return (
    <div className="off" title={pendencia.motivo}>
      {/* `pointer-events:none` vem do CSS; os controles internos também vão `disabled`,
          para não ficarem alcançáveis pelo Tab. */}
      <div className="off-c" aria-disabled="true">{children}</div>
      <p className="mut off-n">
        {pendencia.motivo}{' '}
        <span className="mut">Ver <code>PENDENCIAS-MOTOR.md</code>.</span>
      </p>
    </div>
  )
}
