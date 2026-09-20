/** Sparkline de 18 meses, porte fiel do `spark()` do protótipo.
 *
 * A faixa cinza marca os 3 últimos meses — a mesma janela que o motor usa como "recente"
 * na comparação das evidências (`motor/saida.py::_comparacao`).
 */

import type { PontoHistorico } from '../api/tipos'
import { mesFmt, nf } from '../formato'

const L = 360, A = 38

export function Sparkline({ historico, chave, rotulo }: {
  historico: PontoHistorico[]
  chave: string
  rotulo: string
}) {
  const pts = historico.map((h) => h.valores?.[chave])
  const ok = pts.filter((v): v is number => typeof v === 'number')
  if (ok.length < 2) return null

  const n = pts.length
  const mn = Math.min(...ok), mx = Math.max(...ok)
  const faixa = mx - mn || 1
  const X = (i: number) => 3 + (i * (L - 6)) / Math.max(1, n - 1)
  const Y = (v: number) => A - 4 - ((v - mn) / faixa) * (A - 8)

  // a linha pula os meses sem dado, ligando os pontos que existem
  const linha = pts
    .map((v, i) => (typeof v === 'number' ? [X(i), Y(v)] : null))
    .filter((p): p is number[] => p !== null)
  const ultimo = linha[linha.length - 1]

  const desc = `${rotulo}: evolução mensal de ${mesFmt(historico[0]?.mes_ref)} a `
    + `${mesFmt(historico[n - 1]?.mes_ref)}, de ${nf(ok[0])} a ${nf(ok[ok.length - 1])}.`

  return (
    <svg className="spark" viewBox={`0 0 ${L} ${A}`} role="img" aria-label={desc}>
      <rect x={X(n - 3) - 3} y={0} width={L - X(n - 3) + 3} height={A}
            fill="var(--edge)" opacity=".45" />
      <polyline
        points={linha.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(' ')}
        fill="none" stroke="var(--pri)" strokeWidth="2"
        strokeLinejoin="round" strokeLinecap="round"
      />
      {ultimo && <circle cx={ultimo[0].toFixed(1)} cy={ultimo[1].toFixed(1)} r="3" fill="var(--pri)" />}
    </svg>
  )
}
