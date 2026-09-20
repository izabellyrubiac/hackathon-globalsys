/** Mediana e faixa P25–P75 de uma métrica, quem saiu contra quem ficou, mês a mês até a saída.
 * Porte do `grafico()` do protótipo: SVG à mão, com título descritivo, eixos nomeados e legenda —
 * o gráfico tem de se explicar sozinho.
 */

import type { Serie } from '../api/tipos'
import { nf, sg } from '../formato'

const L = 960, A = 240
const MARG = { esq: 56, dir: 16, topo: 16, base: 40 }

const COR = { c: 'var(--hi)', a: 'var(--lo)' }
const NOME = { c: 'Cancelaram', a: 'Continuam ativos' }

/** Marcas de eixo redondas dentro de [mn, mx]. */
function ticks(mn: number, mx: number, alvo = 5): number[] {
  const bruto = (mx - mn) / alvo || 1
  const mag = 10 ** Math.floor(Math.log10(bruto))
  const passo = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((p) => p >= bruto) ?? mag * 10
  const saida: number[] = []
  for (let v = Math.ceil(mn / passo) * passo; v <= mx + 1e-9; v += passo) saida.push(+v.toFixed(6))
  return saida
}

export function GraficoSeries({ serie }: { serie: Serie }) {
  const pontos = serie.pontos.filter((p) => p.c || p.a)
  if (pontos.length < 2) return null

  const todos: number[] = []
  for (const p of pontos) for (const g of [p.c, p.a]) if (g) todos.push(g.p25, g.p75, g.med)
  const mn = Math.min(...todos), mx = Math.max(...todos)
  const folga = (mx - mn) * 0.08 || 1
  const y0 = mn - folga, y1 = mx + folga

  const rels = pontos.map((p) => p.rel)
  const rMin = Math.min(...rels), rMax = Math.max(...rels)
  const X = (r: number) => MARG.esq + ((r - rMin) / (rMax - rMin || 1)) * (L - MARG.esq - MARG.dir)
  const Y = (v: number) => A - MARG.base - ((v - y0) / (y1 - y0 || 1)) * (A - MARG.topo - MARG.base)
  const f = (n: number) => n.toFixed(1)

  const area = (g: 'c' | 'a') => {
    const ps = pontos.filter((p) => p[g])
    if (ps.length < 2) return null
    const cima = ps.map((p) => `${f(X(p.rel))},${f(Y(p[g]!.p75))}`).join(' ')
    const baixo = [...ps].reverse().map((p) => `${f(X(p.rel))},${f(Y(p[g]!.p25))}`).join(' ')
    return `${cima} ${baixo}`
  }
  const linha = (g: 'c' | 'a') =>
    pontos.filter((p) => p[g]).map((p) => `${f(X(p.rel))},${f(Y(p[g]!.med))}`).join(' ')

  const desc = `${serie.rotulo}: mediana e faixa entre o primeiro e o terceiro quartil, por mês `
    + `até a saída. Quem cancelou em vermelho, quem ficou em azul-acinzentado. `
    + (serie.antecedencia
        ? `Os dois grupos se separam desde ${serie.antecedencia} meses antes da saída.`
        : 'Os dois grupos não se separam de forma clara.')

  return (
    <div style={{ marginTop: 14 }}>
      <h3>
        {serie.rotulo}{' '}
        <span className="mut" style={{ fontWeight: 400 }}>
          · {serie.tabela} · {serie.antecedencia
            ? `separa desde o mês −${serie.antecedencia}`
            : 'não separa'} · AUC {nf(+serie.auc.toFixed(3))}
        </span>
      </h3>
      <svg viewBox={`0 0 ${L} ${A}`} role="img" aria-label={desc} style={{ width: '100%', height: 'auto' }}>
        {ticks(y0, y1).map((v) => (
          <g key={v}>
            <line x1={MARG.esq} x2={L - MARG.dir} y1={f(Y(v))} y2={f(Y(v))}
                  stroke="var(--line)" strokeWidth="1" />
            <text x={MARG.esq - 8} y={Y(v) + 4} textAnchor="end" fontSize="11" fill="var(--mut)">
              {nf(v)}
            </text>
          </g>
        ))}
        {pontos.map((p) => (
          <text key={p.rel} x={f(X(p.rel))} y={A - MARG.base + 16} textAnchor="middle"
                fontSize="11" fill="var(--mut)">{sg(p.rel)}</text>
        ))}
        <text x={(L + MARG.esq) / 2} y={A - 6} textAnchor="middle" fontSize="12" fill="var(--mut)">
          meses até a saída (−1 = último mês com dados nos dois grupos)
        </text>

        {(['a', 'c'] as const).map((g) => {
          const pol = area(g)
          return pol ? <polygon key={g} points={pol} fill={COR[g]} opacity=".12" /> : null
        })}
        {(['a', 'c'] as const).map((g) => (
          <polyline key={g} points={linha(g)} fill="none" stroke={COR[g]} strokeWidth="2"
                    strokeLinejoin="round" strokeLinecap="round" />
        ))}
        {(['a', 'c'] as const).map((g) =>
          pontos.filter((p) => p[g]).map((p) => (
            <circle key={g + p.rel} cx={f(X(p.rel))} cy={f(Y(p[g]!.med))} r="3" fill={COR[g]}>
              {/* o <title> do SVG é a dica ao passar o mouse; tem de ser um texto só */}
              <title>{`${NOME[g]} · mês ${sg(p.rel)}: mediana ${nf(p[g]!.med)} `
                + `(P25–P75: ${nf(p[g]!.p25)} a ${nf(p[g]!.p75)}), ${p[g]!.n} clientes`}</title>
            </circle>
          )),
        )}

        <g fontSize="12">
          <rect x={L - MARG.dir - 190} y={MARG.topo} width="10" height="10" fill={COR.c} />
          <text x={L - MARG.dir - 175} y={MARG.topo + 9} fill="var(--ink)">Cancelaram</text>
          <rect x={L - MARG.dir - 190} y={MARG.topo + 16} width="10" height="10" fill={COR.a} />
          <text x={L - MARG.dir - 175} y={MARG.topo + 25} fill="var(--ink)">Continuam ativos</text>
        </g>
      </svg>
    </div>
  )
}
