/** Pesos do score que ordena a fila: score = (p1·risco + p2·valor mensal + p3·variação do risco) ÷ (p1 + p2 + p3).
 *
 * `CamposPesosScore` serve ao card "Treinar" (já na hora do treino) e ao popup "Pesos" de cada versão pronta, que
 * chama `PATCH /execucoes/{id}` e refaz a ordem da fila no servidor, sem treinar de novo. O front não reordena:
 * quem calcula score e prioridade é o motor.
 */

import { useState } from 'react'
import * as api from '../api/cliente'
import { ErroApi } from '../api/cliente'
import type { PesosScore } from '../api/tipos'
import { CaixaErro } from '../componentes/Estados'
import { Popup } from '../componentes/Popup'
import { useToast } from '../componentes/Toast'

export interface RascunhoPesos { risco: string; valor: string; delta: string }

export const PESOS_PADRAO: RascunhoPesos = { risco: '0.6', valor: '0.3', delta: '0.1' }

const ITENS: { k: keyof PesosScore; rotulo: string }[] = [
  { k: 'risco', rotulo: 'Risco de cancelar (p1)' },
  { k: 'valor', rotulo: 'Valor mensal (p2)' },
  { k: 'delta', rotulo: 'Variação do risco vs. mês anterior (p3)' },
]

export const dePesos = (p?: PesosScore | null): RascunhoPesos =>
  p ? { risco: String(p.risco), valor: String(p.valor), delta: String(p.delta) } : { ...PESOS_PADRAO }

/** Pesos válidos (números ≥ 0, ao menos um > 0) ou null. */
export function lerPesos(r: RascunhoPesos): PesosScore | null {
  const p = { risco: Number(r.risco.replace(',', '.')), valor: Number(r.valor.replace(',', '.')),
              delta: Number(r.delta.replace(',', '.')) }
  const ok = Object.values(p).every((x) => Number.isFinite(x) && x >= 0) && p.risco + p.valor + p.delta > 0
  return ok ? p : null
}

/** "60% · 30% · 10%" — a participação de cada item. */
export function pesosTexto(p?: PesosScore | null): string {
  if (!p) return 'Perda anual ajustada'
  const t = p.risco + p.valor + p.delta || 1
  return [p.risco, p.valor, p.delta].map((x) => `${Math.round((100 * x) / t)}%`).join(' · ')
}

export function CamposPesosScore({ valor, aoMudar, id }: {
  valor: RascunhoPesos
  aoMudar: (v: RascunhoPesos) => void
  id: string
}) {
  const p = lerPesos(valor)
  return (
    <>
      <div className="pesos">
        {ITENS.map(({ k, rotulo }) => (
          <div key={k}>
            <label htmlFor={`${id}-${k}`}>{rotulo}</label>
            <input id={`${id}-${k}`} type="number" min={0} step={0.05} value={valor[k]}
                   onChange={(e) => aoMudar({ ...valor, [k]: e.target.value })} />
          </div>
        ))}
      </div>
      <p className="mut" style={{ marginTop: 12 }}>
        {p ? <>Peso relativo: risco, valor mensal e variação do risco = {pesosTexto(p)}</>
           : 'Informe pesos de 0 em diante, com ao menos um maior que 0.'}
      </p>
      <p className="mut" style={{ marginTop: 6, fontSize: 13 }}>
        Score de 0 a 100 = média ponderada dos três itens, cada um em escala de 0 a 100% dentro da fila
        (o valor é relativo ao maior contrato). A ordem da fila é calculada no servidor.
      </p>
    </>
  )
}

export function PopupPesos({ baseId, execucaoId, rotulo, atual, aoFechar, aoSalvar }: {
  baseId: string
  execucaoId: string
  rotulo: string
  atual: PesosScore | null
  aoFechar: () => void
  aoSalvar: () => void
}) {
  const toast = useToast()
  const [padrao, setPadrao] = useState(!atual)
  const [v, setV] = useState<RascunhoPesos>(dePesos(atual))
  const [salvando, setSalvando] = useState(false)
  const [erro, setErro] = useState<ErroApi | null>(null)
  const p = lerPesos(v)

  async function salvar() {
    if (!padrao && !p) return
    setSalvando(true)
    setErro(null)
    try {
      await api.atualizarExecucao(baseId, execucaoId, { pesos_score: padrao ? null : p })
      toast(padrao ? 'Fila de volta à ordem padrão do motor.' : 'Fila reordenada pelo novo score.')
      aoSalvar()
    } catch (e) {
      setErro(e instanceof ErroApi ? e : new ErroApi(0, String(e)))
      setSalvando(false)
    }
  }

  return (
    <Popup titulo={`Pesos do score · ${rotulo}`} largura="med" aoFechar={salvando ? () => {} : aoFechar}>
      <label style={{ display: 'flex', alignItems: 'center', gap: 8, margin: '14px 0 4px', cursor: 'pointer' }}>
        <input type="checkbox" checked={padrao} onChange={(e) => setPadrao(e.target.checked)} />
        Ordem padrão do motor (perda anual ajustada), sem score
      </label>
      {!padrao && <CamposPesosScore valor={v} aoMudar={setV} id="pp-pesos" />}
      <CaixaErro erro={erro} titulo="Não consegui salvar." />
      <div className="acoes">
        <button className="btn ghost" onClick={aoFechar} disabled={salvando}>Cancelar</button>
        <button className="btn" onClick={salvar} disabled={salvando || (!padrao && !p)}>
          {salvando ? 'Salvando…' : 'Salvar'}
        </button>
      </div>
    </Popup>
  )
}
