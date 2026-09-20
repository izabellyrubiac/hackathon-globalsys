/** "Analisar dados": uma olhada nos dados antes de treinar, calculada no servidor (`POST /api/bases/{id}/analise`).
 *
 * Funciona em qualquer base guardada — enviada agora, de demonstração ou aberta depois de recarregar a página —,
 * porque quem lê os dados é a API. É exploratória: a medida validada por cliente está na aba Validação.
 */

import { useEffect, useState } from 'react'
import * as api from '../api/cliente'
import { ErroApi } from '../api/cliente'
import type { Analise, Mapeamento } from '../api/tipos'
import { CaixaErro } from '../componentes/Estados'
import { GraficoSeries } from '../componentes/GraficoSeries'
import { mesFmt } from '../formato'

export function CardAnalise({ baseId, mapeamento, pronto }: {
  baseId: string
  /** Mapeamento do formulário; só vale quando `pronto`. */
  mapeamento: Mapeamento
  pronto: boolean
}) {
  const [analise, setAnalise] = useState<Analise | null>(null)
  const [erro, setErro] = useState<ErroApi | null>(null)
  const [calculando, setCalculando] = useState(false)

  // Outra base ou outro mapeamento: o resultado antigo deixou de valer.
  const chave = baseId + JSON.stringify(mapeamento)
  useEffect(() => { setAnalise(null); setErro(null) }, [chave])

  async function analisar() {
    setCalculando(true)
    setErro(null)
    try {
      setAnalise(await api.analisarBase(baseId, mapeamento))
    } catch (e) {
      setAnalise(null)
      setErro(e instanceof ErroApi ? e : new ErroApi(0, String(e)))
    } finally {
      setCalculando(false)
    }
  }

  return (
    <div className="card">
      <h2>Análise dos dados</h2>
      <p className="sub mut">
        Compara, métrica a métrica, quem saiu com quem ficou nos meses antes da saída. É uma
        olhada exploratória: a medida validada por cliente está na aba Validação.
      </p>

      <div style={{ marginTop: 12, display: 'flex', gap: 8 }}>
        <button
          className="btn" disabled={!pronto || calculando} onClick={analisar}
          title={pronto ? undefined : 'Escolha a tabela de clientes, o identificador e a coluna de cancelamento.'}
        >
          {calculando ? 'Analisando…' : 'Analisar dados'}
        </button>
        {analise && <button className="btn ghost" onClick={() => setAnalise(null)}>Ocultar</button>}
      </div>

      <CaixaErro erro={erro} titulo="Não consegui analisar." />

      {analise && (
        <>
          <p className="mut" style={{ margin: '16px 0 4px', fontSize: 13 }}>
            {analise.cancelados} clientes que saíram contra {analise.ativos} que continuam, com a
            base indo até {mesFmt(analise.referencia)}. Cada série é a média móvel de 3 meses por
            cliente; a linha é a mediana do grupo e a faixa vai do primeiro ao terceiro quartil.
            Em ordem de separação entre os dois grupos.
          </p>
          {analise.series.map((s) => <GraficoSeries key={s.tabela + '.' + s.coluna} serie={s} />)}
        </>
      )}
    </div>
  )
}
