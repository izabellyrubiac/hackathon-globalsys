/** "Analisar dados": uma olhada nos dados crus antes de treinar, calculada no navegador.
 *
 * Só funciona sobre a planilha que o usuário enviou nesta sessão. Numa base de demonstração —
 * que já estava no servidor e nunca passou por este navegador — não há arquivo para ler, e o
 * botão fica desativado com o motivo. Ver PENDENCIAS-MOTOR.md.
 */

import { useState } from 'react'
import { analisar, type Analise } from '../analise/exploratoria'
import type { Planilhas } from '../analise/planilha'
import { Desativado } from '../componentes/Faixa'
import { GraficoSeries } from '../componentes/GraficoSeries'
import { PENDENCIAS } from '../desativado'
import { mesFmt } from '../formato'

export function CardAnalise({ planilhas, colunaId, alvoTabela, colunaDataSaida, pronto }: {
  planilhas: Planilhas | null
  colunaId: string
  alvoTabela: string
  colunaDataSaida: string | null
  pronto: boolean
}) {
  const [analise, setAnalise] = useState<Analise | null>(null)

  const botao = (
    <button
      className="btn"
      disabled={!pronto || !planilhas}
      title={pronto ? undefined : 'Escolha a tabela de clientes, o identificador e a coluna de cancelamento.'}
      onClick={() => setAnalise(analisar({ planilhas: planilhas!, colunaId, alvoTabela, colunaDataSaida }))}
    >
      Analisar dados
    </button>
  )

  return (
    <div className="card">
      <h2>Análise dos dados</h2>
      <p className="sub mut">
        Compara, métrica a métrica, quem saiu com quem ficou nos meses antes da saída. É uma
        olhada exploratória: a medida validada por cliente está na aba Validação.
      </p>

      {!planilhas ? (
        <Desativado pendencia={PENDENCIAS.analiseSemArquivo}>{botao}</Desativado>
      ) : (
        <div style={{ marginTop: 12, display: 'flex', gap: 8 }}>
          {botao}
          {analise && (
            <button className="btn ghost" onClick={() => setAnalise(null)}>Ocultar</button>
          )}
        </div>
      )}

      {analise && 'erro' in analise && (
        <div className="aviso erro" role="alert" style={{ marginTop: 14 }}>{analise.erro}</div>
      )}

      {analise && 'series' in analise && (
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
