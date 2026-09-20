/** Painel lateral de um cliente: por que ele está na fila e o que fazer.
 *
 * A tabela da fila é carregada sem histórico (resposta bem menor). Aqui o histórico é buscado
 * sob demanda, por posição: `prioridade` é 1..N, única e contígua, e a API pagina a fila na
 * mesma ordem sem nunca reordenar — então `desde = prioridade − 1, limite = 1` devolve
 * exatamente este cliente. A resposta é conferida pelo `cliente_id` antes de ser usada.
 */

import { useMemo } from 'react'
import * as api from '../api/cliente'
import { useRecurso } from '../api/ganchos'
import type { ClienteFila, Evidencia, ModeloFila } from '../api/tipos'
import { Drawer } from '../componentes/Drawer'
import { Faixa } from '../componentes/Faixa'
import { Sparkline } from '../componentes/Sparkline'
import { SETA, desdeQuando, textoComparacao, variacao } from '../dominio'
import { brl, mesFmt, nf, pc, sg } from '../formato'

export function GavetaCliente({ baseId, execucaoId, cliente, modelo, mesReferencia, aoFechar }: {
  baseId: string
  execucaoId: string | null
  cliente: ClienteFila
  modelo: ModeloFila | null
  mesReferencia: string | null
  aoFechar: () => void
}) {
  const prio = cliente.prioridade
  const { dados } = useRecurso(
    (s) => api.fila(baseId, { desde: prio - 1, limite: 1, incluir_historico: true }, s),
    [baseId, execucaoId, prio],
  )

  // só aproveita a resposta se ela for mesmo deste cliente
  const completo: ClienteFila = useMemo(() => {
    const c = dados?.clientes?.[0]
    return c && c.cliente_id === cliente.cliente_id ? c : cliente
  }, [dados, cliente])

  const historico = completo.historico || []
  const v = variacao(completo)
  const desde = desdeQuando(mesReferencia, completo.meses_em_alerta)
  const rotuloSerie = (chave: string) =>
    modelo?.series_historico.find((s) => s.chave === chave)?.rotulo || chave

  const atributos = Object.values(completo.atributos || {}).filter(Boolean).join(' · ')

  return (
    <Drawer rotulo={`Cliente ${completo.cliente_id}`} aoFechar={aoFechar}>
      <h2>{completo.cliente_id}</h2>
      <p className="mut">{atributos || '—'}</p>

      <p style={{ marginTop: 12 }}>
        Posição {prio} na fila · <Faixa faixa={completo.faixa_risco} /> · chance de cancelar{' '}
        <b>{pc(completo.risco)}</b>
        {completo.valor_mensal != null && <> · {brl(completo.valor_mensal)}/mês</>}
      </p>

      {v.tipo === 'sem-dado' ? (
        <p className="mut" style={{ marginTop: 8 }}>
          Variação vs. mês anterior indisponível nesta versão do modelo.
        </p>
      ) : (
        <p className={'var-c ' + v.tipo} style={{ marginTop: 8 }}>
          <span className="mut">Variação vs. mês anterior:</span>{' '}
          <b>{SETA[v.tipo]} {sg(Math.round(v.rel))}%</b>{' '}
          <small style={{ display: 'inline' }}>{sg(+v.pp.toFixed(1))} p.p.</small>
        </p>
      )}

      <p className="mut" style={{ marginTop: 4 }}>
        {completo.perda_anual_esperada != null &&
          <>Perda anual esperada {brl(completo.perda_anual_esperada)} · </>}
        {completo.meses_em_alerta > 0
          ? `em alerta há ${completo.meses_em_alerta} ${completo.meses_em_alerta === 1 ? 'mês' : 'meses'}`
            + (desde ? ` (desde ${mesFmt(desde)})` : '')
          : 'sem meses seguidos em alerta'}
      </p>

      <h3>Evidências</h3>
      {completo.evidencias.length ? (
        completo.evidencias.map((e) => (
          <Sinal key={e.sinal} e={e} historico={historico} rotulo={rotuloSerie(e.chave_serie || '')} />
        ))
      ) : (
        <p className="mut">Nenhum sinal se destaca.</p>
      )}

      {historico.length > 0 && (
        <p className="mut" style={{ fontSize: 12 }}>
          Gráficos: {mesFmt(historico[0]?.mes_ref)} a {mesFmt(historico[historico.length - 1]?.mes_ref)};
          a faixa cinza marca os 3 últimos meses.
        </p>
      )}

      {completo.fatores_secundarios.length > 0 && (
        <details className="mais-sinais">
          <summary>Outros sinais ({completo.fatores_secundarios.length})</summary>
          {completo.fatores_secundarios.map((e) => (
            <Sinal key={e.sinal} e={e} historico={historico} rotulo={rotuloSerie(e.chave_serie || '')} />
          ))}
        </details>
      )}

      <h3>O que fazer</h3>
      <p>{completo.acao_sugerida || 'Manter o acompanhamento de rotina.'}</p>
      {completo.sinal_dominante && (
        <p className="mut" style={{ marginTop: 6, fontSize: 13 }}>
          Sinal que mais pesa: {completo.coluna_dominante || completo.sinal_dominante}.
        </p>
      )}
    </Drawer>
  )
}

function Sinal({ e, historico, rotulo }: {
  e: Evidencia
  historico: ClienteFila['historico']
  rotulo: string
}) {
  return (
    <div className="ev">
      <b>{e.titulo}</b>
      <br />
      <small>{textoComparacao(e, nf)}</small>
      {e.chave_serie && historico && historico.length > 0 && (
        <Sparkline historico={historico} chave={e.chave_serie} rotulo={rotulo} />
      )}
    </div>
  )
}
