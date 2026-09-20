/** Fila de atendimento: com quem falar, por que e em que ordem.
 *
 * A ordem é a do motor (`prioridade`, da perda anual ajustada pelo delta de risco) e **nunca**
 * é refeita aqui: filtrar só esconde linhas. A faixa vem do risco, por corte de probabilidade,
 * nunca da capacidade da equipe — todos os ativos entram na fila.
 */

import { useMemo, useState } from 'react'
import * as api from '../api/cliente'
import { useRecurso } from '../api/ganchos'
import type { BaseResumo, ClienteFila, Faixa as TFaixa, PaginaClientes } from '../api/tipos'
import { Carregando, CaixaErro, Vazio } from '../componentes/Estados'
import { Chip, Desativado, Faixa } from '../componentes/Faixa'
import { CampoSelect, Select } from '../componentes/Select'
import { PENDENCIAS } from '../desativado'
import {
  FILTROS_VAZIOS, SETA, aplicarFiltros, atributosDisponiveis, rotuloAtributo, temFiltro, variacao,
  type Filtros,
} from '../dominio'
import { useSessao } from '../estado/Sessao'
import { brl, mesFmt, pc, sg } from '../formato'
import { irPara } from '../rotas'
import { GavetaCliente } from './GavetaCliente'

const PASSO = 20

export function Fila({ bases, carregandoBases }: { bases: BaseResumo[]; carregandoBases: boolean }) {
  const { baseId, escolherBase, versao } = useSessao()
  const prontas = bases.filter((b) => b.execucao_ativa)
  const atual = prontas.find((b) => b.base_id === baseId) || prontas[0] || null

  const [filtros, setFiltros] = useState<Filtros>(FILTROS_VAZIOS)
  const [mostrar, setMostrar] = useState(PASSO)
  const [aberto, setAberto] = useState<string | null>(null)

  // A tabela dispensa o histórico: a resposta fica muito menor e a gaveta busca o do cliente
  // aberto por conta própria.
  const { dados, carregando, erro } = useRecurso(
    (s) => api.fila(atual!.base_id, { limite: 5000, incluir_historico: false }, s),
    [atual?.base_id, atual?.execucao_ativa, versao],
    Boolean(atual),
  )

  const clientes = dados?.clientes ?? []
  const atributos = useMemo(() => atributosDisponiveis(clientes), [clientes])
  const filtrados = useMemo(() => aplicarFiltros(clientes, filtros), [clientes, filtros])
  const visiveis = filtrados.slice(0, mostrar)
  const clienteAberto = clientes.find((c) => c.cliente_id === aberto) || null
  const temValor = dados?.modelo?.coluna_valor != null

  function mudarFiltro(f: Partial<Filtros>) {
    setFiltros((a) => ({ ...a, ...f }))
    setMostrar(PASSO)
  }

  if (carregandoBases) return <Carregando texto="Procurando as bases…" />

  if (!prontas.length) {
    return (
      <>
        <div className="head"><h1>Fila de atendimento</h1></div>
        <div className="card semmodelo" role="status">
          <h2>Nenhum modelo treinado</h2>
          <p className="mut">
            A fila é gerada por um modelo. Treine o primeiro na aba Modelos — ou use uma das bases
            de demonstração, que já vêm prontas.
          </p>
          <button className="btn" onClick={() => irPara('modelos')}>Ir para Modelos</button>
        </div>
      </>
    )
  }

  return (
    <>
      <div className="head cab-modelo">
        <h1>Fila de atendimento</h1>
        {prontas.length > 1 && (
          <div className="sel-modelo">
            <label htmlFor="f-base">Base</label>
            <Select
              id="f-base" rotulo="Base" valor={atual?.base_id || ''}
              opcoes={prontas.map((b) => ({
                valor: b.base_id,
                texto: b.nome + (b.demonstracao ? ' · demonstração' : ''),
              }))}
              aoMudar={(v) => { escolherBase(v); setFiltros(FILTROS_VAZIOS); setMostrar(PASSO) }}
            />
          </div>
        )}
      </div>

      <CaixaErro erro={erro} />
      {carregando && !dados && <Carregando texto="Montando a fila…" />}

      {dados && (
        <>
          <ResumoFila dados={dados} />
          <div className="fila">
            <div className="card" id="fila">
              <div className="cab">
                <h2>Ordem de atendimento</h2>
                <span className="mut">
                  {filtrados.length === clientes.length
                    ? `${clientes.length} clientes`
                    : `${filtrados.length} de ${clientes.length} clientes`}
                </span>
              </div>
              <p className="mut" style={{ margin: '6px 0 12px', fontSize: 13 }}>
                {dados.modelo?.formula_prioridade
                  ? `Ordem definida pelo motor: ${dados.modelo.formula_prioridade}.`
                  : 'Ordem definida pelo motor.'}{' '}
                A tela não reordena a fila; o filtro só esconde linhas.
              </p>

              {visiveis.length ? (
                <div className="tw">
                  <table>
                    <thead>
                      <tr>
                        <th title="Posição na fila, definida pelo motor. A faixa vem do risco.">#</th>
                        <th>Cliente</th>
                        {temValor && <th>Valor mensal</th>}
                        <th>Chance de cancelamento</th>
                        <th>Variação vs. mês anterior</th>
                        {temValor && <th>Perda anual esperada</th>}
                        <th>Por quê</th>
                        <th>Em alerta</th>
                      </tr>
                    </thead>
                    <tbody>
                      {visiveis.map((c) => (
                        <LinhaFila key={c.cliente_id} c={c} temValor={temValor}
                               aoAbrir={() => setAberto(c.cliente_id)} />
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <Vazio>Nenhum cliente com esses filtros.</Vazio>
              )}

              {filtrados.length > visiveis.length && (
                <div className="mais">
                  <button className="btn sm" onClick={() => setMostrar((m) => m + PASSO)}>
                    Ver mais ({filtrados.length - visiveis.length})
                  </button>
                </div>
              )}
            </div>

            <aside className="card filtros" aria-label="Filtros">
              <h2>Filtros</h2>

              <label htmlFor="f-q" style={{ marginTop: 0 }}>Buscar cliente</label>
              <input id="f-q" type="search" placeholder="Ex.: C017" value={filtros.busca}
                     onChange={(e) => mudarFiltro({ busca: e.target.value })} />

              <span className="lb" id="f-nivel">Nível de risco</span>
              <div className="fxs" role="group" aria-labelledby="f-nivel">
                {([['', 'Todos'], ['alto', 'Alto'], ['atencao', 'Atenção'], ['baixo', 'Baixo']] as const)
                  .map(([k, t]) => (
                    <button key={k} aria-pressed={filtros.faixa === k}
                            onClick={() => mudarFiltro({ faixa: k as TFaixa | '' })}>{t}</button>
                  ))}
              </div>

              {atributos.map((a) => (
                <CampoSelect
                  key={a.chave} rotulo={rotuloAtributo(a.chave)}
                  valor={filtros.atributos[a.chave] || ''} vazio="Todos"
                  opcoes={a.valores.map((v) => ({ valor: v, texto: v }))}
                  aoMudar={(v) => mudarFiltro({ atributos: { ...filtros.atributos, [a.chave]: v } })}
                />
              ))}

              {temValor && (
                <>
                  <label htmlFor="f-vmin">Valor mensal mínimo (R$)</label>
                  <input id="f-vmin" type="number" min={0} step={500} value={filtros.valorMinimo}
                         onChange={(e) => mudarFiltro({ valorMinimo: e.target.value })} />
                </>
              )}

              <Desativado pendencia={PENDENCIAS.inicioContrato}>
                <label htmlFor="f-desde">Contrato iniciado desde</label>
                <input id="f-desde" type="month" disabled tabIndex={-1} />
              </Desativado>

              <button className="btn ghost" disabled={!temFiltro(filtros)}
                      onClick={() => { setFiltros(FILTROS_VAZIOS); setMostrar(PASSO) }}>
                Limpar filtros
              </button>
            </aside>
          </div>
        </>
      )}

      {clienteAberto && atual && (
        <GavetaCliente
          baseId={atual.base_id} execucaoId={dados?.execucao_id ?? null}
          cliente={clienteAberto} modelo={dados?.modelo ?? null}
          mesReferencia={dados?.mes_referencia ?? null}
          aoFechar={() => setAberto(null)}
        />
      )}
    </>
  )
}

export function LinhaFila({ c, temValor, aoAbrir }: {
  c: ClienteFila; temValor: boolean; aoAbrir: () => void
}) {
  const v = variacao(c)
  return (
    <tr className="r" tabIndex={0} onClick={aoAbrir}
        onKeyDown={(e) => { if (e.key === 'Enter') aoAbrir() }}>
      <td>
        <div className={'sc ' + c.faixa_risco}>
          <b>{c.prioridade}</b>
          <div className="sbar"><i style={{ width: `${Math.round(c.risco * 100)}%` }} /></div>
        </div>
        <Faixa faixa={c.faixa_risco} />
      </td>
      <td>
        <b>{c.cliente_id}</b>
        <br />
        <small className="mut">
          {Object.values(c.atributos || {}).filter(Boolean).join(' · ') || '—'}
        </small>
      </td>
      {temValor && <td className="num">{brl(c.valor_mensal)}</td>}
      <td className="num"><b>{pc(c.risco)}</b></td>
      {v.tipo === 'sem-dado' ? (
        <td className="num mut" title="Esta versão do modelo é anterior ao campo de variação do risco. Retreine a base para vê-la.">—</td>
      ) : (
        <td className={'var-c ' + v.tipo}>
          <b>{SETA[v.tipo]} {sg(Math.round(v.rel))}%</b>
          <small>{sg(+v.pp.toFixed(1))} p.p.</small>
        </td>
      )}
      {temValor && <td className="num">{brl(c.perda_anual_esperada)}</td>}
      <td>
        {c.evidencias.length
          ? c.evidencias.slice(0, 3).map((e) => (
              <Chip key={e.sinal} title={e.detalhe}>{e.titulo}</Chip>
            ))
          : <span className="mut">—</span>}
      </td>
      <td className="num">
        {c.meses_em_alerta > 0
          ? `${c.meses_em_alerta} ${c.meses_em_alerta === 1 ? 'mês' : 'meses'}`
          : <span className="mut">—</span>}
      </td>
    </tr>
  )
}

export function ResumoFila({ dados }: { dados: PaginaClientes }) {
  const r = dados.resumo
  if (!r) return null
  return (
    <>
      <p className="mut" style={{ margin: '-8px 0 14px' }}>
        {dados.modelo?.rotulo} · mês de referência {mesFmt(dados.mes_referencia)}
        {dados.modelo && <> · prevê cancelamento nos próximos {dados.modelo.horizonte_meses} meses</>}
      </p>
      <div className="kpis">
        <div className="kpi"><b>{r.clientes_ativos}</b><span>clientes ativos</span></div>
        <div className="kpi alto"><b>{r.em_risco_alto}</b><span>em risco alto</span></div>
        <div className="kpi atencao"><b>{r.em_atencao}</b><span>em atenção</span></div>
        {r.receita_em_risco_mensal != null && (
          <div className="kpi"><b>{brl(r.receita_em_risco_mensal)}</b><span>receita mensal em risco</span></div>
        )}
        {r.perda_anual_esperada_total != null && (
          <div className="kpi"><b>{brl(r.perda_anual_esperada_total)}</b><span>perda anual esperada</span></div>
        )}
      </div>
    </>
  )
}
