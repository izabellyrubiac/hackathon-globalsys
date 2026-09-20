/** Modelos: enviar uma base, dizer o que prever, escolher as variáveis e treinar uma versão.
 *
 * Cada treino cria uma **execução** — uma versão do modelo com seus próprios resultados. Uma
 * delas fica ativa por base, e é a que a Fila e a Validação mostram.
 */

import { useEffect, useMemo, useState } from 'react'
import * as api from '../api/cliente'
import { ErroApi } from '../api/cliente'
import { useRecurso } from '../api/ganchos'
import type { BaseDetalhe, BaseResumo, Execucao, Mapeamento, Opcoes, Tabela } from '../api/tipos'
import { Barra } from '../componentes/Barra'
import { Avisos, CaixaErro, Carregando } from '../componentes/Estados'
import { Chip } from '../componentes/Faixa'
import { CampoSelect, Select } from '../componentes/Select'
import { useToast } from '../componentes/Toast'
import { EM_ANDAMENTO, achatarExecucoes, type LinhaVersao } from '../dominio'
import { useSessao } from '../estado/Sessao'
import { dataHora, norm, num } from '../formato'
import { irPara } from '../rotas'
import { CardAnalise } from './CardAnalise'
import {
  CamposPesosScore, PESOS_PADRAO, PopupPesos, lerPesos, pesosTexto, type RascunhoPesos,
} from './PopupPesos'
import { PopupEnvio } from './PopupEnvio'

export function Modelos({ bases, carregandoBases, recarregarBases }: {
  bases: BaseResumo[]
  carregandoBases: boolean
  recarregarBases: () => void
}) {
  const { baseId, escolherBase, avisarMudanca } = useSessao()
  const toast = useToast()
  const [popup, setPopup] = useState(false)
  const [recarga, setRecarga] = useState(0)

  const atual = bases.find((b) => b.base_id === baseId) || bases[0] || null

  const detalhe = useRecurso(
    (s) => api.detalheBase(atual!.base_id, s),
    [atual?.base_id, recarga],
    Boolean(atual),
  )

  // Uma chamada por base para montar a tabela de versões de todas elas.
  const todas = useRecurso(
    async (s) => {
      const pares = await Promise.all(
        bases.map(async (b) => [b.base_id, (await api.listarExecucoes(b.base_id, s)).execucoes] as const),
      )
      return Object.fromEntries(pares) as Record<string, Execucao[]>
    },
    [bases.map((b) => b.base_id + b.n_execucoes + b.estado).join('|'), recarga],
    bases.length > 0,
  )

  const linhas = useMemo(
    () => achatarExecucoes(bases, todas.dados ?? {}),
    [bases, todas.dados],
  )
  const emAndamento = linhas.some((l) => EM_ANDAMENTO(l.estado))

  // Enquanto algo treina, pergunta de tempos em tempos; a API não empurra progresso.
  // Com a aba em segundo plano o relógio para, para não ficar perguntando à toa.
  useEffect(() => {
    if (!emAndamento) return
    let vivo = true
    const id = window.setInterval(() => {
      if (!vivo || document.hidden) return
      setRecarga((r) => r + 1)
      recarregarBases()
    }, 1500)
    return () => { vivo = false; clearInterval(id) }
  }, [emAndamento, recarregarBases])

  function atualizar() {
    setRecarga((r) => r + 1)
    recarregarBases()
    avisarMudanca()
  }

  return (
    <>
      <div className="head"><h1>Modelos</h1></div>

      <TabelaVersoes
        linhas={linhas} carregando={todas.carregando && !todas.dados} erro={todas.erro}
        aoAtualizar={atualizar} aoVerFila={(l) => { escolherBase(l.base_id); irPara('fila') }}
      />

      {/* ------------------------------------------------------------- 1. Dados */}
      <div className={'card' + (atual ? ' ok' : '')}>
        <h2><span className="n">1</span>Dados</h2>
        <p className="sub mut">
          Envie um .xlsx com uma tabela por aba, ou vários .csv. As bases de demonstração já vêm
          prontas no servidor.
        </p>
        <div style={{ marginTop: 12, display: 'flex', flexWrap: 'wrap', alignItems: 'flex-end', gap: '8px 16px' }}>
          <button className={'btn' + (atual ? ' ghost' : '')} onClick={() => setPopup(true)}>
            {atual ? 'Enviar outra base' : 'Enviar base de dados'}
          </button>
          {bases.length > 0 && (
            <div className="campo" style={{ minWidth: 260 }}>
              <label htmlFor="m-base">Base em uso</label>
              <Select
                id="m-base" rotulo="Base em uso" valor={atual?.base_id || ''}
                opcoes={bases.map((b) => ({
                  valor: b.base_id,
                  texto: b.nome + (b.demonstracao ? ' · demonstração' : ''),
                  dica: `${b.n_execucoes} ${b.n_execucoes === 1 ? 'versão' : 'versões'} · ${b.arquivos.join(', ')}`,
                }))}
                aoMudar={escolherBase}
              />
            </div>
          )}
        </div>

        {carregandoBases && !bases.length && <p className="mut" style={{ marginTop: 12 }}>Procurando as bases…</p>}
        {!carregandoBases && !bases.length && (
          <p className="mut" style={{ marginTop: 12 }}>
            Nenhuma base ainda. Envie a primeira para começar.
          </p>
        )}

        {detalhe.dados && (
          <>
            <p className="mut" style={{ marginTop: 12 }}>{detalhe.dados.arquivos.join(', ')}</p>
            <div className="tabs">
              {detalhe.dados.inspecao.tabelas.map((t) => (
                <Chip key={t.nome} title={`Papel sugerido: ${t.papel_sugerido}${t.motivo ? ' — ' + t.motivo : ''}`}>
                  {t.nome} · {t.linhas} linhas · {t.colunas.length} colunas
                </Chip>
              ))}
            </div>
            <Avisos avisos={detalhe.dados.inspecao.avisos} titulo="Ao ler a base" />
          </>
        )}
        <CaixaErro erro={detalhe.erro} />
      </div>

      {detalhe.carregando && !detalhe.dados && <Carregando texto="Lendo a inspeção da base…" />}

      {detalhe.dados && (
        <Treino
          base={detalhe.dados}
          andamento={linhas.filter((l) => l.base_id === detalhe.dados!.base_id && EM_ANDAMENTO(l.estado))}
          aoTreinar={() => { atualizar(); toast('Treino começou. Acompanhe na lista acima.') }}
          aoCancelar={() => { atualizar(); toast('Cancelamento pedido. O motor para na próxima etapa.') }}
        />
      )}

      {popup && (
        <PopupEnvio
          aoFechar={() => setPopup(false)}
          aoEnviar={(base) => {
            setPopup(false)
            escolherBase(base.base_id)
            atualizar()
            toast(`Base "${base.nome}" enviada e inspecionada.`)
          }}
        />
      )}
    </>
  )
}

// --------------------------------------------------------------------------- versões
function TabelaVersoes({ linhas, carregando, erro, aoAtualizar, aoVerFila }: {
  linhas: LinhaVersao[]
  carregando: boolean
  erro: ErroApi | null
  aoAtualizar: () => void
  aoVerFila: (l: LinhaVersao) => void
}) {
  const toast = useToast()
  const [ocupado, setOcupado] = useState<string | null>(null)
  const [falha, setFalha] = useState<ErroApi | null>(null)
  const [editando, setEditando] = useState<string | null>(null)   // chave da linha com o nome em edição
  const [nome, setNome] = useState('')
  const [pesosDe, setPesosDe] = useState<LinhaVersao | null>(null)

  async function agir(l: LinhaVersao, acao: 'ativar' | 'apagar' | 'cancelar' | 'renomear') {
    setOcupado(l.chave)
    setFalha(null)
    try {
      if (acao === 'ativar') {
        await api.ativarExecucao(l.base_id, l.execucao_id)
        toast(`"${l.rotulo}" agora é a versão em uso em ${l.base_nome}.`)
      } else if (acao === 'cancelar') {
        await api.cancelarExecucao(l.base_id, l.execucao_id)
        toast(`Cancelamento de "${l.rotulo}" pedido. O motor para na próxima etapa.`)
      } else if (acao === 'renomear') {
        await api.atualizarExecucao(l.base_id, l.execucao_id, { rotulo: nome.trim() || null })
        setEditando(null)
        toast('Versão renomeada.')
      } else {
        await api.apagarExecucao(l.base_id, l.execucao_id)
        toast(`Versão "${l.rotulo}" apagada.`)
      }
      aoAtualizar()
    } catch (e) {
      setFalha(e instanceof ErroApi ? e : new ErroApi(0, String(e)))
    } finally {
      setOcupado(null)
    }
  }

  if (carregando) return <Carregando texto="Procurando as versões já treinadas…" />
  if (!linhas.length && !erro) return null

  return (
    <div className="card">
      <div className="cab">
        <h2>Modelos treinados</h2>
        <span className="mut">{linhas.length} {linhas.length === 1 ? 'versão' : 'versões'}</span>
      </div>
      <CaixaErro erro={erro} />
      <CaixaErro erro={falha} />

      <div className="tw" style={{ marginTop: 8 }}>
        <table>
          <thead>
            <tr>
              <th>Versão</th><th>Base</th><th>Algoritmo</th><th>Criada em</th>
              <th>Variáveis</th>
              <th title="Pesos do score que ordena a fila: risco · valor mensal · variação do risco">Ordem da fila</th>
              <th title="AUC no último mês antes da saída">AUC −1</th>
              <th title="Cancelados que estavam na faixa alto um mês antes de sair">Pegou</th>
              <th title="Ativos hoje na faixa alto">Alarme hoje</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {linhas.map((l) => (
              <tr key={l.chave}>
                <td>
                  {editando === l.chave ? (
                    <span style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                      <input aria-label={`Novo nome de ${l.rotulo}`} value={nome} maxLength={80} autoFocus
                             onChange={(e) => setNome(e.target.value)}
                             onKeyDown={(e) => { if (e.key === 'Enter') agir(l, 'renomear'); if (e.key === 'Escape') setEditando(null) }} />
                      <button className="btn sm" disabled={ocupado === l.chave} onClick={() => agir(l, 'renomear')}>Salvar</button>
                      <button className="btn ghost sm" onClick={() => setEditando(null)}>Cancelar</button>
                    </span>
                  ) : (
                    <b>{l.rotulo}</b>
                  )}
                  {l.ativa && <> <Chip title="É a versão que a Fila e a Validação mostram para esta base.">em uso</Chip></>}
                  {l.estado === 'cancelada' && <><br /><small className="mut">Treino cancelado.</small></>}
                  {EM_ANDAMENTO(l.estado) && (
                    <>
                      <br /><small className="mut">{l.etapa || l.estado}…</small>
                      <Barra fracao={l.fracao} />
                    </>
                  )}
                  {l.estado === 'erro' && (
                    <>
                      <br /><small style={{ color: 'var(--hi)' }}>{l.mensagem || 'Falhou.'}</small>
                      {l.problemas.map((p, i) => (
                        <small key={i} className="mut" style={{ display: 'block' }}>{p}</small>
                      ))}
                    </>
                  )}
                </td>
                <td>
                  {l.base_nome}
                  {l.demonstracao && <><br /><small className="mut">demonstração</small></>}
                </td>
                <td>{l.algoritmo ?? <span className="mut">—</span>}</td>
                <td className="num">{dataHora(l.criada_em)}</td>
                <td className="num">{l.variaveis ?? '—'}</td>
                <td className="num" title={l.pesos_score ? 'Score: risco · valor mensal · variação do risco' : 'Ordem padrão do motor'}>
                  {l.estado === 'pronta' ? (l.pesos_score ? pesosTexto(l.pesos_score) : <span className="mut">perda ajustada</span>) : '—'}
                </td>
                <td className="num">{l.auc_k1 != null ? num(l.auc_k1, 3) : '—'}</td>
                <td className="num">{l.cancelados_pegos ?? '—'}</td>
                <td className="num">{l.alarme_hoje ?? '—'}</td>
                <td style={{ whiteSpace: 'nowrap' }}>
                  {EM_ANDAMENTO(l.estado) && (
                    <><button className="btn ghost sm" disabled={ocupado === l.chave}
                              title="Um treino na fila sai da fila; um em andamento para na próxima etapa."
                              onClick={() => agir(l, 'cancelar')}>Cancelar</button>{' '}</>
                  )}
                  <button className="btn ghost sm" disabled={editando === l.chave}
                          onClick={() => { setNome(l.rotulo); setEditando(l.chave) }}>Renomear</button>{' '}
                  <button className="btn ghost sm" disabled={l.estado !== 'pronta'}
                          title={l.estado !== 'pronta' ? 'Só uma versão pronta tem fila para reordenar.' : 'Mudar os pesos do score que ordena a fila.'}
                          onClick={() => setPesosDe(l)}>Pesos</button>{' '}
                  <button className="btn ghost sm" disabled={l.estado !== 'pronta'}
                          title={l.estado !== 'pronta' ? 'A versão ainda não ficou pronta.' : undefined}
                          onClick={() => aoVerFila(l)}>Ver fila</button>{' '}
                  <button className="btn ghost sm"
                          disabled={l.ativa || l.estado !== 'pronta' || ocupado === l.chave}
                          title={l.ativa ? 'Já é a versão em uso nesta base.'
                                : l.estado !== 'pronta' ? 'Só uma versão pronta pode ser ativada.' : undefined}
                          onClick={() => agir(l, 'ativar')}>Usar</button>{' '}
                  <button className="btn ghost sm"
                          disabled={l.ativa || EM_ANDAMENTO(l.estado) || ocupado === l.chave}
                          title={l.ativa ? 'É a versão em uso. Use outra antes de apagar esta.'
                                : EM_ANDAMENTO(l.estado) ? 'A versão ainda está na fila ou treinando.' : undefined}
                          onClick={() => agir(l, 'apagar')}>Apagar</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {pesosDe && (
        <PopupPesos
          baseId={pesosDe.base_id} execucaoId={pesosDe.execucao_id} rotulo={pesosDe.rotulo}
          atual={pesosDe.pesos_score} aoFechar={() => setPesosDe(null)}
          aoSalvar={() => { setPesosDe(null); aoAtualizar() }}
        />
      )}
    </div>
  )
}

// --------------------------------------------------------------------------- 2, 3 e 4
interface Rascunho {
  tabela_clientes: string
  coluna_id: string
  alvo_tabela: string
  coluna_situacao: string
  valor_cancelado: string
  coluna_data_saida: string
  valor_tabela: string
  coluna_valor: string
  horizonte_meses: number
  rotulo: string
  ignoradas: Set<string>
  /** Pesos por coluna: Manual mostra os sliders; cada coluna vale 1 até o usuário mexer. */
  manual: boolean
  pesos: Record<string, number>
  /** Ordenar a fila por score (pesos do risco, do valor e da variação). Desligado = ordem padrão do motor. */
  usarScore: boolean
  score: RascunhoPesos
}

function doSugerido(b: BaseDetalhe): Rascunho {
  const s = b.inspecao.sugestao
  const m = b.mapeamento
  return {
    tabela_clientes: m?.tabela_clientes || s.tabela_clientes || '',
    coluna_id: m?.coluna_id || s.coluna_id || '',
    alvo_tabela: m?.alvo_tabela || s.alvo?.tabela || '',
    coluna_situacao: m?.coluna_situacao || s.alvo?.coluna_situacao || '',
    valor_cancelado: String(m?.valor_cancelado ?? s.alvo?.valor_cancelado ?? ''),
    coluna_data_saida: m?.coluna_data_saida || s.alvo?.coluna_data_saida || '',
    valor_tabela: m?.valor_tabela || s.coluna_valor?.tabela || '',
    coluna_valor: m?.coluna_valor || s.coluna_valor?.coluna || '',
    horizonte_meses: m?.horizonte_meses || s.horizonte_meses || 3,
    rotulo: '',
    ignoradas: new Set(m?.colunas_ignoradas ?? []),
    manual: false,
    pesos: {},
    usarScore: false,
    score: { ...PESOS_PADRAO },
  }
}

function Treino({ base, andamento, aoTreinar, aoCancelar }: {
  base: BaseDetalhe
  /** Versões desta base que estão na fila ou treinando. */
  andamento: LinhaVersao[]
  aoTreinar: () => void
  aoCancelar: () => void
}) {
  const toast = useToast()
  const [r, setR] = useState<Rascunho>(() => doSugerido(base))
  const [enviando, setEnviando] = useState(false)
  const [cancelando, setCancelando] = useState(false)
  const [erro, setErro] = useState<ErroApi | null>(null)

  async function cancelarTreinos() {
    setCancelando(true)
    setErro(null)
    try {
      await Promise.all(andamento.map((l) => api.cancelarExecucao(l.base_id, l.execucao_id)))
      aoCancelar()
    } catch (e) {
      setErro(e instanceof ErroApi ? e : new ErroApi(0, String(e)))
    } finally {
      setCancelando(false)
    }
  }

  // Trocar de base recomeça o formulário com o que a inspeção daquela base sugeriu.
  useEffect(() => { setR(doSugerido(base)); setErro(null) }, [base.base_id])

  const tabelas = base.inspecao.tabelas.filter((t) => t.papel_sugerido !== 'ignorada')
  const achar = (nome: string): Tabela | undefined => tabelas.find((t) => t.nome === nome)
  const opTabelas = tabelas.map((t) => ({
    valor: t.nome, texto: t.nome,
    dica: `${t.linhas} linhas · papel sugerido: ${t.papel_sugerido}`,
  }))
  const opColunas = (tab: string) => (achar(tab)?.colunas ?? []).map((c) => ({
    valor: c.nome, texto: c.nome,
    dica: `${c.tipo}${c.exemplos.length ? ' · ex.: ' + c.exemplos.slice(0, 3).join(', ') : ''}`,
  }))

  const mudar = (p: Partial<Rascunho>) => setR((a) => ({ ...a, ...p }))
  const pronto = Boolean(r.tabela_clientes && r.coluna_id && r.alvo_tabela
                         && (r.coluna_situacao || r.coluna_data_saida))

  // Candidatas: as colunas das tabelas ligadas ao cliente, menos o id, as colunas de tempo e
  // tudo que está na tabela do cancelamento — que o motor descarta inteira, para não vazar.
  const candidatas = useMemo(() => {
    if (!r.coluna_id) return []
    const idn = norm(r.coluna_id)
    const saida: { tabela: string; coluna: string; chave: string }[] = []
    for (const t of tabelas) {
      if (t.nome === r.alvo_tabela) continue
      if (!t.colunas.some((c) => norm(c.nome) === idn)) continue
      for (const c of t.colunas) {
        if (norm(c.nome) === idn) continue
        if (c.nome === t.coluna_data) continue
        saida.push({ tabela: t.nome, coluna: c.nome, chave: `${t.nome}.${c.nome}` })
      }
    }
    return saida
  }, [tabelas, r.coluna_id, r.alvo_tabela])

  const porTabela = useMemo(() => {
    const g: Record<string, typeof candidatas> = {}
    for (const c of candidatas) (g[c.tabela] ??= []).push(c)
    return g
  }, [candidatas])

  function marcarTodas(marcar: boolean) {
    mudar({ ignoradas: marcar ? new Set() : new Set(candidatas.map((c) => c.chave)) })
  }

  const mapeamento: Mapeamento = {
    tabela_clientes: r.tabela_clientes,
    coluna_id: r.coluna_id,
    alvo_tabela: r.alvo_tabela,
    coluna_situacao: r.coluna_situacao || null,
    valor_cancelado: r.coluna_situacao ? r.valor_cancelado : null,
    coluna_data_saida: r.coluna_data_saida || null,
    valor_tabela: r.coluna_valor ? (r.valor_tabela || r.tabela_clientes) : null,
    coluna_valor: r.coluna_valor || null,
    horizonte_meses: r.horizonte_meses,
    colunas_ignoradas: [...r.ignoradas],
    acoes: {},
    rotulos: {},
  }
  const pesosScore = r.usarScore ? lerPesos(r.score) : null
  const scoreInvalido = r.usarScore && !pesosScore
  const peso = (chave: string) => r.pesos[chave] ?? 1

  async function treinar() {
    if (!pronto || enviando || scoreInvalido) return
    setEnviando(true)
    setErro(null)
    const opcoes: Opcoes = { modelo: 'auto' }
    if (pesosScore) opcoes.pesos_score = pesosScore
    if (r.manual) {                       // só as colunas marcadas e com peso diferente de 1
      const alterados = candidatas.filter((c) => !r.ignoradas.has(c.chave) && peso(c.chave) !== 1)
      if (alterados.length) opcoes.pesos_colunas = Object.fromEntries(alterados.map((c) => [c.chave, peso(c.chave)]))
    }
    try {
      const resp = await api.treinar(base.base_id, { mapeamento, opcoes, rotulo: r.rotulo.trim() || null })
      if (resp.avisos?.length) toast(resp.avisos[0]!)
      aoTreinar()
    } catch (e) {
      setErro(e instanceof ErroApi ? e : new ErroApi(0, String(e)))
    } finally {
      setEnviando(false)
    }
  }

  return (
    <>
      {/* ------------------------------------------------------------- 2. O que prever */}
      <div className="card">
        <h2><span className="n">2</span>O que prever</h2>
        <p className="sub mut">
          Sugerimos tudo a partir da leitura da base; confira e corrija se preciso. As colunas do
          cancelamento nunca viram variável, para o modelo não colar a resposta.
        </p>
        <div className="form">
          <div>
            <label>Tabela de clientes e coluna do identificador</label>
            <div className="par">
              <Select rotulo="Tabela de clientes" valor={r.tabela_clientes} opcoes={opTabelas}
                      vazio="Tabela" aoMudar={(v) => mudar({ tabela_clientes: v, coluna_id: '' })} />
              <Select rotulo="Coluna do identificador" valor={r.coluna_id}
                      opcoes={opColunas(r.tabela_clientes)} vazio="Coluna"
                      desabilitado={!r.tabela_clientes} aoMudar={(v) => mudar({ coluna_id: v })} />
            </div>
          </div>

          <div>
            <label>Onde está o cancelamento</label>
            <Select rotulo="Tabela do cancelamento" valor={r.alvo_tabela} opcoes={opTabelas}
                    vazio="Tabela"
                    aoMudar={(v) => mudar({ alvo_tabela: v, coluna_situacao: '', coluna_data_saida: '' })} />
          </div>

          <div>
            <label>Coluna de situação e o valor que significa cancelado</label>
            <div className="par">
              <Select rotulo="Coluna de situação" valor={r.coluna_situacao}
                      opcoes={opColunas(r.alvo_tabela)} vazio="Nenhuma"
                      desabilitado={!r.alvo_tabela} aoMudar={(v) => mudar({ coluna_situacao: v })} />
              <input aria-label="Valor que significa cancelado" placeholder="Ex.: Cancelado"
                     value={r.valor_cancelado} disabled={!r.coluna_situacao}
                     onChange={(e) => mudar({ valor_cancelado: e.target.value })} />
            </div>
          </div>

          <div>
            <label>Coluna com o mês da saída</label>
            <Select rotulo="Coluna do mês da saída" valor={r.coluna_data_saida}
                    opcoes={opColunas(r.alvo_tabela)} vazio="Nenhuma"
                    desabilitado={!r.alvo_tabela} aoMudar={(v) => mudar({ coluna_data_saida: v })} />
          </div>

          <div>
            <label>Valor do contrato (opcional, define a prioridade da fila)</label>
            <div className="par">
              <Select rotulo="Tabela do valor" valor={r.valor_tabela || r.tabela_clientes}
                      opcoes={opTabelas} vazio="Tabela"
                      aoMudar={(v) => mudar({ valor_tabela: v, coluna_valor: '' })} />
              <Select rotulo="Coluna do valor" valor={r.coluna_valor}
                      opcoes={opColunas(r.valor_tabela || r.tabela_clientes)} vazio="Nenhuma"
                      aoMudar={(v) => mudar({ coluna_valor: v })} />
            </div>
          </div>

          <CampoSelect
            rotulo="Prever cancelamento nos próximos" valor={String(r.horizonte_meses)}
            opcoes={[1, 2, 3, 4, 5, 6, 9, 12].map((n) => ({
              valor: String(n), texto: `${n} ${n === 1 ? 'mês' : 'meses'}`,
            }))}
            aoMudar={(v) => mudar({ horizonte_meses: Number(v) })}
          />
        </div>

        {!r.coluna_situacao && !r.coluna_data_saida && r.alvo_tabela && (
          <div className="aviso" role="status">
            Informe a coluna de situação (com o valor que significa cancelado) ou a coluna com o mês
            da saída — sem cancelamento não há como aprender. As duas juntas também valem.
          </div>
        )}
      </div>

      {/* ------------------------------------------------------------- análise */}
      <CardAnalise baseId={base.base_id} mapeamento={mapeamento} pronto={pronto} />

      {/* ------------------------------------------------------------- 3. Variáveis */}
      <div className="card">
        <h2><span className="n">3</span>Variáveis</h2>
        <p className="sub mut">
          Todas as outras colunas são candidatas. O motor escolhe sozinho quais pesam, comparando
          quem saiu com quem ficou; desmarque só o que não deve ser usado.
        </p>

        <div className="seg" role="group" aria-label="Como definir os pesos">
          <button aria-pressed={!r.manual} onClick={() => mudar({ manual: false })}>Automático (recomendado)</button>
          <button aria-pressed={r.manual} onClick={() => mudar({ manual: true })}>Manual</button>
        </div>
        <p className="mut" style={{ margin: '10px 0', fontSize: 13 }}>
          {r.manual
            ? 'Dê um peso de 0 (ignorar) a 1 (normal) a cada variável marcada. O peso escala a penalização da seleção da logística; a seleção e a validação continuam automáticas e são refeitas com esses pesos. Nas árvores só o 0 vale.'
            : 'O motor escolhe sozinho quais variáveis pesam. Desmarque apenas o que não deve ser usado.'}
        </p>

        {candidatas.length ? (
          <>
            <div style={{ display: 'flex', gap: 8, marginBottom: 6 }}>
              <button className="btn ghost sm" onClick={() => marcarTodas(true)}>Marcar todas</button>
              <button className="btn ghost sm" onClick={() => marcarTodas(false)}>Desmarcar todas</button>
              <span className="mut" style={{ alignSelf: 'center', fontSize: 13 }}>
                {candidatas.length - r.ignoradas.size} de {candidatas.length} marcadas
              </span>
            </div>
            {Object.entries(porTabela).map(([tab, cols]) => (
              <div className="vgrp" key={tab}>
                <h3>{tab}</h3>
                {cols.map((c) => (
                  <div className="var" key={c.chave}>
                    <label>
                      <input type="checkbox" checked={!r.ignoradas.has(c.chave)}
                             onChange={(e) => {
                               const s = new Set(r.ignoradas)
                               if (e.target.checked) s.delete(c.chave)
                               else s.add(c.chave)
                               mudar({ ignoradas: s })
                             }} />
                      {' '}{c.coluna}
                    </label>
                    {r.manual && !r.ignoradas.has(c.chave) && (
                      <>
                        <input type="range" min={0} max={1} step={0.05} value={peso(c.chave)}
                               aria-label={`Peso de ${c.coluna}`}
                               onChange={(e) => mudar({ pesos: { ...r.pesos, [c.chave]: Number(e.target.value) } })} />
                        <span className="pv">{peso(c.chave).toFixed(2)}</span>
                      </>
                    )}
                  </div>
                ))}
              </div>
            ))}
          </>
        ) : (
          <p className="mut">Escolha a tabela de clientes e a coluna do identificador para listar as variáveis.</p>
        )}

      </div>

      {/* ------------------------------------------------------------- 4. Treinar */}
      <div className="card">
        <h2><span className="n">4</span>Treinar</h2>
        <p className="sub mut">
          O treino roda em segundo plano e vira uma nova versão. A primeira versão pronta de uma
          base passa a ser a que a Fila mostra.
        </p>

        <div style={{ maxWidth: 420, marginBottom: 14 }}>
          <label htmlFor="t-rotulo" style={{ display: 'block', fontSize: 13, fontWeight: 600, marginBottom: 4 }}>
            Nome desta versão (opcional)
          </label>
          <input id="t-rotulo" maxLength={80} placeholder="Ex.: sem NPS, horizonte 2"
                 value={r.rotulo} onChange={(e) => mudar({ rotulo: e.target.value })} />
        </div>

        <label style={{ display: 'flex', alignItems: 'center', gap: 8, margin: '4px 0 10px', cursor: 'pointer' }}>
          <input type="checkbox" checked={r.usarScore} onChange={(e) => mudar({ usarScore: e.target.checked })} />
          Ordenar a fila por um score com pesos meus (senão vale a perda anual ajustada do motor)
        </label>
        {r.usarScore && (
          <div style={{ maxWidth: 640, marginBottom: 14 }}>
            <CamposPesosScore valor={r.score} aoMudar={(score) => mudar({ score })} id="t-pesos" />
          </div>
        )}

        <CaixaErro erro={erro} titulo="Não consegui começar o treino." />

        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
          <button className="btn" disabled={!pronto || enviando || scoreInvalido} onClick={treinar}
                  title={!pronto ? 'Complete a tabela de clientes, o identificador e o cancelamento.'
                        : scoreInvalido ? 'Corrija os pesos do score.' : undefined}>
            {enviando ? 'Enviando…' : 'Treinar modelo'}
          </button>
          {andamento.length > 0 && (
            <button className="btn ghost" disabled={cancelando} onClick={cancelarTreinos}
                    title="Cancela os treinos desta base que estão na fila ou em andamento.">
              {cancelando ? 'Cancelando…' : `Cancelar treino${andamento.length > 1 ? 's' : ''}`}
            </button>
          )}
        </div>
      </div>
    </>
  )
}
