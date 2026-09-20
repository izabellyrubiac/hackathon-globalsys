/** Validação: com quanta antecedência o sinal aparece, o quanto ele separa quem ficou, e o que
 * acontece no topo da fila. São os três critérios que a banca avalia.
 *
 * Tudo vem de `validacao.json`, como o motor gravou — nada é recalculado aqui. Os textos de
 * `metodo` e `honestidade` também vêm prontos do motor e são exibidos como estão.
 */

import { useState } from 'react'
import * as api from '../api/cliente'
import { useRecurso } from '../api/ganchos'
import type { BaseResumo, Validacao as TValidacao } from '../api/tipos'
import { CaixaErro, Carregando, Avisos } from '../componentes/Estados'
import { Faixa } from '../componentes/Faixa'
import { Select } from '../componentes/Select'
import { useSessao } from '../estado/Sessao'
import { brl, mesFmt, num, pc } from '../formato'
import { irPara } from '../rotas'

export function Validacao({ bases, carregandoBases }: {
  bases: BaseResumo[]; carregandoBases: boolean
}) {
  const { baseId, escolherBase, versao } = useSessao()
  const prontas = bases.filter((b) => b.execucao_ativa)
  const atual = prontas.find((b) => b.base_id === baseId) || prontas[0] || null

  const { dados, carregando, erro } = useRecurso(
    (s) => api.validacao(atual!.base_id, s),
    [atual?.base_id, atual?.execucao_ativa, versao],
    Boolean(atual),
  )

  if (carregandoBases) return <Carregando texto="Procurando as bases…" />

  if (!prontas.length) {
    return (
      <>
        <div className="head"><h1>Validação</h1></div>
        <div className="card semmodelo" role="status">
          <h2>Nenhum modelo treinado</h2>
          <p className="mut">Treine um modelo para ver como ele se saiu contra quem já cancelou.</p>
          <button className="btn" onClick={() => irPara('modelos')}>Ir para Modelos</button>
        </div>
      </>
    )
  }

  return (
    <>
      <div className="head cab-modelo">
        <h1>Validação</h1>
        {prontas.length > 1 && (
          <div className="sel-modelo">
            <label htmlFor="v-base">Base</label>
            <Select
              id="v-base" rotulo="Base" valor={atual?.base_id || ''}
              opcoes={prontas.map((b) => ({
                valor: b.base_id, texto: b.nome + (b.demonstracao ? ' · demonstração' : ''),
              }))}
              aoMudar={escolherBase}
            />
          </div>
        )}
      </div>

      <CaixaErro erro={erro} />
      {carregando && !dados && <Carregando texto="Lendo a validação…" />}
      {dados && <ConteudoValidacao v={dados} />}
    </>
  )
}

/** Mostra as primeiras `passo` linhas e vai revelando o resto sob demanda: numa base grande a
 * lista de quem saiu tem milhares de linhas e não cabe de uma vez. */
function useMostrarMais<T>(itens: T[], passo = 25) {
  const [n, setN] = useState(passo)
  return {
    visiveis: itens.slice(0, n),
    restantes: Math.max(0, itens.length - n),
    mais: () => setN((x) => x + passo),
  }
}

function VerMais({ restantes, aoPedir }: { restantes: number; aoPedir: () => void }) {
  if (!restantes) return null
  return (
    <div className="mais">
      <button className="btn sm" onClick={aoPedir}>Ver mais ({restantes})</button>
    </div>
  )
}

export function ConteudoValidacao({ v }: { v: TValidacao }) {
  const desligada = !v.metodos?.length
  const p = v.painel || {}
  const aucMotor = v.auc_por_mes?.filter((a) => a.score === 'motor') ?? []
  const aucVar = v.auc_por_mes?.filter((a) => a.score === 'melhor_variavel') ?? []
  const logistica = v.escolha?.modelo === 'logistica'
  const quemSaiu = useMostrarMais(v.cancelados ?? [])
  const variaveis = useMostrarMais(
    [...(v.coeficientes ?? [])].sort((a, b) => b.importancia - a.importancia))

  return (
    <>
      {/* --------------------------------------------------- modelo escolhido */}
      <div className="card">
        <h2>Modelo escolhido</h2>
        <p style={{ marginTop: 10 }}>
          <b>{v.escolha?.rotulo}</b>{v.escolha?.forcado ? ' (forçado nas opções)' : ''} — {v.escolha?.motivo}
        </p>
        <p className="mut" style={{ marginTop: 8, fontSize: 13 }}>{v.escolha?.criterio}</p>
        {v.modelos?.length > 0 && (
          <div className="tw" style={{ marginTop: 14 }}>
            <table>
              <thead>
                <tr>
                  <th>Modelo</th><th>Elegível</th><th>Log-loss fora da amostra</th>
                  <th>AUC −1</th><th>AUC −2</th><th>AUC −3</th><th>Por quê</th>
                </tr>
              </thead>
              <tbody>
                {v.modelos.map((m) => (
                  <tr key={m.nome}>
                    <td><b>{m.rotulo}</b>{m.escolhido && <> <span className="chip">escolhido</span></>}</td>
                    <td>{m.elegivel ? 'sim' : 'não'}</td>
                    <td className="num">
                      {m.avaliado && m.log_loss != null
                        ? `${num(m.log_loss, 4)} ± ${num(m.log_loss_ep, 4)}`
                        : <span className="mut">não avaliado</span>}
                    </td>
                    <td className="num">{m.auc_k1 != null ? num(m.auc_k1, 4) : '—'}</td>
                    <td className="num">{m.auc_k2 != null ? num(m.auc_k2, 4) : '—'}</td>
                    <td className="num">{m.auc_k3 != null ? num(m.auc_k3, 4) : '—'}</td>
                    <td><small className="mut">{m.motivo_elegibilidade}</small></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {desligada ? (
        <div className="card">
          <h2>Validação desligada</h2>
          <p className="mut" style={{ marginTop: 10 }}>
            Esta versão foi treinada sem validação cruzada, então não há antecedência, separação
            nem alarme falso para mostrar. Treine de novo com a validação ligada.
          </p>
        </div>
      ) : (
        <>
          {/* ----------------------------------------- antecedência e alarme falso */}
          <div className="card">
            <h2>Antecedência e alarme falso</h2>
            <p className="sub mut">
              Quantos dos {p.cancelados} que saíram o método pegou, em cada mês antes da saída, e
              quantos dos {p.ativos} que ficaram ele incomodou à toa.
            </p>
            <div className="tw">
              <table>
                <thead>
                  <tr>
                    <th>Método</th>
                    <th>Pegos em −1</th><th>em −2</th><th>em −3</th>
                    <th>Antecedência mediana</th>
                    <th>Ativos com alarme hoje</th>
                    <th>Ativos com alarme em algum mês</th>
                    <th>Meses de alarme entre os ativos</th>
                  </tr>
                </thead>
                <tbody>
                  {v.metodos.map((m) => (
                    <tr key={m.id}>
                      <td>
                        <b>{m.nome}</b>
                        {m.rotulo && <><br /><small className="mut">{m.rotulo}</small></>}
                      </td>
                      <td className="num">{m.canc_k1}</td>
                      <td className="num">{m.canc_k2}</td>
                      <td className="num">{m.canc_k3}</td>
                      <td className="num">{m.antecedencia_mediana ?? '—'}</td>
                      <td className="num">{m.ativos_hoje}</td>
                      <td className="num">{m.ativos_algum_mes}</td>
                      <td className="num">{num(m.meses_alarme_ativos_pct, 1)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* ----------------------------------------- separação */}
          <div className="card">
            <h2>Separação</h2>
            <p className="sub mut">
              AUC de quem saiu contra quem ficou, olhando o risco a 1, 2 e 3 meses da saída.
              O motor é comparado com a melhor variável sozinha.
            </p>
            <div className="tw">
              <table>
                <thead>
                  <tr>
                    <th>Meses antes da saída</th><th>Motor</th><th>Melhor variável sozinha</th>
                    <th>Clientes</th><th>Cancelados</th>
                  </tr>
                </thead>
                <tbody>
                  {aucMotor.map((a) => {
                    const alt = aucVar.find((x) => x.k === a.k)
                    return (
                      <tr key={a.k}>
                        <td>−{a.k}</td>
                        <td className="num"><b>{num(a.auc, 4)}</b></td>
                        <td className="num">{alt ? num(alt.auc, 4) : '—'}</td>
                        <td className="num">{a.n}</td>
                        <td className="num">{a.n_canc}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* ----------------------------------------- topo da fila */}
          {v.precisao_topo?.length > 0 && (
            <div className="card">
              <h2>Topo da fila</h2>
              <p className="sub mut">
                Quantos dos que saíram estavam entre os primeiros da fila, em cada mês antes da saída.
              </p>
              <div className="tw">
                <table>
                  <thead>
                    <tr><th>Meses antes da saída</th><th>Topo</th><th>Cancelados no topo</th></tr>
                  </thead>
                  <tbody>
                    {v.precisao_topo.map((t, i) => (
                      <tr key={i}>
                        <td>−{t.k}</td>
                        <td className="num">primeiros {t.top}</td>
                        <td className="num"><b>{t.cancelados}</b> de {t.top}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* ----------------------------------------- os que saíram */}
          {v.cancelados?.length > 0 && (
            <div className="card">
              <h2>Quem saiu</h2>
              <p className="sub mut">
                O risco que o modelo dava a cada um deles 1, 2 e 3 meses antes da saída — calculado
                sem nunca ter visto esse cliente no treino.
              </p>
              <div className="tw">
                <table>
                  <thead>
                    <tr>
                      <th>Cliente</th><th>Saiu em</th><th>Valor mensal</th>
                      <th>−1 mês</th><th>−2 meses</th><th>−3 meses</th>
                    </tr>
                  </thead>
                  <tbody>
                    {quemSaiu.visiveis.map((c) => (
                      <tr key={c.cliente_id}>
                        <td><b>{c.cliente_id}</b></td>
                        <td>{mesFmt(c.mes_saida)}</td>
                        <td className="num">{brl(c.valor_mensal)}</td>
                        {([['risco_k1', 'faixa_k1'], ['risco_k2', 'faixa_k2'], ['risco_k3', 'faixa_k3']] as const)
                          .map(([r, f]) => (
                            <td key={r} className="num">
                              {c[r] != null ? pc(c[r]) : '—'}{' '}
                              {c[f] && <Faixa faixa={c[f]!} />}
                            </td>
                          ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <VerMais restantes={quemSaiu.restantes} aoPedir={quemSaiu.mais} />
            </div>
          )}

          {/* ----------------------------------------- variáveis */}
          {v.coeficientes?.length > 0 && (
            <div className="card">
              <h2>O que o modelo olha</h2>
              <p className="sub mut">
                Ordenado por importância. A frequência nas dobras mostra em quantas partições da
                validação a variável foi escolhida de novo.
              </p>
              <div className="tw">
                <table>
                  <thead>
                    <tr>
                      <th>Variável</th><th>Direção</th>
                      {logistica && <><th>Coeficiente</th><th>Razão de chances</th></>}
                      <th>Importância</th><th>Frequência nas dobras</th>
                    </tr>
                  </thead>
                  <tbody>
                    {variaveis.visiveis.map((c) => (
                      <tr key={c.variavel}>
                        <td><b>{c.rotulo}</b><br /><small className="mut">{c.coluna}</small></td>
                        <td>{c.direcao > 0 ? 'maior = mais risco' : 'menor = mais risco'}</td>
                        {logistica && (
                          <>
                            <td className="num">{c.coef != null ? num(c.coef, 4) : '—'}</td>
                            <td className="num">{c.odds_ratio != null ? num(c.odds_ratio, 4) : '—'}</td>
                          </>
                        )}
                        <td className="num">{num(c.importancia, 4)}</td>
                        <td className="num">{pc(c.frequencia_dobras)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <VerMais restantes={variaveis.restantes} aoPedir={variaveis.mais} />
            </div>
          )}
        </>
      )}

      {/* ----------------------------------------- método e honestidade */}
      <div className="card">
        <h2>Como isto foi medido</h2>
        <p style={{ marginTop: 10 }}>{v.metodo}</p>
        {v.honestidade && <p style={{ marginTop: 12 }}>{v.honestidade}</p>}
        {v.cortes && (
          <p className="mut" style={{ marginTop: 12, fontSize: 13 }}>
            Corte de risco alto: {num(v.cortes.alto, 2)} ({v.cortes.criterio_alto}). Corte de
            atenção: {num(v.cortes.atencao, 2)} ({v.cortes.criterio_atencao}).
          </p>
        )}
        <Avisos avisos={v.avisos ?? []} />
      </div>
    </>
  )
}
