/** Renderiza as telas com os dados reais da API e confere que o HTML sai como se espera.
 *
 * Não é conferência visual — é execução: pega campo nulo, acesso fora do array, chave errada e
 * qualquer coisa que só apareceria quando a tela monta de verdade. Rode com a API de pé:
 *
 *     npm run renderizar
 */

import { renderToStaticMarkup } from 'react-dom/server'
import type { ReactElement } from 'react'
import type { PaginaClientes, Validacao } from './api/tipos'
import { GraficoSeries } from './componentes/GraficoSeries'
import { Sparkline } from './componentes/Sparkline'
import { LinhaFila, ResumoFila } from './paginas/Fila'
import { GavetaCliente } from './paginas/GavetaCliente'
import { ConteudoValidacao } from './paginas/Validacao'
import { analisar } from './analise/exploratoria'

const API = process.env.API || 'http://localhost:8000'
let falhas = 0

function conferir(nome: string, elemento: ReactElement, esperado: (string | RegExp)[]) {
  let html: string
  try {
    html = renderToStaticMarkup(elemento)
  } catch (e) {
    falhas++
    console.log(`  QUEBROU ${nome}: ${(e as Error).message}`)
    return
  }
  const faltando = esperado.filter((e) =>
    typeof e === 'string' ? !html.includes(e) : !e.test(html))
  if (faltando.length) {
    falhas++
    console.log(`  FALHA ${nome}: não achei ${faltando.map(String).join(', ')}`)
    console.log(`        ${html.slice(0, 400)}`)
  } else {
    console.log(`  ok   ${nome}  (${html.length} caracteres)`)
  }
}

async function pegar<T>(rota: string): Promise<T> {
  const r = await fetch(API + rota)
  if (!r.ok) throw new Error(`${rota} respondeu ${r.status}`)
  return r.json() as Promise<T>
}

async function main() {
  const fila = await pegar<PaginaClientes>('/api/bases/inovaapps/clientes?limite=5000&incluir_historico=false')
  const detalhado = await pegar<PaginaClientes>('/api/bases/inovaapps/clientes?desde=0&limite=1&incluir_historico=true')
  const validacao = await pegar<Validacao>('/api/bases/inovaapps/validacao')
  const primeiro = fila.clientes[0]!
  const comHistorico = detalhado.clientes[0]!

  console.log('tela Fila')
  conferir('tira de resumo', <ResumoFila dados={fila} />,
           ['58', 'clientes ativos', 'em risco alto', 'R$ 242.145', 'jun/2026'])
  conferir('linha do primeiro da fila',
           <table><tbody><LinhaFila c={primeiro} temValor aoAbrir={() => {}} /></tbody></table>,
           [primeiro.cliente_id, '>1<', 'fx atencao', 'R$ ', '16%', primeiro.evidencias[0]!.titulo])

  // todas as linhas: é onde um campo nulo de um cliente qualquer apareceria
  conferir('as 58 linhas',
           <table><tbody>{fila.clientes.map((c) => (
             <LinhaFila key={c.cliente_id} c={c} temValor aoAbrir={() => {}} />
           ))}</tbody></table>,
           [fila.clientes[57]!.cliente_id, '<tr class="r"'])

  console.log('\npainel do cliente')
  conferir('gaveta com histórico',
           <GavetaCliente baseId="inovaapps" execucaoId={null} cliente={comHistorico}
                          modelo={detalhado.modelo} mesReferencia={detalhado.mes_referencia}
                          aoFechar={() => {}} />,
           [comHistorico.cliente_id, 'Evidências', 'O que fazer', '<svg', 'faixa cinza',
            comHistorico.acao_sugerida])
  conferir('gaveta sem histórico (fila enxuta)',
           <GavetaCliente baseId="inovaapps" execucaoId={null} cliente={primeiro}
                          modelo={fila.modelo} mesReferencia={fila.mes_referencia}
                          aoFechar={() => {}} />,
           ['Evidências', 'O que fazer'])

  console.log('\nsparklines de todas as séries')
  const series = detalhado.modelo!.series_historico
  conferir('uma por série',
           <div>{series.map((s) => (
             <Sparkline key={s.chave} historico={comHistorico.historico!} chave={s.chave} rotulo={s.rotulo} />
           ))}</div>,
           [/<svg[^>]+class="spark"/, 'jun/2026', 'polyline'])

  console.log('\ntela Validação')
  conferir('validação inteira', <ConteudoValidacao v={validacao} />,
           ['Regressão logística L2', 'Antecedência e alarme falso', 'Separação', 'Topo da fila',
            'Quem saiu', 'O que o modelo olha', 'Como isto foi medido',
            '0,9788', 'Melhor variável sozinha', 'C004'])

  // Na INOVAAPPS só a logística é elegível. O caminho das árvores é coberto aqui, derivando os
  // dados reais: nas árvores o motor manda `coef` e `odds_ratio` nulos, e a tela some com essas
  // duas colunas.
  console.log('\nmodelo de árvore (coeficientes nulos)')
  conferir('validação sem coeficientes',
           <ConteudoValidacao v={{
             ...validacao,
             escolha: { ...validacao.escolha, modelo: 'lightgbm', rotulo: 'LightGBM (gradient boosting)' },
             coeficientes: validacao.coeficientes.map((c) => ({ ...c, coef: null, odds_ratio: null })),
           }} />,
           ['LightGBM', 'O que o modelo olha', 'Antecedência e alarme falso'])

  console.log('\ngráfico da análise exploratória')
  // série fabricada: os dois grupos se separam a partir do mês −3
  const pontos = Array.from({ length: 12 }, (_, i) => {
    const rel = i - 12
    const ativos = { med: 80, p25: 72, p75: 88, n: 40 }
    return { rel, a: ativos, c: rel >= -3 ? { med: 40, p25: 30, p75: 52, n: 12 } : ativos }
  })
  conferir('gráfico de cancelados × ativos',
           <GraficoSeries serie={{ tabela: 'atendimento_mensal', coluna: 'uso_plataforma_pct',
                                   rotulo: 'Uso da plataforma (%)', auc: 0.91,
                                   antecedencia: 3, pontos }} />,
           ['Uso da plataforma (%)', 'separa desde o mês −3', 'Cancelaram', 'Continuam ativos',
            'meses até a saída', '<polygon'])

  console.log('\nanálise exploratória sobre dados sintéticos')
  const planilhas = {
    clientes: [{ cliente_id: 'A', v: 1 }, { cliente_id: 'B', v: 1 }, { cliente_id: 'C', v: 1 }],
    situacao: [{ cliente_id: 'A', mes_saida: '2026-05' }, { cliente_id: 'B', mes_saida: '' },
               { cliente_id: 'C', mes_saida: '' }],
    mensal: [] as Record<string, unknown>[],
  }
  for (const id of ['A', 'B', 'C']) {
    for (let m = 0; m < 18; m++) {
      const caindo = id === 'A' && m >= 12
      planilhas.mensal.push({ cliente_id: id, mes_ref: `${2025 + (m > 11 ? 1 : 0)}-${String((m % 12) + 1).padStart(2, '0')}`,
                              uso_pct: caindo ? 30 : 80 })
    }
  }
  const a = analisar({ planilhas, colunaId: 'cliente_id', alvoTabela: 'situacao',
                       colunaDataSaida: 'mes_saida' })
  if ('erro' in a) { falhas++; console.log('  FALHA analisar: ' + a.erro) }
  else {
    const s = a.series[0]!
    const ok = a.cancelados === 1 && a.ativos === 2 && s.coluna === 'uso_pct' && s.auc === 0
    console.log(`  ${ok ? 'ok  ' : 'FALHA'} analisar: ${a.cancelados} cancelado, ${a.ativos} ativos, `
                + `série "${s.coluna}", AUC ${s.auc}, separa desde ${s.antecedencia}`)
    if (!ok) falhas++
  }

  // O avaliador pode trazer uma base sem coluna de valor, sem NPS, com um cliente sem nenhum
  // sinal, ou treinada sem validação. Nenhum desses casos pode quebrar a tela.
  console.log('\ncasos-limite')
  const semValor: PaginaClientes = {
    ...fila,
    modelo: { ...fila.modelo!, coluna_valor: null },
    resumo: { ...fila.resumo!, receita_em_risco_mensal: null,
              receita_risco_alto_mensal: null, perda_anual_esperada_total: null },
    clientes: fila.clientes.map((c) => ({ ...c, valor_mensal: null,
                                          perda_anual_esperada: null, perda_anual_ajustada: null })),
  }
  conferir('resumo sem coluna de valor', <ResumoFila dados={semValor} />,
           ['clientes ativos', 'em risco alto'])
  conferir('fila sem coluna de valor',
           <table><tbody>{semValor.clientes.slice(0, 5).map((c) => (
             <LinhaFila key={c.cliente_id} c={c} temValor={false} aoAbrir={() => {}} />
           ))}</tbody></table>,
           [semValor.clientes[0]!.cliente_id])
  conferir('cliente sem sinal nenhum',
           <table><tbody><LinhaFila
             c={{ ...primeiro, evidencias: [], fatores_secundarios: [], meses_em_alerta: 0,
                  delta_risco: null, atributos: {} }}
             temValor aoAbrir={() => {}} /></tbody></table>,
           [primeiro.cliente_id, '—'])
  conferir('gaveta de cliente sem sinal',
           <GavetaCliente baseId="inovaapps" execucaoId={null}
                          cliente={{ ...primeiro, evidencias: [], fatores_secundarios: [],
                                     acao_sugerida: '', sinal_dominante: null, historico: [] }}
                          modelo={fila.modelo} mesReferencia={fila.mes_referencia}
                          aoFechar={() => {}} />,
           ['Nenhum sinal se destaca', 'Manter o acompanhamento de rotina'])
  conferir('validação desligada',
           <ConteudoValidacao v={{ ...validacao, metodo: 'Validação desligada', metodos: [],
                                   auc_por_mes: [], precisao_topo: [], curva: [],
                                   coeficientes: [], cancelados: [] }} />,
           ['Validação desligada', 'Modelo escolhido'])

  console.log(falhas ? `\n${falhas} FALHA(S)` : '\nTodas as telas renderizam com os dados reais.')
  process.exit(falhas ? 1 : 0)
}

main().catch((e) => { console.error('\nErro:', e.message, '\nA API está de pé em ' + API + '?'); process.exit(1) })
