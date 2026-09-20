/** Conferência das funções puras contra o resultado real do motor.
 *
 * Os imports levam a extensão `.ts` porque este arquivo roda direto no Node, que resolve o
 * caminho exato; o bundler aceita as duas formas.
 *
 * Neste projeto a IA não faz revisão visual: a verificação é rodar o código e checar os números.
 * Este script busca a fila e a validação da API e confere que as funções que a tela usa devolvem
 * o que se espera. Rode com a API de pé:
 *
 *     npm run verificar
 */

import { aplicarFiltros, atributosDisponiveis, variacao, FILTROS_VAZIOS } from './dominio.ts'
import { brl, dataHora, mesFmt, pc, recuarMes, sg } from './formato.ts'
import type { ClienteFila, PaginaClientes, Validacao } from './api/tipos.ts'

const API = process.env.API || 'http://localhost:8000'

let falhas = 0
function conferir(nome: string, obtido: unknown, esperado: unknown) {
  const a = JSON.stringify(obtido), b = JSON.stringify(esperado)
  if (a === b) console.log(`  ok   ${nome}`)
  else { falhas++; console.log(`  FALHA ${nome}\n        obtido:   ${a}\n        esperado: ${b}`) }
}

async function pegar<T>(rota: string): Promise<T> {
  const r = await fetch(API + rota)
  if (!r.ok) throw new Error(`${rota} respondeu ${r.status}`)
  return r.json() as Promise<T>
}

async function main() {
  console.log('formato pt-BR')
  conferir('brl', brl(12345.6), 'R$ 12.346')
  conferir('brl nulo', brl(null), '—')
  conferir('pc', pc(0.1603), '16%')
  conferir('mesFmt', mesFmt('2026-06'), 'jun/2026')
  conferir('dataHora', dataHora('2026-09-20T00:16:43-03:00'), '20/set/2026 00:16')
  conferir('sg positivo', sg(12.3), '+12,3')
  conferir('sg negativo', sg(-4), '−4')
  conferir('recuarMes', recuarMes('2026-06', 3), '2026-03')
  conferir('recuarMes virando o ano', recuarMes('2026-01', 2), '2025-11')

  console.log('\nfila real da API')
  const fila = await pegar<PaginaClientes>('/api/bases/inovaapps/clientes?limite=5000&incluir_historico=false')
  const cs = fila.clientes
  conferir('58 ativos', cs.length, 58)
  conferir('ordem do motor preservada',
           cs.map((c) => c.prioridade), Array.from({ length: cs.length }, (_, i) => i + 1))
  conferir('atributos dinâmicos',
           atributosDisponiveis(cs).map((a) => a.chave), ['plano', 'porte', 'segmento'])

  const emRisco = aplicarFiltros(cs, { ...FILTROS_VAZIOS, faixa: 'alto' })
  const atencao = aplicarFiltros(cs, { ...FILTROS_VAZIOS, faixa: 'atencao' })
  conferir('faixa alto bate com o resumo', emRisco.length, fila.resumo!.em_risco_alto)
  conferir('faixa atenção bate com o resumo', atencao.length, fila.resumo!.em_atencao)
  conferir('filtrar não reordena',
           emRisco.map((c) => c.prioridade),
           [...emRisco].sort((a, b) => a.prioridade - b.prioridade).map((c) => c.prioridade))
  conferir('busca por id', aplicarFiltros(cs, { ...FILTROS_VAZIOS, busca: 'c071' }).length, 1)
  // filtro "contrato iniciado desde": cliente sem a data ou anterior ao mês pedido sai da lista
  const comData = [
    { ...cs[0]!, datas: { inicio_contrato: '2024-05' } }, { ...cs[1]!, datas: { inicio_contrato: '2022-11' } },
    { ...cs[2]!, datas: { inicio_contrato: null } }, { ...cs[3]! },
  ]
  conferir('filtro por início de contrato',
           aplicarFiltros(comData, { ...FILTROS_VAZIOS, datas: { inicio_contrato: '2023-01' } }).map((c) => c.cliente_id),
           [cs[0]!.cliente_id])

  const soma = cs.reduce((s, c) => s + (c.perda_anual_esperada ?? 0), 0)
  conferir('perda anual soma o resumo',
           Math.round(soma), Math.round(fila.resumo!.perda_anual_esperada_total!))

  conferir('variação vem do motor', variacao(cs[0]!).tipo !== 'sem-dado', true)
  // risco 0,20 hoje contra 0,10 no mês anterior: +10 pontos percentuais, o dobro do risco
  const comHistorico = variacao({ ...cs[0]!, risco: 0.2, delta_risco: null,
    historico: [{ mes_ref: '2026-05', risco: 0.1, valores: {} },
                { mes_ref: '2026-06', risco: 0.2, valores: {} }] } as ClienteFila)
  conferir('variação cai no histórico quando falta o delta',
           [comHistorico.tipo, 'origem' in comHistorico ? comHistorico.origem : null,
            'pp' in comHistorico ? Math.round(comHistorico.pp) : null,
            'rel' in comHistorico ? Math.round(comHistorico.rel) : null],
           ['pior', 'historico', 10, 100])
  conferir('sem delta e sem histórico', variacao({ ...cs[0]!, delta_risco: null, historico: undefined } as ClienteFila),
           { tipo: 'sem-dado' })

  console.log('\nvalidação real da API')
  const v = await pegar<Validacao>('/api/bases/inovaapps/validacao')
  conferir('painel', [v.painel.clientes, v.painel.cancelados, v.painel.ativos], [80, 22, 58])
  conferir('modelo escolhido', v.escolha.modelo, 'logistica')
  conferir('um único modelo avaliado', v.modelos.filter((m) => m.avaliado).length, 1)
  conferir('três métodos comparados', v.metodos.map((m) => m.id),
           ['melhor_variavel', 'motor_alto', 'motor_alto_ou_atencao'])
  conferir('22 cancelados listados', v.cancelados.length, 22)
  conferir('coeficientes = variáveis selecionadas', v.coeficientes.length, 11)

  console.log(falhas ? `\n${falhas} FALHA(S)` : '\nTudo conferido.')
  process.exit(falhas ? 1 : 0)
}

main().catch((e) => { console.error('\nErro:', e.message, '\nA API está de pé em ' + API + '?'); process.exit(1) })
