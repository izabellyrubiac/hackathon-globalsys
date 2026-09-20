/** Moldura do app: menu lateral fixo e a tela da rota atual. */

import * as api from './api/cliente'
import { useRecurso } from './api/ganchos'
import { CaixaErro } from './componentes/Estados'
import { useSessao } from './estado/Sessao'
import { Fila } from './paginas/Fila'
import { Modelos } from './paginas/Modelos'
import { Validacao } from './paginas/Validacao'
import { ROTAS, TITULO, useRota } from './rotas'

export function App() {
  const rota = useRota()
  const { versao } = useSessao()

  // Uma só busca da lista de bases para as três telas; refeita quando um treino termina.
  const bases = useRecurso((s) => api.listarBases(s), [versao])

  return (
    <div className="app">
      <aside className="side">
        <div className="logo">Global <span>Oráculo</span></div>
        <nav>
          {ROTAS.map((r) => (
            <a key={r} href={'#/' + r} aria-current={r === rota ? 'page' : undefined}>
              {TITULO[r]}
            </a>
          ))}
        </nav>
      </aside>

      <main className={'main' + (rota === 'modelos' ? ' estreito' : '')}>
        <CaixaErro erro={bases.erro} titulo="Não consegui listar as bases." />
        {rota === 'fila' && <Fila bases={bases.dados ?? []} carregandoBases={bases.carregando} />}
        {rota === 'modelos' && (
          <Modelos bases={bases.dados ?? []} carregandoBases={bases.carregando}
                   recarregarBases={bases.recarregar} />
        )}
        {rota === 'validacao' && (
          <Validacao bases={bases.dados ?? []} carregandoBases={bases.carregando} />
        )}
      </main>
    </div>
  )
}
