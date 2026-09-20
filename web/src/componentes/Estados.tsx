/** Carregando, vazio, aviso e erro — um caminho visual só para tudo que vem do backend.
 *
 * Todo erro da API tem o mesmo formato `{detalhe, problemas[]}` (garantido pelos exception
 * handlers de `api/main.py`), então um componente cobre o 422 do treino, o 409 do ativar,
 * o 413 do upload e o 404 de base sem execução pronta.
 */

import type { ReactNode } from 'react'
import { ErroApi } from '../api/cliente'

export function Carregando({ texto = 'Carregando…' }: { texto?: string }) {
  return (
    <div className="card" style={{ textAlign: 'center' }} role="status" aria-live="polite">
      <div className="ring" />
      <p className="mut">{texto}</p>
    </div>
  )
}

export function Vazio({ children }: { children: ReactNode }) {
  return <p className="mut vazio">{children}</p>
}

export function CaixaErro({ erro, titulo }: { erro: ErroApi | null; titulo?: string }) {
  if (!erro) return null
  return (
    <div className="aviso erro" role="alert">
      <b>{titulo || erro.detalhe}</b>
      {titulo && <span>{erro.detalhe}</span>}
      {erro.problemas.length > 0 && (
        <ul>{erro.problemas.map((p, i) => <li key={i}>{p}</li>)}</ul>
      )}
    </div>
  )
}

/** Os avisos que o motor devolve (tabela ignorada, histórico curto, perfil rápido…). */
export function Avisos({ avisos, titulo = 'Avisos do motor' }: { avisos: string[]; titulo?: string }) {
  if (!avisos?.length) return null
  return (
    <div className="aviso">
      <b>{titulo}</b>
      <ul>{avisos.map((a, i) => <li key={i}>{a}</li>)}</ul>
    </div>
  )
}
