/** Combobox do protótipo, agora como componente React.
 *
 * No HTML original isso era feito reescrevendo o `<select>` nativo depois de cada render
 * (`melhorarSelect`). Aqui é um listbox de verdade, com o mesmo comportamento de teclado:
 * setas, Home/End, Enter, Esc, busca por digitação, e abre para cima quando não cabe embaixo.
 */

import { useEffect, useId, useRef, useState } from 'react'
import { norm } from '../formato'

export interface Opcao {
  valor: string
  texto: string
  /** Vira o `title` da linha — usado para mostrar tipo e exemplos de uma coluna. */
  dica?: string
}

export function Select({ valor, opcoes, aoMudar, vazio, rotulo, id, desabilitado }: {
  valor: string
  opcoes: Opcao[]
  aoMudar: (v: string) => void
  /** Texto da opção vazia; sem ele, a lista não tem opção vazia. */
  vazio?: string
  rotulo?: string
  id?: string
  desabilitado?: boolean
}) {
  const auto = useId()
  const meuId = id || auto
  const [aberto, setAberto] = useState(false)
  const [ativo, setAtivo] = useState(0)
  const [acima, setAcima] = useState(false)
  const botao = useRef<HTMLButtonElement>(null)
  const lista = useRef<HTMLUListElement>(null)
  const busca = useRef({ texto: '', t: 0 })

  const itens: Opcao[] = vazio != null ? [{ valor: '', texto: vazio }, ...opcoes] : opcoes
  const iSel = Math.max(0, itens.findIndex((o) => o.valor === valor))
  const atual = itens[iSel]

  useEffect(() => {
    if (!aberto) return
    setAtivo(iSel)
    const r = botao.current?.getBoundingClientRect()
    if (r) {
      const alt = Math.min(270, itens.length * 34 + 12)
      setAcima(innerHeight - r.bottom < alt && r.top > innerHeight - r.bottom)
    }
    const fora = (e: MouseEvent) => {
      if (!botao.current?.parentElement?.contains(e.target as Node)) setAberto(false)
    }
    document.addEventListener('click', fora)
    return () => document.removeEventListener('click', fora)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [aberto])

  useEffect(() => {
    if (!aberto) return
    lista.current?.children[ativo]?.scrollIntoView({ block: 'nearest' })
  }, [ativo, aberto])

  function escolher(i: number) {
    const o = itens[i]
    if (o) aoMudar(o.valor)
    setAberto(false)
    botao.current?.focus()
  }

  function tecla(e: React.KeyboardEvent) {
    const k = e.key
    const n = itens.length
    if (k === 'ArrowDown' || k === 'ArrowUp') {
      e.preventDefault()
      if (!aberto) return setAberto(true)
      setAtivo((a) => (a + (k === 'ArrowDown' ? 1 : -1) + n) % n)
    } else if ((k === 'Home' || k === 'End') && aberto) {
      e.preventDefault()
      setAtivo(k === 'Home' ? 0 : n - 1)
    } else if (k === 'Enter' || k === ' ') {
      e.preventDefault()
      if (aberto) escolher(ativo)
      else setAberto(true)
    } else if (k === 'Escape' && aberto) {
      e.preventDefault()
      setAberto(false)
    } else if (k.length === 1) {
      // busca por digitação: acumula as letras por 700 ms, como num <select> nativo
      const b = busca.current
      clearTimeout(b.t)
      b.texto += k
      b.t = window.setTimeout(() => { b.texto = '' }, 700)
      const j = itens.findIndex((o) => norm(o.texto).startsWith(norm(b.texto)))
      if (j >= 0) { setAberto(true); setAtivo(j) }
    }
  }

  return (
    <div className={'dd' + (aberto ? ' aberto' : '') + (acima ? ' acima' : '')}>
      <button
        ref={botao} type="button" id={meuId} className="dd-b"
        disabled={desabilitado}
        aria-haspopup="listbox" aria-expanded={aberto} aria-label={rotulo}
        onClick={() => !desabilitado && setAberto((a) => !a)}
        onKeyDown={tecla}
      >
        <span className={atual && atual.valor === '' ? 'mut' : undefined}>{atual?.texto ?? ''}</span>
      </button>
      {aberto && (
        <ul ref={lista} className="dd-l" role="listbox" aria-label={rotulo} tabIndex={-1}>
          {itens.map((o, i) => (
            <li
              key={o.valor || '__vazio'} role="option"
              aria-selected={o.valor === valor}
              className={(i === ativo ? 'ativo' : '') + (o.valor === '' ? ' ph' : '')}
              title={o.dica}
              onMouseDown={(e) => e.preventDefault()}
              onMouseMove={() => setAtivo(i)}
              onClick={() => escolher(i)}
            >{o.texto}</li>
          ))}
        </ul>
      )}
    </div>
  )
}

/** Rótulo + Select, o par que se repete nos formulários. */
export function CampoSelect({ rotulo, ...resto }: { rotulo: string } & Omit<Parameters<typeof Select>[0], 'rotulo'>) {
  const id = useId()
  return (
    <div className="campo">
      <label htmlFor={id}>{rotulo}</label>
      <Select id={id} rotulo={rotulo} {...resto} />
    </div>
  )
}
