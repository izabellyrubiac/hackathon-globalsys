/** Hooks de dados: um `useEffect` com `AbortController` por recurso, e o polling do treino.
 * Sem biblioteca de cache — são três telas e um punhado de recursos.
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import { ErroApi } from './cliente'

export interface Recurso<T> {
  dados: T | null
  carregando: boolean
  erro: ErroApi | null
  recarregar: () => void
}

/** Busca um recurso e refaz a busca quando as `chaves` mudam ou quando se pede `recarregar()`. */
export function useRecurso<T>(
  buscar: (s: AbortSignal) => Promise<T>,
  chaves: unknown[],
  ativo = true,
): Recurso<T> {
  const [dados, setDados] = useState<T | null>(null)
  const [carregando, setCarregando] = useState(ativo)
  const [erro, setErro] = useState<ErroApi | null>(null)
  const [n, setN] = useState(0)
  const ref = useRef(buscar)
  ref.current = buscar

  useEffect(() => {
    if (!ativo) {
      setCarregando(false)
      return
    }
    const ctrl = new AbortController()
    let vivo = true
    setCarregando(true)
    setErro(null)
    ref.current(ctrl.signal).then(
      (d) => { if (vivo) { setDados(d); setCarregando(false) } },
      (e) => {
        if (!vivo || ctrl.signal.aborted) return
        setErro(e instanceof ErroApi ? e : new ErroApi(0, String(e)))
        setCarregando(false)
      },
    )
    return () => { vivo = false; ctrl.abort() }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...chaves, ativo, n])

  return { dados, carregando, erro, recarregar: useCallback(() => setN((x) => x + 1), []) }
}

/** Acompanha uma execução até ela ficar `pronta` ou dar `erro`.
 *
 * A API não empurra progresso (não há SSE): o jeito é perguntar. Um `setTimeout` recursivo de
 * 1 s evita empilhar chamadas quando uma resposta demora. Para sozinho no estado final e no
 * desmonte — o `vivo` descarta qualquer resposta que chegue depois.
 */
export function usePolling<T extends { estado: string }>(
  buscar: (s: AbortSignal) => Promise<T>,
  parar: (d: T) => boolean,
  ativo: boolean,
  intervalo = 1000,
): { dados: T | null; erro: ErroApi | null } {
  const [dados, setDados] = useState<T | null>(null)
  const [erro, setErro] = useState<ErroApi | null>(null)
  const ref = useRef(buscar)
  ref.current = buscar

  useEffect(() => {
    if (!ativo) return
    let vivo = true
    let timer: number | undefined
    const ctrl = new AbortController()

    const passo = async () => {
      try {
        const d = await ref.current(ctrl.signal)
        if (!vivo) return
        setDados(d)
        setErro(null)
        if (parar(d)) return
      } catch (e) {
        if (!vivo || ctrl.signal.aborted) return
        // Uma falha isolada (API reiniciando) não derruba o acompanhamento: registra e tenta de novo.
        setErro(e instanceof ErroApi ? e : new ErroApi(0, String(e)))
      }
      if (vivo) timer = window.setTimeout(passo, intervalo)
    }
    passo()

    return () => { vivo = false; ctrl.abort(); if (timer) clearTimeout(timer) }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ativo, intervalo])

  return { dados, erro }
}
