/** Hooks de dados: um `useEffect` com `AbortController` por recurso.
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
