/** Aviso curto no canto, como no protótipo: `role="status"` e some sozinho em 3,2 s. */

import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from 'react'

const Ctx = createContext<(t: string) => void>(() => {})

export function ProvedorToast({ children }: { children: ReactNode }) {
  const [msgs, setMsgs] = useState<{ id: number; texto: string }[]>([])
  const avisar = useCallback((texto: string) => {
    const id = Date.now() + Math.random()
    setMsgs((a) => [...a, { id, texto }])
    setTimeout(() => setMsgs((a) => a.filter((m) => m.id !== id)), 3200)
  }, [])
  return (
    <Ctx.Provider value={avisar}>
      {children}
      {msgs.map((m) => <div key={m.id} className="toast" role="status">{m.texto}</div>)}
    </Ctx.Provider>
  )
}

export const useToast = () => useContext(Ctx)

/** Esc fecha o que estiver aberto por cima. */
export function useEsc(fechar: () => void, ativo = true) {
  useEffect(() => {
    if (!ativo) return
    const h = (e: KeyboardEvent) => { if (e.key === 'Escape') fechar() }
    document.addEventListener('keydown', h)
    return () => document.removeEventListener('keydown', h)
  }, [fechar, ativo])
}
