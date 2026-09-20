/** Estado que atravessa as telas: qual base está em foco e um contador que avisa que algo mudou. */

import { createContext, useContext, useMemo, useState, type ReactNode } from 'react'

interface Estado {
  baseId: string | null
  escolherBase: (id: string | null) => void
  /** Vira um número novo a cada treino concluído: as telas escutam para recarregar. */
  versao: number
  avisarMudanca: () => void
}

const Ctx = createContext<Estado | null>(null)

export function ProvedorSessao({ children }: { children: ReactNode }) {
  const [baseId, setBaseId] = useState<string | null>(null)
  const [versao, setVersao] = useState(0)

  const valor = useMemo<Estado>(() => ({
    baseId,
    escolherBase: setBaseId,
    versao,
    avisarMudanca: () => setVersao((v) => v + 1),
  }), [baseId, versao])

  return <Ctx.Provider value={valor}>{children}</Ctx.Provider>
}

export function useSessao(): Estado {
  const c = useContext(Ctx)
  if (!c) throw new Error('useSessao fora do ProvedorSessao.')
  return c
}
