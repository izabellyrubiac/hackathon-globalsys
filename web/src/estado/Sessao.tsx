/** Estado que atravessa as telas: qual base está em foco e as planilhas lidas nesta sessão.
 *
 * As planilhas só existem quando o usuário enviou o arquivo aqui, no navegador: são o insumo do
 * botão "Analisar dados", que roda client-side. Numa base de demonstração (que já estava no
 * servidor) não há planilha, e o botão fica desativado — ver PENDENCIAS-MOTOR.md.
 */

import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from 'react'

export type Planilhas = Record<string, Record<string, unknown>[]>

interface Estado {
  baseId: string | null
  escolherBase: (id: string | null) => void
  /** base_id → planilhas lidas nesta sessão (só as que passaram pelo navegador). */
  planilhas: Record<string, Planilhas>
  guardarPlanilhas: (baseId: string, p: Planilhas) => void
  /** Vira um número novo a cada treino concluído: as telas escutam para recarregar. */
  versao: number
  avisarMudanca: () => void
}

const Ctx = createContext<Estado | null>(null)

export function ProvedorSessao({ children }: { children: ReactNode }) {
  const [baseId, setBaseId] = useState<string | null>(null)
  const [planilhas, setPlanilhas] = useState<Record<string, Planilhas>>({})
  const [versao, setVersao] = useState(0)

  const guardarPlanilhas = useCallback((id: string, p: Planilhas) => {
    setPlanilhas((a) => ({ ...a, [id]: p }))
  }, [])

  const valor = useMemo<Estado>(() => ({
    baseId,
    escolherBase: setBaseId,
    planilhas,
    guardarPlanilhas,
    versao,
    avisarMudanca: () => setVersao((v) => v + 1),
  }), [baseId, planilhas, guardarPlanilhas, versao])

  return <Ctx.Provider value={valor}>{children}</Ctx.Provider>
}

export function useSessao(): Estado {
  const c = useContext(Ctx)
  if (!c) throw new Error('useSessao fora do ProvedorSessao.')
  return c
}
