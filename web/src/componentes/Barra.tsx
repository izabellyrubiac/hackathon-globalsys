/** Barra de progresso do treino. A fração vem do motor (`motor/treino.py` chama o callback
 * de progresso a cada etapa), não de um contador inventado no front. */

export function Barra({ fracao }: { fracao: number }) {
  const p = Math.max(0, Math.min(1, fracao || 0))
  return (
    <div className="bar" role="progressbar" aria-valuemin={0} aria-valuemax={100}
         aria-valuenow={Math.round(p * 100)}>
      <i style={{ width: `${p * 100}%` }} />
    </div>
  )
}
