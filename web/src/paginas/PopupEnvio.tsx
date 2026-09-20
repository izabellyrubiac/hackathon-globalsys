/** Envio de uma base: um .xlsx com várias abas ou vários .csv. A API lê, inspeciona e guarda os arquivos. */

import { useRef, useState } from 'react'
import * as api from '../api/cliente'
import { ErroApi } from '../api/cliente'
import type { BaseCriada } from '../api/tipos'
import { CaixaErro } from '../componentes/Estados'
import { Popup } from '../componentes/Popup'
import { kb } from '../formato'

const ACEITA = /\.(xlsx|xlsm|xls|csv|tsv|txt)$/i

export function PopupEnvio({ aoFechar, aoEnviar }: {
  aoFechar: () => void
  aoEnviar: (base: BaseCriada) => void
}) {
  const [arquivos, setArquivos] = useState<File[]>([])
  const [enviando, setEnviando] = useState(false)
  const [erro, setErro] = useState<ErroApi | null>(null)
  const [nota, setNota] = useState<string | null>(null)
  const entrada = useRef<HTMLInputElement>(null)

  function acrescentar(lista: FileList | null) {
    if (!lista) return
    const novos = [...lista]
    const ok = novos.filter((f) => ACEITA.test(f.name))
    if (ok.length < novos.length) setNota('Só arquivos .xlsx, .xls, .csv, .tsv e .txt.')
    setArquivos((a) => {
      const nomes = new Set(a.map((x) => x.name))
      return [...a, ...ok.filter((f) => !nomes.has(f.name))]
    })
  }

  async function confirmar() {
    if (!arquivos.length || enviando) return
    setEnviando(true)
    setErro(null)
    try {
      aoEnviar(await api.enviarBase(arquivos))
    } catch (e) {
      setErro(e instanceof ErroApi ? e : new ErroApi(0, String(e)))
      setEnviando(false)
    }
  }

  return (
    <Popup titulo="Enviar base de dados" aoFechar={enviando ? () => {} : aoFechar}>
      {enviando ? (
        <div style={{ flex: 1, display: 'grid', placeContent: 'center', textAlign: 'center' }}
             role="status" aria-live="polite">
          <div className="ring" />
          <p className="mut">Enviando {arquivos.map((a) => a.name).join(', ')} e inspecionando as tabelas…</p>
        </div>
      ) : (
        <>
          <div
            className="drop" tabIndex={0} role="button"
            onClick={() => entrada.current?.click()}
            onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); entrada.current?.click() } }}
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => { e.preventDefault(); acrescentar(e.dataTransfer.files) }}
          >
            Arraste aqui um .xlsx com uma tabela por aba, ou vários .csv — ou clique para escolher.
          </div>
          <input ref={entrada} type="file" multiple accept=".xlsx,.xlsm,.xls,.csv,.tsv,.txt"
                 style={{ display: 'none' }}
                 onChange={(e) => { acrescentar(e.target.files); e.target.value = '' }} />

          {nota && <p className="mut" style={{ marginTop: 10 }}>{nota}</p>}
          <CaixaErro erro={erro} />

          {arquivos.length > 0 && (
            <ul className="arqs">
              {arquivos.map((a, i) => (
                <li key={a.name}>
                  <span>{a.name}</span>
                  <span className="mut">{kb(a.size)}</span>
                  <button className="btn ghost sm" aria-label={`Tirar ${a.name}`}
                          onClick={() => setArquivos((l) => l.filter((_, j) => j !== i))}>×</button>
                </li>
              ))}
            </ul>
          )}

          <div className="acoes">
            <button className="btn ghost" onClick={aoFechar}>Cancelar</button>
            <button className="btn" disabled={!arquivos.length} onClick={confirmar}>
              Enviar {arquivos.length > 0 && `(${arquivos.length})`}
            </button>
          </div>
        </>
      )}
    </Popup>
  )
}
