/** Leitura das planilhas no próprio navegador, com SheetJS — igual ao protótipo.
 *
 * Serve só ao botão "Analisar dados", que é exploratório e roda antes do treino. O arquivo é o
 * mesmo que vai para a API; o motor faz a leitura de verdade do lado do servidor.
 *
 * A biblioteca entra por `import()` dinâmico: só é baixada quando alguém envia um arquivo.
 */

export type Planilhas = Record<string, Record<string, unknown>[]>

export async function lerArquivo(f: File): Promise<Planilhas> {
  const XLSX = await import('xlsx')
  const buffer = await f.arrayBuffer()
  const wb = XLSX.read(buffer, { type: 'array', cellDates: true })
  const saida: Planilhas = {}
  for (const nome of wb.SheetNames) {
    const linhas = XLSX.utils.sheet_to_json<Record<string, unknown>>(wb.Sheets[nome]!, { defval: '' })
    if (!linhas.length) continue
    // Num .csv a aba se chama "Sheet1"; o nome do arquivo diz mais.
    const chave = /\.csv$/i.test(f.name) ? f.name.replace(/\.csv$/i, '') : nome
    saida[chave] = linhas
  }
  return saida
}

export async function lerArquivos(arquivos: File[]): Promise<Planilhas> {
  const partes = await Promise.all(arquivos.map(lerArquivo))
  return Object.assign({}, ...partes) as Planilhas
}

export const colunasDe = (linhas: Record<string, unknown>[]): string[] =>
  Object.keys(linhas[0] || {})
