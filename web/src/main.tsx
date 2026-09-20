import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { App } from './App'
import { ProvedorToast } from './componentes/Toast'
import { ProvedorSessao } from './estado/Sessao'
import './estilo.css'

createRoot(document.getElementById('raiz')!).render(
  <StrictMode>
    <ProvedorToast>
      <ProvedorSessao>
        <App />
      </ProvedorSessao>
    </ProvedorToast>
  </StrictMode>,
)
