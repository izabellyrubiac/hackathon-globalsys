import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// O front fala sempre com caminhos relativos ("/api/..."); o proxy leva para a API FastAPI.
// Assim não há CORS no desenvolvimento nem base-url para configurar em produção.
const API = process.env.API_URL || 'http://localhost:8000'

export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { '/api': { target: API, changeOrigin: true } } },
  preview: { port: 4173, proxy: { '/api': { target: API, changeOrigin: true } } },
})
