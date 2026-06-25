import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(() => {
  const backendUrl = process.env.VITE_BACKEND_URL || 'http://localhost:8000';
  const wsBackendUrl = backendUrl.replace(/^http/, 'ws');

  return {
    plugins: [react()],
    server: {
      port: 5173,
      host: true, // Listen on all network interfaces for Docker
      proxy: {
        '/api': { target: backendUrl, changeOrigin: true },
        '/ws': { target: wsBackendUrl, ws: true, changeOrigin: true },
      },
    },
    build: { outDir: 'dist' },
  }
})
