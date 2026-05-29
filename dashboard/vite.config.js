import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    allowedHosts: [
      'openshorts.app',
      'www.openshorts.app'
    ],
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/videos': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/thumbnails': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/gallery': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/video': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/render': {
        target: 'http://localhost:3100',
        changeOrigin: true,
      }
    }
  }
})
