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
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/videos': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/thumbnails': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/gallery': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/video': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      }
    }
  }
})
