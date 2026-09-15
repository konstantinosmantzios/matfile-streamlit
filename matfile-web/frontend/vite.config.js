import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  build: {
    target: 'es2020',
    sourcemap: false,
    chunkSizeWarningLimit: 2000,
    rollupOptions: {
      output: {
        // Plotly.js is huge — keep it in its own chunk so it caches separately
        // and never blocks React app updates. (rolldown requires a function.)
        manualChunks(id) {
          if (!id.includes('node_modules')) return undefined
          if (id.includes('plotly.js') || id.includes('react-plotly.js')) return 'plotly'
          if (id.includes('react') || id.includes('react-dom') ||
              id.includes('axios') || id.includes('lucide-react')) return 'vendor'
          return 'vendor'
        },
      },
    },
  },
})