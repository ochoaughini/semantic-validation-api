import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  base: './', // Change to relative path for better compatibility
  server: {
    port: 3000
  },
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
    minify: true,
    sourcemap: true,
    // Add rollup options for better production build
    rollupOptions: {
      output: {
        manualChunks: undefined
      }
    }
  }
})

