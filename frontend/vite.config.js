import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    host: '0.0.0.0',
    proxy: {
      // Only proxy POST requests for API endpoints, let Vite serve index.html for GET (client-side routing)
      '/search': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        bypass: (req, _res, _options) => {
          if (req.method !== 'POST') {
            return '/index.html'; // Let Vite serve the SPA entry point
          }
        },
      },
      '/ask': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        bypass: (req, _res, _options) => {
          if (req.method !== 'POST') {
            return '/index.html';
          }
        },
      },
      '/ingest': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        bypass: (req, _res, _options) => {
          if (req.method !== 'POST') {
            return '/index.html';
          }
        },
      },
      '/index': { target: 'http://localhost:8000', changeOrigin: true },
      '/bids': { target: 'http://localhost:8000', changeOrigin: true },
      '/documents': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        bypass: (req, _res, _options) => {
          if (req.method !== 'POST') {
            return '/index.html';
          }
        },
      },
      '/chunks': { target: 'http://localhost:8000', changeOrigin: true },
      '/extract': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        bypass: (req, _res, _options) => {
          if (req.method !== 'POST') {
            return '/index.html';
          }
        },
      },
      '/extraction': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        bypass: (req, _res, _options) => {
          if (req.method !== 'POST') {
            return '/index.html';
          }
        },
      },
      '/upload': { target: 'http://localhost:8000', changeOrigin: true },
      '/health': { target: 'http://localhost:8000', changeOrigin: true },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
  },
});