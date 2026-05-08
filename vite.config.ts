import { defineConfig } from 'vite';

export default defineConfig({
  root: 'static',
  publicDir: 'public',
  base: './',
  build: {
    outDir: '../dist',
    emptyOutDir: true,
    target: 'es2022',
  },
  server: {
    port: 5173,
  },
});
