import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Port 5173 is the origin backend/api.py allows through CORS.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, strictPort: true },
});
