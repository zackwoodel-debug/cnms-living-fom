import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: true,
    // Same-origin /api in development, so the browser never needs CORS and the
    // production nginx config can proxy the same path.
    proxy: {
      "/api": {
        target: process.env.VITE_API_BASE_URL ?? "http://localhost:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
