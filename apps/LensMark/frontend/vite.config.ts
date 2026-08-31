// LensMark front end build. Output goes straight into the Python package (lensmark/static/) and is
// committed, so `lensmark serve` needs no node at runtime. `base: "/"` -> absolute /assets/ URLs.
import { defineConfig } from "vitest/config";

export default defineConfig({
  base: "/",
  build: {
    outDir: "../lensmark/static",
    emptyOutDir: true,
    sourcemap: false,
    target: "es2022",
  },
  server: {
    port: 5173,
    proxy: { "/api": { target: "http://127.0.0.1:8765", changeOrigin: false } },
  },
  test: {
    environment: "node",
    include: ["src/**/*.test.ts"],
  },
});
