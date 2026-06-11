import { defineConfig } from 'tsup';

// Builds the optional framework integrations (currently the LangChain.js
// vector store; the llamaindex entry lands in slice 5) into `dist/` as dual
// ESM + CJS with `.d.ts`. The napi-generated `index.js` / `index.d.ts` live at
// the package root and are intentionally left untouched — `clean: false`.
export default defineConfig({
  entry: {
    langchain: 'ts/langchain.ts',
    // llamaindex: "ts/llamaindex.ts", // slice 5
  },
  format: ['esm', 'cjs'],
  dts: true,
  sourcemap: true,
  // Do NOT wipe dist between builds — and never touch the root-level
  // napi artifacts (index.js / index.d.ts).
  clean: false,
  target: 'node20',
  // `@langchain/core` is an optional peer dep and the local native addon is
  // resolved at runtime; never bundle either into the published output.
  external: ['@langchain/core', '../index.js'],
  outExtension({ format }) {
    return { js: format === 'cjs' ? '.cjs' : '.js' };
  },
});
