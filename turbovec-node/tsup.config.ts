import { defineConfig } from 'tsup';

// Builds the optional framework integrations (currently the LangChain.js
// vector store; the llamaindex entry lands in slice 5) into `dist/` as dual
// ESM + CJS with `.d.ts`. The napi-generated `index.js` / `index.d.ts` live at
// the package root and are intentionally left untouched — `clean: false`.
export default defineConfig({
  entry: {
    langchain: 'ts/langchain.ts',
    llamaindex: 'ts/llamaindex.ts',
  },
  format: ['esm', 'cjs'],
  dts: true,
  sourcemap: true,
  // Do NOT wipe dist between builds — and never touch the root-level
  // napi artifacts (index.js / index.d.ts).
  clean: false,
  target: 'node20',
  // `@langchain/core` and `@llamaindex/core` are optional peer deps and the
  // local native addon is resolved at runtime; never bundle them into the
  // published output.
  external: [/^@langchain\/core/, /^@llamaindex\/core/, '../index.js'],
  // Emit ESM as `.mjs` (and CJS as `.cjs`) so ESM consumers don't trip the
  // Node `MODULE_TYPELESS_PACKAGE_JSON` warning (the package has no
  // `"type":"module"` — and must not, or the CJS napi `index.js` breaks).
  // tsup emits matching `.d.mts` (ESM types) and `.d.ts` (CJS types).
  outExtension({ format }) {
    return { js: format === 'cjs' ? '.cjs' : '.mjs' };
  },
});
