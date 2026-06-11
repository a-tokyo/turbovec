import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    // Native addons must load in a fresh process — forks pool isolates them.
    pool: 'forks',
    coverage: {
      provider: 'v8',
      // Cover only the hand-written TS integration source.
      include: ['ts/**'],
      exclude: [
        // napi-generated loader — not coverable by JS instrumentation.
        'index.js',
        // Build outputs and test helpers.
        'dist/**',
        '**/*.config.*',
        '__test__/**',
        // Platform-specific binary packages.
        'npm/**',
      ],
      // Thresholds are set to the measured baseline (see CONTRIBUTING.md).
      // They form a ratchet: raise them when coverage genuinely improves.
      // Measured baseline (2026-06-11): stmts 94.43, branch 86.61, funcs 95.23, lines 94.43.
      // Thresholds are rounded down to give ~4-6pp headroom while still forming a
      // meaningful ratchet. Raise them as coverage improves.
      thresholds: {
        lines: 90,
        functions: 90,
        branches: 80,
        statements: 90,
      },
    },
  },
});
