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
      // Thresholds form a ratchet — raise them when coverage genuinely improves.
      thresholds: {
        lines: 90,
        functions: 90,
        branches: 80,
        statements: 90,
      },
    },
  },
});
