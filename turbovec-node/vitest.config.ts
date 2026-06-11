import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    // Native addons must load in a fresh process — forks pool isolates them.
    pool: 'forks',
  },
});
