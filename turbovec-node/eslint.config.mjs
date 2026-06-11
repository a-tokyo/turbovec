import js from '@eslint/js';
import tseslint from 'typescript-eslint';
import prettier from 'eslint-config-prettier';

export default tseslint.config(
  {
    // Lint only the hand-written TS (integration source + tests). Everything
    // else — the napi-generated loader (index.js / index.d.ts), the build
    // output, vendored binaries, configs — is out of scope.
    ignores: [
      'dist',
      'node_modules',
      'npm',
      'target',
      'src',
      'index.js',
      'index.d.ts',
      '*.config.*',
    ],
  },
  {
    files: ['ts/**/*.ts', '__test__/**/*.ts'],
    extends: [js.configs.recommended, ...tseslint.configs.recommended, prettier],
    rules: {
      '@typescript-eslint/no-unused-vars': ['error', { argsIgnorePattern: '^_' }],
      '@typescript-eslint/consistent-type-imports': 'error',
    },
  },
  {
    // The pre-existing native-binding tests (id-map / index / filtering) use
    // `catch (e: any)` to read the napi error `.code`. Allow `any` there only;
    // the shipping integration (ts/**) and its test stay strict.
    files: ['__test__/id-map.test.ts', '__test__/index.test.ts', '__test__/filtering.test.ts'],
    rules: {
      '@typescript-eslint/no-explicit-any': 'off',
    },
  },
);
