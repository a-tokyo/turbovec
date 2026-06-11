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
    extends: [js.configs.recommended, ...tseslint.configs.recommendedTypeChecked, prettier],
    languageOptions: {
      parserOptions: {
        projectService: true,
        tsconfigRootDir: import.meta.dirname,
      },
    },
    rules: {
      '@typescript-eslint/no-unused-vars': ['error', { argsIgnorePattern: '^_' }],
      '@typescript-eslint/consistent-type-imports': 'error',
    },
  },
  {
    // Type-checked friction inherent to the framework contracts — NOT real
    // defects, so relaxed only for the integration source. The async/await
    // safety rules that actually matter (no-floating-promises,
    // no-misused-promises — the reason this config is type-checked) stay ON.
    //
    //  - require-await: the VectorStore / BaseVectorStore methods (add, delete,
    //    query, similaritySearchVectorWithScore, save, persist, …) MUST be
    //    `async` to satisfy the abstract `Promise<…>` base signatures even when
    //    a particular override has no `await`.
    //  - no-unsafe-* / no-unsafe-enum-comparison: `@llamaindex/core`'s
    //    `Metadata` is `Record<string, any>` and its filter operators / query
    //    modes are enums whose identity lint cannot match across the package
    //    boundary; reading those values is unavoidably "unsafe" to the linter.
    files: ['ts/**/*.ts'],
    rules: {
      '@typescript-eslint/require-await': 'off',
      '@typescript-eslint/no-unsafe-member-access': 'off',
      '@typescript-eslint/no-unsafe-assignment': 'off',
      '@typescript-eslint/no-unsafe-argument': 'off',
      '@typescript-eslint/no-unsafe-return': 'off',
      '@typescript-eslint/no-unsafe-enum-comparison': 'off',
    },
  },
  {
    // The pre-existing native-binding tests (id-map / index / filtering) use
    // `catch (e: any)` to read the napi error `.code`. Allow `any` there only;
    // the shipping integration (ts/**) and its test stay strict. With
    // type-checked linting on, reading `.code` off that `any` also trips the
    // unsafe-member-access rule, so relax it in the same scope.
    files: ['__test__/id-map.test.ts', '__test__/index.test.ts', '__test__/filtering.test.ts'],
    rules: {
      '@typescript-eslint/no-explicit-any': 'off',
      '@typescript-eslint/no-unsafe-member-access': 'off',
    },
  },
  {
    // The integration tests exercise the framework contracts the same way the
    // integration source does: reading `Record<string, any>` metadata,
    // re-parsing tampered persistence JSON (`JSON.parse` → `any`) and defining
    // async stub embedders to satisfy the `EmbeddingsInterface` signature. Same
    // framework-driven friction as `ts/**`; the floating/misused-promise rules
    // still apply.
    files: ['__test__/langchain.test.ts', '__test__/llamaindex.test.ts', '__test__/helpers.ts'],
    rules: {
      '@typescript-eslint/require-await': 'off',
      '@typescript-eslint/no-unsafe-member-access': 'off',
      '@typescript-eslint/no-unsafe-assignment': 'off',
      '@typescript-eslint/no-unsafe-argument': 'off',
      '@typescript-eslint/no-unsafe-return': 'off',
      '@typescript-eslint/no-unsafe-enum-comparison': 'off',
    },
  },
);
