import js from '@eslint/js'
import globals from 'globals'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import { defineConfig, globalIgnores } from 'eslint/config'

export default defineConfig([
  // public/ holds vendored, minified third-party scripts (OpenPanel's op1.js).
  globalIgnores(['dist', 'public']),
  {
    files: ['**/*.{js,jsx}'],
    extends: [
      js.configs.recommended,
      reactHooks.configs['recommended-latest'],
      reactRefresh.configs.vite,
    ],
    languageOptions: {
      ecmaVersion: 2020,
      globals: globals.browser,
      parserOptions: {
        ecmaVersion: 'latest',
        ecmaFeatures: { jsx: true },
        sourceType: 'module',
      },
    },
    rules: {
      // `catch (e)` / `catch (_)` with the error deliberately ignored is the
      // house style for best-effort localStorage and fetch calls.
      'no-unused-vars': ['error', {
        // A capitalised destructured arg (`{ icon: Icon }`) is a component used
        // as JSX, which this config's core rule does not see.
        varsIgnorePattern: '^[A-Z_]', argsIgnorePattern: '^(_|[A-Z])', caughtErrors: 'none',
      }],
      // Contexts and modals export a hook or a constant next to the component.
      'react-refresh/only-export-components': ['error', { allowConstantExport: true }],
    },
  },
  {
    // The entry point mounts a few one-off components; nothing here hot-reloads.
    files: ['src/main.jsx'],
    rules: { 'react-refresh/only-export-components': 'off' },
  },
  {
    files: ['vite.config.js', 'vite-plugin-seo.js', 'seo/**/*.js'],
    languageOptions: { globals: { ...globals.node } },
  },
])
