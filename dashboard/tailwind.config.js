/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Lumen · Night Foundry — values mirror tokens.css (kept literal so
        // Tailwind alpha modifiers like bg-brass/10 compile)
        paper: "oklch(13% 0.014 265 / <alpha-value>)",
        paper2: "oklch(16.5% 0.015 265 / <alpha-value>)",
        paper3: "oklch(20% 0.016 265 / <alpha-value>)",
        ink: "oklch(96% 0.006 262 / <alpha-value>)",
        ink2: "oklch(86% 0.01 262 / <alpha-value>)",
        muted: "oklch(64% 0.012 262 / <alpha-value>)",
        brass: "oklch(76% 0.17 50 / <alpha-value>)",
        brassink: "oklch(17% 0.03 50 / <alpha-value>)",
        coral: "oklch(68% 0.16 18 / <alpha-value>)",
        ok: "oklch(75% 0.11 150 / <alpha-value>)",
        warn: "oklch(78% 0.14 75 / <alpha-value>)",
        danger: "oklch(66% 0.18 25 / <alpha-value>)",
        // legacy aliases so untouched files degrade gracefully
        background: "oklch(13% 0.014 265 / <alpha-value>)",
        surface: "oklch(16.5% 0.015 265 / <alpha-value>)",
        primary: "oklch(76% 0.17 50 / <alpha-value>)",
        accent: "oklch(68% 0.16 18 / <alpha-value>)",
      },
      fontFamily: {
        display: "var(--font-display)",
        body: "var(--font-body)",
        sans: "var(--font-body)",
        serif: "var(--font-display)",
        mono: "var(--font-mono)",
      },
      borderColor: {
        rule: "var(--color-rule)",
        rule2: "var(--color-rule-2)",
      },
      borderRadius: {
        card: "var(--radius-card)",
        input: "var(--radius-input)",
      },
      fontSize: {
        micro: ["10.5px", { letterSpacing: "0.10em" }],
      },
      transitionTimingFunction: {
        out: "var(--ease-out)",
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'fade': 'fadeIn 0.4s var(--ease-out)',
      },
      keyframes: {
        fadeIn: {
          from: { opacity: '0' },
          to: { opacity: '1' },
        },
      },
    },
  },
  plugins: [],
}
