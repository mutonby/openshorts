/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "#0c0c0c",
        sidebar:    "#111111",
        surface:    "#141414",
        border:     "#1e1e1e",
        primary:    "#5b5ef4",
        accent:     "#5b5ef4",
        success:    "#34d470",
        platform: {
          youtube:   "#f87171",
          tiktok:    "#a5a8fd",
          instagram: "#f0abfc",
          snapchat:  "#facc15",
          facebook:  "#1877f2",
        },
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      }
    },
  },
  plugins: [],
}
