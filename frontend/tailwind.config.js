/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      }
    },
  },
  safelist: [
    // Dynamic color classes used in components
    { pattern: /bg-(blue|red|emerald|yellow|orange|purple|green|indigo|pink)-(500|600)\/(10|15|20|40)/ },
    { pattern: /text-(blue|red|emerald|yellow|orange|purple|green|indigo|pink)-(300|400)/ },
    { pattern: /border-(blue|red|emerald|yellow|orange|purple|green|indigo|pink)-(500|600)\/(20|30|40|50)/ },
    { pattern: /hover:border-(blue|red|emerald|yellow|orange|purple|green|indigo|pink)-(500)\/(30|40)/ },
    { pattern: /hover:bg-(blue|red|emerald|yellow|orange|purple|green|indigo|pink)-(500|600)\/(25|40)/ },
  ],
  plugins: [],
};
