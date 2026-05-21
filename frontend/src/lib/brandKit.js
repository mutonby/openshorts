// Brand Kit persistence + defaults.
//
// The brand kit defines how text overlays look across the app. Layout-level
// settings (size, stroke width, position, words-per-line) are stored
// PER ASPECT RATIO — Shorts (9:16) and YouTube (16:9) usually want very
// different sizing and wrapping. Colors and font are SHARED across ratios
// because they're brand identity, not layout.
//
// Shape:
//   {
//     colors: [{ name, hex }, ...]              // 1+ named brand colors
//     font:   { family, source, url? }          // url present for bundled/user fonts
//     styles: {
//       '9:16': { size, strokeWidth, textColor, strokeColor, position, wordsPerLine },
//       '16:9': { size, strokeWidth, textColor, strokeColor, position, wordsPerLine },
//     }
//   }
//
// Position values use a 3x3 anchor grid:
//   top-left | top-center | top-right
//   middle-left | middle-center | middle-right
//   bottom-left | bottom-center | bottom-right
//
// `size` and `strokeWidth` are in 1080p-equivalent pixels (canvas units).

import { useEffect, useState } from 'react';

const STORAGE_KEY = 'openshorts.brandKit.v2';
const LEGACY_KEY  = 'openshorts.brandKit.v1';

export const ALL_POSITIONS = [
  'top-left',    'top-center',    'top-right',
  'middle-left', 'middle-center', 'middle-right',
  'bottom-left', 'bottom-center', 'bottom-right',
];

export const DEFAULT_PREVIEW_TEXT = 'Stop scrolling and watch this insane clip right now before you regret it';

export const DEFAULT_BRAND_KIT = {
  colors: [
    { name: 'Primary',  hex: '#FFFFFF' },
    { name: 'Accent',   hex: '#FFD60A' },
    { name: 'Stroke',   hex: '#000000' },
  ],
  font: { family: 'Inter', source: 'system', url: null },
  previewText: DEFAULT_PREVIEW_TEXT,
  styles: {
    '9:16': {
      size: 72,
      strokeWidth: 6,
      textColor:    '#FFFFFF',
      strokeColor:  '#000000',
      position:     'bottom-center',
      wordsPerLine: 2,
      textCase:     'upper',   // Hormozi default for shorts
    },
    '16:9': {
      size: 48,
      strokeWidth: 4,
      textColor:    '#FFFFFF',
      strokeColor:  '#000000',
      position:     'bottom-center',
      wordsPerLine: 10,
      textCase:     'original',
    },
  },
};

// Apply a brand-kit text-case setting to a string.
export function applyTextCase(text, textCase) {
  if (textCase === 'upper') return String(text).toUpperCase();
  if (textCase === 'lower') return String(text).toLowerCase();
  return String(text);
}

function migrateLegacy(parsed) {
  // v1 had a single { style: ... } block; convert to per-ratio styles.
  if (parsed?.style && !parsed.styles) {
    const legacyPos = parsed.style.position; // 'top' | 'middle' | 'bottom'
    const legacyAlign = parsed.style.align || 'center'; // 'left' | 'center' | 'right'
    const mapped = `${legacyPos === 'middle' ? 'middle' : legacyPos}-${legacyAlign}`;
    const safePosition = ALL_POSITIONS.includes(mapped) ? mapped : 'bottom-center';
    const base = {
      size: parsed.style.size,
      strokeWidth: parsed.style.strokeWidth,
      textColor:   parsed.style.textColor,
      strokeColor: parsed.style.strokeColor,
      position:    safePosition,
    };
    return {
      colors: parsed.colors || DEFAULT_BRAND_KIT.colors,
      font:   parsed.font   || DEFAULT_BRAND_KIT.font,
      styles: {
        '9:16': { ...DEFAULT_BRAND_KIT.styles['9:16'], ...base, wordsPerLine: 2 },
        '16:9': { ...DEFAULT_BRAND_KIT.styles['16:9'], ...base, wordsPerLine: 10, size: Math.round(base.size * 0.66) },
      },
    };
  }
  return parsed;
}

export function loadBrandKit() {
  try {
    let raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      // Look for legacy v1 and migrate.
      const legacy = localStorage.getItem(LEGACY_KEY);
      if (legacy) {
        const migrated = migrateLegacy(JSON.parse(legacy));
        if (migrated?.styles) {
          localStorage.setItem(STORAGE_KEY, JSON.stringify(migrated));
          localStorage.removeItem(LEGACY_KEY);
          return migrated;
        }
      }
      return DEFAULT_BRAND_KIT;
    }
    const parsed = JSON.parse(raw);
    return {
      colors: parsed.colors?.length ? parsed.colors : DEFAULT_BRAND_KIT.colors,
      font:   { ...DEFAULT_BRAND_KIT.font, ...(parsed.font || {}) },
      previewText: parsed.previewText ?? DEFAULT_PREVIEW_TEXT,
      styles: {
        '9:16': { ...DEFAULT_BRAND_KIT.styles['9:16'], ...(parsed.styles?.['9:16'] || {}) },
        '16:9': { ...DEFAULT_BRAND_KIT.styles['16:9'], ...(parsed.styles?.['16:9'] || {}) },
      },
    };
  } catch {
    return DEFAULT_BRAND_KIT;
  }
}

export function saveBrandKit(kit) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(kit));
    window.dispatchEvent(new CustomEvent('brandKit:changed', { detail: kit }));
  } catch (e) {
    console.error('Failed to save brand kit:', e);
  }
}

export function resetBrandKit() {
  localStorage.removeItem(STORAGE_KEY);
  localStorage.removeItem(LEGACY_KEY);
  window.dispatchEvent(new CustomEvent('brandKit:changed', { detail: DEFAULT_BRAND_KIT }));
}

// React hook: returns the current brand kit and re-renders any time it's
// changed (in this tab or another). Use this in modals so they always see
// the latest settings, not a stale snapshot.
export function useBrandKit() {
  const [kit, setKit] = useState(() => loadBrandKit());
  useEffect(() => {
    const onChange = (e) => setKit(e.detail || loadBrandKit());
    const onStorage = (e) => {
      if (e.key === STORAGE_KEY) setKit(loadBrandKit());
    };
    window.addEventListener('brandKit:changed', onChange);
    window.addEventListener('storage', onStorage);
    return () => {
      window.removeEventListener('brandKit:changed', onChange);
      window.removeEventListener('storage', onStorage);
    };
  }, []);
  return kit;
}

// Helper: wrap a string into lines of N words.
export function wrapByWords(text, wordsPerLine) {
  const words = String(text).split(/\s+/).filter(Boolean);
  if (wordsPerLine <= 0) return [words.join(' ')];
  const lines = [];
  for (let i = 0; i < words.length; i += wordsPerLine) {
    lines.push(words.slice(i, i + wordsPerLine).join(' '));
  }
  return lines.length ? lines : [''];
}

// Register an @font-face for a bundled/user font so the browser can render it.
const _registeredFonts = new Set();
export function ensureFontLoaded(font) {
  if (!font?.url || font.source === 'system') return;
  if (_registeredFonts.has(font.family)) return;
  const fontFace = new FontFace(font.family, `url(${font.url})`);
  fontFace.load().then((loaded) => {
    document.fonts.add(loaded);
    _registeredFonts.add(font.family);
  }).catch((err) => {
    console.warn(`Failed to load font ${font.family}:`, err);
  });
}
