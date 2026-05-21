// AI Restyle preset store. Two dimensions (backgrounds + lightings),
// each a list of { id, label, prompt } records with one marked as default
// via `defaultBackgroundId` / `defaultLightingId`. Persisted to
// localStorage and broadcast via a custom event so any subscribed
// component re-renders.
//
// Mirrors the keysStore.js + brandKit.js pattern. Seeded with 5
// hand-tuned presets per dimension on first load.

import { useEffect, useState } from 'react';

const STORAGE_KEY = 'openshorts.aiRestyle.presets';
const EVENT = 'openshorts:ai-restyle-presets-changed';

const SEED = {
  backgrounds: [
    { id: 'studio-white',     label: 'Studio white',     prompt: 'clean white seamless backdrop, minimalist photo studio, no clutter, perfect color separation' },
    { id: 'sunlit-office',    label: 'Sunlit office',    prompt: 'bright modern office interior with floor-to-ceiling windows, soft natural light, plants, wooden desk' },
    { id: 'bahamas-beach',    label: 'Bahamas beach',    prompt: 'tropical beach with palm trees, turquoise ocean water in the distance, soft white sand' },
    { id: 'cyberpunk-neon',   label: 'Cyberpunk neon',   prompt: 'nighttime city street with vivid neon signs, pink-and-cyan color palette, light fog' },
    { id: 'cinematic-forest', label: 'Cinematic forest', prompt: 'deep forest with dappled sunlight through tall pine trees, mossy ground, atmospheric haze' },
  ],
  lightings: [
    { id: 'studio-softbox',   label: 'Studio softbox',   prompt: 'soft diffused studio softbox lighting from camera-left, gentle fill on the right, no harsh shadows' },
    { id: 'window-daylight',  label: 'Window daylight',  prompt: 'bright daylight pouring through large windows, soft fill on subject\'s face' },
    { id: 'golden-hour',      label: 'Golden hour',      prompt: 'warm golden-hour sun low and to the side, long shadows, amber and rose tones' },
    { id: 'cinematic-moody',  label: 'Cinematic moody',  prompt: 'low-key cinematic lighting with strong directional key, deep shadows, single soft fill' },
    { id: 'neon-nighttime',   label: 'Neon nighttime',   prompt: 'colored neon spill lighting (pink and cyan accents), low ambient, subject lit from multiple sides' },
  ],
  defaultBackgroundId: 'studio-white',
  defaultLightingId: 'studio-softbox',
};

function seedOnce() {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(SEED)); } catch { /* ignore */ }
  return SEED;
}

function read() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return seedOnce();
    const data = JSON.parse(raw);
    if (!data || !Array.isArray(data.backgrounds) || !Array.isArray(data.lightings)) return seedOnce();
    return data;
  } catch {
    return seedOnce();
  }
}

function write(next) {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(next)); } catch { /* ignore */ }
  window.dispatchEvent(new CustomEvent(EVENT, { detail: next }));
}

export function getPresets() { return read(); }

export function setDefault(dimension, id) {
  const cur = read();
  const key = dimension === 'background' ? 'defaultBackgroundId' : 'defaultLightingId';
  write({ ...cur, [key]: id });
}

export function upsertPreset(dimension, preset) {
  const cur = read();
  const list = dimension === 'background' ? cur.backgrounds : cur.lightings;
  const next = list.some((p) => p.id === preset.id)
    ? list.map((p) => (p.id === preset.id ? preset : p))
    : [...list, preset];
  const dimKey = dimension === 'background' ? 'backgrounds' : 'lightings';
  write({ ...cur, [dimKey]: next });
}

export function deletePreset(dimension, id) {
  const cur = read();
  const defaultKey = dimension === 'background' ? 'defaultBackgroundId' : 'defaultLightingId';
  if (cur[defaultKey] === id) return; // can't delete the default
  const list = dimension === 'background' ? cur.backgrounds : cur.lightings;
  const next = list.filter((p) => p.id !== id);
  const dimKey = dimension === 'background' ? 'backgrounds' : 'lightings';
  write({ ...cur, [dimKey]: next });
}

export function useAIRestylePresets() {
  const [state, setState] = useState(() => read());
  useEffect(() => {
    const onChange = (e) => setState(e.detail || read());
    const onStorage = (e) => { if (e.key === STORAGE_KEY) setState(read()); };
    window.addEventListener(EVENT, onChange);
    window.addEventListener('storage', onStorage);
    return () => {
      window.removeEventListener(EVENT, onChange);
      window.removeEventListener('storage', onStorage);
    };
  }, []);
  return state;
}
