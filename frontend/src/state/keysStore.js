// API key store. Holds Gemini, Upload-Post, ElevenLabs, fal.ai keys + the
// selected Upload-Post user profile. Mirrors the brandKit.js pattern:
// localStorage-backed, broadcasts via a custom event so any page subscribed
// to `useKeys()` re-renders on change.

import { useEffect, useState } from 'react';
import { encrypt, decrypt } from '../lib/crypto.js';
import { getApiUrl } from '../config.js';

const STORAGE = {
  gemini:      { key: 'gemini_key',         encrypted: false },
  uploadPost:  { key: 'uploadPostKey_v3',   encrypted: true  },
  elevenLabs:  { key: 'elevenLabsKey_v1',   encrypted: true  },
  fal:         { key: 'falKey_v1',          encrypted: true  },
  uploadUser:  { key: 'uploadUserId',       encrypted: false },
};

const EVENT = 'openshorts:keys-changed';

function readOne(spec) {
  const raw = localStorage.getItem(spec.key);
  if (!raw) return '';
  return spec.encrypted ? decrypt(raw) : raw;
}

export function loadKeys() {
  return {
    gemini:      readOne(STORAGE.gemini),
    uploadPost:  readOne(STORAGE.uploadPost),
    elevenLabs:  readOne(STORAGE.elevenLabs),
    fal:         readOne(STORAGE.fal),
    uploadUserId: readOne(STORAGE.uploadUser),
  };
}

export function setKey(name, value) {
  const spec = STORAGE[name];
  if (!spec) throw new Error(`Unknown key: ${name}`);
  if (!value) {
    localStorage.removeItem(spec.key);
  } else {
    localStorage.setItem(spec.key, spec.encrypted ? encrypt(value) : value);
  }
  window.dispatchEvent(new CustomEvent(EVENT, { detail: loadKeys() }));
}

export function setUploadUserId(value) {
  setKey('uploadUser', value);
}

export function useKeys() {
  const [keys, setKeys] = useState(() => loadKeys());
  useEffect(() => {
    const onChange = (e) => setKeys(e.detail || loadKeys());
    const onStorage = (e) => {
      if (Object.values(STORAGE).some(s => s.key === e.key)) setKeys(loadKeys());
    };
    window.addEventListener(EVENT, onChange);
    window.addEventListener('storage', onStorage);
    return () => {
      window.removeEventListener(EVENT, onChange);
      window.removeEventListener('storage', onStorage);
    };
  }, []);
  return keys;
}

// Fetch Upload-Post profiles for the current API key. Returns
// { profiles: [...] } or throws. Stored separately from keys because
// profile list is server-side state, not credential state.
export async function fetchUploadProfiles(uploadPostKey) {
  if (!uploadPostKey) throw new Error('No Upload-Post key');
  const res = await fetch(getApiUrl('/api/social/user'), {
    headers: { 'X-Upload-Post-Key': uploadPostKey },
  });
  if (!res.ok) throw new Error('Failed to fetch profiles');
  return res.json();
}
