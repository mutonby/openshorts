import React, { useEffect, useRef, useState } from 'react';
import { Check, Type, Upload, X } from 'lucide-react';
import { ensureFontLoaded } from '../lib/brandKit';

export default function FontPicker({ value, onChange }) {
  const [fonts, setFonts] = useState([]);
  const [open, setOpen] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState(null);
  const inputRef = useRef(null);
  const listRef = useRef(null);

  const refresh = async () => {
    try {
      const r = await fetch('/api/fonts');
      const data = await r.json();
      setFonts(data.fonts || []);
      // Pre-register bundled + user fonts so the list previews render correctly.
      (data.fonts || []).forEach(ensureFontLoaded);
    } catch (e) {
      console.warn('Failed to load fonts:', e);
    }
  };

  useEffect(() => { refresh(); }, []);

  // Close on outside click
  useEffect(() => {
    const handler = (e) => {
      if (listRef.current && !listRef.current.contains(e.target)) setOpen(false);
    };
    if (open) document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [open]);

  const handleUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setUploadError(null);
    try {
      const form = new FormData();
      form.append('file', file);
      const r = await fetch('/api/fonts/upload', { method: 'POST', body: form });
      if (!r.ok) {
        const err = await r.json().catch(() => ({}));
        throw new Error(err.detail || `Upload failed (${r.status})`);
      }
      const uploaded = await r.json();
      await refresh();
      onChange(uploaded);
    } catch (err) {
      setUploadError(err.message);
    } finally {
      setUploading(false);
      e.target.value = '';
    }
  };

  const selectedName = value?.family || 'Inter';

  return (
    <div ref={listRef} className="relative">
      <button
        type="button"
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between px-4 py-3 bg-white/5 hover:bg-white/10 border border-white/10 rounded-xl text-left transition-colors"
      >
        <span className="flex items-center gap-3 min-w-0">
          <Type size={16} className="text-zinc-400 shrink-0" />
          <span
            className="truncate text-base"
            style={{ fontFamily: `"${selectedName}", system-ui` }}
          >
            {selectedName}
          </span>
          {value?.source && value.source !== 'system' && (
            <span className="text-[10px] uppercase tracking-wide text-zinc-500 shrink-0">
              {value.source}
            </span>
          )}
        </span>
        <svg width="12" height="12" viewBox="0 0 12 12" className={`text-zinc-400 transition-transform ${open ? 'rotate-180' : ''}`}>
          <path d="M2 4l4 4 4-4" stroke="currentColor" strokeWidth="1.5" fill="none" />
        </svg>
      </button>

      {open && (
        <div className="absolute z-20 mt-2 w-full bg-surface border border-white/10 rounded-xl shadow-2xl overflow-hidden">
          <div className="max-h-72 overflow-y-auto custom-scrollbar">
            {fonts.map((f) => (
              <button
                type="button"
                key={`${f.source}-${f.name}`}
                onClick={() => { onChange(f); setOpen(false); }}
                className={`w-full flex items-center gap-3 px-4 py-3 hover:bg-white/5 text-left transition-colors border-b border-white/5 last:border-b-0 ${value?.family === f.name ? 'bg-primary/10' : ''}`}
              >
                <span
                  className="flex-1 truncate text-base"
                  style={{ fontFamily: `"${f.name}", system-ui` }}
                >
                  {f.name}
                </span>
                <span className="text-[10px] uppercase tracking-wide text-zinc-500">{f.source}</span>
                {value?.family === f.name && <Check size={14} className="text-primary" />}
              </button>
            ))}
          </div>

          <label className="block border-t border-white/10 px-4 py-3 hover:bg-white/5 cursor-pointer text-sm text-zinc-300">
            <span className="flex items-center gap-2">
              {uploading ? (
                <>
                  <div className="w-4 h-4 border-2 border-primary border-t-transparent rounded-full animate-spin" />
                  Uploading…
                </>
              ) : (
                <>
                  <Upload size={14} />
                  Upload .ttf / .otf / .woff
                </>
              )}
            </span>
            <input
              ref={inputRef}
              type="file"
              accept=".ttf,.otf,.woff,.woff2,font/ttf,font/otf,font/woff,font/woff2"
              onChange={handleUpload}
              className="hidden"
              disabled={uploading}
            />
            {uploadError && (
              <span className="text-xs text-red-400 mt-1 flex items-center gap-1">
                <X size={12} /> {uploadError}
              </span>
            )}
          </label>
        </div>
      )}
    </div>
  );
}
