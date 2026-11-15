import React, { useState, useEffect, useCallback } from 'react';
import { Loader2, ChevronUp, Folder, HardDrive, Check } from 'lucide-react';
import { apiJson } from '../lib/api';

const errText = (e, fallback) => (typeof e?.detail === 'string' && e.detail) || fallback;

export default function FolderBrowser({ initial, onPick, onClose }) {
  const [listing, setListing] = useState(null);
  const [error, setError] = useState('');
  const [first] = useState(initial);

  const open = useCallback(async (path) => {
    setError('');
    try {
      const data = await apiJson(`/api/live/browse?path=${encodeURIComponent(path || '')}`);
      setListing(data);
    } catch (e) {
      setError(errText(e, 'Cannot open this folder.'));
      if (path) {
        const data = await apiJson('/api/live/browse?path=').catch(() => null);
        if (data) setListing(data);
      }
    }
  }, []);

  useEffect(() => { open(first); }, [open, first]);

  const join = (base, name) => (base.endsWith('\\') || base.endsWith('/') ? `${base}${name}` : `${base}${base.includes('\\') ? '\\' : '/'}${name}`);

  return (
    <div className="mt-2 rounded-input border border-rule p-3 text-left animate-fade">
      <div className="flex items-center justify-between gap-2 mb-2">
        <span className="text-xs text-ink2 truncate">{listing?.path || 'this computer'}</span>
        <div className="flex gap-1.5 shrink-0">
          {listing?.path && (
            <button type="button" className="btn-quiet text-xs" onClick={() => open(listing.parent || '')}>
              <ChevronUp size={12} /> up
            </button>
          )}
          {listing?.path && (
            <button type="button" className="btn-primary py-1 px-3 text-xs" onClick={() => onPick(listing.path)}>
              <Check size={12} /> use this folder
            </button>
          )}
          <button type="button" className="btn-quiet text-xs" onClick={onClose}>close</button>
        </div>
      </div>
      {error && <p className="text-danger text-xs mb-2">{error}</p>}
      <div className="max-h-56 overflow-y-auto custom-scrollbar space-y-0.5">
        {!listing && <Loader2 size={14} className="animate-spin text-muted" />}
        {listing && listing.dirs.length === 0 && <p className="text-xs text-muted">no subfolders</p>}
        {listing?.dirs.map((d) => (
          <button
            key={d}
            type="button"
            onClick={() => open(listing.path ? join(listing.path, d) : d)}
            className="w-full flex items-center gap-2 text-left text-sm text-ink2 hover:text-brass px-1.5 py-1 rounded"
          >
            {listing.path ? <Folder size={13} className="shrink-0" /> : <HardDrive size={13} className="shrink-0" />}
            <span className="truncate">{d}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
