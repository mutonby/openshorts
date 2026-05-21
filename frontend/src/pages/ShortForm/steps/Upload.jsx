// Step 1: Upload. Drag-drop + browse, up to 5 files, MP4/MOV <= 2 GB.
// Each entry: { id, file (File), name, size, durationSec? }

import { useRef, useState } from 'react';
import { FileVideo, UploadCloud, X } from 'lucide-react';

const MAX_FILES = 5;
const MAX_SIZE_BYTES = 2 * 1024 * 1024 * 1024;
const ALLOWED_TYPES = ['video/mp4', 'video/quicktime'];

function nextId() { return `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`; }

function fmtSize(bytes) {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function fmtDuration(secs) {
  if (secs < 60) return `${Math.round(secs)}s`;
  const m = Math.floor(secs / 60), s = Math.round(secs % 60);
  return s > 0 ? `${m}m ${s}s` : `${m}m`;
}

// Probe video duration via a hidden HTMLVideoElement. Used by Step 3
// (Processing) to estimate ETA. Returns null if the metadata can't be
// read (rare — non-MP4 fakes, corrupt files).
function probeDurationSec(file) {
  return new Promise((resolve) => {
    const url = URL.createObjectURL(file);
    const video = document.createElement('video');
    video.preload = 'metadata';
    const cleanup = () => { URL.revokeObjectURL(url); };
    video.onloadedmetadata = () => {
      const d = Number.isFinite(video.duration) ? video.duration : null;
      cleanup();
      resolve(d);
    };
    video.onerror = () => { cleanup(); resolve(null); };
    video.src = url;
  });
}

export default function Upload({ wizard }) {
  const inputRef = useRef(null);
  const [dragOver, setDragOver] = useState(false);
  const [error, setError] = useState('');

  const files = wizard.data.files || [];

  function addFiles(list) {
    setError('');
    const incoming = Array.from(list);
    const accepted = [];
    for (const f of incoming) {
      if (files.length + accepted.length >= MAX_FILES) {
        setError(`Up to ${MAX_FILES} files per batch.`);
        break;
      }
      const okType = ALLOWED_TYPES.includes(f.type) || /\.(mp4|mov)$/i.test(f.name);
      if (!okType) { setError(`${f.name}: only MP4 / MOV files.`); continue; }
      if (f.size > MAX_SIZE_BYTES) { setError(`${f.name}: over 2 GB.`); continue; }
      accepted.push({ id: nextId(), file: f, name: f.name, size: f.size, durationSec: null });
    }
    if (accepted.length) {
      wizard.setData({ files: [...files, ...accepted] });
      // Probe durations asynchronously — Processing uses them to estimate ETA.
      accepted.forEach(async (entry) => {
        const d = await probeDurationSec(entry.file);
        if (d == null) return;
        wizard.setData((prev) => ({
          ...prev,
          files: (prev.files || []).map((p) => p.id === entry.id ? { ...p, durationSec: d } : p),
        }));
      });
    }
  }

  function removeFile(id) {
    wizard.setData({ files: files.filter((f) => f.id !== id) });
  }

  return (
    <div className="h-full overflow-y-auto custom-scrollbar">
      <div className="p-6 max-w-3xl mx-auto space-y-6">
        <header>
          <h1 className="text-[18px] font-semibold text-white">Upload videos</h1>
          <p className="text-[13px] text-zinc-500 mt-1">
            Drop up to {MAX_FILES} source videos. MP4 or MOV, up to 2 GB each.
          </p>
        </header>

        <div
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragOver(false);
            if (e.dataTransfer.files?.length) addFiles(e.dataTransfer.files);
          }}
          onClick={() => inputRef.current?.click()}
          className={`rounded-xl border-2 border-dashed p-10 text-center cursor-pointer transition-colors ${
            dragOver ? 'border-primary bg-primary/10' : 'border-border bg-surface hover:bg-white/5'
          }`}
        >
          <UploadCloud size={36} className={`mx-auto mb-3 ${dragOver ? 'text-primary' : 'text-zinc-500'}`} />
          <div className="text-[14px] text-white font-medium">
            Drop videos here or click to browse
          </div>
          <div className="text-[11px] text-zinc-500 mt-1">
            MP4 / MOV · up to 2 GB · up to {MAX_FILES} per batch
          </div>
          <input
            ref={inputRef}
            type="file"
            accept="video/mp4,video/quicktime,.mp4,.mov"
            multiple
            className="hidden"
            onChange={(e) => addFiles(e.target.files || [])}
          />
        </div>

        {error && <div className="text-[12px] text-red-400">{error}</div>}

        {files.length > 0 && (
          <div className="space-y-2">
            <div className="text-[11px] uppercase tracking-wider text-zinc-500">
              {files.length} of {MAX_FILES} files
            </div>
            {files.map((f) => (
              <div key={f.id} className="flex items-center gap-3 rounded-lg border border-border bg-surface p-3">
                <FileVideo size={18} className="text-zinc-500 shrink-0" />
                <div className="flex-1 min-w-0">
                  <div className="text-[13px] text-white truncate">{f.name}</div>
                  <div className="text-[11px] text-zinc-500">
                    {fmtSize(f.size)}
                    {f.durationSec != null && ` · ${fmtDuration(f.durationSec)}`}
                  </div>
                </div>
                <button
                  onClick={() => removeFile(f.id)}
                  className="p-1.5 text-zinc-500 hover:text-red-400 transition-colors"
                  aria-label={`Remove ${f.name}`}
                >
                  <X size={14} />
                </button>
              </div>
            ))}
          </div>
        )}

        <div className="flex items-center justify-between pt-4 border-t border-border">
          <span className="text-[11px] text-zinc-500">
            {files.length === 0 ? 'Add at least one file to continue.' : 'Ready for categorization.'}
          </span>
          <button
            onClick={wizard.next}
            disabled={files.length === 0}
            className="btn-primary px-5 py-2 text-[13px] disabled:opacity-40 disabled:cursor-not-allowed"
          >
            Continue
          </button>
        </div>
      </div>
    </div>
  );
}
