// Step 1: Upload one long-form source file. MP4/MOV up to 4K (cap to 8 GB).

import { useRef, useState } from 'react';
import { FileVideo, UploadCloud, X } from 'lucide-react';

const MAX_SIZE_BYTES = 8 * 1024 * 1024 * 1024;
const ALLOWED_TYPES = ['video/mp4', 'video/quicktime'];

function fmtSize(bytes) {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function nextId() { return `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`; }

export default function Upload({ wizard }) {
  const inputRef = useRef(null);
  const [dragOver, setDragOver] = useState(false);
  const [error, setError] = useState('');

  const file = wizard.data.file;

  function accept(f) {
    setError('');
    const okType = ALLOWED_TYPES.includes(f.type) || /\.(mp4|mov)$/i.test(f.name);
    if (!okType) { setError(`${f.name}: only MP4 / MOV files.`); return; }
    if (f.size > MAX_SIZE_BYTES) { setError(`${f.name}: over 8 GB.`); return; }
    wizard.setData({ file: { id: nextId(), file: f, name: f.name, size: f.size } });
  }

  return (
    <div className="h-full overflow-y-auto custom-scrollbar">
      <div className="p-6 max-w-3xl mx-auto space-y-6">
        <header>
          <h1 className="text-[18px] font-semibold text-white">Upload long-form video</h1>
          <p className="text-[13px] text-zinc-500 mt-1">
            One source file. MP4 or MOV at up to 4K — the editor handles chapter detection, segment exports, and intro/outro insertion.
          </p>
        </header>

        <div
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragOver(false);
            const f = e.dataTransfer.files?.[0];
            if (f) accept(f);
          }}
          onClick={() => inputRef.current?.click()}
          className={`rounded-xl border-2 border-dashed p-10 text-center cursor-pointer transition-colors ${
            dragOver ? 'border-primary bg-primary/10' : 'border-border bg-surface hover:bg-white/5'
          }`}
        >
          <UploadCloud size={36} className={`mx-auto mb-3 ${dragOver ? 'text-primary' : 'text-zinc-500'}`} />
          <div className="text-[14px] text-white font-medium">
            Drop a video here or click to browse
          </div>
          <div className="text-[11px] text-zinc-500 mt-1">
            MP4 / MOV · up to 8 GB · 4K supported
          </div>
          <input
            ref={inputRef}
            type="file"
            accept="video/mp4,video/quicktime,.mp4,.mov"
            className="hidden"
            onChange={(e) => { const f = e.target.files?.[0]; if (f) accept(f); }}
          />
        </div>

        {error && <div className="text-[12px] text-red-400">{error}</div>}

        {file && (
          <div className="flex items-center gap-3 rounded-lg border border-border bg-surface p-3">
            <FileVideo size={18} className="text-zinc-500 shrink-0" />
            <div className="flex-1 min-w-0">
              <div className="text-[13px] text-white truncate">{file.name}</div>
              <div className="text-[11px] text-zinc-500">{fmtSize(file.size)}</div>
            </div>
            <button
              onClick={() => wizard.setData({ file: null })}
              className="p-1.5 text-zinc-500 hover:text-red-400 transition-colors"
              aria-label={`Remove ${file.name}`}
            >
              <X size={14} />
            </button>
          </div>
        )}

        <div className="flex items-center justify-between pt-4 border-t border-border">
          <span className="text-[11px] text-zinc-500">
            {!file ? 'Add a video to continue.' : 'Ready for settings.'}
          </span>
          <button
            onClick={wizard.next}
            disabled={!file}
            className="btn-primary px-5 py-2 text-[13px] disabled:opacity-40 disabled:cursor-not-allowed"
          >
            Continue
          </button>
        </div>
      </div>
    </div>
  );
}
