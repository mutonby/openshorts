// AI Restyle Upload step. Single MP4/MOV file, <=30s. Probes duration on
// the client (HTMLVideoElement) before letting the user advance.

import { useRef, useState } from 'react';
import { Upload as UploadIcon, X } from 'lucide-react';

const MAX_SEC = 30;
const ACCEPT = 'video/mp4,video/quicktime,.mp4,.mov';

async function probeDuration(file) {
  return new Promise((resolve) => {
    const url = URL.createObjectURL(file);
    const v = document.createElement('video');
    v.preload = 'metadata';
    v.onloadedmetadata = () => { URL.revokeObjectURL(url); resolve(v.duration); };
    v.onerror = () => { URL.revokeObjectURL(url); resolve(null); };
    v.src = url;
  });
}

export default function Upload({ wizard }) {
  const inputRef = useRef(null);
  const [error, setError] = useState(null);
  const data = wizard.data.file;

  async function onChange(e) {
    const f = e.target.files?.[0];
    if (!f) return;
    setError(null);

    const ext = f.name.toLowerCase().match(/\.(mp4|mov)$/);
    if (!ext) { setError('File must be MP4 or MOV.'); return; }

    const dur = await probeDuration(f);
    if (dur == null) { setError('Could not read video duration.'); return; }
    if (dur > MAX_SEC) {
      setError(`AI Restyle v1 caps at 30s. Your file is ${dur.toFixed(1)}s. Trim it first or use Short-form.`);
      return;
    }

    wizard.setData({
      file: {
        id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
        name: f.name,
        size: f.size,
        durationSec: dur,
        file: f,
      },
    });
  }

  function clearFile() {
    wizard.setData({ file: null });
    if (inputRef.current) inputRef.current.value = '';
  }

  return (
    <div className="h-full overflow-y-auto custom-scrollbar p-8">
      <div className="max-w-2xl mx-auto">
        <h1 className="text-[24px] font-semibold mb-2">Upload a video</h1>
        <p className="text-[13px] text-zinc-400 mb-6">
          MP4 or MOV, up to 30 seconds. We'll relight the lighting and replace
          the background while keeping your motion and original audio.
        </p>

        {!data ? (
          <button
            onClick={() => inputRef.current?.click()}
            className="w-full border-2 border-dashed border-border rounded-lg p-12 flex flex-col items-center gap-3 hover:bg-white/5 transition-colors"
          >
            <UploadIcon size={24} className="text-zinc-500" />
            <div className="text-[13px] text-zinc-300">Drop a video here or click to browse</div>
            <div className="text-[11px] text-zinc-500">MP4 / MOV · ≤30 seconds · ≤2 GB</div>
          </button>
        ) : (
          <div className="rounded-lg border border-border bg-surface p-4 flex items-center justify-between">
            <div className="min-w-0">
              <div className="text-[13px] text-white font-medium truncate">{data.name}</div>
              <div className="text-[11px] text-zinc-500 mt-0.5">
                {(data.size / 1024 / 1024).toFixed(1)} MB · {data.durationSec.toFixed(1)}s
              </div>
            </div>
            <button onClick={clearFile} className="p-1.5 hover:bg-white/10 rounded text-zinc-400 shrink-0" aria-label="Remove">
              <X size={14} />
            </button>
          </div>
        )}

        <input ref={inputRef} type="file" accept={ACCEPT} onChange={onChange} className="hidden" />

        {error && (
          <div className="mt-3 text-[12px] text-red-400" role="alert">{error}</div>
        )}

        <div className="mt-6 flex justify-end">
          <button
            onClick={wizard.next}
            disabled={!data}
            className="px-4 py-2 text-[13px] bg-primary text-white rounded-md hover:bg-primary/90 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            Continue →
          </button>
        </div>
      </div>
    </div>
  );
}
