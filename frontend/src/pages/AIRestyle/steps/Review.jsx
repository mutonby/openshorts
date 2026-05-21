// AI Restyle Review step. Polls GET /api/restyle/{job_id} every 2s until
// the job is terminal (completed/failed), then shows a Before/After phone
// preview with Download + Send-to-Short-form CTAs.

import { useEffect, useState } from 'react';
import { Download, Eye } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import PhoneFrame from '../../../components/ui/PhoneFrame.jsx';
import { getApiUrl } from '../../../config.js';

const HANDOFF_KEY = 'openshorts.shortForm.handoff';

export default function Review({ wizard }) {
  const job = wizard.data.job;
  const file = wizard.data.file;
  const [showOriginal, setShowOriginal] = useState(false);
  const [sourceUrl, setSourceUrl] = useState(null);
  const navigate = useNavigate();

  // Blob URL for the Before view.
  useEffect(() => {
    if (!file?.file) { setSourceUrl(null); return; }
    const u = URL.createObjectURL(file.file);
    setSourceUrl(u);
    return () => URL.revokeObjectURL(u);
  }, [file?.file]);

  // Poll status until terminal.
  useEffect(() => {
    if (!job?.jobId) return;
    if (job.status === 'completed' || job.status === 'failed') return;
    let alive = true;
    const tick = async () => {
      try {
        const res = await fetch(getApiUrl(`/api/restyle/${job.jobId}`));
        if (!res.ok) return;
        const data = await res.json();
        if (!alive) return;
        wizard.setData({ job: { ...job, ...data } });
      } catch { /* transient — retry next tick */ }
    };
    const i = setInterval(tick, 2000);
    tick();
    return () => { alive = false; clearInterval(i); };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [job?.jobId, job?.status]);

  const restyledUrl = job?.result?.video_url ? getApiUrl(job.result.video_url) : null;
  const status = job?.status || 'idle';

  function sendToShortForm() {
    if (!restyledUrl) return;
    sessionStorage.setItem(
      HANDOFF_KEY,
      JSON.stringify({ url: restyledUrl, name: `restyled-${file?.name || 'video.mp4'}` }),
    );
    navigate('/short-form');
  }

  if (status === 'processing' || status === 'idle') {
    return (
      <div className="h-full flex items-center justify-center p-12">
        <div className="max-w-md w-full">
          <div className="text-[14px] text-white font-medium mb-3">Restyling…</div>
          <div className="h-1.5 bg-white/10 rounded-full overflow-hidden">
            <div
              className="h-full bg-primary transition-all"
              style={{ width: `${Math.max(5, job?.progress_pct || 5)}%` }}
            />
          </div>
          <div className="mt-4 text-[11px] text-zinc-500 font-mono leading-relaxed max-h-40 overflow-y-auto custom-scrollbar">
            {(job?.logs || []).slice(-8).map((l, i) => <div key={i}>{l}</div>)}
          </div>
        </div>
      </div>
    );
  }

  if (status === 'failed') {
    return (
      <div className="h-full flex items-center justify-center p-12 text-center">
        <div className="max-w-md">
          <div className="text-[14px] text-red-400 font-medium mb-2">Restyle failed</div>
          <div className="text-[12px] text-zinc-500 font-mono whitespace-pre-line">
            {(job?.logs || []).slice(-6).join('\n')}
          </div>
          <div className="mt-4 flex gap-3 justify-center">
            <button onClick={wizard.back} className="px-3 py-1.5 text-[12px] border border-border rounded-md text-zinc-300 hover:bg-white/5 transition-colors">
              Try again
            </button>
            <button onClick={wizard.reset} className="px-3 py-1.5 text-[12px] bg-primary text-white rounded hover:bg-primary/90 transition-colors">
              Start over
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col p-8">
      <div className="flex-1 flex flex-col items-center gap-4">
        <div className="flex items-center gap-2 text-[12px]">
          <button
            onClick={() => setShowOriginal(false)}
            className={`px-3 py-1.5 rounded-md transition-colors ${!showOriginal ? 'bg-white/10 text-white' : 'text-zinc-400 hover:text-white'}`}
          >
            After
          </button>
          <button
            onClick={() => setShowOriginal(true)}
            disabled={!sourceUrl}
            className={`px-3 py-1.5 rounded-md transition-colors disabled:opacity-30 ${showOriginal ? 'bg-white/10 text-white' : 'text-zinc-400 hover:text-white'}`}
          >
            <Eye size={12} className="inline mr-1" /> Before
          </button>
        </div>

        <PhoneFrame size="md">
          {showOriginal && sourceUrl ? (
            <video key="src" src={sourceUrl} controls className="w-full h-full object-contain" />
          ) : restyledUrl ? (
            <video key="rst" src={restyledUrl} controls className="w-full h-full object-contain" />
          ) : (
            <div className="text-zinc-600 text-[12px] p-4 text-center">No preview available.</div>
          )}
        </PhoneFrame>
      </div>

      <div className="border-t border-border pt-4 mt-4 flex items-center gap-3">
        <a
          href={restyledUrl || '#'}
          download
          className={`px-3 py-2 text-[12px] flex items-center gap-2 bg-primary text-white rounded-md hover:bg-primary/90 transition-colors ${!restyledUrl ? 'opacity-40 pointer-events-none' : ''}`}
        >
          <Download size={12} /> Download
        </a>
        <button
          onClick={sendToShortForm}
          disabled={!restyledUrl}
          className="px-3 py-2 text-[12px] border border-primary/40 text-primary rounded-md hover:bg-primary/10 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          Send to Short-form →
        </button>
        <button
          onClick={wizard.reset}
          className="ml-auto px-3 py-2 text-[12px] text-zinc-400 hover:text-white transition-colors"
        >
          Start another
        </button>
      </div>
    </div>
  );
}
