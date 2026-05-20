// Step 3: Processing. Fires up to N parallel POST /api/process calls (the
// backend has no batch endpoint — see plan TODO #1). Each file gets its
// own progress row that mirrors the existing jobStore polling shape.
//
// User can play SnakeGame while waiting; Skip advances to Review with
// whatever clips have finished (failed jobs are skipped in Review).
//
// TODO(backend): plan TODO #1 — replace the per-file loop with POST
// /api/process/batch returning a list of job ids.

import { useEffect, useRef, useState } from 'react';
import { CheckCircle2, Loader2, XCircle } from 'lucide-react';
import { getApiUrl } from '../../../config';
import { useKeys } from '../../../state/keysStore.js';
import SnakeGame from '../../../components/ui/SnakeGame.jsx';

const POLL_MS = 2000;
const HISTORY_KEY = 'openshorts.shortForm.history';

async function startJob({ file, geminiKey, signal }) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('acknowledged', 'true');
  const res = await fetch(getApiUrl('/api/process'), {
    method: 'POST',
    headers: { 'X-Gemini-Key': geminiKey },
    body: formData,
    signal,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function fetchStatus(jobId, signal) {
  const res = await fetch(getApiUrl(`/api/status/${jobId}`), { signal });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function saveHistory(entry) {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    const list = raw ? JSON.parse(raw) : [];
    list.unshift(entry);
    localStorage.setItem(HISTORY_KEY, JSON.stringify(list.slice(0, 50)));
  } catch {/* ignore */}
}

export default function Processing({ wizard }) {
  const keys = useKeys();
  const files = wizard.data.files || [];
  const jobs = wizard.data.jobs || {};
  const startedRef = useRef(false);
  const historySavedRef = useRef(false);
  const [overallStatus, setOverallStatus] = useState('starting');

  // Kick off jobs once on first mount.
  useEffect(() => {
    if (startedRef.current) return;
    startedRef.current = true;
    if (!keys.gemini) {
      setOverallStatus('error');
      return;
    }
    setOverallStatus('running');
    files.forEach(async (f) => {
      if (jobs[f.id]?.jobId) return;
      // File object may not exist after a wizard rehydrate.
      if (!(f.file instanceof File)) {
        wizard.setData((prev) => ({
          ...prev,
          jobs: { ...prev.jobs, [f.id]: { jobId: null, status: 'error', logs: ['Source file lost — re-upload to retry.'], result: null } },
        }));
        return;
      }
      try {
        const { job_id } = await startJob({ file: f.file, geminiKey: keys.gemini });
        wizard.setData((prev) => ({
          ...prev,
          jobs: { ...prev.jobs, [f.id]: { jobId: job_id, status: 'processing', logs: [], result: null } },
        }));
      } catch (e) {
        wizard.setData((prev) => ({
          ...prev,
          jobs: { ...prev.jobs, [f.id]: { jobId: null, status: 'error', logs: [String(e.message || e)], result: null } },
        }));
      }
    });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Poll status for every still-running job. Cleanup aborts in-flight
  // fetches and a `cancelled` flag stops late responses from committing
  // after unmount or after the effect re-runs. The terminal-status guard
  // in the setData updater drops stale 'processing' responses that race
  // past a newer 'complete'/'error' response.
  useEffect(() => {
    const active = Object.entries(jobs).filter(([, j]) => j.jobId && j.status === 'processing');
    if (active.length === 0) return;
    let cancelled = false;
    const controller = new AbortController();
    const id = setInterval(async () => {
      if (cancelled) return;
      for (const [fileId, j] of active) {
        try {
          const data = await fetchStatus(j.jobId, controller.signal);
          if (cancelled) return;
          wizard.setData((prev) => {
            const cur = prev.jobs[fileId];
            if (cur?.status === 'complete' || cur?.status === 'error') return prev;
            return {
              ...prev,
              jobs: {
                ...prev.jobs,
                [fileId]: {
                  ...cur,
                  status: data.status || cur.status,
                  logs:   data.logs   || cur.logs,
                  result: data.results || cur.result,
                },
              },
            };
          });
        } catch (e) {
          if (e?.name === 'AbortError' || cancelled) return;
          wizard.setData((prev) => {
            const cur = prev.jobs[fileId];
            if (cur?.status === 'complete' || cur?.status === 'error') return prev;
            return {
              ...prev,
              jobs: {
                ...prev.jobs,
                [fileId]: { ...cur, status: 'error', logs: [...(cur?.logs || []), String(e.message || e)] },
              },
            };
          });
        }
      }
    }, POLL_MS);
    return () => {
      cancelled = true;
      controller.abort();
      clearInterval(id);
    };
  // Re-subscribe whenever the set of active job statuses changes.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [Object.values(jobs).map((j) => j.status).join(',')]);

  // Detect all-done + persist a history entry once.
  useEffect(() => {
    const entries = Object.values(jobs);
    if (entries.length < files.length) return;
    const done = entries.every((j) => j.status === 'complete' || j.status === 'error');
    if (!done) return;
    setOverallStatus('complete');
    if (historySavedRef.current) return;
    historySavedRef.current = true;
    saveHistory({
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
      ts: Date.now(),
      clipCount: entries.reduce((sum, j) => sum + (j.result?.clips?.length || 0), 0),
      jobs: entries.map((j) => j.jobId).filter(Boolean),
      title: `Batch of ${files.length} file${files.length === 1 ? '' : 's'}`,
    });
  }, [jobs, files.length]);

  const hasAnyComplete = Object.values(jobs).some((j) => j.status === 'complete');

  return (
    <div className="h-full overflow-y-auto custom-scrollbar">
      <div className="p-6 max-w-5xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-6">
        <section>
          <h1 className="text-[18px] font-semibold text-white">Processing</h1>
          <p className="text-[13px] text-zinc-500 mt-1 mb-4">
            Each file runs through the pipeline in parallel. Backend batch
            (plan TODO #1) will replace these per-file calls.
          </p>

          <div className="space-y-2">
            {files.map((f) => {
              const j = jobs[f.id];
              const status = j?.status || 'queued';
              const lastLog = j?.logs?.[j.logs.length - 1];
              return (
                <div key={f.id} className="rounded-lg border border-border bg-surface p-3">
                  <div className="flex items-center gap-3">
                    <StatusIcon status={status} />
                    <div className="flex-1 min-w-0">
                      <div className="text-[13px] text-white truncate">{f.name}</div>
                      <div className="text-[11px] text-zinc-500 truncate">
                        {status === 'queued'    ? 'Queued…' :
                         status === 'complete'  ? `Generated ${j.result?.clips?.length || 0} clip${(j.result?.clips?.length || 0) === 1 ? '' : 's'}` :
                         status === 'error'     ? (lastLog || 'Failed') :
                                                  (lastLog || 'Processing…')}
                      </div>
                    </div>
                    {j?.jobId && <span className="text-[10px] font-mono text-zinc-600 shrink-0">{j.jobId.slice(0, 8)}</span>}
                  </div>
                </div>
              );
            })}
          </div>

          <div className="flex items-center justify-between mt-6 pt-4 border-t border-border">
            <span className="text-[11px] text-zinc-500">
              {overallStatus === 'complete' ? 'All files done.' :
               overallStatus === 'error'    ? 'Missing Gemini key — set it in Settings.' :
                                              'You can wait, play Snake, or skip to whatever has finished.'}
            </span>
            <div className="flex items-center gap-3">
              <button
                onClick={() => wizard.goto(3)}
                disabled={!hasAnyComplete}
                className="text-[13px] text-zinc-400 hover:text-white transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
              >
                Skip →
              </button>
              <button
                onClick={() => wizard.goto(3)}
                disabled={overallStatus !== 'complete'}
                className="btn-primary px-5 py-2 text-[13px] disabled:opacity-40 disabled:cursor-not-allowed"
              >
                Review →
              </button>
            </div>
          </div>
        </section>

        <aside className="rounded-xl border border-border bg-surface p-5">
          <div className="mb-3">
            <h2 className="text-[14px] font-semibold text-white">Pass the time</h2>
            <p className="text-[12px] text-zinc-500 mt-0.5">Render times scale with clip length — keep your hands busy.</p>
          </div>
          <SnakeGame />
        </aside>
      </div>
    </div>
  );
}

function StatusIcon({ status }) {
  if (status === 'complete') return <CheckCircle2 size={16} className="text-success" />;
  if (status === 'error')    return <XCircle size={16} className="text-red-400" />;
  if (status === 'queued')   return <span className="w-2 h-2 rounded-full bg-zinc-700 inline-block mx-[3px]" />;
  return <Loader2 size={16} className="text-primary animate-spin" />;
}
