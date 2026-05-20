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
import { CheckCircle2, Loader2, XCircle, KeyRound } from 'lucide-react';
import { Link } from 'react-router-dom';
import { getApiUrl } from '../../../config';
import { useKeys } from '../../../state/keysStore.js';
import { useBrandKit } from '../../../lib/brandKit.js';
import SnakeGame from '../../../components/ui/SnakeGame.jsx';

const POLL_MS = 2000;
const HISTORY_KEY = 'openshorts.shortForm.history';

// Processing time ≈ source duration × 1.2 on CPU (transcription + per-frame
// reframing dominate). Refined as logs arrive. Treat as a hint, not a SLA.
const ETA_MULTIPLIER = 1.2;
const ETA_FLOOR_SEC = 25;        // small clips still pay fixed-cost setup
const ETA_FALLBACK_SEC = 60;     // when we couldn't probe duration

// Build the SubtitleStyle JSON the backend's SubtitleStyle pydantic model
// validates against (see backend/app/main.py). Brand-kit positions are 3x3
// grid strings (bottom-center, top-left, ...); the backend aliases those
// down to its top/middle/bottom enum.
function buildSubtitleStyle(brandKit) {
  const style = brandKit?.styles?.['9:16'] || {};
  return {
    position:       style.position     ?? 'bottom-center',
    font_size:      style.size         ?? 16,
    font_name:      brandKit?.font?.family ?? 'Verdana',
    font_color:     style.textColor    ?? '#FFFFFF',
    border_color:   style.strokeColor  ?? '#000000',
    border_width:   style.strokeWidth  ?? 2,
    words_per_line: style.wordsPerLine ?? null,
    text_case:      style.textCase     ?? null,
  };
}

async function startJob({ file, geminiKey, category, settings, subtitleStyle, signal }) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('acknowledged', 'true');
  if (category) formData.append('category', category);
  formData.append('auto_edit',       settings.autoEdit       ? 'true' : 'false');
  formData.append('auto_subtitles',  settings.autoSubtitles  ? 'true' : 'false');
  formData.append('color_grade',     settings.colorGrade     ? 'true' : 'false');
  formData.append('silence_removal', settings.silenceRemoval ? 'true' : 'false');
  formData.append('subtitle_style', JSON.stringify(subtitleStyle));
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

// Backend vocab is `queued | processing | completed | failed` and the
// payload uses `result` (singular). The wizard's done check + StatusIcon
// expect `complete | error` and a `result` key. Normalize once at the
// boundary so the rest of the component speaks the wizard's vocab.
function normalizeJobPayload(data) {
  const status =
    data.status === 'completed' ? 'complete' :
    data.status === 'failed'    ? 'error'    :
    data.status;
  return { status, logs: data.logs, result: data.result };
}

function saveHistory(entry) {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    const list = raw ? JSON.parse(raw) : [];
    list.unshift(entry);
    localStorage.setItem(HISTORY_KEY, JSON.stringify(list.slice(0, 50)));
  } catch {/* ignore */}
}

function fmtRemaining(secs) {
  if (secs == null || !Number.isFinite(secs)) return '';
  const s = Math.max(1, Math.round(secs));
  if (s < 60) return `~${s}s left`;
  const m = Math.floor(s / 60), rem = s % 60;
  return rem > 0 ? `~${m}m ${rem}s left` : `~${m}m left`;
}

function estimatedTotalSec(file) {
  if (file?.durationSec && Number.isFinite(file.durationSec)) {
    return Math.max(ETA_FLOOR_SEC, file.durationSec * ETA_MULTIPLIER);
  }
  return ETA_FALLBACK_SEC;
}

export default function Processing({ wizard }) {
  const keys = useKeys();
  const brandKit = useBrandKit();
  const files = wizard.data.files || [];
  const settings = wizard.data.settings || {};
  const jobs = wizard.data.jobs || {};
  const subtitleStyle = buildSubtitleStyle(brandKit);
  const historySavedRef = useRef(false);
  // Track which files we're already submitting so a re-run of the init
  // effect (e.g. when keys.gemini arrives later) doesn't double-fire
  // startJob. Lives in a ref because we don't want re-renders for it.
  const submittingRef = useRef(new Set());
  // Local "uploading" state so the row says "Uploading…" between the user
  // hitting Start processing and the POST returning a job_id.
  const [uploadingIds, setUploadingIds] = useState(new Set());
  // Tick once a second so the ETA countdown updates smoothly between polls.
  const [, setNowTick] = useState(0);

  // Kick off jobs whenever the Gemini key is present. Re-runs when the
  // key arrives later — the inside-loop guards (`submittingRef`, existing
  // jobId) keep it idempotent. No startedRef gate, so users who land here
  // before setting a key can recover by setting it from another tab/page.
  useEffect(() => {
    if (!keys.gemini) return;
    files.forEach(async (f) => {
      if (jobs[f.id]?.jobId) return;
      if (submittingRef.current.has(f.id)) return;
      if (jobs[f.id]?.status === 'error') return;
      if (!(f.file instanceof File)) {
        wizard.setData((prev) => ({
          ...prev,
          jobs: { ...prev.jobs, [f.id]: { jobId: null, status: 'error', logs: ['Source file lost — re-upload to retry.'], result: null, startedAt: null } },
        }));
        return;
      }
      submittingRef.current.add(f.id);
      setUploadingIds((s) => new Set(s).add(f.id));
      try {
        const { job_id } = await startJob({
          file: f.file,
          geminiKey: keys.gemini,
          category: f.category,
          settings,
          subtitleStyle,
        });
        wizard.setData((prev) => ({
          ...prev,
          jobs: { ...prev.jobs, [f.id]: { jobId: job_id, status: 'processing', logs: [], result: null, startedAt: Date.now() } },
        }));
      } catch (e) {
        wizard.setData((prev) => ({
          ...prev,
          jobs: { ...prev.jobs, [f.id]: { jobId: null, status: 'error', logs: [String(e.message || e)], result: null, startedAt: null } },
        }));
      } finally {
        submittingRef.current.delete(f.id);
        setUploadingIds((s) => { const n = new Set(s); n.delete(f.id); return n; });
      }
    });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [keys.gemini]);

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
          const norm = normalizeJobPayload(data);
          wizard.setData((prev) => {
            const cur = prev.jobs[fileId];
            if (cur?.status === 'complete' || cur?.status === 'error') return prev;
            return {
              ...prev,
              jobs: {
                ...prev.jobs,
                [fileId]: {
                  ...cur,
                  status: norm.status || cur.status,
                  logs:   norm.logs   || cur.logs,
                  result: norm.result || cur.result,
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

  // Tick the ETA countdown every second while any job is in flight.
  useEffect(() => {
    const anyInFlight = uploadingIds.size > 0 || Object.values(jobs).some((j) => j.status === 'processing');
    if (!anyInFlight) return;
    const id = setInterval(() => setNowTick((n) => n + 1), 1000);
    return () => clearInterval(id);
  }, [uploadingIds, jobs]);

  // Detect all-done + persist a history entry once.
  useEffect(() => {
    const entries = Object.values(jobs);
    if (entries.length < files.length) return;
    const done = entries.every((j) => j.status === 'complete' || j.status === 'error');
    if (!done) return;
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

  // Derived footer state. Reactive to keys.gemini and per-file outcomes,
  // not a sticky setOverallStatus from the init effect.
  const entries = Object.values(jobs);
  const allTerminal = files.length > 0
    && entries.length >= files.length
    && entries.every((j) => j.status === 'complete' || j.status === 'error');
  const anyTerminal = entries.some((j) => j.status === 'complete' || j.status === 'error');
  const hasAnyComplete = entries.some((j) => j.status === 'complete');

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
              const isUploading = uploadingIds.has(f.id);
              const status = isUploading
                ? 'uploading'
                : (j?.status || (keys.gemini ? 'queued' : 'awaiting_key'));
              const lastLog = j?.logs?.[j.logs.length - 1];

              // Status caption: collapse the multiple wait states into
              // specific user-readable text.
              const caption =
                status === 'awaiting_key'   ? 'Waiting for Gemini key…' :
                status === 'uploading'      ? 'Uploading to backend…' :
                status === 'queued'         ? 'Queued for the next slot…' :
                status === 'processing'     ? (lastLog || 'Starting pipeline…') :
                status === 'complete'       ? `Generated ${j.result?.clips?.length || 0} clip${(j.result?.clips?.length || 0) === 1 ? '' : 's'}` :
                status === 'error'          ? (lastLog || 'Failed') :
                                              'Working…';

              // ETA: only meaningful while the job is actually running on
              // the backend (we know startedAt). Before submission we just
              // show the estimated total.
              const total = estimatedTotalSec(f);
              let remaining = null;
              if (status === 'processing' && j?.startedAt) {
                const elapsed = (Date.now() - j.startedAt) / 1000;
                remaining = total - elapsed;
              } else if (status === 'uploading' || status === 'queued') {
                remaining = total;
              }
              const showEta = remaining != null && remaining > 0
                && status !== 'complete' && status !== 'error';
              const overdue = status === 'processing' && j?.startedAt
                && (Date.now() - j.startedAt) / 1000 > total + 10;

              return (
                <div key={f.id} className="rounded-lg border border-border bg-surface p-3">
                  <div className="flex items-center gap-3">
                    <StatusIcon status={status} />
                    <div className="flex-1 min-w-0">
                      <div className="text-[13px] text-white truncate">{f.name}</div>
                      <div className="text-[11px] text-zinc-500 truncate">{caption}</div>
                    </div>
                    <div className="text-[10px] font-mono text-zinc-500 shrink-0 text-right tabular-nums">
                      {status === 'complete' && <span className="text-success">done</span>}
                      {status === 'error'    && <span className="text-red-400">failed</span>}
                      {showEta && !overdue && <span>{fmtRemaining(remaining)}</span>}
                      {overdue && <span>taking longer than expected…</span>}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          <div className="flex items-center justify-between mt-6 pt-4 border-t border-border">
            {!keys.gemini ? (
              <span className="text-[11px] text-amber-300 flex items-center gap-2">
                <KeyRound size={12} />
                No Gemini key set —{' '}
                <Link to="/settings/system/api-keys" className="underline hover:text-white">
                  add one to start
                </Link>.
              </span>
            ) : allTerminal ? (
              <span className="text-[11px] text-zinc-500">All files done.</span>
            ) : (
              <span className="text-[11px] text-zinc-500">
                You can wait, play Snake, or skip to whatever has finished.
              </span>
            )}
            <div className="flex items-center gap-3">
              <button
                onClick={() => wizard.goto(3)}
                disabled={!hasAnyComplete && !allTerminal}
                className="text-[13px] text-zinc-400 hover:text-white transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
              >
                Skip →
              </button>
              <button
                onClick={() => wizard.goto(3)}
                disabled={!anyTerminal}
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
  if (status === 'complete')     return <CheckCircle2 size={16} className="text-success" />;
  if (status === 'error')        return <XCircle size={16} className="text-red-400" />;
  if (status === 'awaiting_key') return <KeyRound size={14} className="text-amber-400" />;
  if (status === 'queued')       return <span className="w-2 h-2 rounded-full bg-zinc-700 inline-block mx-[3px]" />;
  // uploading / processing / anything else in-flight
  return <Loader2 size={16} className="text-primary animate-spin" />;
}
