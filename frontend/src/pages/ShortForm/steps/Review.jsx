// Step 4: Review. Split view — clip list (left) + phone preview + export bar.
//
// Phase 3: per-clip stage selector. The auto-pipeline emits a chain of variants
// (original → edited → graded → silencecut → subtitled) per clip; here we let
// the user step through them. Missing variants can be generated inline via the
// existing /api/edit, /api/colorgrade, /api/silencecut, /api/subtitle endpoints.
// Selection + LUT choice persist in wizard.data so reloads keep them.
//
// Export wiring:
//   - Download: opens the currently-displayed variant URL.
//   - Publish:  pushes a notification + would call POST /api/social/post.
//               Backend doesn't queue these yet (plan TODO #9), so we
//               surface the intent locally via the bell.
//   - Schedule: same path as Publish with status='scheduled'.
//   - Send to CapCut: placeholder — backend integration TODO.

import { useEffect, useMemo, useState } from 'react';
import { ArrowDown, ArrowUp, Combine, Download, Eye, Loader2, Plus, Scissors, X } from 'lucide-react';
import PhoneFrame from '../../../components/ui/PhoneFrame.jsx';
import PlatformBadge from '../../../components/ui/PlatformBadge.jsx';
import { getApiUrl } from '../../../config';
import { pushNotification } from '../../../state/notificationsStore.js';
import { useKeys } from '../../../state/keysStore.js';
import { useBrandKit } from '../../../lib/brandKit.js';

const PLATFORMS = ['youtube', 'tiktok', 'instagram', 'snapchat', 'facebook'];

// Chain order must match backend/app/main.py:_run_auto_pipeline.
const STAGES = [
  { key: 'original',   label: 'Original',     short: 'Original'    },
  { key: 'edited',     label: '+ AI Edit',    short: '+ Edit'      },
  { key: 'graded',     label: '+ Color Grade', short: '+ Grade'    },
  { key: 'silencecut', label: '+ Silence Cut', short: '+ Cut'      },
  { key: 'subtitled',  label: '+ Subtitles',  short: '+ Subs'      },
];

// Must match backend/app/editing/color_grade.py:LUT_PRESETS (allowlist enforced
// server-side via the ColorGradeRequest validator).
const LUTS = ['teal_orange', 'warm', 'cool', 'vivid', 'noir'];
const DEFAULT_LUT = 'teal_orange';

function clipKey(c) {
  return `${c.jobId}-${c.clipIndex}`;
}

function flattenClips(jobs, files) {
  const out = [];
  for (const f of files) {
    const j = jobs[f.id];
    if (!j?.result?.clips) continue;
    j.result.clips.forEach((clip, i) => {
      out.push({
        jobId: j.jobId,
        fileId: f.id,
        sourceName: f.name,
        sourceFile: f.file instanceof File ? f.file : null,
        clipIndex: i,
        clip,
      });
    });
  }
  return out;
}

// Pick the deepest stage whose variant exists, walking the chain backwards.
// Used as the initial display when the wizard first lands on Review.
function pickInitialStage(variants) {
  if (!variants) return 'original';
  for (let i = STAGES.length - 1; i >= 0; i--) {
    if (variants[STAGES[i].key]) return STAGES[i].key;
  }
  return 'original';
}

// Walk backwards from `targetStage` to find the most recent existing variant —
// that's the input to feed when generating the missing one. Falls back to original.
function priorVariantFilename(variants, targetStage) {
  const idx = STAGES.findIndex((s) => s.key === targetStage);
  for (let i = idx - 1; i >= 0; i--) {
    const f = variants?.[STAGES[i].key];
    if (f) return f;
  }
  return variants?.original || null;
}

export default function Review({ wizard }) {
  const files = wizard.data.files || [];
  const jobs = wizard.data.jobs || {};
  const clips = useMemo(() => flattenClips(jobs, files), [jobs, files]);
  const mergedClips = wizard.data.mergedClips || [];
  const [selected, setSelected] = useState(0);
  const [selectedMergedId, setSelectedMergedId] = useState(null);
  const [showOriginal, setShowOriginal] = useState(false);
  const [sourceUrl, setSourceUrl] = useState(null);

  // Per-clip transient state — loading flag + last error. Lost on reload (OK).
  const [pendingStage, setPendingStage] = useState(null); // { clipKey, stageKey }
  const [stageError, setStageError] = useState(null);     // { clipKey, message }

  // Merge UI state.
  const [mergeChecked, setMergeChecked] = useState({});   // clipKey -> bool
  const [mergeModalOpen, setMergeModalOpen] = useState(false);
  const [modalOrder, setModalOrder] = useState([]);       // clipKey[]
  const [merging, setMerging] = useState(false);
  const [mergeError, setMergeError] = useState(null);

  const keys = useKeys();
  const brand = useBrandKit();

  const current = clips[Math.min(selected, clips.length - 1)] || null;

  // Persisted per-clip state (auto-saves through wizard.data).
  const clipStages = wizard.data.clipStages || {};
  const clipLuts = wizard.data.clipLuts || {};

  const variants = current?.clip?.variants || null;
  const currentClipKey = current ? clipKey(current) : null;
  const selectedStage = currentClipKey
    ? (clipStages[currentClipKey] || pickInitialStage(variants))
    : 'original';
  const lutName = currentClipKey ? (clipLuts[currentClipKey] || DEFAULT_LUT) : DEFAULT_LUT;

  // Resolve the URL for the currently-selected stage, falling back to the
  // deepest existing variant, then to the polished URL the backend set, then
  // to the raw clip URL (covers legacy clips without a variants dict).
  const stageFilename = variants?.[selectedStage] || variants?.polished || null;
  const baseClipUrl = stageFilename
    ? getApiUrl(`/videos/${current.jobId}/${stageFilename}`)
    : (current?.clip?.video_url ? getApiUrl(current.clip.video_url) : null);
  // Merged previews override everything: a merged output has no variants.
  const activeMergedClip = selectedMergedId
    ? (wizard.data.mergedClips || []).find((m) => m.id === selectedMergedId)
    : null;
  const clipUrl = activeMergedClip ? getApiUrl(activeMergedClip.url) : baseClipUrl;

  // Build a blob URL for the original source file — only available when
  // the wizard has the in-memory File (lost after reload).
  useEffect(() => {
    if (!current?.sourceFile) { setSourceUrl(null); return; }
    const url = URL.createObjectURL(current.sourceFile);
    setSourceUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [current?.sourceFile]);

  if (clips.length === 0) {
    return (
      <div className="h-full flex flex-col items-center justify-center p-12 text-center text-zinc-500">
        <p className="text-[14px] text-white font-medium">No finished clips yet.</p>
        <p className="text-[12px] mt-1">Go back to Processing and wait, or restart the wizard.</p>
        <button onClick={wizard.reset} className="mt-4 btn-primary px-4 py-2 text-[13px]">
          Start over
        </button>
      </div>
    );
  }

  function setClipStage(stageKey) {
    if (!currentClipKey) return;
    wizard.setData({ clipStages: { ...clipStages, [currentClipKey]: stageKey } });
  }

  function setClipLut(lut) {
    if (!currentClipKey) return;
    wizard.setData({ clipLuts: { ...clipLuts, [currentClipKey]: lut } });
  }

  // Merge a new variant into wizard.data.jobs[fileId].result.clips[i].variants.
  // This is the central state update — every successful generation flows through here.
  function mergeVariant(stageKey, newFilename) {
    if (!current) return;
    wizard.setData((prev) => {
      const job = prev.jobs?.[current.fileId];
      if (!job?.result?.clips) return prev;
      const newClips = job.result.clips.map((c, i) => {
        if (i !== current.clipIndex) return c;
        const newVariants = { ...(c.variants || { original: c.video_url?.split('/').pop() }) };
        newVariants[stageKey] = newFilename;
        return { ...c, variants: newVariants };
      });
      return {
        ...prev,
        jobs: {
          ...prev.jobs,
          [current.fileId]: {
            ...job,
            result: { ...job.result, clips: newClips },
          },
        },
      };
    });
  }

  async function generateStage(stageKey) {
    if (!current) return;
    const cKey = clipKey(current);
    setPendingStage({ clipKey: cKey, stageKey });
    setStageError(null);

    try {
      const inputFilename = priorVariantFilename(variants, stageKey);
      if (!inputFilename) throw new Error('No source variant available');

      let newFilename = null;

      if (stageKey === 'edited') {
        if (!keys.gemini) throw new Error('Set your Gemini key in Settings first');
        const res = await fetch(getApiUrl('/api/edit'), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'X-Gemini-Key': keys.gemini },
          body: JSON.stringify({
            job_id: current.jobId,
            clip_index: current.clipIndex,
            input_filename: inputFilename,
          }),
        });
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        newFilename = data.new_video_url?.split('/').pop();
      } else if (stageKey === 'graded') {
        const res = await fetch(getApiUrl('/api/colorgrade'), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            job_id: current.jobId,
            clip_index: current.clipIndex,
            input_filename: inputFilename,
            lut_name: lutName,
          }),
        });
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        newFilename = data.new_video_url?.split('/').pop();
      } else if (stageKey === 'silencecut') {
        const res = await fetch(getApiUrl('/api/silencecut'), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            job_id: current.jobId,
            clip_index: current.clipIndex,
            input_filename: inputFilename,
          }),
        });
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        newFilename = data.new_video_url?.split('/').pop();
      } else if (stageKey === 'subtitled') {
        const res = await fetch(getApiUrl('/api/subtitle'), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            job_id: current.jobId,
            clip_index: current.clipIndex,
            input_filename: inputFilename,
            position: brand?.position || 'bottom',
            font_size: brand?.font_size || 50,
            font_name: brand?.font_name || 'Anton',
            font_color: brand?.font_color || '#FFFF00',
            border_color: brand?.border_color || '#000000',
            border_width: brand?.border_width || 4,
            bg_opacity: brand?.bg_opacity ?? 0,
            words_per_line: brand?.words_per_line || 3,
            text_case: brand?.text_case || 'upper',
          }),
        });
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        newFilename = data.new_video_url?.split('/').pop();
      }

      if (!newFilename) throw new Error('Empty response from backend');
      mergeVariant(stageKey, newFilename);
      // Switch to the freshly-generated stage so the preview swaps immediately.
      wizard.setData((prev) => ({
        ...prev,
        clipStages: { ...(prev.clipStages || {}), [cKey]: stageKey },
      }));
    } catch (e) {
      setStageError({ clipKey: cKey, message: String(e.message || e) });
      setTimeout(() => setStageError(null), 6000);
    } finally {
      setPendingStage(null);
    }
  }

  async function regenerateGrade(newLut) {
    if (!current) return;
    setClipLut(newLut);
    // If we already have a graded variant, regenerate with the new LUT.
    if (variants?.graded) {
      setPendingStage({ clipKey: clipKey(current), stageKey: 'graded' });
      setStageError(null);
      try {
        const inputFilename = priorVariantFilename(variants, 'graded');
        const res = await fetch(getApiUrl('/api/colorgrade'), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            job_id: current.jobId,
            clip_index: current.clipIndex,
            input_filename: inputFilename,
            lut_name: newLut,
          }),
        });
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        const newFilename = data.new_video_url?.split('/').pop();
        if (newFilename) mergeVariant('graded', newFilename);
      } catch (e) {
        setStageError({ clipKey: clipKey(current), message: String(e.message || e) });
        setTimeout(() => setStageError(null), 6000);
      } finally {
        setPendingStage(null);
      }
    }
  }

  // --- Merge helpers --------------------------------------------------------
  // Restrict selection to a single job's clips so /api/merge (which takes one
  // job_id) stays valid. Once a clip is checked, lock the candidate set.
  const checkedClipKeys = Object.keys(mergeChecked).filter((k) => mergeChecked[k]);
  const lockedJobId = checkedClipKeys.length > 0
    ? clips.find((c) => clipKey(c) === checkedClipKeys[0])?.jobId
    : null;

  function isMergeable(c) {
    return !lockedJobId || c.jobId === lockedJobId;
  }

  function toggleMergeCheck(c) {
    const k = clipKey(c);
    setMergeChecked((prev) => {
      const next = { ...prev };
      if (next[k]) delete next[k];
      else next[k] = true;
      return next;
    });
  }

  function clearMergeSelection() {
    setMergeChecked({});
    setMergeError(null);
  }

  function openMergeModal() {
    const ordered = clips
      .filter((c) => mergeChecked[clipKey(c)])
      .map((c) => clipKey(c));
    setModalOrder(ordered);
    setMergeError(null);
    setMergeModalOpen(true);
  }

  function reorderModal(fromIdx, toIdx) {
    setModalOrder((prev) => {
      if (toIdx < 0 || toIdx >= prev.length) return prev;
      const next = [...prev];
      const [moved] = next.splice(fromIdx, 1);
      next.splice(toIdx, 0, moved);
      return next;
    });
  }

  function removeFromModal(k) {
    setModalOrder((prev) => prev.filter((x) => x !== k));
    setMergeChecked((prev) => {
      const next = { ...prev };
      delete next[k];
      return next;
    });
  }

  async function submitMerge() {
    if (modalOrder.length < 2) {
      setMergeError('Need at least 2 clips to merge');
      return;
    }
    const ordered = modalOrder
      .map((k) => clips.find((c) => clipKey(c) === k))
      .filter(Boolean);
    if (ordered.length < 2) {
      setMergeError('Selected clips no longer available');
      return;
    }
    const jobId = ordered[0].jobId;
    const indices = ordered.map((c) => c.clipIndex);
    setMerging(true);
    setMergeError(null);
    try {
      const res = await fetch(getApiUrl('/api/merge'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          job_id: jobId,
          clip_indices: indices,
          use_processed: true,
          transition: 'cut',
        }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      const url = data.new_video_url;
      if (!url) throw new Error('Empty response from backend');
      const merged = {
        id: `m-${jobId}-${indices.join('_')}-${Date.now()}`,
        jobId,
        indices,
        url,
        label: `Merged ${indices.map((i) => `#${i + 1}`).join(' + ')}`,
        createdAt: Date.now(),
      };
      wizard.setData((prev) => ({
        ...prev,
        mergedClips: [...(prev.mergedClips || []), merged],
      }));
      setSelectedMergedId(merged.id);
      setMergeModalOpen(false);
      setMergeChecked({});
      setModalOrder([]);
    } catch (e) {
      setMergeError(String(e.message || e));
    } finally {
      setMerging(false);
    }
  }

  function publish(platform, scheduled) {
    if (!current) return;
    pushNotification({
      type: 'publish',
      platform,
      status: scheduled ? 'scheduled' : 'submitted',
      jobId: current.jobId,
      message: scheduled
        ? `Clip ${current.clipIndex + 1} scheduled to ${platform}`
        : `Clip ${current.clipIndex + 1} sent to ${platform}`,
    });
    // TODO(backend): plan TODO #9 — wire to /api/social/post once the
    // publish_jobs queue + status endpoint land.
  }

  const title = current?.clip?.video_title_for_youtube_short || current?.clip?.title || '';
  const description =
    current?.clip?.video_description_for_instagram ||
    current?.clip?.video_description_for_tiktok ||
    current?.clip?.description ||
    '';

  const isPending = pendingStage?.clipKey === currentClipKey;
  const isErr = stageError?.clipKey === currentClipKey;

  return (
    <div className="h-full flex">
      <aside className="w-[230px] shrink-0 border-r border-border bg-background overflow-y-auto custom-scrollbar p-3 space-y-1">
        <div className="text-[11px] uppercase tracking-wider text-zinc-500 px-2 mb-2">
          {clips.length} clip{clips.length === 1 ? '' : 's'}
        </div>
        {clips.map((c, i) => {
          const active = i === selected && !selectedMergedId;
          const clipTitle = c.clip?.video_title_for_youtube_short || c.clip?.title;
          const k = clipKey(c);
          const checked = !!mergeChecked[k];
          const mergeable = isMergeable(c);
          return (
            <div
              key={k}
              className={`w-full rounded-lg p-2 transition-colors flex items-start gap-2 ${
                active ? 'bg-primary/15 border border-primary/30' : 'border border-transparent hover:bg-white/5'
              }`}
            >
              <input
                type="checkbox"
                aria-label={`Select clip ${i + 1} to merge`}
                checked={checked}
                disabled={!mergeable && !checked}
                onChange={() => toggleMergeCheck(c)}
                title={mergeable ? 'Include in merge' : 'Merge only works within one source video'}
                className="mt-0.5 accent-primary disabled:opacity-30 cursor-pointer disabled:cursor-not-allowed"
                onClick={(e) => e.stopPropagation()}
              />
              <button
                type="button"
                onClick={() => { setSelected(i); setSelectedMergedId(null); setShowOriginal(false); }}
                className="flex-1 text-left min-w-0"
              >
                <div className={`text-[12px] font-medium truncate ${active ? 'text-white' : 'text-zinc-300'}`}>
                  Clip {i + 1}
                </div>
                <div className="text-[10px] text-zinc-500 truncate mt-0.5">{c.sourceName}</div>
                {clipTitle && (
                  <div className="text-[10px] text-zinc-400 truncate mt-1 italic">"{clipTitle}"</div>
                )}
              </button>
            </div>
          );
        })}

        {mergedClips.length > 0 && (
          <div className="pt-3 mt-2 border-t border-border space-y-1">
            <div className="text-[11px] uppercase tracking-wider text-zinc-500 px-2 mb-1">
              Merged outputs
            </div>
            {mergedClips.map((m) => {
              const active = selectedMergedId === m.id;
              return (
                <button
                  key={m.id}
                  type="button"
                  onClick={() => { setSelectedMergedId(m.id); setShowOriginal(false); }}
                  className={`w-full text-left rounded-lg p-2 transition-colors flex items-center gap-2 ${
                    active ? 'bg-primary/15 border border-primary/30' : 'border border-transparent hover:bg-white/5'
                  }`}
                >
                  <Combine size={12} className="text-zinc-400 shrink-0" />
                  <div className="min-w-0 flex-1">
                    <div className={`text-[12px] font-medium truncate ${active ? 'text-white' : 'text-zinc-300'}`}>
                      {m.label}
                    </div>
                    <div className="text-[10px] text-zinc-500 truncate">{m.indices.length} clips</div>
                  </div>
                </button>
              );
            })}
          </div>
        )}
      </aside>

      <div className="flex-1 flex flex-col overflow-hidden">
        <div className="flex-1 overflow-y-auto custom-scrollbar p-6 flex flex-col items-center gap-4">
          {!activeMergedClip && (
            <div className="flex items-center gap-2 text-[12px]">
              <button
                onClick={() => setShowOriginal(false)}
                className={`px-3 py-1.5 rounded-md ${!showOriginal ? 'bg-white/10 text-white' : 'text-zinc-400 hover:text-white'}`}
              >
                After
              </button>
              <button
                onClick={() => setShowOriginal(true)}
                disabled={!sourceUrl}
                className={`px-3 py-1.5 rounded-md disabled:opacity-30 disabled:cursor-not-allowed ${showOriginal ? 'bg-white/10 text-white' : 'text-zinc-400 hover:text-white'}`}
              >
                <Eye size={12} className="inline mr-1" /> Before
              </button>
            </div>
          )}

          <PhoneFrame size="md">
            {showOriginal && sourceUrl ? (
              <video key={`src-${selected}`} src={sourceUrl} controls className="w-full h-full object-contain" />
            ) : clipUrl ? (
              <video key={`clip-${selected}-${selectedStage}`} src={clipUrl} controls className="w-full h-full object-cover" />
            ) : (
              <div className="text-zinc-600 text-[12px] p-4 text-center">No preview available.</div>
            )}
          </PhoneFrame>

          {activeMergedClip && (
            <div className="text-[11px] text-zinc-400 max-w-md text-center" role="status">
              Preview of the merged output — no further editing available. Download from the bar below.
            </div>
          )}

          {/* Stage selector — segmented row. Lights up the currently-displayed
              variant and exposes a [+] on each missing stage so the user can
              fill it in without leaving Review. */}
          {!showOriginal && current && !activeMergedClip && (
            <div className="flex flex-col items-center gap-2">
              <div className="inline-flex rounded-lg border border-border bg-surface p-0.5 text-[12px]">
                {STAGES.map((stage) => {
                  const has = !!variants?.[stage.key];
                  const isActive = stage.key === selectedStage;
                  const isThisPending = isPending && pendingStage.stageKey === stage.key;
                  const clickable = has || !isThisPending;
                  return (
                    <button
                      key={stage.key}
                      disabled={!clickable}
                      onClick={() => has ? setClipStage(stage.key) : generateStage(stage.key)}
                      title={has
                        ? `Show ${stage.label.replace(/^\+ /, '').toLowerCase()} variant`
                        : `Generate ${stage.label.replace(/^\+ /, '').toLowerCase()}`}
                      className={`px-3 py-1.5 rounded-md transition-colors flex items-center gap-1.5 disabled:opacity-50 disabled:cursor-not-allowed ${
                        isActive
                          ? 'bg-primary/20 text-white border border-primary/40'
                          : has
                            ? 'text-zinc-300 hover:bg-white/5'
                            : 'text-zinc-500 hover:text-zinc-300 hover:bg-white/5'
                      }`}
                    >
                      {isThisPending ? (
                        <Loader2 size={12} className="animate-spin" />
                      ) : !has ? (
                        <Plus size={11} />
                      ) : null}
                      <span>{stage.short}</span>
                    </button>
                  );
                })}
              </div>

              {/* LUT picker — only relevant when the user is viewing or about
                  to generate the graded stage. Sliding the dropdown re-grades
                  in place if a graded variant already exists. */}
              {(selectedStage === 'graded' || (!variants?.graded && pendingStage?.stageKey === 'graded')) && (
                <div className="flex items-center gap-2 text-[11px] text-zinc-400">
                  <label htmlFor="lut-picker">LUT:</label>
                  <select
                    id="lut-picker"
                    value={lutName}
                    onChange={(e) => regenerateGrade(e.target.value)}
                    disabled={isPending}
                    className="bg-surface border border-border rounded-md px-2 py-1 text-zinc-200 text-[11px] disabled:opacity-50"
                  >
                    {LUTS.map((l) => (
                      <option key={l} value={l}>{l.replace('_', ' ')}</option>
                    ))}
                  </select>
                </div>
              )}

              {isErr && (
                <div className="text-[11px] text-red-400 max-w-md text-center" role="alert">
                  {stageError.message}
                </div>
              )}
            </div>
          )}

          {title && !activeMergedClip && (
            <div className="text-center max-w-md">
              <div className="text-[13px] text-white font-medium">{title}</div>
              {description && (
                <p className="text-[11px] text-zinc-500 mt-1 leading-snug whitespace-pre-line">{description}</p>
              )}
            </div>
          )}
        </div>

        <div className="border-t border-border bg-surface px-4 py-3 flex flex-wrap items-center gap-3 shrink-0">
          {checkedClipKeys.length >= 2 && (
            <button
              type="button"
              onClick={openMergeModal}
              className="btn-primary px-3 py-2 text-[12px] flex items-center gap-2 bg-emerald-600 hover:bg-emerald-500"
              title="Stitch the selected clips into one MP4"
            >
              <Combine size={12} /> Merge {checkedClipKeys.length} selected
            </button>
          )}
          {checkedClipKeys.length > 0 && checkedClipKeys.length < 2 && (
            <span className="text-[11px] text-zinc-500">Select at least 2 clips to merge</span>
          )}
          {checkedClipKeys.length > 0 && (
            <button
              type="button"
              onClick={clearMergeSelection}
              className="text-[11px] text-zinc-400 hover:text-white underline-offset-2 hover:underline"
            >
              Clear
            </button>
          )}
          <a
            href={clipUrl || '#'}
            download
            className={`btn-primary px-3 py-2 text-[12px] flex items-center gap-2 ${!clipUrl ? 'opacity-40 pointer-events-none' : ''}`}
          >
            <Download size={12} /> Download
          </a>
          <div className="flex items-center gap-1">
            <span className="text-[11px] text-zinc-500 mr-1">Publish:</span>
            {PLATFORMS.map((p) => (
              <button key={p} onClick={() => publish(p, false)} className="hover:opacity-80 transition-opacity" title={`Publish to ${p}`}>
                <PlatformBadge platform={p} withLabel={false} size="sm" />
              </button>
            ))}
          </div>
          <div className="flex items-center gap-1">
            <span className="text-[11px] text-zinc-500 mr-1">Schedule:</span>
            {PLATFORMS.map((p) => (
              <button key={p} onClick={() => publish(p, true)} className="hover:opacity-80 transition-opacity" title={`Schedule to ${p}`}>
                <PlatformBadge platform={p} withLabel={false} size="sm" />
              </button>
            ))}
          </div>
          <button
            disabled
            title="CapCut export — coming soon"
            className="ml-auto px-3 py-2 text-[12px] flex items-center gap-2 rounded-md border border-border text-zinc-500 cursor-not-allowed"
          >
            <Scissors size={12} /> Send to CapCut
          </button>
        </div>
      </div>

      {mergeModalOpen && (
        <div
          className="fixed inset-0 z-50 bg-black/70 flex items-center justify-center p-6"
          role="dialog"
          aria-modal="true"
          aria-label="Confirm merge order"
          onClick={() => !merging && setMergeModalOpen(false)}
        >
          <div
            className="w-full max-w-md bg-surface border border-border rounded-xl p-5 space-y-4"
            onClick={(e) => e.stopPropagation()}
          >
            <div>
              <h2 className="text-[14px] text-white font-semibold flex items-center gap-2">
                <Combine size={14} /> Merge clips
              </h2>
              <p className="text-[11px] text-zinc-500 mt-1">
                Reorder, remove, then re-render. The merged file appears under "Merged outputs" in the sidebar.
              </p>
            </div>

            <ul className="space-y-1.5">
              {modalOrder.map((k, idx) => {
                const c = clips.find((x) => clipKey(x) === k);
                if (!c) return null;
                const title = c.clip?.video_title_for_youtube_short || c.clip?.title;
                return (
                  <li
                    key={k}
                    className="flex items-center gap-2 bg-background border border-border rounded-md p-2"
                  >
                    <div className="text-[11px] text-zinc-500 w-5 text-right">{idx + 1}.</div>
                    <div className="flex-1 min-w-0">
                      <div className="text-[12px] text-white truncate">Clip {c.clipIndex + 1}</div>
                      {title && <div className="text-[10px] text-zinc-500 truncate italic">"{title}"</div>}
                    </div>
                    <button
                      type="button"
                      aria-label="Move up"
                      disabled={idx === 0 || merging}
                      onClick={() => reorderModal(idx, idx - 1)}
                      className="p-1 text-zinc-400 hover:text-white disabled:opacity-20 disabled:cursor-not-allowed"
                    >
                      <ArrowUp size={12} />
                    </button>
                    <button
                      type="button"
                      aria-label="Move down"
                      disabled={idx === modalOrder.length - 1 || merging}
                      onClick={() => reorderModal(idx, idx + 1)}
                      className="p-1 text-zinc-400 hover:text-white disabled:opacity-20 disabled:cursor-not-allowed"
                    >
                      <ArrowDown size={12} />
                    </button>
                    <button
                      type="button"
                      aria-label="Remove"
                      disabled={merging}
                      onClick={() => removeFromModal(k)}
                      className="p-1 text-zinc-400 hover:text-red-400 disabled:opacity-20"
                    >
                      <X size={12} />
                    </button>
                  </li>
                );
              })}
            </ul>

            {mergeError && (
              <div className="text-[11px] text-red-400" role="alert">{mergeError}</div>
            )}

            <div className="flex items-center justify-end gap-2 pt-1">
              <button
                type="button"
                onClick={() => setMergeModalOpen(false)}
                disabled={merging}
                className="px-3 py-1.5 text-[12px] text-zinc-400 hover:text-white"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={submitMerge}
                disabled={merging || modalOrder.length < 2}
                className="btn-primary px-3 py-1.5 text-[12px] flex items-center gap-2 disabled:opacity-50"
              >
                {merging ? (
                  <><Loader2 size={12} className="animate-spin" /> Re-rendering…</>
                ) : (
                  <>Re-render</>
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
