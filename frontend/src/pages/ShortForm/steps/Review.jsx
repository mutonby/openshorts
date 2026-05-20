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
import { Download, Eye, Loader2, Plus, Scissors } from 'lucide-react';
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
  const [selected, setSelected] = useState(0);
  const [showOriginal, setShowOriginal] = useState(false);
  const [sourceUrl, setSourceUrl] = useState(null);

  // Per-clip transient state — loading flag + last error. Lost on reload (OK).
  const [pendingStage, setPendingStage] = useState(null); // { clipKey, stageKey }
  const [stageError, setStageError] = useState(null);     // { clipKey, message }

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
  const clipUrl = stageFilename
    ? getApiUrl(`/videos/${current.jobId}/${stageFilename}`)
    : (current?.clip?.video_url ? getApiUrl(current.clip.video_url) : null);

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
          const active = i === selected;
          const clipTitle = c.clip?.video_title_for_youtube_short || c.clip?.title;
          return (
            <button
              key={`${c.jobId}-${c.clipIndex}`}
              onClick={() => { setSelected(i); setShowOriginal(false); }}
              className={`w-full text-left rounded-lg p-2 transition-colors ${
                active ? 'bg-primary/15 border border-primary/30' : 'border border-transparent hover:bg-white/5'
              }`}
            >
              <div className={`text-[12px] font-medium truncate ${active ? 'text-white' : 'text-zinc-300'}`}>
                Clip {i + 1}
              </div>
              <div className="text-[10px] text-zinc-500 truncate mt-0.5">{c.sourceName}</div>
              {clipTitle && (
                <div className="text-[10px] text-zinc-400 truncate mt-1 italic">"{clipTitle}"</div>
              )}
            </button>
          );
        })}
      </aside>

      <div className="flex-1 flex flex-col overflow-hidden">
        <div className="flex-1 overflow-y-auto custom-scrollbar p-6 flex flex-col items-center gap-4">
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

          <PhoneFrame size="md">
            {showOriginal && sourceUrl ? (
              <video key={`src-${selected}`} src={sourceUrl} controls className="w-full h-full object-contain" />
            ) : clipUrl ? (
              <video key={`clip-${selected}-${selectedStage}`} src={clipUrl} controls className="w-full h-full object-cover" />
            ) : (
              <div className="text-zinc-600 text-[12px] p-4 text-center">No preview available.</div>
            )}
          </PhoneFrame>

          {/* Stage selector — segmented row. Lights up the currently-displayed
              variant and exposes a [+] on each missing stage so the user can
              fill it in without leaving Review. */}
          {!showOriginal && current && (
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

          {title && (
            <div className="text-center max-w-md">
              <div className="text-[13px] text-white font-medium">{title}</div>
              {description && (
                <p className="text-[11px] text-zinc-500 mt-1 leading-snug whitespace-pre-line">{description}</p>
              )}
            </div>
          )}
        </div>

        <div className="border-t border-border bg-surface px-4 py-3 flex flex-wrap items-center gap-3 shrink-0">
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
    </div>
  );
}
