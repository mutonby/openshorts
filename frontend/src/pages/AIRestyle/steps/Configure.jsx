// AI Restyle Configure step. Pick a Background + Lighting preset; either
// can be overridden per-job via an editable textarea. POSTs /api/restyle
// with the resolved prompts on submit.
//
// Plan deviation: the spec sketched a single combined-prompt textarea
// (split by newline). That's fragile (multi-line user edits break the
// boundary). Use two separate textareas instead — matches the backend's
// two form-field contract 1:1.

import { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { useAIRestylePresets } from '../../../state/aiRestylePresets.js';
import { useKeys } from '../../../state/keysStore.js';
import { getApiUrl } from '../../../config.js';

export default function Configure({ wizard }) {
  const presets = useAIRestylePresets();
  const keys = useKeys();
  const sel = wizard.data.selection;

  // First-render init: seed selection from defaults if not already chosen.
  useEffect(() => {
    if (sel.backgroundPresetId && sel.lightingPresetId) return;
    wizard.setData({
      selection: {
        ...sel,
        backgroundPresetId: sel.backgroundPresetId || presets.defaultBackgroundId,
        lightingPresetId:   sel.lightingPresetId   || presets.defaultLightingId,
      },
    });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [presets.defaultBackgroundId, presets.defaultLightingId]);

  const bgPreset = presets.backgrounds.find((p) => p.id === sel.backgroundPresetId);
  const ltPreset = presets.lightings.find((p) => p.id === sel.lightingPresetId);

  const bgPrompt = useMemo(
    () => sel.backgroundPromptOverride ?? bgPreset?.prompt ?? '',
    [sel.backgroundPromptOverride, bgPreset?.prompt],
  );
  const ltPrompt = useMemo(
    () => sel.lightingPromptOverride ?? ltPreset?.prompt ?? '',
    [sel.lightingPromptOverride, ltPreset?.prompt],
  );

  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);

  function setBg(id) {
    wizard.setData({ selection: { ...sel, backgroundPresetId: id, backgroundPromptOverride: null } });
  }
  function setLt(id) {
    wizard.setData({ selection: { ...sel, lightingPresetId: id, lightingPromptOverride: null } });
  }
  function setBgOverride(text) {
    wizard.setData({ selection: { ...sel, backgroundPromptOverride: text } });
  }
  function setLtOverride(text) {
    wizard.setData({ selection: { ...sel, lightingPromptOverride: text } });
  }

  const canSubmit =
    !!wizard.data.file?.file
    && !!keys.gemini
    && !!keys.fal
    && bgPrompt.trim().length > 0
    && ltPrompt.trim().length > 0
    && bgPrompt.length <= 500
    && ltPrompt.length <= 500
    && !submitting;

  async function start() {
    setError(null);
    if (!keys.gemini) { setError('Set your Gemini key in Settings → API Keys first.'); return; }
    if (!keys.fal)    { setError('Set your fal.ai key in Settings → API Keys first.'); return; }

    const fd = new FormData();
    fd.append('file', wizard.data.file.file);
    fd.append('background_prompt', bgPrompt.slice(0, 500));
    fd.append('lighting_prompt',   ltPrompt.slice(0, 500));

    setSubmitting(true);
    try {
      const res = await fetch(getApiUrl('/api/restyle'), {
        method: 'POST',
        headers: {
          'X-Gemini-Key': keys.gemini,
          'X-Fal-Key': keys.fal,
        },
        body: fd,
      });
      if (!res.ok) {
        const detail = await res.text();
        throw new Error(`HTTP ${res.status}: ${detail}`);
      }
      const { job_id } = await res.json();
      wizard.setData({
        job: { jobId: job_id, status: 'processing', result: null, progress_pct: 0, logs: [] },
      });
      wizard.next();
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setSubmitting(false);
    }
  }

  const keysReady = keys.gemini && keys.fal;

  return (
    <div className="h-full overflow-y-auto custom-scrollbar p-8">
      <div className="max-w-2xl mx-auto">
        <h1 className="text-[24px] font-semibold mb-2">Configure restyle</h1>
        <p className="text-[13px] text-zinc-400 mb-6">
          Pick a background and lighting preset. Tweak either prompt below if you want — overrides apply to this job only.
        </p>

        {!keysReady && (
          <div className="mb-4 rounded-md border border-yellow-500/30 bg-yellow-500/5 p-3 text-[12px] text-yellow-300">
            Missing API keys. <Link to="/settings/system/api-keys" className="underline">Set them in Settings →</Link>
            <div className="mt-1 text-zinc-500">Required: Gemini (Nano-Banana relight) + fal.ai (v2v restyle).</div>
          </div>
        )}

        <div className="grid grid-cols-2 gap-4 mb-6">
          <PresetSelect
            label="Background"
            value={sel.backgroundPresetId || ''}
            onChange={setBg}
            options={presets.backgrounds}
            defaultId={presets.defaultBackgroundId}
          />
          <PresetSelect
            label="Lighting"
            value={sel.lightingPresetId || ''}
            onChange={setLt}
            options={presets.lightings}
            defaultId={presets.defaultLightingId}
          />
        </div>

        <PromptArea
          label="Background prompt"
          value={bgPrompt}
          onChange={setBgOverride}
          maxLength={500}
          overridden={sel.backgroundPromptOverride !== null}
        />

        <div className="h-4" />

        <PromptArea
          label="Lighting prompt"
          value={ltPrompt}
          onChange={setLtOverride}
          maxLength={500}
          overridden={sel.lightingPromptOverride !== null}
        />

        {error && <div className="mt-3 text-[12px] text-red-400" role="alert">{error}</div>}

        <div className="mt-6 flex items-center justify-between">
          <button onClick={wizard.back} className="px-4 py-2 text-[13px] text-zinc-400 hover:text-white transition-colors">
            ← Back
          </button>
          <button
            onClick={start}
            disabled={!canSubmit}
            className="px-4 py-2 text-[13px] bg-primary text-white rounded-md hover:bg-primary/90 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            {submitting ? 'Starting…' : 'Start restyle →'}
          </button>
        </div>
      </div>
    </div>
  );
}

function PresetSelect({ label, value, onChange, options, defaultId }) {
  return (
    <div>
      <label className="block text-[11px] uppercase tracking-wider text-zinc-500 mb-2">{label}</label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full bg-surface border border-border rounded-md px-3 py-2 text-[13px] text-zinc-200 focus:outline-none focus:border-primary"
      >
        {options.map((p) => (
          <option key={p.id} value={p.id}>
            {p.label}{p.id === defaultId ? '  ★' : ''}
          </option>
        ))}
      </select>
    </div>
  );
}

function PromptArea({ label, value, onChange, maxLength, overridden }) {
  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <label className="text-[11px] uppercase tracking-wider text-zinc-500">{label}</label>
        {overridden && (
          <span className="text-[10px] text-yellow-400 uppercase tracking-wider">Custom (this job only)</span>
        )}
      </div>
      <textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        rows={3}
        maxLength={maxLength}
        className="w-full bg-surface border border-border rounded-md p-3 text-[12px] text-zinc-200 font-mono leading-relaxed focus:outline-none focus:border-primary"
      />
      <div className="text-[10px] text-zinc-500 text-right mt-1">{value.length}/{maxLength}</div>
    </div>
  );
}
