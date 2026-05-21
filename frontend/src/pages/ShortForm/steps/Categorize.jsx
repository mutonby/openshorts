// Step 2: Categorize. Four category cards per uploaded clip + an auto-edit
// settings block.
//
// AI categorization is stubbed — see plan TODO #2. We pre-fill 'educational'
// for new clips and let the user override per clip.

import { useEffect } from 'react';
import { GraduationCap, Mic, Sparkles, Tv, KeyRound } from 'lucide-react';
import { Link } from 'react-router-dom';
import { useKeys } from '../../../state/keysStore.js';

const CATEGORIES = [
  {
    id: 'educational',
    label: 'Educational',
    icon: GraduationCap,
    description: 'Talking-head explainers, lectures, walk-throughs.',
  },
  {
    id: 'yap',
    label: 'Yap',
    icon: Mic,
    description: 'Podcasts, casual rants, multi-speaker chats.',
  },
  {
    id: 'live',
    label: 'Live',
    icon: Tv,
    description: 'Streams, gameplay reactions, IRL moments.',
  },
  {
    id: 'viral',
    label: 'Viral',
    icon: Sparkles,
    description: 'Fast-cut highlights, memes, micro-moments.',
  },
];

// Order matches the run_job auto-pipeline chain: AI edit → grade → silence → subtitles.
const TOGGLES = [
  { id: 'autoEdit',       label: 'AI edits (zoom / pan / cuts)', hint: 'Gemini picks zoompan + cut points per clip.' },
  { id: 'colorGrade',     label: 'Color grade',                   hint: 'Apply a cinematic LUT (teal-orange by default).' },
  { id: 'silenceRemoval', label: 'Silence removal',               hint: 'Auto-cut dead air from your clip.' },
  { id: 'autoSubtitles',  label: 'Auto subtitles',                hint: 'Transcribe + burn captions with brand-kit style.' },
];

export default function Categorize({ wizard }) {
  const files = wizard.data.files || [];
  const settings = wizard.data.settings || {};
  const keys = useKeys();
  const hasGeminiKey = !!keys.gemini;

  // TODO(backend): plan TODO #2 — replace this pre-fill with POST /api/categorize
  // on the file's transcript or thumbnail.
  useEffect(() => {
    const next = files.map((f) => (f.category ? f : { ...f, category: 'educational' }));
    if (next.some((f, i) => f !== files[i])) {
      wizard.setData({ files: next });
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [files.length]);

  function setCategory(fileId, categoryId) {
    wizard.setData({ files: files.map((f) => f.id === fileId ? { ...f, category: categoryId } : f) });
  }

  function toggle(key) {
    wizard.setData({ settings: { ...settings, [key]: !settings[key] } });
  }

  return (
    <div className="h-full overflow-y-auto custom-scrollbar">
      <div className="p-6 max-w-5xl mx-auto space-y-8">
        <header>
          <h1 className="text-[18px] font-semibold text-white">Categorize</h1>
          <p className="text-[13px] text-zinc-500 mt-1">
            Pick a category per clip — it tunes the layout and editing style.
            AI categorization lands with a backend update; defaults are pre-selected.
          </p>
        </header>

        <section className="space-y-3">
          <h2 className="text-[12px] uppercase tracking-wider text-zinc-500">Clips</h2>
          <div className="space-y-3">
            {files.map((f) => (
              <div key={f.id} className="rounded-xl border border-border bg-surface p-4">
                <div className="text-[13px] text-white font-medium mb-3 truncate">{f.name}</div>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                  {CATEGORIES.map((c) => {
                    const Icon = c.icon;
                    const active = f.category === c.id;
                    return (
                      <button
                        key={c.id}
                        onClick={() => setCategory(f.id, c.id)}
                        className={`text-left rounded-lg border p-3 transition-colors ${
                          active ? 'border-primary bg-primary/10' : 'border-border hover:bg-white/5'
                        }`}
                      >
                        <div className={`flex items-center gap-2 text-[12px] font-medium ${active ? 'text-primary' : 'text-white'}`}>
                          <Icon size={14} />
                          {c.label}
                        </div>
                        <p className="text-[10px] text-zinc-500 mt-1 leading-snug">{c.description}</p>
                      </button>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
        </section>

        <section className="rounded-xl border border-border bg-surface p-5 space-y-3">
          <h2 className="text-[12px] uppercase tracking-wider text-zinc-500">Auto-edit settings</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {TOGGLES.map((t) => (
              <label key={t.id} className="flex items-start gap-3 rounded-lg border border-border p-3 cursor-pointer hover:bg-white/5">
                <input
                  type="checkbox"
                  checked={!!settings[t.id]}
                  onChange={() => toggle(t.id)}
                  className="mt-1 accent-primary"
                />
                <div>
                  <div className="text-[13px] text-white">{t.label}</div>
                  <div className="text-[11px] text-zinc-500 mt-0.5">{t.hint}</div>
                </div>
              </label>
            ))}
          </div>
        </section>

        {!hasGeminiKey && (
          <div className="rounded-xl border border-amber-500/30 bg-amber-500/10 p-4 flex items-start gap-3">
            <KeyRound size={16} className="text-amber-400 shrink-0 mt-0.5" />
            <div className="flex-1">
              <div className="text-[13px] text-amber-200 font-medium">Gemini API key required</div>
              <div className="text-[12px] text-amber-200/80 mt-0.5">
                Processing needs a Gemini key to identify viral moments. Add one before continuing.
              </div>
            </div>
            <Link
              to="/settings/system/api-keys"
              className="text-[12px] text-amber-200 hover:text-white underline shrink-0"
            >
              Open Settings →
            </Link>
          </div>
        )}

        <div className="flex items-center justify-between pt-4 border-t border-border">
          <button onClick={wizard.back} className="text-[13px] text-zinc-400 hover:text-white transition-colors">
            ← Back
          </button>
          <button
            onClick={wizard.next}
            disabled={!hasGeminiKey}
            title={!hasGeminiKey ? 'Set your Gemini API key in Settings first.' : ''}
            className="btn-primary px-5 py-2 text-[13px] disabled:opacity-40 disabled:cursor-not-allowed"
          >
            Start processing →
          </button>
        </div>
      </div>
    </div>
  );
}
