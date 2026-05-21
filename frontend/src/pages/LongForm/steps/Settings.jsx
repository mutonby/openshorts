// Step 2: Long-form processing settings.

const TOGGLES = [
  { id: 'colorGrade',       label: 'Color grade',       hint: 'Apply a cinematic LUT (backend TODO #5).' },
  { id: 'autoSubtitles',    label: 'Auto subtitles',    hint: 'Transcribe + burn captions with brand-kit style.' },
  { id: 'chapterDetection', label: 'Chapter detection', hint: 'Run PySceneDetect for chapter markers (backend TODO #6).' },
  { id: 'descriptionTags',  label: 'Description + tags', hint: 'Generate YouTube description and tag suggestions.' },
  { id: 'introOutro',       label: 'Intro / outro',     hint: 'Splice a brand-kit intro and outro (backend TODO #8).' },
];

export default function Settings({ wizard }) {
  const settings = wizard.data.settings || {};
  const file = wizard.data.file;

  function toggle(key) {
    wizard.setData({ settings: { ...settings, [key]: !settings[key] } });
  }

  return (
    <div className="h-full overflow-y-auto custom-scrollbar">
      <div className="p-6 max-w-3xl mx-auto space-y-6">
        <header>
          <h1 className="text-[18px] font-semibold text-white">Settings</h1>
          <p className="text-[13px] text-zinc-500 mt-1">
            Pick what runs during processing. Each setting maps to a feature the editor exposes in Step 4.
          </p>
        </header>

        {file && (
          <div className="rounded-lg border border-border bg-surface p-3 flex items-center gap-3">
            <div className="flex-1 min-w-0">
              <div className="text-[12px] text-zinc-500">Source</div>
              <div className="text-[13px] text-white truncate">{file.name}</div>
            </div>
          </div>
        )}

        <section className="rounded-xl border border-border bg-surface p-5 space-y-3">
          <h2 className="text-[12px] uppercase tracking-wider text-zinc-500">Pipeline</h2>
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

        <div className="flex items-center justify-between pt-4 border-t border-border">
          <button onClick={wizard.back} className="text-[13px] text-zinc-400 hover:text-white transition-colors">
            ← Back
          </button>
          <button onClick={wizard.next} className="btn-primary px-5 py-2 text-[13px]">
            Start processing →
          </button>
        </div>
      </div>
    </div>
  );
}
