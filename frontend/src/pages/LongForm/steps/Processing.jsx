// Step 3: Long-form processing. The chapter-detection / segmentation
// pipeline isn't wired yet (plan TODOs #4–#8), so this step runs a
// simulated progress bar — the SnakeGame on the side handles the wait.
//
// TODO(backend): plan TODOs #4 (silence removal), #5 (LUT color grade),
// #6 (chapter detection), #7 (segment export), #8 (intro/outro). Replace
// the timer with a real polling loop once those routes ship.

import { useEffect, useRef } from 'react';
import { CheckCircle2, Loader2 } from 'lucide-react';
import SnakeGame from '../../../components/ui/SnakeGame.jsx';

const STAGES = [
  { id: 'transcribe', label: 'Transcribing',         until: 25 },
  { id: 'scenes',     label: 'Detecting chapters',   until: 55 },
  { id: 'grade',      label: 'Applying color grade', until: 75 },
  { id: 'subs',       label: 'Generating subtitles', until: 90 },
  { id: 'finalize',   label: 'Finalizing',           until: 100 },
];

const HISTORY_KEY = 'openshorts.longForm.history';

function saveHistory(entry) {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    const list = raw ? JSON.parse(raw) : [];
    list.unshift(entry);
    localStorage.setItem(HISTORY_KEY, JSON.stringify(list.slice(0, 50)));
  } catch {/* ignore */}
}

export default function Processing({ wizard }) {
  const proc = wizard.data.processing || { progress: 0, status: 'idle' };
  const file = wizard.data.file;
  const startedRef = useRef(false);
  const savedRef = useRef(false);

  // Drive the fake progress timer once.
  useEffect(() => {
    if (startedRef.current) return;
    startedRef.current = true;
    if (proc.status === 'complete') return;
    wizard.setData((prev) => ({ ...prev, processing: { progress: 0, status: 'running' } }));

    const id = setInterval(() => {
      wizard.setData((prev) => {
        const cur = prev.processing?.progress ?? 0;
        const next = Math.min(100, cur + 2 + Math.random() * 2);
        if (next >= 100) {
          clearInterval(id);
          return { ...prev, processing: { progress: 100, status: 'complete' } };
        }
        return { ...prev, processing: { progress: next, status: 'running' } };
      });
    }, 200);
    return () => clearInterval(id);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // When complete, persist a history entry once and seed chapters.
  useEffect(() => {
    if (proc.status !== 'complete' || savedRef.current) return;
    savedRef.current = true;
    saveHistory({
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
      ts: Date.now(),
      title: file?.name ? `Edit: ${file.name}` : 'Untitled edit',
      chapters: 3,
    });
    // Placeholder chapters (real ones come with backend TODO #6).
    if (!wizard.data.chapters?.length) {
      wizard.setData({
        chapters: [
          { id: 'c1', label: 'Intro',    startSec: 0,    endSec: 60 },
          { id: 'c2', label: 'Main',     startSec: 60,   endSec: 540 },
          { id: 'c3', label: 'Outro',    startSec: 540,  endSec: 600 },
        ],
      });
    }
  }, [proc.status, file, wizard]);

  const currentStage = STAGES.find((s) => proc.progress <= s.until) || STAGES[STAGES.length - 1];

  return (
    <div className="h-full overflow-y-auto custom-scrollbar">
      <div className="p-6 max-w-5xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-6">
        <section>
          <h1 className="text-[18px] font-semibold text-white">Processing</h1>
          <p className="text-[13px] text-zinc-500 mt-1 mb-4">
            Chapter detection and long-form pipeline branches are stubbed —
            backend TODOs #4–#8. The progress here is simulated.
          </p>

          <div className="rounded-xl border border-border bg-surface p-5 space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-[13px] text-white flex items-center gap-2">
                {proc.status === 'complete'
                  ? <CheckCircle2 size={16} className="text-success" />
                  : <Loader2 size={16} className="text-primary animate-spin" />}
                {proc.status === 'complete' ? 'Complete' : currentStage.label}
              </span>
              <span className="text-[11px] font-mono text-zinc-500">{Math.round(proc.progress)}%</span>
            </div>
            <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
              <div
                className="h-full bg-primary transition-all"
                style={{ width: `${proc.progress}%` }}
              />
            </div>
            <ul className="text-[11px] text-zinc-500 space-y-1">
              {STAGES.map((s) => (
                <li key={s.id} className="flex items-center gap-2">
                  <span className={`w-1.5 h-1.5 rounded-full ${
                    proc.progress >= s.until ? 'bg-success' :
                    proc.progress + 5 >= s.until ? 'bg-primary' : 'bg-zinc-700'
                  }`} />
                  {s.label}
                </li>
              ))}
            </ul>
          </div>

          <div className="flex items-center justify-between mt-6 pt-4 border-t border-border">
            <span className="text-[11px] text-zinc-500">
              {proc.status === 'complete' ? 'Ready to edit.' : 'Hang tight — try Snake while you wait.'}
            </span>
            <button
              onClick={() => wizard.goto(3)}
              disabled={proc.status !== 'complete'}
              className="btn-primary px-5 py-2 text-[13px] disabled:opacity-40 disabled:cursor-not-allowed"
            >
              Open editor →
            </button>
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
