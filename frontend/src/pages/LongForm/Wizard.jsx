// 4-step long-form wizard. Owns wizard state via useWizard; step
// components stay dumb. Mirrors the short-form layout but each step
// targets the long-form pipeline (chapter detection, segment exports).

import { Check } from 'lucide-react';
import { useWizard } from '../../hooks/useWizard.js';
import Upload from './steps/Upload.jsx';
import Settings from './steps/Settings.jsx';
import Processing from './steps/Processing.jsx';
import Editor from './steps/Editor.jsx';

const STEPS = [
  { id: 'upload',     label: 'Upload' },
  { id: 'settings',   label: 'Settings' },
  { id: 'processing', label: 'Processing', lock: true },
  { id: 'editor',     label: 'Editor' },
];

const INITIAL = {
  file: null,            // { id, file, name, size, durationSec? }
  settings: {
    colorGrade:       true,
    autoSubtitles:    true,
    chapterDetection: true,
    descriptionTags:  true,
    introOutro:       false,
  },
  processing: {
    progress: 0,         // 0–100 (stubbed timer until backend ships)
    status:   'idle',    // idle | running | complete
  },
  chapters: [],          // [{ id, label, startSec, endSec }]
};

const STORAGE_KEY = 'openshorts.longForm.wizard';

// File objects can't be JSON-serialized; after a reload `data.file.file`
// is a plain object instead of a real File and Settings/Processing/Editor
// would all fail. Force the wizard back to Upload in that case.
function longFormNeedsFreshUpload(data) {
  return !!data?.file && !(data.file.file instanceof File);
}

export default function Wizard() {
  const w = useWizard({
    steps: STEPS,
    initialData: INITIAL,
    storageKey: STORAGE_KEY,
    resetOnRehydrate: longFormNeedsFreshUpload,
  });

  return (
    <div className="h-full flex flex-col">
      <StepIndicator wizard={w} />
      <div className="flex-1 overflow-hidden">
        {w.currentStep.id === 'upload'     && <Upload wizard={w} />}
        {w.currentStep.id === 'settings'   && <Settings wizard={w} />}
        {w.currentStep.id === 'processing' && <Processing wizard={w} />}
        {w.currentStep.id === 'editor'     && <Editor wizard={w} />}
      </div>
    </div>
  );
}

function StepIndicator({ wizard }) {
  return (
    <div className="px-6 py-4 border-b border-border bg-background shrink-0">
      <div className="flex items-center gap-3">
        {wizard.steps.map((s, i) => {
          const active = i === wizard.step;
          const done = i < wizard.step;
          const reachable = i <= wizard.step && !wizard.isLocked;
          return (
            <div key={s.id} className="flex items-center gap-3 flex-1">
              <button
                onClick={() => reachable && wizard.goto(i)}
                disabled={!reachable}
                className={`flex items-center gap-2 disabled:cursor-not-allowed ${
                  active ? 'text-white' : done ? 'text-zinc-300' : 'text-zinc-600'
                }`}
              >
                <span className={`w-6 h-6 flex items-center justify-center rounded-full text-[11px] font-medium ${
                  active ? 'bg-primary text-white' :
                  done  ? 'bg-success/20 text-success border border-success/40' :
                          'bg-white/5 text-zinc-500 border border-border'
                }`}>
                  {done ? <Check size={12} /> : i + 1}
                </span>
                <span className="text-[12px]">{s.label}</span>
              </button>
              {i < wizard.steps.length - 1 && (
                <div className={`flex-1 h-px ${done ? 'bg-success/40' : 'bg-border'}`} />
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
