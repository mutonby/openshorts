// 3-step AI Restyle wizard. Same useWizard pattern as ShortForm/Wizard.jsx.
// File objects can't be JSON-serialized, so a rehydrate with a stale file
// stub forces the wizard back to Upload.

import { Check } from 'lucide-react';
import { useWizard } from '../../hooks/useWizard.js';
import Upload from './steps/Upload.jsx';
import Configure from './steps/Configure.jsx';
import Review from './steps/Review.jsx';

const STEPS = [
  { id: 'upload',    label: 'Upload' },
  { id: 'configure', label: 'Configure' },
  { id: 'review',    label: 'Review' },
];

const INITIAL = {
  file: null,
  selection: {
    backgroundPresetId: null,
    lightingPresetId: null,
    backgroundPromptOverride: null,
    lightingPromptOverride: null,
  },
  job: null,
};

const STORAGE_KEY = 'openshorts.aiRestyle.wizard';

function needsFreshUpload(data) {
  return !!data?.file && !(data.file?.file instanceof File);
}

export default function Wizard() {
  const w = useWizard({
    steps: STEPS,
    initialData: INITIAL,
    storageKey: STORAGE_KEY,
    resetOnRehydrate: needsFreshUpload,
  });

  return (
    <div className="h-full flex flex-col">
      <StepIndicator wizard={w} />
      <div className="flex-1 overflow-hidden">
        {w.currentStep.id === 'upload'    && <Upload wizard={w} />}
        {w.currentStep.id === 'configure' && <Configure wizard={w} />}
        {w.currentStep.id === 'review'    && <Review wizard={w} />}
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
