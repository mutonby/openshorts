import { Check } from 'lucide-react';

/**
 * The single wizard stepper (design.md) — shared by ThumbnailStudio and
 * SaaShortsTab. Mono ordinals, hairline connectors, brass current step.
 *
 * Props:
 *  - steps: string[] (labels)
 *  - current: index of the active step
 *  - onStepClick(index)? — optional, only past steps are clickable
 */
export default function StepIndicator({ steps, current, onStepClick }) {
  return (
    <ol className="flex items-center w-full" aria-label="progress">
      {steps.map((label, i) => {
        const done = i < current;
        const active = i === current;
        const clickable = done && typeof onStepClick === 'function';
        return (
          <li key={label} className={`flex items-center ${i < steps.length - 1 ? 'flex-1' : ''}`}>
            <button
              type="button"
              disabled={!clickable}
              onClick={() => clickable && onStepClick(i)}
              className={`flex items-center gap-2 shrink-0 ${clickable ? 'cursor-pointer' : 'cursor-default'}`}
              aria-current={active ? 'step' : undefined}
            >
              <span
                className={`w-6 h-6 rounded-full border flex items-center justify-center font-mono text-[10px] transition-colors duration-200
                  ${active ? 'border-brass text-brass' : done ? 'border-rule2 text-ok' : 'border-rule text-muted'}`}
              >
                {done ? <Check size={11} /> : String(i + 1).padStart(2, '0')}
              </span>
              <span className={`hidden sm:block text-micro font-mono uppercase tracking-widest ${active ? 'text-ink' : 'text-muted'}`}>
                {label}
              </span>
            </button>
            {i < steps.length - 1 && (
              <span className={`h-px flex-1 mx-1.5 sm:mx-3 ${done ? 'bg-brass/40' : 'bg-[color:var(--color-rule)]'}`} aria-hidden="true" />
            )}
          </li>
        );
      })}
    </ol>
  );
}
