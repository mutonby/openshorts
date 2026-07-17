/**
 * The single option-toggle group (design.md). Replaces every ad-hoc
 * grid of toggle buttons (position/size/animation pickers, platform
 * selectors, mode switches).
 *
 * Props:
 *  - options: [{ value, label, icon?, hint?, disabled? }]
 *  - value: selected value (or array when multi)
 *  - onChange(value)
 *  - multi: allow multiple selections (value must be an array)
 *  - columns: grid column count (default: options.length, capped at 4)
 *  - size: 'sm' | 'md'
 */
export default function SegmentedControl({ options, value, onChange, multi = false, columns, size = 'md' }) {
  const cols = columns || Math.min(options.length, 4);
  const isActive = (v) => (multi ? Array.isArray(value) && value.includes(v) : value === v);

  const toggle = (v) => {
    if (!multi) return onChange(v);
    const arr = Array.isArray(value) ? value : [];
    onChange(arr.includes(v) ? arr.filter((x) => x !== v) : [...arr, v]);
  };

  const pad = size === 'sm' ? 'px-2.5 py-1.5' : 'px-3 py-2.5';

  // Responsive tracks: `cols` columns when they fit, wrapping to fewer on
  // narrow containers so options never crush below a readable width.
  const minCol = size === 'sm' ? 48 : 88;
  const gridTemplateColumns = `repeat(auto-fill, minmax(max(${minCol}px, calc((100% - ${(cols - 1) * 6}px) / ${cols})), 1fr))`;

  return (
    <div className="grid gap-1.5" style={{ gridTemplateColumns }} role={multi ? 'group' : 'radiogroup'}>
      {options.map((opt) => {
        const active = isActive(opt.value);
        return (
          <button
            key={opt.value}
            type="button"
            role={multi ? 'checkbox' : 'radio'}
            aria-checked={active}
            disabled={opt.disabled}
            onClick={() => toggle(opt.value)}
            className={`${pad} rounded-input border text-xs lowercase flex flex-col items-center justify-center gap-1 transition-colors duration-200
              ${active
                ? 'border-brass bg-paper3 text-ink'
                : 'border-rule bg-paper text-muted hover:text-ink2 hover:border-rule2'}
              disabled:opacity-40 disabled:cursor-not-allowed`}
          >
            {opt.icon && <span className={active ? 'text-brass' : 'text-muted'}>{opt.icon}</span>}
            <span className="font-medium">{opt.label}</span>
            {opt.hint && <span className="readout normal-case">{opt.hint}</span>}
          </button>
        );
      })}
    </div>
  );
}
