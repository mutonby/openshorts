import React from 'react';

const POSITIONS = [
  'top-left', 'top', 'top-right',
  'left', 'center', 'right',
  'bottom-left', 'bottom', 'bottom-right',
];

export default function WatermarkControls({ value, onChange }) {
  const update = (patch) => onChange({ ...value, ...patch });
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between gap-3">
        <p className="eyebrow">Text watermark</p>
        <label className="flex items-center gap-2 text-xs text-ink2 cursor-pointer select-none">
          <input
            type="checkbox"
            checked={!!value.enabled}
            onChange={(e) => update({ enabled: e.target.checked })}
            className="w-4 h-4 accent-[var(--color-accent)] cursor-pointer"
          />
          on
        </label>
      </div>
      <input
        className="input-field"
        maxLength={80}
        placeholder="@yourchannel"
        value={value.text || ''}
        onChange={(e) => update({ text: e.target.value })}
      />
      <div className="flex gap-4 items-start">
        <div className="flex-1 space-y-3 min-w-0">
          <div>
            <div className="flex justify-between mb-1">
              <span className="readout">Opacity</span>
              <span className="readout">{Math.round((value.opacity ?? 0.5) * 100)}%</span>
            </div>
            <input
              type="range" min="5" max="100" value={Math.round((value.opacity ?? 0.5) * 100)}
              onChange={(e) => update({ opacity: Number(e.target.value) / 100 })}
              className="w-full accent-[var(--color-accent)]"
            />
          </div>
          <div>
            <div className="flex justify-between mb-1">
              <span className="readout">Size</span>
              <span className="readout">{value.size ?? 4}% of height</span>
            </div>
            <input
              type="range" min="1" max="15" step="0.5" value={value.size ?? 4}
              onChange={(e) => update({ size: Number(e.target.value) })}
              className="w-full accent-[var(--color-accent)]"
            />
          </div>
        </div>
        <div className="grid grid-cols-3 gap-1 w-24 shrink-0" title="Position">
          {POSITIONS.map((p) => (
            <button
              key={p}
              type="button"
              title={p}
              onClick={() => update({ position: p })}
              className={`h-7 rounded-input border transition-colors flex items-center justify-center
                ${value.position === p ? 'border-[color:var(--color-accent)] bg-brass/20' : 'border-rule2 hover:border-[color:var(--color-accent)]'}`}
            >
              <span className={`w-1.5 h-1.5 rounded-full ${value.position === p ? 'bg-brass' : 'bg-muted'}`} />
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
