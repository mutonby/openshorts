import React from 'react';
import { ALL_POSITIONS } from '../lib/brandKit';

// 3x3 anchor selector. Each cell represents the text's anchor point on the
// video frame. Same UI pattern Figma uses for object alignment.

const LABELS = {
  'top-left':      'Top L',
  'top-center':    'Top',
  'top-right':     'Top R',
  'middle-left':   'Left',
  'middle-center': 'Center',
  'middle-right':  'Right',
  'bottom-left':   'Bot L',
  'bottom-center': 'Bottom',
  'bottom-right':  'Bot R',
};

export default function PositionGrid({ value, onChange, label = 'Text position' }) {
  return (
    <div>
      {label && (
        <label className="text-sm font-semibold text-zinc-200 mb-2 block">
          {label}
          <span className="ml-2 text-zinc-500 font-normal text-xs">
            {LABELS[value] || value}
          </span>
        </label>
      )}
      <div className="inline-block p-2 bg-white/5 border border-white/10 rounded-xl">
        <div className="grid grid-cols-3 gap-1 w-32 aspect-[9/16]">
          {ALL_POSITIONS.map((pos) => {
            const isActive = value === pos;
            return (
              <button
                key={pos}
                type="button"
                onClick={() => onChange(pos)}
                title={LABELS[pos]}
                aria-label={LABELS[pos]}
                className={`relative rounded transition-all flex items-center justify-center ${
                  isActive
                    ? 'bg-primary text-white shadow-md scale-105'
                    : 'bg-white/5 hover:bg-white/15 text-zinc-500 hover:text-zinc-200'
                }`}
              >
                <span className={`w-1.5 h-1.5 rounded-full ${isActive ? 'bg-white' : 'bg-current'}`} />
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}
