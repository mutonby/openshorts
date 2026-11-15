import React from 'react';

const PLACE = {
  'top-left': 'items-start justify-start text-left',
  top: 'items-start justify-center text-center',
  'top-right': 'items-start justify-end text-right',
  left: 'items-center justify-start text-left',
  center: 'items-center justify-center text-center',
  right: 'items-center justify-end text-right',
  'bottom-left': 'items-end justify-start text-left',
  bottom: 'items-end justify-center text-center',
  'bottom-right': 'items-end justify-end text-right',
};

export default function WatermarkOverlay({ value, height, width }) {
  if (!value?.enabled || !(value.text || '').trim()) return null;
  const fontPx = Math.max(6, height * (value.size ?? 4) / 100);
  const margin = Math.max(4, Math.min(width, height) * 0.04);
  return (
    <div
      className={`absolute inset-0 flex pointer-events-none z-10 ${PLACE[value.position] || PLACE['bottom-right']}`}
      style={{ padding: margin }}
    >
      <span
        className="font-extrabold leading-tight break-words"
        style={{
          fontFamily: 'Montserrat, sans-serif',
          fontSize: fontPx,
          color: `rgba(255,255,255,${value.opacity ?? 0.5})`,
          WebkitTextStroke: `${Math.max(0.5, fontPx * 0.07)}px rgba(0,0,0,${value.opacity ?? 0.5})`,
          maxWidth: '100%',
        }}
      >
        {value.text}
      </span>
    </div>
  );
}
