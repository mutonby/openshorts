import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Smartphone, Monitor, MessageSquare, RotateCcw, Pause, Play } from 'lucide-react';
import { wrapByWords, applyTextCase, DEFAULT_PREVIEW_TEXT } from '../lib/brandKit';

// Live preview canvas that mirrors how subtitles actually flash on a video:
// ONE chunk on screen at a time. We split the user's text into chunks of N
// words (N = brand kit's wordsPerLine) and auto-cycle through them like real
// SRT playback. Dots beneath the canvas let the user jump to any chunk.

const FRAME_CONFIG = {
  '9:16': { w: 270, h: 480, scaleBase: 1920 },
  '16:9': { w: 480, h: 270, scaleBase: 1080 },
};

const CYCLE_MS = 1500;

function anchorToCoords(position, w, h) {
  const [vert, horiz] = position.split('-');
  const padX = w * 0.06;
  const padY = h * 0.08;
  let x, textAlign;
  if (horiz === 'left')        { x = padX;     textAlign = 'left'; }
  else if (horiz === 'right')  { x = w - padX; textAlign = 'right'; }
  else                          { x = w / 2;    textAlign = 'center'; }
  let yBand;
  if (vert === 'top')          yBand = padY;
  else if (vert === 'middle')  yBand = h / 2;
  else                          yBand = h - padY;
  return { x, yBand, textAlign, vert };
}

export default function BrandPreview({ brandKit, activeRatio, onRatioChange, onPreviewTextChange }) {
  const canvasRef = useRef(null);
  const ratio = activeRatio || '9:16';
  const previewText = brandKit.previewText ?? DEFAULT_PREVIEW_TEXT;
  const style = brandKit.styles[ratio];

  // Split the user's text into chunks of N words — each chunk is one
  // "subtitle moment" on screen, just like the real burn-in.
  const chunks = useMemo(
    () => wrapByWords(previewText, style.wordsPerLine),
    [previewText, style.wordsPerLine]
  );

  const [chunkIdx, setChunkIdx] = useState(0);
  const [paused, setPaused] = useState(false);

  // Clamp index when chunks change (e.g., user dragged words-per-line)
  useEffect(() => {
    if (chunkIdx >= chunks.length) setChunkIdx(0);
  }, [chunks.length, chunkIdx]);

  // Auto-cycle
  useEffect(() => {
    if (paused || chunks.length <= 1) return;
    const id = setInterval(() => {
      setChunkIdx(i => (i + 1) % chunks.length);
    }, CYCLE_MS);
    return () => clearInterval(id);
  }, [paused, chunks.length]);

  const draw = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const { w, h, scaleBase } = FRAME_CONFIG[ratio];
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');

    // Mock frame background
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, '#1a1a2e');
    grad.addColorStop(0.5, '#16213e');
    grad.addColorStop(1, '#0f3460');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = 'rgba(255,255,255,0.08)';
    ctx.beginPath();
    ctx.arc(w / 2, h * 0.35, h * 0.18, 0, Math.PI * 2);
    ctx.fill();

    const { font } = brandKit;
    const previewScale = h / scaleBase;
    const fontSize = Math.max(10, Math.round(style.size * previewScale * 2.6));
    const strokeWidth = Math.max(0, Math.round(style.strokeWidth * previewScale * 2.6));

    ctx.font = `bold ${fontSize}px "${font.family}", system-ui, sans-serif`;
    ctx.lineJoin = 'round';

    // The "current moment" — exactly one chunk on screen, with brand-kit case applied.
    const text = applyTextCase(chunks[chunkIdx] ?? '', style.textCase);

    const { x, yBand, textAlign, vert } = anchorToCoords(style.position, w, h);
    ctx.textAlign = textAlign;
    ctx.textBaseline = 'middle';
    const lineHeight = fontSize * 1.15;
    let y;
    if (vert === 'top')        y = yBand + lineHeight / 2;
    else if (vert === 'middle') y = yBand;
    else                        y = yBand - lineHeight / 2;

    if (strokeWidth > 0) {
      ctx.strokeStyle = style.strokeColor;
      ctx.lineWidth = strokeWidth;
      ctx.strokeText(text, x, y);
    }
    ctx.fillStyle = style.textColor;
    ctx.fillText(text, x, y);
  };

  useEffect(() => {
    draw();
    if (document.fonts && document.fonts.ready) {
      document.fonts.ready.then(draw);
    }
  }, [brandKit, ratio, chunkIdx, chunks]);

  const wordCount = previewText.trim().split(/\s+/).filter(Boolean).length;

  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <div className="text-sm font-semibold text-zinc-300">Live preview</div>
        <div className="flex gap-1 p-1 bg-white/5 rounded-lg">
          {Object.entries(FRAME_CONFIG).map(([key]) => {
            const Icon = key === '9:16' ? Smartphone : Monitor;
            return (
              <button
                key={key}
                type="button"
                onClick={() => onRatioChange?.(key)}
                className={`flex items-center gap-1.5 px-3 py-1 rounded text-xs font-medium transition-colors ${ratio === key ? 'bg-primary text-white' : 'text-zinc-400 hover:text-white'}`}
              >
                <Icon size={12} />
                {key}
              </button>
            );
          })}
        </div>
      </div>

      {/* Editable preview text */}
      <div className="mb-3">
        <label className="flex items-center justify-between text-xs text-zinc-400 mb-1.5">
          <span className="flex items-center gap-1.5">
            <MessageSquare size={12} /> Try your own text
          </span>
          <span className="flex items-center gap-2">
            <span className="font-mono">{wordCount} {wordCount === 1 ? 'word' : 'words'}</span>
            {previewText !== DEFAULT_PREVIEW_TEXT && (
              <button
                type="button"
                onClick={() => onPreviewTextChange?.(DEFAULT_PREVIEW_TEXT)}
                className="text-zinc-500 hover:text-white flex items-center gap-1"
              >
                <RotateCcw size={10} /> reset
              </button>
            )}
          </span>
        </label>
        <textarea
          value={previewText}
          onChange={(e) => onPreviewTextChange?.(e.target.value)}
          rows={2}
          placeholder="Type a sample sentence to see how it wraps…"
          className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-zinc-200 placeholder-zinc-600 focus:outline-none focus:border-primary resize-y"
        />
      </div>

      {/* Canvas */}
      <div className="flex items-center justify-center bg-black/50 border border-white/10 rounded-xl p-6">
        <canvas
          ref={canvasRef}
          className="rounded-md shadow-2xl"
          style={{ maxWidth: '100%', height: 'auto' }}
        />
      </div>

      {/* Chunk navigation */}
      <div className="mt-3 flex items-center justify-between gap-3">
        <button
          type="button"
          onClick={() => setPaused(p => !p)}
          className="flex items-center gap-1 px-2 py-1 text-xs text-zinc-400 hover:text-white border border-white/10 hover:border-white/20 rounded transition-colors"
        >
          {paused ? <><Play size={10} /> Play</> : <><Pause size={10} /> Pause</>}
        </button>

        <div className="flex-1 flex items-center justify-center gap-1.5 flex-wrap">
          {chunks.map((chunk, i) => (
            <button
              key={i}
              type="button"
              onClick={() => { setChunkIdx(i); setPaused(true); }}
              title={chunk}
              className={`h-1.5 rounded-full transition-all ${i === chunkIdx ? 'bg-primary w-8' : 'bg-white/15 hover:bg-white/30 w-2'}`}
            />
          ))}
        </div>

        <div className="text-xs text-zinc-500 font-mono shrink-0">
          {chunkIdx + 1}/{chunks.length}
        </div>
      </div>

      <p className="text-[11px] text-zinc-500 mt-2 text-center leading-relaxed">
        Each <span className="text-zinc-300">chunk = {style.wordsPerLine || '∞'} word{style.wordsPerLine === 1 ? '' : 's'}</span> on screen at once — exactly what your subtitle burn-in produces. Cycling automatically every {(CYCLE_MS / 1000).toFixed(1)}s.
      </p>
    </div>
  );
}
