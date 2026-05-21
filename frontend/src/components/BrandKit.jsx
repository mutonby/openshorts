import React, { useEffect, useState } from 'react';
import { Palette, Plus, X, RotateCcw, Smartphone, Monitor, TextSelect, CaseUpper, CaseLower, CaseSensitive } from 'lucide-react';
import { DEFAULT_BRAND_KIT, loadBrandKit, saveBrandKit, resetBrandKit, ensureFontLoaded } from '../lib/brandKit';
import FontPicker from './FontPicker';
import BrandPreview from './BrandPreview';
import PositionGrid from './PositionGrid';

const RATIO_META = {
  '9:16': { label: 'Shorts',  icon: Smartphone },
  '16:9': { label: 'YouTube', icon: Monitor },
};

export default function BrandKit() {
  const [kit, setKit] = useState(() => loadBrandKit());
  const [ratio, setRatio] = useState('9:16');

  useEffect(() => {
    saveBrandKit(kit);
    ensureFontLoaded(kit.font);
  }, [kit]);

  const style = kit.styles[ratio];

  const updateStyle = (patch) => setKit(k => ({
    ...k,
    styles: { ...k.styles, [ratio]: { ...k.styles[ratio], ...patch } },
  }));
  const updateFont = (f) => setKit(k => ({ ...k, font: f }));
  const updateColor  = (i, hex)  => setKit(k => ({ ...k, colors: k.colors.map((c, idx) => idx === i ? { ...c, hex }  : c) }));
  const renameColor  = (i, name) => setKit(k => ({ ...k, colors: k.colors.map((c, idx) => idx === i ? { ...c, name } : c) }));
  const addColor     = ()        => setKit(k => ({ ...k, colors: [...k.colors, { name: `Color ${k.colors.length + 1}`, hex: '#FF6B6B' }] }));
  const removeColor  = (i)       => setKit(k => ({ ...k, colors: k.colors.filter((_, idx) => idx !== i) }));

  const handleReset = () => {
    if (window.confirm('Reset brand kit to defaults?')) {
      resetBrandKit();
      setKit(DEFAULT_BRAND_KIT);
    }
  };

  return (
    <div className="bg-surface border border-white/5 rounded-2xl p-6 mt-6 animate-[fadeIn_0.5s_ease-out]">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-fuchsia-500/20 rounded-lg text-fuchsia-400">
            <Palette size={20} />
          </div>
          <div>
            <h2 className="text-lg font-semibold">Brand Kit</h2>
            <p className="text-xs text-zinc-500">Colors and font are shared; size, position, and word-wrap can differ per aspect ratio.</p>
          </div>
        </div>
        <button
          onClick={handleReset}
          className="flex items-center gap-1.5 px-3 py-1.5 text-xs text-zinc-400 hover:text-white border border-white/10 hover:border-white/20 rounded-lg transition-colors"
        >
          <RotateCcw size={12} /> Reset
        </button>
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        {/* LEFT: controls */}
        <div className="space-y-6">
          {/* Brand colors (shared) */}
          <div>
            <label className="text-sm font-semibold text-zinc-200 mb-3 flex items-center justify-between">
              <span>Brand colors</span>
              <span className="text-[10px] uppercase tracking-wider text-zinc-500">Shared</span>
            </label>
            <div className="space-y-2">
              {kit.colors.map((c, i) => (
                <div key={i} className="flex items-center gap-2">
                  <label className="relative cursor-pointer shrink-0">
                    <div className="w-10 h-10 rounded-lg border-2 border-white/10" style={{ backgroundColor: c.hex }} />
                    <input type="color" value={c.hex} onChange={(e) => updateColor(i, e.target.value.toUpperCase())} className="absolute inset-0 opacity-0 cursor-pointer" />
                  </label>
                  <input type="text" value={c.name} onChange={(e) => renameColor(i, e.target.value)} className="flex-1 px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm focus:outline-none focus:border-primary" />
                  <input type="text" value={c.hex} onChange={(e) => updateColor(i, e.target.value.toUpperCase())} className="w-24 px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm font-mono focus:outline-none focus:border-primary" />
                  {kit.colors.length > 1 && (
                    <button onClick={() => removeColor(i)} className="p-2 text-zinc-500 hover:text-red-400 transition-colors">
                      <X size={14} />
                    </button>
                  )}
                </div>
              ))}
              <button onClick={addColor} className="w-full flex items-center justify-center gap-1.5 py-2 text-xs text-zinc-400 hover:text-white border border-dashed border-white/10 hover:border-white/30 rounded-lg transition-colors">
                <Plus size={12} /> Add color
              </button>
            </div>
          </div>

          {/* Font (shared) */}
          <div>
            <label className="text-sm font-semibold text-zinc-200 mb-3 flex items-center justify-between">
              <span>Font</span>
              <span className="text-[10px] uppercase tracking-wider text-zinc-500">Shared</span>
            </label>
            <FontPicker value={kit.font} onChange={updateFont} />
          </div>

          {/* Ratio toggle */}
          <div className="border-t border-white/5 pt-4">
            <div className="flex items-center justify-between mb-3">
              <label className="text-sm font-semibold text-zinc-200">Layout settings</label>
              <div className="flex gap-1 p-1 bg-white/5 rounded-lg">
                {Object.entries(RATIO_META).map(([key, { label, icon: I }]) => (
                  <button
                    key={key}
                    onClick={() => setRatio(key)}
                    className={`flex items-center gap-1.5 px-3 py-1 rounded text-xs font-medium transition-colors ${ratio === key ? 'bg-primary text-white' : 'text-zinc-400 hover:text-white'}`}
                  >
                    <I size={12} />
                    {key}
                  </button>
                ))}
              </div>
            </div>

            {/* Size + Stroke width (per-ratio) */}
            <div className="grid grid-cols-2 gap-4 mb-4">
              <div>
                <label className="text-sm text-zinc-300 mb-2 flex items-center justify-between">
                  <span>Size</span>
                  <span className="text-zinc-500 text-xs font-mono">{style.size}px</span>
                </label>
                <input type="range" min="24" max="200" value={style.size} onChange={(e) => updateStyle({ size: parseInt(e.target.value, 10) })} className="w-full accent-fuchsia-500" />
              </div>
              <div>
                <label className="text-sm text-zinc-300 mb-2 flex items-center justify-between">
                  <span>Stroke width</span>
                  <span className="text-zinc-500 text-xs font-mono">{style.strokeWidth}px</span>
                </label>
                <input type="range" min="0" max="20" value={style.strokeWidth} onChange={(e) => updateStyle({ strokeWidth: parseInt(e.target.value, 10) })} className="w-full accent-fuchsia-500" />
              </div>
            </div>

            {/* Words per line (per-ratio) */}
            <div className="mb-4">
              <label className="text-sm text-zinc-300 mb-2 flex items-center justify-between">
                <span className="flex items-center gap-2"><TextSelect size={14} /> Words per line</span>
                <span className="text-zinc-500 text-xs font-mono">{style.wordsPerLine === 0 ? 'no wrap' : `${style.wordsPerLine} word${style.wordsPerLine === 1 ? '' : 's'}`}</span>
              </label>
              <input type="range" min="0" max="15" value={style.wordsPerLine} onChange={(e) => updateStyle({ wordsPerLine: parseInt(e.target.value, 10) })} className="w-full accent-fuchsia-500" />
              <p className="text-[11px] text-zinc-500 mt-1">
                {style.wordsPerLine === 0
                  ? 'No wrapping — single line until it overflows.'
                  : style.wordsPerLine <= 3
                    ? 'Hormozi style — short bursts, max impact.'
                    : style.wordsPerLine <= 7
                      ? 'Balanced — readable on mobile.'
                      : 'Full sentences — good for wider 16:9 frames.'}
              </p>
            </div>

            {/* Text case (per-ratio) */}
            <div className="mb-4">
              <label className="text-sm text-zinc-300 mb-2 block">Text case</label>
              <div className="flex gap-1 p-1 bg-white/5 rounded-lg">
                {[
                  { key: 'original', icon: CaseSensitive, label: 'As typed' },
                  { key: 'upper',    icon: CaseUpper,     label: 'UPPERCASE' },
                  { key: 'lower',    icon: CaseLower,     label: 'lowercase' },
                ].map(({ key, icon: I, label }) => (
                  <button
                    key={key}
                    onClick={() => updateStyle({ textCase: key })}
                    className={`flex-1 flex items-center justify-center gap-1.5 py-2 rounded text-xs font-medium transition-colors ${style.textCase === key ? 'bg-primary text-white' : 'text-zinc-400 hover:text-white'}`}
                  >
                    <I size={14} />
                    {label}
                  </button>
                ))}
              </div>
            </div>

            {/* Text color (uses brand palette) */}
            <div className="grid grid-cols-2 gap-4 mb-4">
              <div>
                <label className="text-sm text-zinc-300 mb-2 block">Text color</label>
                <div className="flex gap-2 flex-wrap">
                  {kit.colors.map((c, i) => (
                    <button key={i} onClick={() => updateStyle({ textColor: c.hex })} className={`w-8 h-8 rounded-lg border-2 transition-all ${style.textColor === c.hex ? 'border-white scale-110' : 'border-white/10 hover:border-white/30'}`} style={{ backgroundColor: c.hex }} title={c.name} />
                  ))}
                  <input type="color" value={style.textColor} onChange={(e) => updateStyle({ textColor: e.target.value.toUpperCase() })} className="w-8 h-8 rounded-lg cursor-pointer bg-transparent border border-white/10" title="Custom" />
                </div>
              </div>
              <div>
                <label className="text-sm text-zinc-300 mb-2 block">Stroke color</label>
                <div className="flex gap-2 flex-wrap">
                  {kit.colors.map((c, i) => (
                    <button key={i} onClick={() => updateStyle({ strokeColor: c.hex })} className={`w-8 h-8 rounded-lg border-2 transition-all ${style.strokeColor === c.hex ? 'border-white scale-110' : 'border-white/10 hover:border-white/30'}`} style={{ backgroundColor: c.hex }} title={c.name} />
                  ))}
                  <input type="color" value={style.strokeColor} onChange={(e) => updateStyle({ strokeColor: e.target.value.toUpperCase() })} className="w-8 h-8 rounded-lg cursor-pointer bg-transparent border border-white/10" title="Custom" />
                </div>
              </div>
            </div>

            {/* 9-anchor position grid (per-ratio) */}
            <PositionGrid value={style.position} onChange={(p) => updateStyle({ position: p })} />
          </div>
        </div>

        {/* RIGHT: live preview */}
        <div>
          <BrandPreview
            brandKit={kit}
            activeRatio={ratio}
            onRatioChange={setRatio}
            onPreviewTextChange={(text) => setKit(k => ({ ...k, previewText: text }))}
          />
          <div className="mt-3 p-3 rounded-lg bg-white/5 border border-white/10 text-xs text-zinc-400 leading-relaxed">
            Applied automatically to <span className="text-white font-medium">subtitles</span>, <span className="text-white font-medium">hook overlays</span>, and <span className="text-white font-medium">AI effect text</span>. Per-clip overrides remain possible.
          </div>
        </div>
      </div>
    </div>
  );
}
