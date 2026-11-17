import React from 'react';
import SegmentedControl from './ui/SegmentedControl';

// The caption look burned while the clips render, chosen at submit time.
// Mirrors the SubtitleModal controls one-to-one (same presets, same looks,
// same request fields) but without the Remotion preview — the submit form has
// no clip to preview — so a static CSS sample shows the look instead.

const FONT_OPTIONS = [
    { value: 'Anton', label: 'Anton' },
    { value: 'Montserrat ExtraBold', label: 'Montserrat' },
    { value: 'Verdana', label: 'Verdana' },
    { value: 'Arial', label: 'Arial' },
    { value: 'Impact', label: 'Impact' },
    { value: 'Helvetica', label: 'Helvetica' },
    { value: 'Georgia', label: 'Georgia' },
    { value: 'Courier New', label: 'Courier New' },
];

const COLOR_PRESETS = [
    { color: '#FFFFFF', label: 'White' },
    { color: '#FFFF00', label: 'Yellow' },
    { color: '#00FFFF', label: 'Cyan' },
    { color: '#00FF00', label: 'Green' },
    { color: '#FF0000', label: 'Red' },
    { color: '#FF69B4', label: 'Pink' },
];

const HIGHLIGHT_PRESETS = [
    { color: '#FFE500', label: 'Yellow' },
    { color: '#FFDD00', label: 'Gold' },
    { color: '#A3FF12', label: 'Lime' },
    { color: '#7C3AED', label: 'Purple' },
    { color: '#FF4444', label: 'Red' },
    { color: '#00BBFF', label: 'Blue' },
    { color: '#00FF88', label: 'Green' },
    { color: '#FF69B4', label: 'Pink' },
];

const ANIMATION_OPTIONS = [
    { value: 'pop', label: 'Pop' },
    { value: 'word-highlight', label: 'Glow' },
    { value: 'karaoke', label: 'Box' },
    { value: 'none', label: 'None' },
];

// Same mapping as SubtitleModal: the preview animation name vs the effect
// burned server-side ('karaoke' in the preview is the 'highlight' effect).
const ANIMATION_TO_EFFECT = { pop: 'pop', 'word-highlight': 'glow', karaoke: 'highlight', none: 'none' };
const EFFECT_TO_ANIMATION = { pop: 'pop', glow: 'word-highlight', highlight: 'karaoke', box: 'karaoke', none: 'none' };

const SIZE_OPTIONS = [
    { value: 34, label: 'S' },
    { value: 44, label: 'M' },
    { value: 56, label: 'L' },
    { value: 70, label: 'XL' },
];

// Characters per line at size M, per font — mirrors subtitles._LINE_CHARS.
const LINE_CHARS = { Anton: 16, 'Montserrat ExtraBold': 9, Impact: 16 };

export const lineBudget = (fontName, fontSize, oneWord) =>
    oneWord ? 1 : Math.max(6, Math.round((LINE_CHARS[fontName] || 14) * 44 / fontSize));

const POSITION_OPTIONS = [
    { value: 'top', label: 'top' },
    { value: 'middle', label: 'middle' },
    { value: 'bottom', label: 'bottom' },
];

// Same looks the captions modal offers, so a look picked at submit is what
// the modal would burn too.
export const CAPTION_PRESETS = [
    { id: 'default', label: 'Default',  style: 'karaoke', effect: 'pop',       highlightColor: '#FFE500', baseOpacity: 1.0, uppercase: true,  fontName: 'Anton', borderWidth: 4, fontSize: 44 },
    { id: 'hormozi', label: 'Hormozi',  style: 'karaoke', effect: 'pop',       highlightColor: '#FFE500', baseOpacity: 1.0, uppercase: true,  fontName: 'Montserrat ExtraBold', borderWidth: 4, shadow: 2, reveal: true, fontSize: 44 },
    { id: 'pill',    label: 'Pill',     style: 'karaoke', effect: 'highlight', highlightColor: '#7C3AED', baseOpacity: 1.0, uppercase: true,  fontName: 'Montserrat ExtraBold', borderWidth: 3, fontSize: 44 },
    { id: 'oneword', label: 'One word', style: 'karaoke', effect: 'pop',       highlightColor: '#FFFFFF', baseOpacity: 1.0, uppercase: true,  fontName: 'Anton', borderWidth: 5, oneWord: true, fontSize: 70 },
    { id: 'clean',   label: 'Clean',    style: 'karaoke', effect: 'none',      highlightColor: '#FFFFFF', baseOpacity: 0.7, uppercase: false, fontName: 'Montserrat ExtraBold', borderWidth: 0, shadow: 2, fontSize: 34 },
    { id: 'lime',    label: 'Lime box', style: 'karaoke', effect: 'highlight', highlightColor: '#A3FF12', baseOpacity: 1.0, uppercase: true,  fontName: 'Montserrat ExtraBold', borderWidth: 3, fontSize: 44 },
    { id: 'tiktok',  label: 'TikTok',     style: 'karaoke', effect: 'none', highlightColor: '#FE2C55', baseOpacity: 0.75, uppercase: false, fontName: 'Verdana', borderWidth: 2 },
    { id: 'reels',   label: 'Reels',      style: 'karaoke', effect: 'none', highlightColor: '#E1306C', baseOpacity: 0.7,  uppercase: false, fontName: 'Verdana', borderWidth: 2 },
    { id: 'shorts',  label: 'Shorts Pop', style: 'karaoke', effect: 'pop',  highlightColor: '#FF0000', baseOpacity: 0.7,  uppercase: false, fontName: 'Verdana', borderWidth: 2 },
    { id: 'gold',    label: 'Gold Glow',  style: 'karaoke', effect: 'glow', highlightColor: '#FFD700', baseOpacity: 0.6,  uppercase: false, fontName: 'Verdana', borderWidth: 2 },
    { id: 'neon',    label: 'Neon',       style: 'karaoke', effect: 'glow', highlightColor: '#00FF88', baseOpacity: 0.55, uppercase: false, fontName: 'Verdana', borderWidth: 2 },
    { id: 'cyber',   label: 'Cyber',      style: 'karaoke', effect: 'glow', highlightColor: '#00FFFF', baseOpacity: 0.5,  uppercase: false, fontName: 'Verdana', borderWidth: 2 },
    { id: 'karaoke', label: 'Karaoke',    style: 'karaoke', effect: 'none', highlightColor: '#FF6B6B', baseOpacity: 0.6,  uppercase: false, fontName: 'Verdana', borderWidth: 2 },
    { id: 'minimal', label: 'Minimal',    style: 'karaoke', effect: 'none', highlightColor: '#FFFFFF', baseOpacity: 0.65, uppercase: false, fontName: 'Verdana', borderWidth: 1 },
    { id: 'beast',   label: 'Beast',      style: 'karaoke', effect: 'pop',  highlightColor: '#FFD700', baseOpacity: 1.0,  uppercase: true,  fontName: 'Impact',  borderWidth: 3 },
    { id: 'boxed',   label: 'Boxed',      style: 'karaoke', effect: 'box',  highlightColor: '#7C3AED', baseOpacity: 0.85, uppercase: false, fontName: 'Verdana', borderWidth: 2 },
    { id: 'classic', label: 'Classic',    style: 'classic', effect: 'none', highlightColor: '#FFD700', baseOpacity: 1.0,  uppercase: false, fontName: 'Verdana', borderWidth: 2 },
];

// Mirrors subtitles._luminance (Rec. 601).
const luminance = (hex) => {
    const m = /^#?([0-9a-fA-F]{6})$/.exec(hex || '');
    if (!m) return 1;
    const [r, g, b] = [0, 2, 4].map((i) => parseInt(m[1].slice(i, i + 2), 16));
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255;
};

const swatchClass = (selected) =>
    `w-6 h-6 rounded-full transition-all ${selected
        ? 'ring-2 ring-[color:var(--color-accent)] ring-offset-2 ring-offset-[color:var(--color-paper-2)]'
        : 'ring-1 ring-[color:var(--color-rule-2)] hover:ring-[color:var(--color-accent)]'}`;

// The defaults are the AUTO_CAPTION_STYLE look every clip ships with, so
// applying without touching anything reproduces the built-in burn.
export const DEFAULT_CAPTION_STYLE = {
    position: 'bottom',
    fontSize: 44,
    fontName: 'Anton',
    fontColor: '#FFFFFF',
    highlightColor: '#FFE500',
    borderColor: '#000000',
    borderWidth: 4,
    bgColor: '#000000',
    bgOpacity: 0.0,
    style: 'karaoke',
    effect: 'pop',
    baseOpacity: 1.0,
    uppercase: true,
    reveal: false,
    shadow: 0,
    oneWord: false,
};

const Toggle = ({ checked, onChange }) => (
    <label className="relative inline-flex items-center cursor-pointer">
        <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} className="sr-only peer" />
        <div className="w-8 h-4 rounded-full bg-paper3 peer-checked:bg-brass transition-colors after:content-[''] after:absolute after:top-0 after:left-0 after:h-4 after:w-4 after:rounded-full after:bg-ink after:transition-all peer-checked:after:translate-x-full"></div>
    </label>
);

// The request-field object /api/process `caption_style` expects (the same
// fields the modal sends to /api/subtitle).
export function toRequestFields(value) {
    const s = { ...DEFAULT_CAPTION_STYLE, ...value };
    return {
        style: s.style,
        position: s.position,
        font_size: s.fontSize,
        font_name: s.fontName,
        font_color: s.fontColor,
        border_color: s.borderColor,
        border_width: s.borderWidth,
        bg_color: s.bgColor,
        bg_opacity: s.bgOpacity,
        highlight_color: s.highlightColor,
        effect: s.effect,
        base_opacity: s.baseOpacity,
        uppercase: s.uppercase,
        reveal: !s.oneWord && s.reveal,
        shadow: s.shadow,
        max_chars: lineBudget(s.fontName, s.fontSize, s.oneWord),
        max_duration: 1.4,
    };
}

export default function CaptionStyleEditor({ value, onChange }) {
    const s = { ...DEFAULT_CAPTION_STYLE, ...value };
    const set = (patch) => onChange({ ...s, ...patch });

    const karaoke = s.style === 'karaoke';
    const animation = EFFECT_TO_ANIMATION[s.effect] || 'none';

    const setAnimation = (anim) => {
        if (karaoke) set({ effect: ANIMATION_TO_EFFECT[anim] || 'none' });
    };

    const applyPreset = (p) => {
        set({
            preset: p.id,
            style: p.style,
            effect: p.effect,
            highlightColor: p.highlightColor,
            baseOpacity: p.baseOpacity,
            uppercase: p.uppercase,
            fontName: p.fontName,
            borderWidth: p.borderWidth,
            fontColor: '#FFFFFF',
            bgOpacity: 0,
            reveal: !!p.reveal,
            shadow: p.shadow || 0,
            oneWord: !!p.oneWord,
            ...(p.fontSize ? { fontSize: p.fontSize } : {}),
        });
    };

    // Static sample of the burned look (same shapes as the modal's fallback):
    // dim words around the active word with its effect on top.
    const bw = Math.max(s.borderWidth, 0);
    const outlineShadow = bw > 0 ? [
        `-${bw}px -${bw}px 0 ${s.borderColor}`, `${bw}px -${bw}px 0 ${s.borderColor}`,
        `-${bw}px ${bw}px 0 ${s.borderColor}`, `${bw}px ${bw}px 0 ${s.borderColor}`,
        `0 -${bw}px 0 ${s.borderColor}`, `0 ${bw}px 0 ${s.borderColor}`,
        `-${bw}px 0 0 ${s.borderColor}`, `${bw}px 0 0 ${s.borderColor}`,
    ].join(', ') : 'none';

    // Dim inactive words via a scaled color (the ASS burn dims by RGB, not
    // alpha — subtitles._dim_hex_color).
    const dimColor = (hex) => {
        const m = /^#?([0-9a-fA-F]{6})$/.exec(hex || '');
        if (!m) return hex;
        const factor = 0.5 + 0.5 * s.baseOpacity;
        const [r, g, b] = [0, 2, 4].map((i) => Math.min(255, Math.round(parseInt(m[1].slice(i, i + 2), 16) * factor)));
        return `rgb(${r},${g},${b})`;
    };

    const onBox = luminance(s.highlightColor) > 0.6 ? '#000000' : s.fontColor;
    const activeStyle = {
        padding: s.effect === 'highlight' ? '2px 6px' : 0,
        borderRadius: 4,
        ...(s.effect === 'highlight'
            ? { backgroundColor: s.highlightColor, color: onBox }
            : s.effect === 'pop'
                ? { color: s.highlightColor }
                : s.effect === 'glow'
                    ? { color: s.highlightColor, textShadow: `0 0 6px ${s.highlightColor}, 0 0 12px ${s.highlightColor}` }
                    : {}),
        display: 'inline-block',
        transform: s.effect === 'pop' ? 'scale(1.15)' : 'none',
    };
    const baseStyle = { color: karaoke ? dimColor(s.fontColor) : s.fontColor };
    const hidden = karaoke && s.reveal;

    return (
        <div className="space-y-4">
            {/* Static preview */}
            <div
                className={`relative bg-black rounded-card border border-rule flex items-center justify-center overflow-hidden ${s.position === 'top' ? 'items-start' : s.position === 'middle' ? '' : 'items-end'}`}
                style={{ aspectRatio: '9 / 16', maxHeight: '180px', minHeight: '120px' }}
            >
                <div
                    className="px-4 text-center max-w-[85%] text-sm font-bold leading-snug"
                    style={{
                        fontFamily: s.fontName,
                        ...(karaoke && s.bgOpacity > 0
                            ? { backgroundColor: `${s.bgColor}${Math.round(s.bgOpacity * 255).toString(16).padStart(2, '0')}`, textShadow: 'none', padding: '6px 12px', borderRadius: 4 }
                            : { textShadow: outlineShadow }),
                        ...(!karaoke
                            ? { backgroundColor: s.bgOpacity > 0 ? `${s.bgColor}${Math.round(s.bgOpacity * 255).toString(16).padStart(2, '0')}` : undefined }
                            : {}),
                    }}
                >
                    {!hidden && <span style={baseStyle}>{s.uppercase ? 'WATCH THIS ' : 'watch this '}</span>}
                    <span style={{ ...activeStyle, display: hidden ? 'inline' : 'inline-block' }}>WORD</span>
                    {!hidden && <span style={baseStyle}>{s.uppercase ? ' NEXT' : ' next'}</span>}
                </div>
            </div>

            {/* Presets */}
            <div>
                <p className="eyebrow mb-2">Preset</p>
                <div className="grid grid-cols-3 gap-1.5">
                    {CAPTION_PRESETS.map((p) => (
                        <button
                            key={p.id}
                            type="button"
                            onClick={() => applyPreset(p)}
                            className={`px-2 py-1.5 rounded-input border text-xs transition-colors flex items-center gap-1.5 justify-center
                                ${s.preset === p.id
                                    ? 'border-[color:var(--color-accent)] text-ink'
                                    : 'border-rule2 text-muted hover:border-[color:var(--color-accent)]'}`}
                            title={p.label}
                        >
                            <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: p.highlightColor }} />
                            {p.label}
                        </button>
                    ))}
                </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                <div>
                    <p className="eyebrow mb-2">Position</p>
                    <SegmentedControl
                        options={POSITION_OPTIONS}
                        value={s.position}
                        onChange={(v) => set({ position: v })}
                        size="sm"
                    />
                </div>
                <div>
                    <p className="eyebrow mb-2">Size</p>
                    <SegmentedControl
                        options={SIZE_OPTIONS}
                        value={s.fontSize}
                        onChange={(v) => set({ fontSize: v })}
                        size="sm"
                    />
                </div>
                <div>
                    <p className="eyebrow mb-2">Animation</p>
                    <SegmentedControl
                        options={ANIMATION_OPTIONS}
                        value={animation}
                        onChange={setAnimation}
                        columns={2}
                        size="sm"
                    />
                </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                <div>
                    <p className="eyebrow mb-2">Font</p>
                    <select
                        value={s.fontName}
                        onChange={(e) => set({ fontName: e.target.value })}
                        className="input-field"
                    >
                        {FONT_OPTIONS.map((f) => (
                            <option key={f.value} value={f.value} style={{ fontFamily: f.value }}>{f.label}</option>
                        ))}
                    </select>
                </div>
                <div>
                    <p className="eyebrow mb-2">Highlight</p>
                    <div className="flex flex-wrap items-center gap-2.5">
                        {HIGHLIGHT_PRESETS.map((c) => (
                            <button
                                key={c.color}
                                type="button"
                                onClick={() => set({ highlightColor: c.color })}
                                className={swatchClass(s.highlightColor === c.color)}
                                style={{ backgroundColor: c.color }}
                                title={c.label}
                            />
                        ))}
                    </div>
                </div>
            </div>

            <div>
                <p className="eyebrow mb-2">Text color</p>
                <div className="flex flex-wrap items-center gap-2.5">
                    {COLOR_PRESETS.map((c) => (
                        <button
                            key={c.color}
                            type="button"
                            onClick={() => set({ fontColor: c.color })}
                            className={swatchClass(s.fontColor === c.color)}
                            style={{ backgroundColor: c.color }}
                            title={c.label}
                        />
                    ))}
                    <label className="w-6 h-6 rounded-full border border-dashed border-rule2 cursor-pointer flex items-center justify-center hover:border-brass transition-colors overflow-hidden relative" title="Custom color">
                        <span className="text-xs text-muted leading-none">+</span>
                        <input type="color" value={s.fontColor} onChange={(e) => set({ fontColor: e.target.value })} className="absolute inset-0 opacity-0 cursor-pointer" />
                    </label>
                </div>
            </div>

            <div>
                <p className="eyebrow mb-2">Border</p>
                <div className="flex items-center gap-3">
                    <label className="relative w-8 h-8 rounded-input border border-rule2 cursor-pointer overflow-hidden shrink-0" title="Border color">
                        <div className="w-full h-full" style={{ backgroundColor: s.borderColor }} />
                        <input type="color" value={s.borderColor} onChange={(e) => set({ borderColor: e.target.value })} className="absolute inset-0 opacity-0 cursor-pointer" />
                    </label>
                    <div className="flex-1">
                        <input
                            type="range"
                            min="0"
                            max="5"
                            value={s.borderWidth}
                            onChange={(e) => set({ borderWidth: parseInt(e.target.value) })}
                            className="w-full accent-[var(--color-accent)]"
                        />
                        <div className="flex justify-between">
                            <span className="readout">None</span>
                            <span className="readout">Thick</span>
                        </div>
                    </div>
                </div>
            </div>

            {karaoke && (
                <div className="space-y-3">
                    <div className="flex items-center justify-between">
                        <span className="readout">Uppercase</span>
                        <Toggle checked={s.uppercase} onChange={(v) => set({ uppercase: v })} />
                    </div>
                    <div>
                        <div className="flex justify-between mb-1">
                            <span className="readout">Dim inactive words</span>
                            <span className="readout">{Math.round(s.baseOpacity * 100)}%</span>
                        </div>
                        <input
                            type="range"
                            min="30"
                            max="100"
                            value={Math.round(s.baseOpacity * 100)}
                            onChange={(e) => set({ baseOpacity: parseInt(e.target.value) / 100 })}
                            className="w-full accent-[var(--color-accent)]"
                        />
                    </div>
                    <div className="flex items-center justify-between">
                        <span className="readout">One word at a time</span>
                        <Toggle checked={s.oneWord} onChange={(v) => set({ oneWord: v })} />
                    </div>
                    {!s.oneWord && (
                        <div className="flex items-center justify-between">
                            <span className="readout">Reveal word by word</span>
                            <Toggle checked={s.reveal} onChange={(v) => set({ reveal: v })} />
                        </div>
                    )}
                    <div>
                        <div className="flex justify-between mb-1">
                            <span className="readout">Shadow</span>
                            <span className="readout">{s.shadow}</span>
                        </div>
                        <input
                            type="range"
                            min="0"
                            max="6"
                            value={s.shadow}
                            onChange={(e) => set({ shadow: parseInt(e.target.value) })}
                            className="w-full accent-[var(--color-accent)]"
                        />
                    </div>
                </div>
            )}

            <div>
                <div className="flex items-center justify-between mb-2">
                    <p className="eyebrow">Background</p>
                    <Toggle
                        checked={s.bgOpacity > 0}
                        onChange={(v) => set({ bgOpacity: v ? 0.5 : 0 })}
                    />
                </div>
                {s.bgOpacity > 0 && (
                    <div className="flex items-center gap-3 animate-fade">
                        <label className="relative w-8 h-8 rounded-input border border-rule2 cursor-pointer overflow-hidden shrink-0" title="Background color">
                            <div className="w-full h-full" style={{ backgroundColor: s.bgColor }} />
                            <input type="color" value={s.bgColor} onChange={(e) => set({ bgColor: e.target.value })} className="absolute inset-0 opacity-0 cursor-pointer" />
                        </label>
                        <div className="flex-1">
                            <input
                                type="range"
                                min="10"
                                max="100"
                                value={Math.round(s.bgOpacity * 100)}
                                onChange={(e) => set({ bgOpacity: parseInt(e.target.value) / 100 })}
                                className="w-full accent-[var(--color-accent)]"
                            />
                            <div className="flex justify-between">
                                <span className="readout">Transparent</span>
                                <span className="readout">{Math.round(s.bgOpacity * 100)}%</span>
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
