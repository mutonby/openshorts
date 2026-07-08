import React, { useState } from 'react';
import { X, Type, Loader2 } from 'lucide-react';

const FONT_OPTIONS = [
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
    { color: '#FFD700', label: 'Gold' },
    { color: '#22C55E', label: 'Green' },
    { color: '#EC4899', label: 'Pink' },
    { color: '#38BDF8', label: 'Blue' },
    { color: '#F97316', label: 'Orange' },
    { color: '#EF4444', label: 'Red' },
];

// Ready-made caption looks (inspired by remotion-captioneer): dimmed base
// text + strong active word, optional glow/pop/box effect.
const CAPTION_PRESETS = [
    { id: 'tiktok',  label: 'TikTok',    style: 'karaoke', effect: 'none', highlightColor: '#FE2C55', baseOpacity: 0.75, uppercase: false, fontName: 'Verdana', borderWidth: 2 },
    { id: 'reels',   label: 'Reels',     style: 'karaoke', effect: 'none', highlightColor: '#E1306C', baseOpacity: 0.7,  uppercase: false, fontName: 'Verdana', borderWidth: 2 },
    { id: 'shorts',  label: 'Shorts Pop', style: 'karaoke', effect: 'pop',  highlightColor: '#FF0000', baseOpacity: 0.7,  uppercase: false, fontName: 'Verdana', borderWidth: 2 },
    { id: 'gold',    label: 'Gold Glow', style: 'karaoke', effect: 'glow', highlightColor: '#FFD700', baseOpacity: 0.6,  uppercase: false, fontName: 'Verdana', borderWidth: 2 },
    { id: 'neon',    label: 'Neon',      style: 'karaoke', effect: 'glow', highlightColor: '#00FF88', baseOpacity: 0.55, uppercase: false, fontName: 'Verdana', borderWidth: 2 },
    { id: 'cyber',   label: 'Cyber',     style: 'karaoke', effect: 'glow', highlightColor: '#00FFFF', baseOpacity: 0.5,  uppercase: false, fontName: 'Verdana', borderWidth: 2 },
    { id: 'karaoke', label: 'Karaoke',   style: 'karaoke', effect: 'none', highlightColor: '#FF6B6B', baseOpacity: 0.6,  uppercase: false, fontName: 'Verdana', borderWidth: 2 },
    { id: 'minimal', label: 'Minimal',   style: 'karaoke', effect: 'none', highlightColor: '#FFFFFF', baseOpacity: 0.65, uppercase: false, fontName: 'Verdana', borderWidth: 1 },
    { id: 'beast',   label: 'Beast',     style: 'karaoke', effect: 'pop',  highlightColor: '#FFD700', baseOpacity: 1.0,  uppercase: true,  fontName: 'Impact',  borderWidth: 3 },
    { id: 'boxed',   label: 'Boxed',     style: 'karaoke', effect: 'box',  highlightColor: '#7C3AED', baseOpacity: 0.85, uppercase: false, fontName: 'Verdana', borderWidth: 2 },
    { id: 'classic', label: 'Classic',   style: 'classic', effect: 'none', highlightColor: '#FFD700', baseOpacity: 1.0,  uppercase: false, fontName: 'Verdana', borderWidth: 2 },
];

// CSS approximation of the ASS effects for previews
const effectShadow = (effect, color) => {
    if (effect === 'glow') return `0 0 6px ${color}, 0 0 14px ${color}`;
    return 'none';
};

// Mirrors the backend: dimming is a fully-opaque scaled color, not CSS opacity,
// so the preview matches exactly what gets burned into the video.
const dimmedWhite = (opacity) => {
    const v = Math.round(255 * (0.5 + 0.5 * opacity));
    return `rgb(${v}, ${v}, ${v})`;
};

export default function SubtitleModal({ isOpen, onClose, onGenerate, isProcessing, videoUrl, bulkCount = 0 }) {
    const [position, setPosition] = useState('bottom');
    const [fontSize, setFontSize] = useState(24);
    const [fontName, setFontName] = useState('Verdana');
    const [fontColor, setFontColor] = useState('#FFFFFF');
    const [borderColor, setBorderColor] = useState('#000000');
    const [borderWidth, setBorderWidth] = useState(2);
    const [bgColor, setBgColor] = useState('#000000');
    const [bgOpacity, setBgOpacity] = useState(0.0);
    const [style, setStyle] = useState('karaoke'); // classic | karaoke (word highlight)
    const [highlightColor, setHighlightColor] = useState('#FFD700');
    const [effect, setEffect] = useState('none'); // none | glow | pop | box
    const [baseOpacity, setBaseOpacity] = useState(1.0);
    const [uppercase, setUppercase] = useState(false);
    const [activePreset, setActivePreset] = useState(null);

    if (!isOpen) return null;

    const applyPreset = (p) => {
        setActivePreset(p.id);
        setStyle(p.style);
        setEffect(p.effect);
        setHighlightColor(p.highlightColor);
        setBaseOpacity(p.baseOpacity);
        setUppercase(p.uppercase);
        setFontName(p.fontName);
        setBorderWidth(p.borderWidth);
        setFontColor('#FFFFFF');
        setBgOpacity(0);
    };

    // Scale border width for preview (preview font is small, so amplify the effect)
    const bw = Math.max(borderWidth, 0);
    const bc = borderColor;
    // Build outline text-shadow (8-direction) — always applied for visibility
    const outlineShadow = bw > 0 ? [
        `-${bw}px -${bw}px 0 ${bc}`, `${bw}px -${bw}px 0 ${bc}`,
        `-${bw}px ${bw}px 0 ${bc}`, `${bw}px ${bw}px 0 ${bc}`,
        `0 -${bw}px 0 ${bc}`, `0 ${bw}px 0 ${bc}`,
        `-${bw}px 0 0 ${bc}`, `${bw}px 0 0 ${bc}`,
    ].join(', ') : 'none';

    const previewStyle = {
        fontFamily: fontName,
        color: fontColor,
        // Preview scales with the chosen size (24 ≈ the old fixed 20px look)
        fontSize: `${Math.round(fontSize * 0.83)}px`,
        fontWeight: 'bold',
        maxWidth: '85%',
        padding: '6px 12px',
        borderRadius: '4px',
        textAlign: 'center',
        lineHeight: '1.3',
        ...(bgOpacity > 0
            ? {
                backgroundColor: `${bgColor}${Math.round(bgOpacity * 255).toString(16).padStart(2, '0')}`,
                textShadow: 'none',
            }
            : { textShadow: outlineShadow }
        ),
    };

    return (
        <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-[fadeIn_0.2s_ease-out]">
            <div className="bg-[#121214] border border-white/10 p-6 rounded-2xl w-full max-w-6xl shadow-2xl relative flex flex-col md:flex-row gap-6 max-h-[90vh]">
                <button
                    onClick={onClose}
                    className="absolute top-4 right-4 text-zinc-500 hover:text-white z-10"
                >
                    <X size={20} />
                </button>

                {/* Left: Preview */}
                <div className="flex-1 flex flex-col items-center justify-center bg-black rounded-lg border border-white/5 overflow-hidden relative aspect-[9/16] max-h-[600px]">
                     <video src={videoUrl} className="w-full h-full object-contain opacity-50" muted playsInline />

                     {/* Subtitle Overlay Preview */}
                     <div className={`absolute w-full px-8 text-center transition-all duration-300 pointer-events-none flex flex-col items-center justify-center
                        ${position === 'top' ? 'top-20' : ''}
                        ${position === 'middle' ? 'top-0 bottom-0' : ''}
                        ${position === 'bottom' ? 'bottom-20' : ''}
                     `}>
                        <span style={{ ...previewStyle, textTransform: uppercase ? 'uppercase' : 'none' }}>
                            {style === 'karaoke' ? (
                                <>
                                    <span style={{ color: dimmedWhite(baseOpacity) }}>This is </span>
                                    <span style={{
                                        color: effect === 'glow' || effect === 'box' ? '#FFFFFF' : highlightColor,
                                        textShadow: effectShadow(effect, highlightColor),
                                        ...(effect === 'pop' ? { display: 'inline-block', transform: 'scale(1.12)' } : {}),
                                        ...(effect === 'box' ? { backgroundColor: highlightColor, borderRadius: '4px', padding: '0 4px' } : {}),
                                    }}>how</span>
                                    <span style={{ color: dimmedWhite(baseOpacity) }}> your subtitles<br/>will appear on the video</span>
                                </>
                            ) : (
                                <>This is how your subtitles<br/>will appear on the video</>
                            )}
                        </span>
                     </div>
                </div>

                {/* Right: Controls */}
                <div className="w-full md:w-[28rem] flex flex-col overflow-y-auto custom-scrollbar">
                    <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
                        <Type className="text-primary" /> Auto Subtitles
                    </h3>

                    <div className="space-y-5 flex-1">
                        {/* Preset Gallery */}
                        <div>
                            <label className="text-xs font-bold text-zinc-400 uppercase tracking-wider mb-2 block">Presets</label>
                            <div className="grid grid-cols-3 gap-2">
                                {CAPTION_PRESETS.map((p) => (
                                    <button
                                        key={p.id}
                                        onClick={() => applyPreset(p)}
                                        className={`group rounded-xl border transition-all overflow-hidden ${activePreset === p.id ? 'border-primary ring-2 ring-primary/40' : 'border-white/10 hover:border-white/40 hover:scale-[1.03]'}`}
                                        title={p.label}
                                    >
                                        <span className="flex h-14 items-center justify-center bg-gradient-to-br from-zinc-700/70 via-zinc-900 to-black px-1 overflow-hidden">
                                            <span
                                                className="text-[15px] font-extrabold tracking-tight leading-none whitespace-nowrap"
                                                style={{
                                                    fontFamily: p.fontName,
                                                    textTransform: p.uppercase ? 'uppercase' : 'none',
                                                    textShadow: '0 1px 2px rgba(0,0,0,0.9)',
                                                }}
                                            >
                                                <span style={{ color: p.style === 'classic' ? '#FFFFFF' : dimmedWhite(p.baseOpacity) }}>So </span>
                                                <span style={{
                                                    color: p.style === 'classic' || p.effect === 'glow' || p.effect === 'box' ? '#FFFFFF' : p.highlightColor,
                                                    textShadow: effectShadow(p.effect, p.highlightColor) !== 'none'
                                                        ? effectShadow(p.effect, p.highlightColor)
                                                        : '0 1px 2px rgba(0,0,0,0.9)',
                                                    ...(p.effect === 'pop' ? { display: 'inline-block', transform: 'scale(1.15)' } : {}),
                                                    ...(p.effect === 'box' ? { backgroundColor: p.highlightColor, borderRadius: '4px', padding: '0 3px' } : {}),
                                                }}>gut</span>
                                            </span>
                                        </span>
                                        <span className={`block py-1 text-center text-[9px] font-medium transition-colors ${activePreset === p.id ? 'text-primary' : 'text-zinc-500 group-hover:text-zinc-300'}`}>{p.label}</span>
                                    </button>
                                ))}
                            </div>
                        </div>

                        {/* Caption Style */}
                        <div>
                            <label className="text-xs font-bold text-zinc-400 uppercase tracking-wider mb-2 block">Style</label>
                            <div className="grid grid-cols-2 gap-2">
                                <button
                                    onClick={() => setStyle('karaoke')}
                                    className={`p-2 rounded-lg border text-center text-xs font-medium transition-all ${style === 'karaoke' ? 'bg-primary/20 border-primary text-white' : 'bg-white/5 border-white/5 text-zinc-400 hover:bg-white/10'}`}
                                >
                                    ✨ Word Highlight
                                </button>
                                <button
                                    onClick={() => setStyle('classic')}
                                    className={`p-2 rounded-lg border text-center text-xs font-medium transition-all ${style === 'classic' ? 'bg-primary/20 border-primary text-white' : 'bg-white/5 border-white/5 text-zinc-400 hover:bg-white/10'}`}
                                >
                                    Classic
                                </button>
                            </div>
                        </div>

                        {/* Effect (karaoke only) */}
                        {style === 'karaoke' && (
                            <div className="animate-[fadeIn_0.2s_ease-out]">
                                <label className="text-xs font-bold text-zinc-400 uppercase tracking-wider mb-2 block">Effect</label>
                                <div className="grid grid-cols-4 gap-2">
                                    {[
                                        { id: 'none', label: 'None' },
                                        { id: 'glow', label: 'Glow' },
                                        { id: 'pop', label: 'Pop' },
                                        { id: 'box', label: 'Box' },
                                    ].map((e) => (
                                        <button
                                            key={e.id}
                                            onClick={() => { setEffect(e.id); setActivePreset(null); }}
                                            className={`p-2 rounded-lg border text-center text-xs font-medium transition-all ${effect === e.id ? 'bg-primary/20 border-primary text-white' : 'bg-white/5 border-white/5 text-zinc-400 hover:bg-white/10'}`}
                                        >
                                            {e.label}
                                        </button>
                                    ))}
                                </div>
                                <div className="flex items-center justify-between mt-3">
                                    <label className="text-[10px] text-zinc-500">Dim inactive words</label>
                                    <input
                                        type="range"
                                        min="20"
                                        max="100"
                                        value={Math.round(baseOpacity * 100)}
                                        onChange={(e) => { setBaseOpacity(parseInt(e.target.value) / 100); setActivePreset(null); }}
                                        className="w-32 accent-primary"
                                    />
                                    <span className="text-[10px] text-zinc-500 w-8 text-right">{Math.round(baseOpacity * 100)}%</span>
                                </div>
                                <div className="flex items-center justify-between mt-2">
                                    <label className="text-[10px] text-zinc-500">UPPERCASE</label>
                                    <label className="relative inline-flex items-center cursor-pointer">
                                        <input type="checkbox" checked={uppercase} onChange={(e) => { setUppercase(e.target.checked); setActivePreset(null); }} className="sr-only peer" />
                                        <div className="w-8 h-4 bg-zinc-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[0px] after:left-[0px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-primary"></div>
                                    </label>
                                </div>
                            </div>
                        )}

                        {/* Highlight Color (karaoke only) */}
                        {style === 'karaoke' && (
                            <div className="animate-[fadeIn_0.2s_ease-out]">
                                <label className="text-xs font-bold text-zinc-400 uppercase tracking-wider mb-2 block">Highlight Color</label>
                                <div className="flex flex-wrap gap-2">
                                    {HIGHLIGHT_PRESETS.map((c) => (
                                        <button
                                            key={c.color}
                                            onClick={() => setHighlightColor(c.color)}
                                            className={`w-7 h-7 rounded-full border-2 transition-all ${highlightColor === c.color ? 'border-white scale-110' : 'border-white/20 hover:border-white/50'}`}
                                            style={{ backgroundColor: c.color }}
                                            title={c.label}
                                        />
                                    ))}
                                    <label className="w-7 h-7 rounded-full border-2 border-dashed border-white/20 cursor-pointer flex items-center justify-center hover:border-white/50 transition-all overflow-hidden relative" title="Custom color">
                                        <span className="text-[10px] text-zinc-400">+</span>
                                        <input type="color" value={highlightColor} onChange={(e) => setHighlightColor(e.target.value)} className="absolute inset-0 opacity-0 cursor-pointer" />
                                    </label>
                                </div>
                            </div>
                        )}

                        {/* Position Selector */}
                        <div>
                            <label className="text-xs font-bold text-zinc-400 uppercase tracking-wider mb-2 block">Position</label>
                            <div className="grid grid-cols-3 gap-2">
                                {['top', 'middle', 'bottom'].map((pos) => (
                                    <button
                                        key={pos}
                                        onClick={() => setPosition(pos)}
                                        className={`p-2 rounded-lg border text-center text-xs font-medium transition-all ${position === pos ? 'bg-primary/20 border-primary text-white' : 'bg-white/5 border-white/5 text-zinc-400 hover:bg-white/10'}`}
                                    >
                                        {pos.charAt(0).toUpperCase() + pos.slice(1)}
                                    </button>
                                ))}
                            </div>
                        </div>

                        {/* Text Size */}
                        <div>
                            <div className="flex items-center justify-between mb-2">
                                <label className="text-xs font-bold text-zinc-400 uppercase tracking-wider">Text Size</label>
                                <span className="text-[10px] text-zinc-500">{fontSize}</span>
                            </div>
                            <input
                                type="range"
                                min="14"
                                max="40"
                                value={fontSize}
                                onChange={(e) => setFontSize(parseInt(e.target.value))}
                                className="w-full accent-primary"
                            />
                            <div className="flex justify-between text-[10px] text-zinc-500">
                                <span>Small</span>
                                <span>Large</span>
                            </div>
                        </div>

                        {/* Font Family */}
                        <div>
                            <label className="text-xs font-bold text-zinc-400 uppercase tracking-wider mb-2 block">Font</label>
                            <select
                                value={fontName}
                                onChange={(e) => setFontName(e.target.value)}
                                className="w-full bg-black/40 border border-white/10 rounded-lg p-2 text-sm text-white focus:outline-none focus:border-primary/50"
                            >
                                {FONT_OPTIONS.map((f) => (
                                    <option key={f.value} value={f.value} style={{ fontFamily: f.value }}>{f.label}</option>
                                ))}
                            </select>
                        </div>

                        {/* Text Color */}
                        <div>
                            <label className="text-xs font-bold text-zinc-400 uppercase tracking-wider mb-2 block">Text Color</label>
                            <div className="flex flex-wrap gap-2">
                                {COLOR_PRESETS.map((c) => (
                                    <button
                                        key={c.color}
                                        onClick={() => setFontColor(c.color)}
                                        className={`w-7 h-7 rounded-full border-2 transition-all ${fontColor === c.color ? 'border-white scale-110' : 'border-white/20 hover:border-white/50'}`}
                                        style={{ backgroundColor: c.color }}
                                        title={c.label}
                                    />
                                ))}
                                <label className="w-7 h-7 rounded-full border-2 border-dashed border-white/20 cursor-pointer flex items-center justify-center hover:border-white/50 transition-all overflow-hidden relative" title="Custom color">
                                    <span className="text-[10px] text-zinc-400">+</span>
                                    <input type="color" value={fontColor} onChange={(e) => setFontColor(e.target.value)} className="absolute inset-0 opacity-0 cursor-pointer" />
                                </label>
                            </div>
                        </div>

                        {/* Border / Outline */}
                        <div>
                            <label className="text-xs font-bold text-zinc-400 uppercase tracking-wider mb-2 block">Border</label>
                            <div className="flex items-center gap-3">
                                <label className="relative w-8 h-8 rounded-lg border border-white/10 cursor-pointer overflow-hidden shrink-0" title="Border color">
                                    <div className="w-full h-full" style={{ backgroundColor: borderColor }} />
                                    <input type="color" value={borderColor} onChange={(e) => setBorderColor(e.target.value)} className="absolute inset-0 opacity-0 cursor-pointer" />
                                </label>
                                <div className="flex-1">
                                    <input
                                        type="range"
                                        min="0"
                                        max="5"
                                        value={borderWidth}
                                        onChange={(e) => setBorderWidth(parseInt(e.target.value))}
                                        className="w-full accent-primary"
                                    />
                                    <div className="flex justify-between text-[10px] text-zinc-500">
                                        <span>None</span>
                                        <span>Thick</span>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Background Box */}
                        <div>
                            <div className="flex items-center justify-between mb-2">
                                <label className="text-xs font-bold text-zinc-400 uppercase tracking-wider">Background Box</label>
                                <label className="relative inline-flex items-center cursor-pointer">
                                    <input type="checkbox" checked={bgOpacity > 0} onChange={(e) => setBgOpacity(e.target.checked ? 0.5 : 0)} className="sr-only peer" />
                                    <div className="w-8 h-4 bg-zinc-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[0px] after:left-[0px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-primary"></div>
                                </label>
                            </div>
                            {bgOpacity > 0 && (
                                <div className="space-y-3 animate-[fadeIn_0.2s_ease-out]">
                                    <div className="flex items-center gap-3">
                                        <label className="relative w-8 h-8 rounded-lg border border-white/10 cursor-pointer overflow-hidden shrink-0" title="Background color">
                                            <div className="w-full h-full" style={{ backgroundColor: bgColor }} />
                                            <input type="color" value={bgColor} onChange={(e) => setBgColor(e.target.value)} className="absolute inset-0 opacity-0 cursor-pointer" />
                                        </label>
                                        <div className="flex-1">
                                            <input
                                                type="range"
                                                min="10"
                                                max="100"
                                                value={Math.round(bgOpacity * 100)}
                                                onChange={(e) => setBgOpacity(parseInt(e.target.value) / 100)}
                                                className="w-full accent-primary"
                                            />
                                            <div className="flex justify-between text-[10px] text-zinc-500">
                                                <span>Transparent</span>
                                                <span>{Math.round(bgOpacity * 100)}%</span>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>

                    <button
                        onClick={() => onGenerate({ position, fontSize, fontName, fontColor, borderColor, borderWidth, bgColor, bgOpacity, style, highlightColor, effect, baseOpacity, uppercase })}
                        disabled={isProcessing}
                        className="w-full py-4 mt-6 bg-gradient-to-r from-yellow-500 to-orange-500 hover:from-yellow-400 hover:to-orange-400 text-black font-bold rounded-xl shadow-lg shadow-orange-500/20 transition-all active:scale-[0.98] flex items-center justify-center gap-2"
                    >
                        {isProcessing ? <Loader2 size={20} className="animate-spin" /> : <Type size={20} />}
                        {isProcessing
                            ? 'Generating...'
                            : bulkCount > 0
                                ? `Apply to all ${bulkCount} clips`
                                : 'Generate Subtitles'}
                    </button>

                    <p className="text-[10px] text-zinc-500 text-center mt-3">
                        Uses AI word-level timestamps to sync perfectly.
                    </p>
                </div>
            </div>
        </div>
    );
}
