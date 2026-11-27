import React, { useEffect, useId, useRef, useState } from 'react';
import { Loader2, X } from 'lucide-react';
import RemotionPreview from './RemotionPreview';
import Modal from './ui/Modal';
import SegmentedControl from './ui/SegmentedControl';
import { useAuth } from '../contexts/AuthContext';
import { apiJson } from '../lib/api';
import { DEFAULT_HOOK_DESIGN, readHookDesigns, writeHookDesigns } from '../lib/hookDesigns';

const ENTRANCE_OPTIONS = [
    { value: 'spring', label: 'Bounce' },
    { value: 'fade', label: 'Fade' },
    { value: 'slide-up', label: 'Slide Up' },
    { value: 'none', label: 'None' },
];

// Must mirror hooks.py HOOK_STYLES.
const HOOK_STYLES = [
    { value: 'pill', label: 'Pills', box: 'rgba(255,255,255,0.98)', text: '#000', sans: true },
    { value: 'classic', label: 'Classic', box: 'rgba(255,255,255,0.94)', text: '#000' },
    { value: 'dark', label: 'Dark', box: 'rgba(18,18,20,0.92)', text: '#fff' },
    { value: 'yellow', label: 'Yellow', box: 'rgba(255,214,0,0.96)', text: '#000' },
    { value: 'red', label: 'Red', box: 'rgba(220,38,38,0.96)', text: '#fff' },
    { value: 'outline', label: 'Outline', box: 'transparent', text: '#fff', outline: true },
    { value: 'outline_yellow', label: 'Outline+', box: 'transparent', text: '#FFD600', outline: true },
];

// Must mirror hooks.py HOOK_FONTS (all bundled: /fonts/*.ttf).
const FONT_OPTIONS = [
    { value: 'montserrat', label: 'Montserrat' },
    { value: 'anton', label: 'Anton' },
    { value: 'serif', label: 'Serif' },
];
const FONT_CSS = {
    montserrat: { fontFamily: "'Montserrat-ExtraBold', 'Montserrat', sans-serif", fontWeight: 800 },
    anton: { fontFamily: "'Anton-Regular', Impact, sans-serif", fontWeight: 400 },
    serif: { fontFamily: "'Noto Serif', Georgia, serif", fontWeight: 700 },
};
// The style's own typeface when none is picked (what hooks.py renders).
const styleFont = (st) => (st === 'pill' ? 'montserrat' : 'serif');

// auto: the server keeps the hook off the speaker's face and the captions
// (hook_placement.py). The preview cannot see faces and draws it at the top.
const POSITION_OPTIONS = [
    { value: 'auto', label: 'auto' },
    { value: 'top', label: 'top' },
    { value: 'center', label: 'center' },
    { value: 'bottom', label: 'bottom' },
];

const SIZE_OPTIONS = [
    { value: 'S', label: 'Small' },
    { value: 'M', label: 'Medium' },
    { value: 'L', label: 'Large' },
];

// Last-used hook settings, restored on the next open (style always reset to
// classic otherwise, which users read as the picker being broken).
function loadHookPrefs(key) {
    try { return JSON.parse(localStorage.getItem(key) || localStorage.getItem('os_hook_prefs')) || {}; }
    catch { return {}; }
}

export default function HookModal({ isOpen, onClose, onGenerate, onRemove, isProcessing, error, videoUrl, initialText, initialHook, durationInSeconds, existingSubtitles, hasCaptions, serverRender, mediaInfo }) {
    const { user } = useAuth();
    const accountId = user?.id;
    const preferencesKey = `os_hook_prefs:${accountId || 'local'}`;
    const prefs = loadHookPrefs(preferencesKey);
    const id = useId();
    const playerRef = useRef(null);
    const [text, setText] = useState(initialHook?.text ?? initialText ?? '');
    const [position, setPosition] = useState(initialHook?.position || prefs.pos || 'auto');
    const [size, setSize] = useState(initialHook?.size || prefs.size || 'M');
    const [style, setStyle] = useState(initialHook?.style || prefs.style || 'pill');
    const [design, setDesign] = useState(initialHook ? initialHook.design || null : prefs.design || null);
    const [savedDesigns, setSavedDesigns] = useState(() => readHookDesigns(accountId));
    const [designName, setDesignName] = useState('');
    const [designError, setDesignError] = useState('');
    const [artworkPreview, setArtworkPreview] = useState(null);
    // null = follow the style's own typeface.
    const [font, setFont] = useState(initialHook ? initialHook.font : prefs.font || null);
    const effectiveFont = font || styleFont(design ? design.shape : style);
    const [entranceAnimation, setEntranceAnimation] = useState(initialHook?.entranceAnimation || prefs.entranceAnimation || 'spring');
    const [displayDuration, setDisplayDuration] = useState(initialHook?.displayDurationSec ?? 5);
    const [wholeClip, setWholeClip] = useState(initialHook?.displayDurationSec === null);
    const [ctaText, setCtaText] = useState(initialHook?.cta?.text || '');
    const [ctaDuration, setCtaDuration] = useState(initialHook?.cta?.displayDurationSec ?? 3);
    const [ctaPosition, setCtaPosition] = useState(initialHook?.cta?.position || 'auto');
    const [ctaSize, setCtaSize] = useState(initialHook?.cta?.size || 'M');

    const cta = ctaText.trim() ? {
        text: ctaText.trim(), position: ctaPosition, size: ctaSize, displayDurationSec: ctaDuration,
    } : null;
    const renderOnServer = serverRender || !!cta || !!design;
    const hookConfig = {
        text: text.trim(),
        position,
        size,
        style,
        font: effectiveFont,
        entranceAnimation: renderOnServer ? 'none' : entranceAnimation,
        displayDurationSec: wholeClip ? null : displayDuration,
        ...(design ? { design } : {}),
        ...(cta ? { cta } : {}),
    };
    const previewWidth = mediaInfo?.width || 1080;
    const previewHeight = mediaInfo?.height || 1920;
    const customTextTooLong = Boolean(design && hookConfig.text.length > 240);
    const previewRequest = design && !customTextTooLong && (hookConfig.text || cta)
        ? JSON.stringify({ width: previewWidth, text: hookConfig.text, size, cta_text: cta?.text || '',
            cta_size: ctaSize, font: effectiveFont, design })
        : null;
    const artwork = artworkPreview?.request === previewRequest ? artworkPreview.artwork : null;
    const artworkError = artworkPreview?.request === previewRequest ? artworkPreview.error : '';

    useEffect(() => {
        if (!isOpen || !videoUrl || !previewRequest) return undefined;
        const controller = new AbortController();
        const timer = setTimeout(async () => {
            try {
                const artwork = await apiJson('/api/hook/design-preview', {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: previewRequest, signal: controller.signal,
                });
                if (!controller.signal.aborted) setArtworkPreview({ request: previewRequest, artwork });
            } catch (error) {
                if (!controller.signal.aborted) setArtworkPreview({ request: previewRequest, error: error.detail || 'Design preview is unavailable.' });
            }
        }, 200);
        return () => { clearTimeout(timer); controller.abort(); };
    }, [isOpen, videoUrl, previewRequest]);

    if (!isOpen) return null;

    function updateDesign(field, value) {
        setDesign((current) => ({ ...current, [field]: value }));
    }

    function updateSavedDesigns(items) {
        if (!writeHookDesigns(accountId, items)) {
            setDesignError('Could not save designs in this browser.');
            return false;
        }
        setSavedDesigns(items);
        setDesignError('');
        return true;
    }

    function saveDesign() {
        const name = designName.trim();
        if (savedDesigns.some((item) => item.name.toLowerCase() === name.toLowerCase())) {
            setDesignError('Choose a unique design name.');
            return;
        }
        if (name && design && updateSavedDesigns([...savedDesigns, { name, design, style, font, position, size }])) {
            setDesignName('');
        }
    }

    function applyDesign(item) {
        setDesign({ ...DEFAULT_HOOK_DESIGN, ...item.design });
        setStyle(item.style || 'pill');
        setFont(item.font || null);
        setPosition(item.position || 'auto');
        setSize(item.size || 'M');
        setDesignError('');
    }

    const pillFontFace = "@font-face{font-family:'Montserrat-ExtraBold';src:url('/fonts/Montserrat-ExtraBold.ttf') format('truetype');font-weight:800;}"
        + "@font-face{font-family:'Anton-Regular';src:url('/fonts/Anton-Regular.ttf') format('truetype');font-weight:400;}";

    return (
        <Modal isOpen={isOpen} onClose={onClose} size="lg" eyebrow="CLIP EDITOR" title="Hook & closing text">
            <style>{pillFontFace}</style>
            <div className="flex flex-col md:flex-row gap-6">
                <div className="w-full max-w-[280px] mx-auto md:mx-0 md:w-[280px] shrink-0 self-start">
                    <div className="flex items-center justify-center bg-black rounded-card border border-rule overflow-hidden"
                        style={{ aspectRatio: `${previewWidth} / ${previewHeight}` }}>
                        {videoUrl ? (
                            <RemotionPreview
                                videoUrl={videoUrl}
                                durationInSeconds={durationInSeconds}
                                width={previewWidth}
                                height={previewHeight}
                                hook={artwork ? { ...hookConfig, artwork } : hookConfig}
                                subtitles={existingSubtitles || null}
                                playerRef={playerRef}
                            />
                        ) : (
                            <p className="p-4 text-sm text-white">Preview unavailable</p>
                        )}
                    </div>
                    {videoUrl && previewRequest && !artwork && (
                        <p role="status" className="text-xs text-muted mt-2">
                            {artworkError || 'Updating design preview…'}
                        </p>
                    )}
                    {videoUrl && cta && (
                        <button type="button" className="btn-ghost mt-2 w-full"
                            onClick={() => {
                                const player = playerRef.current;
                                player?.seekTo(Math.max(0, Math.round((durationInSeconds - ctaDuration) * 30)));
                                player?.pause();
                            }}>
                            Preview closing text
                        </button>
                    )}
                </div>

                <div className="w-full min-w-0 flex flex-col">
                    <div className="space-y-5 flex-1 pr-1">
                        <div>
                            <label className="eyebrow mb-2 block" htmlFor={`${id}-hook`}>Opening hook</label>
                            <textarea
                                id={`${id}-hook`}
                                value={text}
                                onChange={(e) => setText(e.target.value)}
                                rows={2}
                                maxLength={design ? 240 : undefined}
                                className="input-field resize-none"
                                placeholder="Opening title"
                            />
                        </div>

                        {/* Style (new) */}
                        <div>
                            <p className="eyebrow mb-2">Style</p>
                            <div className="grid grid-cols-3 gap-1.5">
                                {HOOK_STYLES.map((s) => (
                                    <button
                                        key={s.value}
                                        onClick={() => { setStyle(s.value); setDesign(null); }}
                                        className={`px-1 py-2 rounded-input border text-xs transition-colors
                                            ${style === s.value && !design ? 'border-[color:var(--color-accent)]' : 'border-rule2 hover:border-[color:var(--color-accent)]'}`}
                                        title={s.label}
                                    >
                                        {s.value === 'pill' ? (
                                            <span className="flex flex-col items-center gap-0.5 py-0.5" style={FONT_CSS[font || styleFont(s.value)]}>
                                                <span className="rounded-md px-1.5 text-[13px] leading-5 bg-white text-black">Aa Bb</span>
                                                <span className="rounded-md px-1.5 text-[13px] leading-5 bg-white text-black">Cc</span>
                                            </span>
                                        ) : (
                                            <span
                                                className="block rounded-md px-1 py-1.5 text-[15px] leading-5"
                                                style={{
                                                    ...FONT_CSS[font || styleFont(s.value)],
                                                    backgroundColor: s.box,
                                                    color: s.text,
                                                    textShadow: s.outline ? '-1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000' : 'none',
                                                }}
                                            >Aa</span>
                                        )}
                                        <span className="block mt-1 text-muted">{s.label}</span>
                                    </button>
                                ))}
                                <button type="button" aria-pressed={!!design} onClick={() => setDesign(design || { ...DEFAULT_HOOK_DESIGN })}
                                    className={`px-1 py-2 rounded-input border text-xs ${design ? 'border-[color:var(--color-accent)]' : 'border-rule2'}`}>
                                    Custom
                                </button>
                            </div>
                        </div>

                        {design && (
                            <div className="grid grid-cols-2 gap-3" aria-label="Custom hook design">
                                <label className="text-xs text-ink2">Background shape
                                    <select className="input-field mt-1" value={design.shape} onChange={(e) => updateDesign('shape', e.target.value)}>
                                        <option value="pill">Box per line</option>
                                        <option value="box">One box</option>
                                        <option value="none">Text only</option>
                                    </select>
                                </label>
                                <label className="text-xs text-ink2">Letter case
                                    <select className="input-field mt-1" value={design.text_case} onChange={(e) => updateDesign('text_case', e.target.value)}>
                                        <option value="as_written">As written</option>
                                        <option value="uppercase">Uppercase</option>
                                    </select>
                                </label>
                                {[
                                    ['text_color', 'Text color'], ['outline_color', 'Outline color'],
                                    ...(design.shape === 'none' ? [] : [['background_color', 'Background color']]),
                                ].map(([field, label]) => (
                                    <label key={field} className="flex items-center justify-between gap-2 text-xs text-ink2">
                                        {label}
                                        <input type="color" value={design[field]} onChange={(e) => updateDesign(field, e.target.value)}
                                            className="h-8 w-10 cursor-pointer rounded-input bg-transparent" />
                                    </label>
                                ))}
                                <label className="flex flex-col gap-1 text-xs text-ink2">Outline width <span className="sr-only">in pixels</span>
                                    <input type="range" min="0" max="20" value={design.outline_width}
                                        onChange={(e) => updateDesign('outline_width', Number(e.target.value))} />
                                </label>
                                {design.shape !== 'none' && (
                                    <label className="flex flex-col gap-1 text-xs text-ink2">Corner radius <span className="sr-only">in pixels</span>
                                        <input type="range" min="0" max="120" value={design.corner_radius}
                                            onChange={(e) => updateDesign('corner_radius', Number(e.target.value))} />
                                    </label>
                                )}
                            </div>
                        )}

                        {(design || savedDesigns.length > 0) && (
                            <div className="space-y-2">
                                <p className="eyebrow">Saved designs</p>
                                <div className="flex flex-wrap gap-1.5">
                                    {savedDesigns.map((item) => (
                                        <div key={item.name} className="inline-flex items-center rounded-input border border-rule2">
                                            <button type="button" className="text-xs px-2 py-1" onClick={() => applyDesign(item)}>{item.name}</button>
                                            <button type="button" className="p-1 text-muted hover:text-danger" aria-label={`Remove ${item.name}`}
                                                onClick={() => updateSavedDesigns(savedDesigns.filter(({ name }) => name !== item.name))}>
                                                <X size={13} />
                                            </button>
                                        </div>
                                    ))}
                                </div>
                                {design && (
                                    <div className="flex gap-2">
                                        <input aria-label="Design name" placeholder="Design name" className="input-field min-w-0" maxLength={48}
                                            value={designName} onChange={(e) => setDesignName(e.target.value)} />
                                        <button type="button" className="btn-quiet shrink-0" disabled={!designName.trim()} onClick={saveDesign}>
                                            Save design
                                        </button>
                                    </div>
                                )}
                                {designError && <p role="alert" className="text-xs text-warn">{designError}</p>}
                            </div>
                        )}

                        {/* Typeface */}
                        <div>
                            <p className="eyebrow mb-2">Font</p>
                            <div className="grid grid-cols-3 gap-1.5">
                                {FONT_OPTIONS.map((f) => (
                                    <button
                                        key={f.value}
                                        onClick={() => setFont(f.value)}
                                        className={`px-1 py-2 rounded-input border text-[15px] text-ink transition-colors
                                            ${effectiveFont === f.value ? 'border-[color:var(--color-accent)]' : 'border-rule2 hover:border-[color:var(--color-accent)]'}`}
                                        style={FONT_CSS[f.value]}
                                    >
                                        {f.label}
                                    </button>
                                ))}
                            </div>
                        </div>

                        {/* Position Control */}
                        <div>
                            <p className="eyebrow mb-2">Position</p>
                            <SegmentedControl
                                options={POSITION_OPTIONS}
                                value={position}
                                onChange={setPosition}
                                size="sm"
                            />
                            {position === 'auto' && (
                                <p className="text-[11px] text-muted mt-1.5 leading-relaxed">
                                    Auto places text around faces and captions. The preview shows the top position.
                                </p>
                            )}
                            {position === 'bottom' && hasCaptions && (
                                <p className="text-[11px] text-warn mt-1.5 leading-relaxed">
                                    A bottom hook may overlap captions or playback controls.
                                </p>
                            )}
                        </div>

                        {/* Size Control */}
                        <div>
                            <p className="eyebrow mb-2">Size</p>
                            <SegmentedControl
                                options={SIZE_OPTIONS}
                                value={size}
                                onChange={setSize}
                                size="sm"
                            />
                        </div>

                        {/* Entrance Animation (new) */}
                        {!renderOnServer && (
                        <div>
                            <p className="eyebrow mb-2">Entrance</p>
                            <SegmentedControl
                                options={ENTRANCE_OPTIONS}
                                value={entranceAnimation}
                                onChange={setEntranceAnimation}
                                columns={2}
                                size="sm"
                            />
                        </div>
                        )}

                        <div>
                            <div className="flex items-center justify-between mb-2">
                                <label className="eyebrow" htmlFor={`${id}-duration`}>Hook duration</label>
                                <label className="flex items-center gap-1.5 text-xs text-ink">
                                    <input type="checkbox" checked={wholeClip}
                                        onChange={(e) => setWholeClip(e.target.checked)} />
                                    Whole clip
                                </label>
                            </div>
                            {!wholeClip && (
                                <div className="flex items-center gap-2">
                                    <input id={`${id}-duration`} type="range" min="1" max={Math.max(15, Math.floor(durationInSeconds))}
                                        value={displayDuration} onChange={(e) => setDisplayDuration(Number(e.target.value))}
                                        className="w-full accent-[var(--color-accent)]" />
                                    <output className="readout" htmlFor={`${id}-duration`}>{displayDuration}s</output>
                                </div>
                            )}
                        </div>

                        <div className="space-y-3 border-t border-rule pt-4">
                            <label className="eyebrow block" htmlFor={`${id}-cta`}>Closing call to action</label>
                            <textarea id={`${id}-cta`} value={ctaText} maxLength={180} rows={2}
                                onChange={(e) => setCtaText(e.target.value)} className="input-field resize-none"
                                placeholder="Follow for more" />
                            {cta && (
                                <>
                                    <div className="flex items-center gap-2">
                                        <label className="eyebrow shrink-0" htmlFor={`${id}-cta-duration`}>Last</label>
                                        <input type="range" min="1" max="15" id={`${id}-cta-duration`} value={ctaDuration}
                                            onChange={(e) => setCtaDuration(Number(e.target.value))}
                                            className="w-full accent-[var(--color-accent)]" />
                                        <output className="readout" htmlFor={`${id}-cta-duration`}>{ctaDuration}s</output>
                                    </div>
                                    <div>
                                        <p className="eyebrow mb-2">Closing text position</p>
                                        <SegmentedControl options={POSITION_OPTIONS} value={ctaPosition} onChange={setCtaPosition} size="sm" />
                                    </div>
                                    <div>
                                        <p className="eyebrow mb-2">Closing text size</p>
                                        <SegmentedControl options={SIZE_OPTIONS} value={ctaSize} onChange={setCtaSize} size="sm" />
                                    </div>
                                    {text.trim() && <p className="text-xs text-muted">The opening hook ends before the closing text appears.</p>}
                                </>
                            )}
                        </div>
                        {onRemove && (
                            <button type="button" onClick={onRemove} disabled={isProcessing}
                                className="text-xs text-warn underline underline-offset-2 disabled:opacity-50">
                                Remove text overlays
                            </button>
                        )}
                    </div>

                    {customTextTooLong && <p role="alert" className="mt-4 text-sm text-warn">Custom hooks support up to 240 characters.</p>}
                    {error && <p role="alert" className="mt-4 text-sm text-warn">{error}</p>}
                    <div className="flex gap-2 mt-5 shrink-0">
                        <button onClick={onClose} className="btn-ghost">
                            cancel
                        </button>
                        <button
                            onClick={() => {
                                try {
                                    localStorage.setItem(preferencesKey, JSON.stringify({
                                        style, font, pos: position, size, entranceAnimation, design,
                                    }));
                                } catch { /* ignore */ }
                                onGenerate(hookConfig);
                            }}
                            disabled={isProcessing || customTextTooLong || !(text.trim() || cta)}
                            className="btn-primary flex-1"
                        >
                            {isProcessing && <Loader2 size={16} className="animate-spin text-brassink" />}
                            {isProcessing ? 'saving…' : 'save text'}
                        </button>
                    </div>
                </div>
            </div>
        </Modal>
    );
}
