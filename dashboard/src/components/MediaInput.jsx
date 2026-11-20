import React, { useState, useEffect, useId, useRef } from 'react';
import { Link2, Upload, FileVideo, X, Info, Loader2, ChevronDown } from 'lucide-react';
import { track } from '../lib/analytics';
import { getApiUrl } from '../config';
import SourceCost from './SourceCost';
import { useAuth } from '../contexts/AuthContext';
import { FREE_CLIPS_PER_VIDEO, FREE_CLIPS_QUIET, MAX_CLIPS_PER_VIDEO } from '../lib/clipProgress';

const SUPPORTED_PLATFORMS = [
    'YouTube', 'Vimeo', 'TikTok', 'X / Twitter', 'Twitch', 'Kick',
    'Facebook', 'Instagram', 'Dailymotion', 'Reddit', 'Streamable',
];

const FRAMING_OPTIONS = [
    { value: 'auto', label: 'Smart framing', hint: 'Uses motion and visual detail, and keeps the whole picture when the focus is unclear.' },
    { value: 'motion', label: 'Follow motion', hint: 'Frames the action in moving footage, such as gameplay and sports.' },
    { value: 'detail', label: 'Follow detail', hint: 'Frames the strongest visual detail in steadier shots.' },
    { value: 'center', label: 'Center crop', hint: 'Fills the output frame with a fixed crop from the center.' },
    { value: 'fit', label: 'Keep full picture', hint: 'Keeps the entire source visible, with a blurred background around it.' },
    { value: 'legacy', label: 'Speaker and screen layouts', hint: 'Uses face detection for speaker tracking, stacked speakers, and presenter layouts.' },
];

// Mirrors the server's MIN_SOURCE_SECONDS: shorter sources are rejected with a
// 400 after the whole file was uploaded. Checking the duration in the browser
// says so the moment the file is picked (a 27 s upload used to end in a bare
// "That run failed").
const MIN_SOURCE_SECONDS = 45;

// Duration of a local video file in seconds, or null when the browser cannot
// read it (unsupported codec): then the server stays the judge.
const readVideoDuration = (file) => new Promise((resolve) => {
    try {
        const url = URL.createObjectURL(file);
        const v = document.createElement('video');
        v.preload = 'metadata';
        const timer = setTimeout(() => done(null), 8000);
        const done = (d) => {
            clearTimeout(timer);
            v.onloadedmetadata = null;
            v.onerror = null;
            URL.revokeObjectURL(url);
            resolve(d);
        };
        v.onloadedmetadata = () => done(Number.isFinite(v.duration) ? v.duration : null);
        v.onerror = () => done(null);
        v.src = url;
    } catch { resolve(null); }
});

export default function MediaInput({ onProcess, isProcessing }) {
    const { plan } = useAuth();
    const [youtubeUrlEnabled, setYoutubeUrlEnabled] = useState(true);
    // File upload is the primary path; the link is secondary.
    const [mode, setMode] = useState('file'); // 'file' | 'url'
    const [url, setUrl] = useState('');
    const [file, setFile] = useState(null);
    const [fileSeconds, setFileSeconds] = useState(null);
    const fileTooShort = fileSeconds != null && fileSeconds < MIN_SOURCE_SECONDS;
    const [acknowledged, setAcknowledged] = useState(false);
    // {start, end} in seconds when the user picked a stretch under the cost line.
    const [range, setRange] = useState(null);
    const [outputFormat, setOutputFormat] = useState('vertical'); // vertical | horizontal | square
    const [framingMethod, setFramingMethod] = useState(() => {
        try {
            const saved = localStorage.getItem('os_framing_method');
            if (FRAMING_OPTIONS.some(({ value }) => value === saved)) return saved;
            // Existing explicit speaker layouts belong to the face-based renderer.
            return ['split', 'screencast', 'none'].includes(localStorage.getItem('os_layout')) ? 'legacy' : 'auto';
        } catch { return 'auto'; }
    });
    const [captions, setCaptions] = useState(true);
    const [showInfo, setShowInfo] = useState(false);
    // Advanced generation controls — empty string means "let the AI decide",
    // which keeps the default pipeline behavior untouched.
    const [showAdvanced, setShowAdvanced] = useState(false);
    const [targetClips, setTargetClips] = useState('');
    const [clipMinSeconds, setClipMinSeconds] = useState('');
    const [clipMaxSeconds, setClipMaxSeconds] = useState('');
    // Auto-hook: burn the AI hook text into every clip. On by default; the
    // choice persists so turning it off sticks across sessions.
    const [autoHook, setAutoHook] = useState(() => {
        try { return localStorage.getItem('os_auto_hook') !== '0'; } catch { return true; }
    });
    const [autoHookStyle, setAutoHookStyle] = useState(() => {
        // v2 key: the default became 'pill'; an old saved 'classic' was just the old default.
        try { return localStorage.getItem('os_auto_hook_style_v2') || 'pill'; } catch { return 'pill'; }
    });
    // Layout: 'auto' lets the AI pick per video (server default); the others
    // force one on so a podcast host who knows what they uploaded doesn't
    // depend on the detector, and 'none' keeps the plain single crop.
    const [layout, setLayout] = useState(() => {
        try { return localStorage.getItem('os_layout') || 'auto'; } catch { return 'auto'; }
    });
    const infoRef = useRef(null);
    const controlsId = useId();
    const invalidLength = clipMinSeconds !== '' && clipMaxSeconds !== ''
        && Number(clipMaxSeconds) < Number(clipMinSeconds) + 5;
    const showFraming = outputFormat !== 'horizontal';
    const showLegacyLayout = showFraming && framingMethod === 'legacy';
    const framing = FRAMING_OPTIONS.find(({ value }) => value === framingMethod);

    // Close the compatibility popover on any outside click.
    useEffect(() => {
        if (!showInfo) return;
        const onClick = (e) => {
            if (infoRef.current && !infoRef.current.contains(e.target)) setShowInfo(false);
        };
        document.addEventListener('mousedown', onClick);
        return () => document.removeEventListener('mousedown', onClick);
    }, [showInfo]);

    useEffect(() => {
        let cancelled = false;
        setFileSeconds(null);
        if (!file) return undefined;
        readVideoDuration(file).then((d) => {
            if (cancelled) return;
            setFileSeconds(d);
            if (d != null && d < MIN_SOURCE_SECONDS) {
                track('SourceTooShort', { props: { seconds: String(Math.round(d)) } });
            }
        });
        return () => { cancelled = true; };
    }, [file]);

    useEffect(() => {
        fetch(getApiUrl('/api/config'))
            .then((r) => r.ok ? r.json() : null)
            .then((cfg) => {
                if (cfg && cfg.youtubeUrlEnabled === false) {
                    setYoutubeUrlEnabled(false);
                    setMode('file');
                }
            })
            .catch(() => {});
    }, []);

    // A link pasted in the landing hero: preload it here so the user picks up
    // where they left off. Not auto-submitted — the rights attestation below
    // has to be ticked by the user.
    useEffect(() => {
        let pending = null;
        try {
            pending = localStorage.getItem('os_pending_url');
            if (pending) localStorage.removeItem('os_pending_url');
        } catch { /* ignore */ }
        if (pending) {
            setMode('url');
            setUrl(pending);
        }
    }, []);

    const handleSubmit = (e) => {
        e.preventDefault();
        if (!acknowledged || invalidLength || isProcessing) return;
        const advanced = {
            targetClips: targetClips || null,
            clipMinSeconds: clipMinSeconds || null,
            clipMaxSeconds: clipMaxSeconds || null,
            autoHook,
            autoHookStyle,
            framingMethod: showFraming ? framingMethod : null,
            layout: showLegacyLayout ? layout : null,
            captions,
            sourceStart: range ? Math.round(range.start) : null,
            sourceEnd: range ? Math.round(range.end) : null,
        };
        try {
            localStorage.setItem('os_auto_hook', autoHook ? '1' : '0');
            localStorage.setItem('os_auto_hook_style_v2', autoHookStyle);
            localStorage.setItem('os_layout', layout);
            localStorage.setItem('os_framing_method', framingMethod);
        } catch { /* ignore */ }
        if (mode === 'url' && url.trim()) {
            onProcess({ type: 'url', payload: url.trim(), acknowledged: true, outputFormat, ...advanced });
        } else if (mode === 'file' && file && !fileTooShort) {
            onProcess({ type: 'file', payload: file, acknowledged: true, outputFormat, ...advanced });
        }
    };

    const handleDrop = (e) => {
        e.preventDefault();
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            setFile(e.dataTransfer.files[0]);
            setMode('file');
        }
    };

    return (
        <div className="card p-4 sm:p-6 animate-fade text-left">
            <div className="grid grid-cols-2 gap-1 rounded-input border border-rule bg-paper p-1 mb-4" role="group" aria-label="Video source" data-tutorial="source-tabs">
                <button
                    type="button"
                    aria-pressed={mode === 'file'}
                    onClick={() => setMode('file')}
                    className={`flex items-center justify-center gap-2 rounded-input border px-2 py-2.5 text-sm lowercase transition-colors ${mode === 'file'
                        ? 'text-ink border-rule2 bg-paper3'
                        : 'text-muted border-transparent hover:text-ink2'
                        }`}
                >
                    <Upload size={16} className={`hidden sm:block ${mode === 'file' ? 'text-brass' : ''}`} />
                    Upload File
                </button>
                {youtubeUrlEnabled && (
                    <button
                        type="button"
                        aria-pressed={mode === 'url'}
                        onClick={() => setMode('url')}
                        className={`flex items-center justify-center gap-2 rounded-input border px-2 py-2.5 text-sm lowercase transition-colors ${mode === 'url'
                            ? 'text-ink border-rule2 bg-paper3'
                            : 'text-muted border-transparent hover:text-ink2'
                            }`}
                    >
                        <Link2 size={16} className={`hidden sm:block ${mode === 'url' ? 'text-brass' : ''}`} />
                        Video URL
                    </button>
                )}
            </div>

            <form onSubmit={handleSubmit}>
                {mode === 'url' ? (
                    <div className="space-y-4" data-tutorial="drop-zone">
                        <div className="relative">
                            <input
                                type="url"
                                value={url}
                                onChange={(e) => setUrl(e.target.value)}
                                placeholder="https://... paste a video link"
                                aria-label="Video URL"
                                className="input-field pr-11"
                                required
                            />
                            <div className="absolute inset-y-0 right-2 flex items-center" ref={infoRef}>
                                <button
                                    type="button"
                                    onClick={() => setShowInfo((v) => !v)}
                                    aria-label="Supported platforms"
                                    aria-expanded={showInfo}
                                    className="p-1.5 text-muted hover:text-brass transition-colors"
                                >
                                    <Info size={16} />
                                </button>
                                {showInfo && (
                                    <div className="absolute right-0 top-full mt-2 w-64 z-20 card p-4 text-left animate-fade">
                                        <p className="eyebrow mb-2">Paste a link from</p>
                                        <div className="flex flex-wrap gap-1.5">
                                            {SUPPORTED_PLATFORMS.map((p) => (
                                                <span key={p} className="text-xs px-2 py-0.5 rounded-full bg-paper3 text-ink2">
                                                    {p}
                                                </span>
                                            ))}
                                        </div>
                                        <p className="text-xs text-muted mt-2.5 leading-relaxed">
                                            …and 1,000+ more sites. If a link has a public video, we can usually fetch it.
                                        </p>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                ) : (
                    <div
                        data-tutorial="drop-zone"
                        className={`border-2 border-dashed rounded-card px-4 py-4 sm:p-8 text-center transition-colors ${file ? 'border-brass' : 'border-rule2 hover:border-brass'
                            }`}
                        onDragOver={(e) => e.preventDefault()}
                        onDrop={handleDrop}
                    >
                        {file ? (
                            <div className="flex items-center justify-center gap-3 text-ok min-w-0">
                                <FileVideo size={18} className="shrink-0" />
                                <span className="font-medium truncate">{file.name}</span>
                                <button
                                    type="button"
                                    onClick={() => setFile(null)}
                                    aria-label="Remove video"
                                    className="p-1 text-muted hover:text-ink hover:bg-paper3 rounded-full transition-colors"
                                >
                                    <X size={16} />
                                </button>
                            </div>
                        ) : null}
                        {file && fileTooShort ? (
                            <p className="text-danger text-sm mt-3" role="alert">
                                This video is {Math.round(fileSeconds)}s long. Clip generation needs at least {MIN_SOURCE_SECONDS}s
                                of footage to cut from: it already is a short. Pick a longer video.
                            </p>
                        ) : null}
                        {!file && (
                            <label className="cursor-pointer block rounded-input focus-within:ring-2 focus-within:ring-brass">
                                <input
                                    type="file"
                                    accept="video/*"
                                    onChange={(e) => setFile(e.target.files?.[0] || null)}
                                    className="sr-only"
                                    aria-label="Choose a video file"
                                />
                                <Upload className="mx-auto mb-2.5 text-muted" size={18} />
                                <p className="text-ink2 lowercase">Choose a video or drop it here</p>
                                <p className="readout mt-2">{MIN_SOURCE_SECONDS}s minimum</p>
                            </label>
                        )}
                    </div>
                )}

                <SourceCost
                    mode={mode}
                    url={url}
                    fileSeconds={fileSeconds}
                    range={range}
                    onRangeChange={setRange}
                />

                {/* Output format selector */}
                <div className="mt-4 sm:mt-5" data-tutorial="output-format">
                    <p className="eyebrow mb-2">Output format</p>
                    <div className="grid grid-cols-3 gap-2" role="group" aria-label="Output format">
                        {[
                            { value: 'vertical', label: '9:16', hint: 'Vertical', w: 15, h: 26 },
                            { value: 'square', label: '1:1', hint: 'Square', w: 22, h: 22 },
                            { value: 'horizontal', label: '16:9', hint: 'Landscape', w: 30, h: 17 },
                        ].map((f) => {
                            const active = outputFormat === f.value;
                            return (
                                <button
                                    key={f.value}
                                    type="button"
                                    aria-pressed={active}
                                    onClick={() => setOutputFormat(f.value)}
                                    className={`py-2 sm:py-3 px-2 rounded-input border flex flex-col items-center gap-1.5 transition-colors
                                        ${active ? 'border-brass bg-paper3 text-ink' : 'border-rule2 bg-paper text-muted hover:border-brass'}`}
                                >
                                    {/* Aspect-ratio glyph */}
                                    <span className="flex h-7 items-center" aria-hidden="true">
                                        <span
                                            className={`block rounded-[3px] border-2 transition-colors ${active ? 'border-brass bg-brass/10' : 'border-rule2'}`}
                                            style={{ width: f.w, height: f.h }}
                                        />
                                    </span>
                                    <span className="block font-mono text-sm leading-none">{f.label}</span>
                                    <span className="block text-[11px] sm:text-[10px] leading-tight text-center text-muted">{f.hint}</span>
                                </button>
                            );
                        })}
                    </div>
                </div>

                {showFraming && (
                    <div className="mt-4 rounded-input border border-rule bg-paper px-3 py-3 sm:px-4">
                        <div className="grid gap-2 sm:grid-cols-[1fr_1fr] sm:items-center">
                            <label htmlFor={`${controlsId}-framing`} className="eyebrow block">Framing</label>
                            <select
                                id={`${controlsId}-framing`}
                                aria-describedby={`${controlsId}-framing-help`}
                                className="input-field text-sm"
                                value={framingMethod}
                                onChange={(e) => setFramingMethod(e.target.value)}
                            >
                                {FRAMING_OPTIONS.map(({ value, label }) => <option key={value} value={value}>{label}</option>)}
                            </select>
                        </div>
                        <p id={`${controlsId}-framing-help`} className="text-xs text-muted mt-2">
                            {framingMethod !== 'legacy' && <span className="text-ink2">No face detection. </span>}{framing.hint}
                        </p>
                        {showLegacyLayout && (
                            <label className="mt-3 grid gap-2 sm:grid-cols-[1fr_1fr] sm:items-center">
                                <span className="eyebrow">Speaker layout</span>
                                <select value={layout} onChange={(e) => setLayout(e.target.value)} className="input-field text-sm">
                                    <option value="auto">Automatic layout</option>
                                    <option value="split">Two speakers stacked</option>
                                    <option value="screencast">Screen over presenter</option>
                                    <option value="none">Single crop only</option>
                                </select>
                            </label>
                        )}
                    </div>
                )}

                {/* Advanced generation controls — collapsed by default; blank = AI decides */}
                <div className="mt-4 border-y border-rule">
                    <button
                        type="button"
                        onClick={() => setShowAdvanced((v) => !v)}
                        aria-expanded={showAdvanced}
                        aria-controls={`${controlsId}-options`}
                        className="flex w-full items-center justify-between gap-2 py-3 text-sm text-ink2 hover:text-brass lowercase transition-colors"
                    >
                        <span>Clip settings</span>
                        <ChevronDown size={14} className={`transition-transform ${showAdvanced ? 'rotate-180' : ''}`} />
                    </button>
                    <div id={`${controlsId}-options`} hidden={!showAdvanced}>
                        <div className="pb-3 grid grid-cols-1 sm:grid-cols-3 gap-3 sm:gap-2">
                            <label>
                                <span className="eyebrow mb-1.5 block">Clip target</span>
                                <input
                                    type="number" min="1" max="15" step="1"
                                    value={targetClips}
                                    onChange={(e) => setTargetClips(e.target.value)}
                                    placeholder="auto"
                                    className="input-field"
                                />
                            </label>
                            <label>
                                <span className="eyebrow mb-1.5 block">Min length (s)</span>
                                <input
                                    type="number" min="5" max="175" step="1"
                                    value={clipMinSeconds}
                                    onChange={(e) => setClipMinSeconds(e.target.value)}
                                    placeholder="15"
                                    className="input-field"
                                />
                            </label>
                            <label>
                                <span className="eyebrow mb-1.5 block">Max length (s)</span>
                                <input
                                    type="number" min="10" max="180" step="1"
                                    value={clipMaxSeconds}
                                    onChange={(e) => setClipMaxSeconds(e.target.value)}
                                    placeholder="60"
                                    className="input-field"
                                    aria-invalid={invalidLength}
                                    aria-describedby={invalidLength ? `${controlsId}-length-error` : undefined}
                                />
                            </label>
                            {plan === 'free' && (
                                <p className="sm:col-span-3 text-xs leading-relaxed text-muted">
                                    Free runs can generate up to {FREE_CLIPS_PER_VIDEO}–{FREE_CLIPS_QUIET} clips depending on the queue. Paid runs can generate up to {MAX_CLIPS_PER_VIDEO}.
                                </p>
                            )}
                            <label className="sm:col-span-3 flex items-center gap-2 text-sm text-ink2 pt-2 border-t border-rule">
                                <input type="checkbox" checked={captions} onChange={(e) => setCaptions(e.target.checked)} className="w-4 h-4 accent-brass" />
                                Add captions
                            </label>
                            <div className="col-span-1 sm:col-span-3 flex flex-wrap items-center justify-between gap-3 pt-3 sm:pt-1 border-t border-rule">
                                <label className="flex items-center gap-2 text-xs text-ink2 cursor-pointer select-none">
                                    <input
                                        type="checkbox"
                                        checked={autoHook}
                                        onChange={(e) => setAutoHook(e.target.checked)}
                                        className="w-4 h-4 shrink-0 accent-[var(--color-accent)] cursor-pointer"
                                    />
                                    Add opening title
                                </label>
                                {autoHook && (
                                    <select
                                        value={autoHookStyle}
                                        onChange={(e) => setAutoHookStyle(e.target.value)}
                                        className="input-field !w-auto text-xs py-1.5"
                                        aria-label="Opening title style"
                                    >
                                        <option value="pill">Pills</option>
                                        <option value="classic">Classic</option>
                                        <option value="dark">Dark</option>
                                        <option value="yellow">Yellow</option>
                                        <option value="red">Red</option>
                                        <option value="outline">Outline</option>
                                        <option value="outline_yellow">Outline+</option>
                                    </select>
                                )}
                            </div>
                        </div>
                    </div>
                </div>
                {invalidLength && (
                    <p id={`${controlsId}-length-error`} className="mt-2 text-xs text-danger" role="alert">
                        Set the maximum length at least 5 seconds above the minimum.
                    </p>
                )}

                <label className="flex items-start gap-2.5 mt-5 text-left text-[13px] sm:text-xs leading-relaxed text-muted cursor-pointer select-none">
                    <input
                        type="checkbox"
                        checked={acknowledged}
                        onChange={(e) => setAcknowledged(e.target.checked)}
                        className="mt-0.5 w-4 h-4 shrink-0 accent-[var(--color-accent)] cursor-pointer"
                    />
                    <span>
                        I confirm I own this content or have the rights to process it. I am responsible for any content I submit. See our <a href="/terms" target="_blank" rel="noopener noreferrer" className="text-ink2 underline underline-offset-2 hover:text-brass transition-colors" onClick={(e) => e.stopPropagation()}>Terms</a> and <a href="/privacy" target="_blank" rel="noopener noreferrer" className="text-ink2 underline underline-offset-2 hover:text-brass transition-colors" onClick={(e) => e.stopPropagation()}>Privacy Policy</a>.
                    </span>
                </label>

                <button
                    type="submit"
                    data-tutorial="generate"
                    disabled={isProcessing || !acknowledged || invalidLength || (mode === 'url' && !url.trim()) || (mode === 'file' && (!file || fileTooShort))}
                    className="w-full btn-primary mt-4"
                >
                    {isProcessing ? (
                        <>
                            <Loader2 size={16} className="animate-spin" />
                            Processing Video...
                        </>
                    ) : (
                        <>
                            Generate Clips
                        </>
                    )}
                </button>
            </form>
        </div>
    );
}
