import React, { useState, useEffect, useId, useRef } from 'react';
import { Link2, Upload, FileVideo, X, Info, Loader2, ChevronDown } from 'lucide-react';
import { track } from '../lib/analytics';
import { getApiUrl } from '../config';
import { apiJson } from '../lib/api';
import SourceCost from './SourceCost';
import { useAuth } from '../contexts/AuthContext';
import { FREE_CLIPS_PER_VIDEO, FREE_CLIPS_QUIET, MAX_CLIPS_PER_VIDEO } from '../lib/clipProgress';

const SUPPORTED_PLATFORMS = [
    'YouTube', 'Vimeo', 'TikTok', 'X / Twitter', 'Twitch', 'Kick',
    'Facebook', 'Instagram', 'Dailymotion', 'Reddit', 'Streamable',
];

const FRAMING_OPTIONS = [
    { value: 'auto', label: 'Smart framing', hint: 'Uses motion and visual detail, and keeps the whole picture when the focus is unclear.' },
    { value: 'gameplay', label: 'Gameplay priority', hint: 'The game large, with its original facecam kept whole inside it. When the moment is the streamer\'s reaction, the facecam gets its own band. Auto captions use a smaller style.' },
    { value: 'streamer', label: 'Streamer stack', hint: 'Facecam on top, gameplay below, the classic stream-clip format. The facecam is sized from its own resolution and the streamer appears once. Falls back to Gameplay priority when no webcam overlay is found.' },
    { value: 'motion', label: 'Follow motion', hint: 'Frames the action in moving footage, such as gameplay and sports.' },
    { value: 'sports', label: 'Sports action', hint: 'Follows the ball when it can be tracked through a shot. Otherwise it follows clear motion or keeps the whole picture.' },
    { value: 'detail', label: 'Follow detail', hint: 'Frames the strongest visual detail in steadier shots.' },
    { value: 'center', label: 'Center crop', hint: 'Fills the output frame with a fixed crop from the center.' },
    { value: 'fit', label: 'Keep full picture', hint: 'Keeps the entire source visible, with a blurred background around it.' },
    { value: 'legacy', label: 'Speaker and screen layouts', hint: 'Uses face detection for speaker tracking, stacked speakers, and presenter layouts.' },
];
// How large the streamer is in a game framing (composition.py). 'auto' follows
// the moment: the streamer's reaction gets a larger facecam, a play the game.
const FOCUS_OPTIONS = [
    { value: 'auto', label: 'Auto (follows the moment)' },
    { value: 'streamer', label: 'Large streamer' },
    { value: 'balanced', label: 'Balanced' },
    { value: 'gameplay', label: 'Small streamer, more game' },
];
const configuredFraming = import.meta.env.VITE_DEFAULT_FRAMING_METHOD;
const defaultFraming = FRAMING_OPTIONS.some(({ value }) => value === configuredFraming)
    ? configuredFraming : 'auto';

// Mirrors the server's MIN_SOURCE_SECONDS: shorter sources are rejected with a
// 400 after the whole file was uploaded. Checking the duration in the browser
// says so the moment the file is picked (a 27 s upload used to end in a bare
// "That run failed").
const MIN_SOURCE_SECONDS = 45;

const youtubeLink = /^https?:\/\/(?:(?:www\.|m\.|music\.)?youtube\.com|youtu\.be|(?:www\.)?youtube-nocookie\.com)\//i;

function audioLanguageLabel(code) {
    const descriptive = code.endsWith('-desc');
    const tag = descriptive ? code.slice(0, -5) : code;
    let name = tag;
    try { name = new Intl.DisplayNames(undefined, { type: 'language' }).of(tag) || tag; } catch { /* show the code */ }
    return descriptive ? `${name} (audio description)` : name;
}

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
    const [reviewAvailable, setReviewAvailable] = useState(false);
    const [reviewCandidates, setReviewCandidates] = useState(true);
    const [discoveryAvailable, setDiscoveryAvailable] = useState(false);
    const [discovery, setDiscovery] = useState(false);
    const [audioFirst, setAudioFirst] = useState(false);
    const [chatReplay, setChatReplay] = useState(null);
    const [chatOffset, setChatOffset] = useState('');
    // File upload is the primary path; the link is secondary.
    const [mode, setMode] = useState('file'); // 'file' | 'url'
    const [url, setUrl] = useState('');
    const [audioLookup, setAudioLookup] = useState(null);
    const [sourceAudio, setSourceAudio] = useState(null);
    const [file, setFile] = useState(null);
    const [fileSeconds, setFileSeconds] = useState(null);
    const [exactRange, setExactRange] = useState(false);
    const fileTooShort = !exactRange && fileSeconds != null && fileSeconds < MIN_SOURCE_SECONDS;
    const [acknowledged, setAcknowledged] = useState(false);
    // {start, end} in seconds when the user picked a stretch under the cost line.
    const [range, setRange] = useState(null);
    const [outputFormat, setOutputFormat] = useState('vertical'); // vertical | horizontal | square
    const [framingMethod, setFramingMethod] = useState(() => {
        try {
            const saved = localStorage.getItem('os_framing_method');
            if (FRAMING_OPTIONS.some(({ value }) => value === saved)) return saved;
            // Existing explicit speaker layouts belong to the face-based renderer.
            return ['split', 'screencast', 'none'].includes(localStorage.getItem('os_layout')) ? 'legacy' : defaultFraming;
        } catch { return defaultFraming; }
    });
    const [captions, setCaptions] = useState(true);
    const [endcard, setEndcard] = useState(null);
    // The viral edit for game framings (viral_edit.py), remembered per browser.
    const [viralEdit, setViralEdit] = useState(() => {
        try { return localStorage.getItem('os_viral_edit') !== '0'; } catch (e) { return true; }
    });
    const [framingFocus, setFramingFocus] = useState(() => {
        try {
            const saved = localStorage.getItem('os_framing_focus');
            return FOCUS_OPTIONS.some(({ value }) => value === saved) ? saved : 'auto';
        } catch (e) { return 'auto'; }
    });
    const [soundEffects, setSoundEffects] = useState(() => {
        try { return localStorage.getItem('os_sound_effects') !== '0'; } catch (e) { return true; }
    });
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
    const invalidLength = !exactRange && clipMinSeconds !== '' && clipMaxSeconds !== ''
        && Number(clipMaxSeconds) < Number(clipMinSeconds) + 5;
    const reviewDiscovery = !exactRange && discoveryAvailable && reviewCandidates && discovery;
    const sourceUrl = url.trim();
    const canChooseAudio = mode === 'url' && youtubeLink.test(sourceUrl);
    const audioTracks = audioLookup?.url === sourceUrl ? audioLookup : null;
    const audioLanguage = sourceAudio?.url === sourceUrl ? sourceAudio.language : '';
    const canDeferVideo = mode === 'url' && !range
        && (canChooseAudio || /^https?:\/\/(?:www\.|m\.)?twitch\.tv\/videos\/\d+/i.test(sourceUrl));
    const showFraming = outputFormat !== 'horizontal';
    const showLegacyLayout = showFraming && framingMethod === 'legacy';
    const gameplayLike = showFraming && (framingMethod === 'gameplay' || framingMethod === 'streamer');
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
                if (cfg?.candidateReview === true) setReviewAvailable(true);
                if (cfg?.highlightDiscovery === true) setDiscoveryAvailable(true);
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

    const checkAudioTracks = async () => {
        const request = { url: sourceUrl, loading: true };
        setAudioLookup(request);
        try {
            const data = await apiJson('/api/process/audio-tracks', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ url: sourceUrl }),
            });
            setAudioLookup((current) => current === request
                ? { url: sourceUrl, languages: data.audio_languages } : current);
        } catch (err) {
            setAudioLookup((current) => current === request
                ? { url: sourceUrl, error: err.detail || 'Could not check the source audio tracks.' } : current);
        }
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        if (!acknowledged || invalidLength || isProcessing) return;
        const advanced = {
            exactRange,
            targetClips: exactRange ? null : targetClips || null,
            clipMinSeconds: exactRange ? null : clipMinSeconds || null,
            clipMaxSeconds: exactRange ? null : clipMaxSeconds || null,
            autoHook: !exactRange && autoHook,
            autoHookStyle,
            framingMethod: showFraming ? framingMethod : null,
            sourceAudioLanguage: canChooseAudio ? audioLanguage || null : null,
            layout: showLegacyLayout ? layout : null,
            captions,
            endcard,
            viralEdit: gameplayLike ? viralEdit : null,
            framingFocus: gameplayLike && framingFocus !== 'auto' ? framingFocus : null,
            soundEffects: gameplayLike && viralEdit ? soundEffects : null,
            reviewCandidates: !exactRange && reviewAvailable && reviewCandidates,
            highlightDiscovery: reviewDiscovery,
            audioFirst: reviewDiscovery && canDeferVideo && audioFirst,
            chatReplay: reviewDiscovery ? chatReplay : null,
            chatOffset: reviewDiscovery && chatReplay && chatOffset ? Number(chatOffset) : null,
            sourceStart: range ? Number(range.start) : null,
            sourceEnd: range ? Number(range.end) : null,
        };
        try {
            localStorage.setItem('os_auto_hook', autoHook ? '1' : '0');
            localStorage.setItem('os_viral_edit', viralEdit ? '1' : '0');
            localStorage.setItem('os_sound_effects', soundEffects ? '1' : '0');
            localStorage.setItem('os_auto_hook_style_v2', autoHookStyle);
            localStorage.setItem('os_layout', layout);
            localStorage.setItem('os_framing_method', framingMethod);
            localStorage.setItem('os_framing_focus', framingFocus);
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
                        {canChooseAudio && (
                            <div className="space-y-1.5">
                                {audioTracks?.languages?.length ? (
                                    <label className="block">
                                        <span className="eyebrow mb-1.5 block">Source audio</span>
                                        <select aria-label="Source audio" className="input-field" value={audioLanguage}
                                            onChange={(e) => setSourceAudio({ url: sourceUrl, language: e.target.value })}>
                                            <option value="">Default audio</option>
                                            {audioTracks.languages.map((code) => (
                                                <option key={code} value={code}>{audioLanguageLabel(code)}</option>
                                            ))}
                                        </select>
                                    </label>
                                ) : (
                                    <button type="button" onClick={checkAudioTracks} disabled={audioTracks?.loading}
                                        className="flex items-center gap-2 text-xs text-brass hover:underline disabled:opacity-70">
                                        {audioTracks?.loading && <Loader2 size={12} className="animate-spin" />}
                                        {audioTracks?.loading ? 'Checking source audio…' : 'Check source audio tracks'}
                                    </button>
                                )}
                                {audioTracks?.error && <p className="text-xs text-danger" role="alert">{audioTracks.error}</p>}
                                {audioTracks?.languages?.length === 0 && (
                                    <p className="text-xs text-muted" role="status">This source will use its default audio.</p>
                                )}
                            </div>
                        )}
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
                                <p className="readout mt-2">{exactRange ? 'MP4, MOV or WebM' : `${MIN_SOURCE_SECONDS}s minimum`}</p>
                            </label>
                        )}
                    </div>
                )}

                <label className="flex items-start gap-2.5 mt-4 text-sm text-ink2">
                    <input type="checkbox" checked={exactRange} className="mt-0.5 w-4 h-4 accent-brass"
                        onChange={(e) => { setExactRange(e.target.checked); setRange(null); }} />
                    <span>Export one clip
                        <span className="block text-xs text-muted mt-1">Keep the full source or select a section to export with your chosen framing and captions.</span>
                    </span>
                </label>
                <SourceCost
                    key={exactRange ? 'exact' : 'highlights'}
                    mode={mode}
                    url={url}
                    file={file}
                    fileSeconds={fileSeconds}
                    range={range}
                    onRangeChange={setRange}
                    localRanges={exactRange || reviewAvailable}
                    exactRange={exactRange}
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
                            {framingMethod !== 'legacy' && framingMethod !== 'streamer' && <span className="text-ink2">No face detection. </span>}{framing.hint}
                        </p>
                        {gameplayLike && (
                            <label className="mt-3 grid gap-2 sm:grid-cols-[1fr_1fr] sm:items-center">
                                <span className="eyebrow">Streamer size</span>
                                <select value={framingFocus} onChange={(e) => setFramingFocus(e.target.value)}
                                    className="input-field text-sm" aria-describedby={`${controlsId}-focus-hint`}>
                                    {FOCUS_OPTIONS.map(({ value, label }) => <option key={value} value={value}>{label}</option>)}
                                </select>
                                <span id={`${controlsId}-focus-hint`} className="text-xs text-muted sm:col-span-2">
                                    Sized from the webcam's own resolution, never blown up past it. The streamer is shown once.
                                </span>
                            </label>
                        )}
                        {gameplayLike && !exactRange && (
                            <div className="mt-3 space-y-2 text-sm text-ink2">
                                <label className="flex items-start gap-2">
                                    <input type="checkbox" checked={viralEdit} className="mt-0.5 w-4 h-4 accent-brass"
                                        onChange={(e) => setViralEdit(e.target.checked)} />
                                    <span>Viral edit
                                        <span className="block text-xs text-muted mt-0.5">Opens on the reaction, then the lead-up. Zooms in on the streamer, adds emoji and bold captions, tightens dead air.</span>
                                    </span>
                                </label>
                                {viralEdit && (
                                    <label className="flex items-center gap-2 pl-6">
                                        <input type="checkbox" checked={soundEffects} className="w-4 h-4 accent-brass"
                                            onChange={(e) => setSoundEffects(e.target.checked)} />
                                        Sound effects
                                    </label>
                                )}
                            </div>
                        )}
                        {showLegacyLayout && (
                            <label className="mt-3 grid gap-2 sm:grid-cols-[1fr_1fr] sm:items-center">
                                <span className="eyebrow">Speaker layout</span>
                                <select value={layout} onChange={(e) => setLayout(e.target.value)} className="input-field text-sm">
                                    <option value="auto">Automatic layout</option>
                                    <option value="split">Two speakers stacked</option>
                                    <option value="screencast">Screen over separate presenter</option>
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
                            {!exactRange && <>
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
                            </>}
                            <label className="sm:col-span-3 flex items-center gap-2 text-sm text-ink2 pt-2 border-t border-rule">
                                <input type="checkbox" checked={captions} onChange={(e) => setCaptions(e.target.checked)} className="w-4 h-4 accent-brass" />
                                Add captions
                            </label>
                            <div className="sm:col-span-3 space-y-2 border-t border-rule pt-3">
                                <label className="flex items-center gap-2 text-sm text-ink2">
                                    <input type="checkbox" checked={!!endcard} className="w-4 h-4 accent-brass"
                                        onChange={(event) => setEndcard(event.target.checked ? { text: 'Follow for more', handle: '', duration: 2, sound: 'chime' } : null)} />
                                    Add end card
                                </label>
                                {endcard && <div className="grid sm:grid-cols-2 gap-2">
                                    <label className="text-sm">Message<input className="input-field mt-1" maxLength={96} value={endcard.text}
                                        onChange={(event) => setEndcard({ ...endcard, text: event.target.value })} /></label>
                                    <label className="text-sm">Channel handle<input className="input-field mt-1" maxLength={40} value={endcard.handle}
                                        onChange={(event) => setEndcard({ ...endcard, handle: event.target.value })} /></label>
                                    <label className="text-sm">End card length<select className="input-field mt-1" value={endcard.duration}
                                        onChange={(event) => setEndcard({ ...endcard, duration: Number(event.target.value) })}>
                                        {[1, 2, 3].map((seconds) => <option key={seconds} value={seconds}>{seconds} seconds</option>)}
                                    </select></label>
                                    <label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={endcard.sound === 'chime'}
                                        className="w-4 h-4 accent-brass" onChange={(event) => setEndcard({ ...endcard, sound: event.target.checked ? 'chime' : 'silent' })} />Chime</label>
                                </div>}
                            </div>
                            {!exactRange && <div className="col-span-1 sm:col-span-3 flex flex-wrap items-center justify-between gap-3 pt-3 sm:pt-1 border-t border-rule">
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
                            </div>}
                        </div>
                    </div>
                </div>
                {invalidLength && (
                    <p id={`${controlsId}-length-error`} className="mt-2 text-xs text-danger" role="alert">
                        Set the maximum length at least 5 seconds above the minimum.
                    </p>
                )}

                {reviewAvailable && !exactRange && (
                    <label className="flex items-start gap-2.5 mt-4 text-sm text-ink2">
                        <input type="checkbox" className="mt-0.5 w-4 h-4 accent-brass"
                            checked={reviewCandidates} onChange={(e) => setReviewCandidates(e.target.checked)} />
                        <span>
                            Review moments before rendering
                            <span className="block text-xs text-muted mt-1">Preview source ranges and render only the ones you choose.</span>
                        </span>
                    </label>
                )}
                {discoveryAvailable && reviewCandidates && !exactRange && (
                    <fieldset className="mt-4 p-3 rounded-input border border-rule space-y-3 text-sm">
                        <label className="flex items-center gap-2 text-ink2">
                            <input type="checkbox" checked={discovery} className="w-4 h-4 accent-brass"
                                onChange={(e) => setDiscovery(e.target.checked)} />
                            Livestream highlights
                        </label>
                        {discovery && (
                            <>
                                {canDeferVideo && (
                                    <label className="flex items-start gap-2 text-ink2">
                                        <input type="checkbox" checked={audioFirst} className="mt-0.5 w-4 h-4 accent-brass"
                                            onChange={(e) => setAudioFirst(e.target.checked)} />
                                        <span>Find moments before downloading full-quality footage
                                            <span className="block text-xs text-muted mt-1">For completed YouTube recordings and Twitch VODs: scans the audio and chat, then fetches only the approved moments in full quality.</span>
                                        </span>
                                    </label>
                                )}
                                <label className="block text-xs text-muted">
                                    Chat replay for this recording (for Twitch, a TwitchDownloader chat JSON)
                                    <input type="file" accept=".csv,.json,.jsonl" className="input-field mt-1 w-full"
                                        onChange={(e) => setChatReplay(e.target.files?.[0] || null)} />
                                </label>
                                {chatReplay && (
                                    <label className="block text-xs text-muted">
                                        Seconds to add to chat timestamps
                                        <input type="number" step="0.1" value={chatOffset}
                                            className="input-field mt-1 w-full" placeholder="0"
                                            onChange={(e) => setChatOffset(e.target.value)} />
                                    </label>
                                )}
                            </>
                        )}
                    </fieldset>
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
                            {exactRange ? 'Export clip' : reviewAvailable && reviewCandidates ? 'Find candidate moments' : 'Generate clips'}
                        </>
                    )}
                </button>
            </form>
        </div>
    );
}
