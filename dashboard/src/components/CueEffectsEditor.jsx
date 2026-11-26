import { useCallback, useEffect, useRef, useState } from 'react';
import { Play, Plus, Trash2 } from 'lucide-react';
import { getApiUrl } from '../config';
import { roundBrollTime as milliseconds } from '../lib/brollTimeline';

const DEFAULT_VOLUME = 0.35;
const PREVIEW_FILTERS = {
    color_pop: 'contrast(1.08) saturate(1.3)',
    bw_moment: 'grayscale(1)',
};

function sourceAtPlayhead(segments, playhead) {
    let elapsed = 0;
    for (const segment of segments) {
        const length = segment.end - segment.start;
        if (playhead < elapsed + length) return milliseconds(segment.start + Math.max(0, playhead - elapsed));
        elapsed += length;
    }
    return null;
}

function cuePosition(segments, time, minimum) {
    if (time == null) return null;
    let elapsed = 0;
    for (const { start, end } of segments) {
        if (time >= start && milliseconds(end - time) >= minimum) return milliseconds(elapsed + time - start);
        elapsed += end - start;
    }
    return null;
}

export default function CueEffectsEditor({
    cues = [], setCues, catalog = [], maxCues = 12, segments, playhead, selectedWord,
    videoUrl, getPreviewTime, seek, disabled,
}) {
    const [selectedKind, setSelectedKind] = useState('');
    const [visual, setVisual] = useState(null);
    const [error, setError] = useState('');
    const previewAudio = useRef(null);
    const previewTimer = useRef(null);
    const previewId = useRef(0);
    const vignetteRef = useRef(null);
    const selected = catalog.find((effect) => effect.kind === selectedKind) || catalog[0];
    const atPlayhead = sourceAtPlayhead(segments, playhead);
    const atWord = selectedWord?.s ?? null;
    const canAdd = !disabled && cues.length < maxCues;

    const stopAudio = useCallback(() => {
        clearTimeout(previewTimer.current);
        previewAudio.current?.pause();
        previewAudio.current = null;
    }, []);

    useEffect(() => stopAudio, [stopAudio]);

    if (!selected) return null;

    const change = (index, patch) => setCues(cues.map((cue, i) => i === index ? { ...cue, ...patch } : cue));
    const add = (time) => {
        if (!canAdd || cuePosition(segments, time, selected.min_duration) == null) return;
        setCues([...cues, {
            kind: selected.kind, source_time: milliseconds(time), duration: selected.duration, volume: DEFAULT_VOLUME,
        }]);
    };

    const preview = async (cue, effect, clipTime = playhead) => {
        stopAudio();
        setError('');
        setVisual(null);
        if (effect.category === 'Sound effect') {
            const audio = new Audio(getApiUrl(`/api/clip-cues/sounds/${effect.kind}`));
            previewAudio.current = audio;
            audio.volume = cue.volume;
            try {
                await audio.play();
                if (previewAudio.current === audio) previewTimer.current = setTimeout(stopAudio, cue.duration * 1000);
            } catch {
                if (previewAudio.current === audio) setError('Could not play this sound effect.');
            }
        } else if (videoUrl) {
            previewId.current += 1;
            setVisual({
                kind: effect.kind, duration: cue.duration, id: previewId.current,
                time: getPreviewTime(clipTime) ?? getPreviewTime(playhead) ?? 0,
            });
        }
    };

    const animatePreview = (event) => {
        const video = event.currentTarget;
        const duration = visual.duration * 1000;
        if (visual.kind === 'dip_to_black') {
            video.animate([{ filter: 'brightness(1)' }, { filter: 'brightness(0)' }, { filter: 'brightness(1)' }], duration);
        } else if (visual.kind === 'vignette') {
            vignetteRef.current?.animate([{ opacity: 1 }, { opacity: 1 }], duration);
        } else {
            const filter = PREVIEW_FILTERS[visual.kind];
            if (filter) video.animate([{ filter }, { filter }], duration);
        }
    };

    return (
        <section aria-label="Timed effects" className="space-y-2 border-t border-rule pt-3">
            <div className="flex items-center justify-between gap-2">
                <p className="eyebrow">Timed effects</p>
                <span className="readout">{cues.length}/{maxCues}</span>
            </div>
            <label className="block">
                <span className="sr-only">Choose an effect</span>
                <select value={selected.kind} disabled={disabled} className="input-field py-1.5"
                    onChange={(event) => { setSelectedKind(event.target.value); stopAudio(); setVisual(null); }}>
                    {['Sound effect', 'Transition', 'Video effect'].map((category) => (
                        <optgroup key={category} label={category}>
                            {catalog.filter((effect) => effect.category === category).map((effect) => (
                                <option key={effect.kind} value={effect.kind}>{effect.label}</option>
                            ))}
                        </optgroup>
                    ))}
                </select>
            </label>
            <div className="flex flex-wrap gap-1.5">
                <button type="button" className="btn-ghost py-1 text-xs" onClick={() => add(atPlayhead)}
                    disabled={!canAdd || cuePosition(segments, atPlayhead, selected.min_duration) == null}>
                    <Plus size={13} /> at playhead
                </button>
                <button type="button" className="btn-ghost py-1 text-xs" onClick={() => add(atWord)}
                    disabled={!canAdd || cuePosition(segments, atWord, selected.min_duration) == null}>
                    <Plus size={13} /> at selected word
                </button>
                <button type="button" className="btn-ghost py-1 text-xs"
                    onClick={() => preview({ volume: DEFAULT_VOLUME, duration: selected.duration }, selected)}
                    disabled={disabled || (!videoUrl && selected.category !== 'Sound effect')}>
                    <Play size={13} /> preview effect
                </button>
            </div>
            {error && <p role="alert" className="text-danger text-xs">{error}</p>}
            {visual && (
                <div className="relative mx-auto max-w-36 overflow-hidden rounded-input bg-black">
                    <video key={visual.id} src={videoUrl} muted playsInline preload="auto" aria-label="Timed video effect preview"
                        className="block h-32 w-full object-contain"
                        onLoadedData={(event) => {
                            const video = event.currentTarget;
                            const when = Math.max(0, Math.min(visual.time, video.duration - 0.01));
                            if (when > 0) video.currentTime = when;
                            else animatePreview(event);
                        }}
                        onSeeked={animatePreview}
                        onError={() => {
                            setVisual(null);
                            setError('Could not preview this frame.');
                        }} />
                    {visual.kind === 'vignette' && (
                        <div ref={vignetteRef} className="pointer-events-none absolute inset-0 opacity-0"
                            style={{ boxShadow: 'inset 0 0 28px 10px black' }} />
                    )}
                </div>
            )}
            {cues.map((cue, index) => {
                const effect = catalog.find((item) => item.kind === cue.kind);
                if (!effect) return null;
                const time = cuePosition(segments, cue.source_time, effect.min_duration);
                return (
                    <div key={index} className="space-y-2 rounded-input border border-rule px-2.5 py-2">
                        <div className="flex flex-wrap items-center gap-2">
                            <span className="mr-auto text-xs text-ink">{effect.label}</span>
                            {time == null ? <span className="text-xs text-muted">outside this cut</span> : (
                                <button type="button" onClick={() => seek?.(time)} disabled={!seek || disabled}
                                    className="readout px-1.5 py-1">{time.toFixed(2)}s</button>
                            )}
                            <button type="button" onClick={() => preview(cue, effect, time ?? playhead)}
                                disabled={disabled || (!videoUrl && effect.category !== 'Sound effect')}
                                className="p-1 text-muted hover:text-ink" aria-label={`Preview ${effect.label}`}>
                                <Play size={13} />
                            </button>
                            <button type="button" disabled={disabled} aria-label={`Remove ${effect.label}`}
                                className="p-1 text-muted hover:text-danger"
                                onClick={() => setCues(cues.filter((_item, i) => i !== index))}>
                                <Trash2 size={13} />
                            </button>
                        </div>
                        <label className="flex items-center gap-2 text-xs text-muted">
                            <span>Duration · {cue.duration.toFixed(2)}s</span>
                            <input type="range" className="min-w-0 flex-1 accent-brass" disabled={disabled}
                                min={effect.min_duration} max={effect.max_duration} step={0.01} value={cue.duration}
                                onChange={(event) => change(index, { duration: Number(event.target.value) })} />
                        </label>
                        {effect.category === 'Sound effect' && (
                            <label className="flex items-center gap-2 text-xs text-muted">
                                <span>Volume · {Math.round(cue.volume * 100)}%</span>
                                <input type="range" className="min-w-0 flex-1 accent-brass" disabled={disabled}
                                    min={0} max={1} step={0.01} value={cue.volume}
                                    onChange={(event) => change(index, { volume: Number(event.target.value) })} />
                            </label>
                        )}
                    </div>
                );
            })}
        </section>
    );
}
