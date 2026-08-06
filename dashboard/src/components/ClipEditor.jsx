import React, { useState, useEffect, useReducer, useRef, useCallback, useMemo } from 'react';
import {
    X, Loader2, Plus, Trash2, ChevronUp, ChevronDown, Scissors,
    AlertCircle, Undo2, Redo2, Film,
} from 'lucide-react';
import { getApiUrl } from '../config';
import { apiFetch, apiJson, QuotaError } from '../lib/api';

// Full-screen clip editor: shows WHICH source segments a clip was cut from,
// lets the user trim/extend/split/reorder them (word-snapped), and re-renders
// through POST /api/clip/rerender. The recipe (EDL) comes from GET .../edl.

const MIN_SEGMENT_SECONDS = 0.5;
const SNAP_WINDOW_SECONDS = 0.35;
const WORD_CONTEXT_SECONDS = 15;

const SEGMENT_COLORS = [
    'oklch(76% .17 50)',   // brass
    'oklch(70% .12 200)',
    'oklch(72% .13 140)',
    'oklch(70% .14 300)',
    'oklch(74% .13 90)',
    'oklch(68% .13 250)',
];

function fmt(t) {
    if (!Number.isFinite(t)) return '–:––';
    const m = Math.floor(t / 60);
    const s = t - m * 60;
    return `${m}:${s.toFixed(1).padStart(4, '0')}`;
}

function totalOf(segments) {
    return segments.reduce((acc, s) => acc + (s.end - s.start), 0);
}

function editorReducer(state, action) {
    switch (action.type) {
        case 'init':
            return { segments: action.segments, selected: 0, past: [], future: [], pendingBase: null };
        case 'select':
            return { ...state, selected: action.index };
        // Live drag feedback: replaces segments without touching history; the
        // pre-drag snapshot is kept so the whole drag undoes as ONE step.
        case 'preview':
            return { ...state, segments: action.segments, pendingBase: state.pendingBase || state.segments };
        case 'commit': {
            const base = state.pendingBase || state.segments;
            return {
                ...state,
                segments: action.segments,
                selected: Math.min(action.select ?? state.selected, action.segments.length - 1),
                past: [...state.past, base],
                future: [],
                pendingBase: null,
            };
        }
        case 'undo': {
            if (!state.past.length) return state;
            const prev = state.past[state.past.length - 1];
            return {
                ...state,
                segments: prev,
                selected: Math.min(state.selected, prev.length - 1),
                past: state.past.slice(0, -1),
                future: [state.segments, ...state.future],
                pendingBase: null,
            };
        }
        case 'redo': {
            if (!state.future.length) return state;
            const next = state.future[0];
            return {
                ...state,
                segments: next,
                selected: Math.min(state.selected, next.length - 1),
                past: [...state.past, state.segments],
                future: state.future.slice(1),
                pendingBase: null,
            };
        }
        default:
            return state;
    }
}

export default function ClipEditor({ jobId, clipIndex, clipTitle, onClose, onRerendered }) {
    const [edl, setEdl] = useState(null);
    const [loadError, setLoadError] = useState(null);
    const [state, dispatch] = useReducer(editorReducer, { segments: [], selected: 0, past: [], future: [], pendingBase: null });
    const { segments, selected } = state;

    const [snapToWords, setSnapToWords] = useState(true);
    const [reapplyCaptions, setReapplyCaptions] = useState(true);
    // The recipe of the currently RENDERED preview (playhead maps onto it).
    const [renderedSegments, setRenderedSegments] = useState(null);
    const [previewUrl, setPreviewUrl] = useState(null);
    const [rendering, setRendering] = useState(false);
    const [renderSeconds, setRenderSeconds] = useState(0);
    const [renderError, setRenderError] = useState(null);
    const [confirmClose, setConfirmClose] = useState(false);
    const [selectedWord, setSelectedWord] = useState(null);
    const [playhead, setPlayhead] = useState(0);
    const [showSource, setShowSource] = useState(false);

    const videoRef = useRef(null);
    const sourceRef = useRef(null);
    const clipTrackRef = useRef(null);
    const sourceTrackRef = useRef(null);
    const dragRef = useRef(null);
    const [ghost, setGhost] = useState(null); // in-progress new segment on the source track

    // ---- load the EDL -------------------------------------------------------
    useEffect(() => {
        let cancelled = false;
        (async () => {
            try {
                const data = await apiJson(`/api/clip/${jobId}/${clipIndex}/edl`);
                if (cancelled) return;
                setEdl(data);
                dispatch({ type: 'init', segments: data.segments.map((s) => ({ ...s })) });
                setRenderedSegments(data.segments.map((s) => ({ ...s })));
                setReapplyCaptions(true);
                setPreviewUrl(getApiUrl(`/videos/${jobId}/${data.current_file}`));
            } catch (e) {
                if (!cancelled) setLoadError(e.detail || e.message || 'could not load the clip recipe');
            }
        })();
        return () => { cancelled = true; };
    }, [jobId, clipIndex]);

    const words = useMemo(() => (edl?.words || []), [edl]);
    const sourceAvailable = !!edl?.source?.available;
    const sourceDuration = edl?.source?.duration || 0;
    const canonical = edl?.canonical_range || { start: 0, end: 0 };
    const limits = edl?.limits || { max_segments: 12, min_segment_seconds: MIN_SEGMENT_SECONDS, max_total_seconds: 180 };
    const minSeg = limits.min_segment_seconds || MIN_SEGMENT_SECONDS;

    const total = totalOf(segments);
    const dirty = useMemo(() => {
        if (!renderedSegments) return false;
        return JSON.stringify(segments) !== JSON.stringify(renderedSegments);
    }, [segments, renderedSegments]);

    // Trim bounds: with the source gone, cuts must stay inside the range the
    // canonical file was rendered from.
    const bounds = sourceAvailable
        ? { lo: 0, hi: sourceDuration || Infinity }
        : { lo: canonical.start, hi: canonical.end };

    const outOfRange = useCallback(
        (seg) => !sourceAvailable && (seg.start < canonical.start - 0.05 || seg.end > canonical.end + 0.05),
        [sourceAvailable, canonical],
    );
    const needsSourcePath = segments.some((s) => s.start < canonical.start - 0.05 || s.end > canonical.end + 0.05);
    const invalidSegments = segments.some(outOfRange);
    const overCaps = segments.length > limits.max_segments || total > limits.max_total_seconds;
    const canRender = !rendering && segments.length > 0 && !invalidSegments && !overCaps
        && segments.every((s) => s.end - s.start >= minSeg);

    // ---- helpers ------------------------------------------------------------
    const snapEdge = useCallback((t, kind) => {
        if (!snapToWords || !words.length) return t;
        let best = null;
        for (const w of words) {
            const c = kind === 'start' ? w.s : w.e;
            if (Math.abs(c - t) <= SNAP_WINDOW_SECONDS && (best === null || Math.abs(c - t) < Math.abs(best - t))) best = c;
        }
        return best ?? t;
    }, [snapToWords, words]);

    const clampSeg = useCallback((seg) => ({
        start: Math.max(bounds.lo, Math.min(seg.start, seg.end - minSeg)),
        end: Math.min(bounds.hi, Math.max(seg.end, seg.start + minSeg)),
    }), [bounds.lo, bounds.hi, minSeg]);

    const setSegment = (index, next, { snap = true } = {}) => {
        const updated = segments.map((s, i) => {
            if (i !== index) return s;
            const seg = { ...s, ...next };
            if (snap) {
                if (next.start !== undefined) seg.start = snapEdge(seg.start, 'start');
                if (next.end !== undefined) seg.end = snapEdge(seg.end, 'end');
            }
            return clampSeg(seg);
        });
        dispatch({ type: 'commit', segments: updated, select: index });
    };

    const addSegment = () => {
        if (segments.length >= limits.max_segments) return;
        const last = segments[segments.length - 1];
        let start = last ? last.end : bounds.lo;
        let end = start + 10;
        if (end > bounds.hi) { end = bounds.hi; start = Math.max(bounds.lo, end - 10); }
        if (end - start < minSeg) return;
        dispatch({ type: 'commit', segments: [...segments, { start: Math.round(start * 1000) / 1000, end: Math.round(end * 1000) / 1000 }], select: segments.length });
    };

    const deleteSegment = (index) => {
        if (segments.length <= 1) return;
        dispatch({ type: 'commit', segments: segments.filter((_, i) => i !== index), select: Math.max(0, index - 1) });
    };

    const moveSegment = (index, dir) => {
        const j = index + dir;
        if (j < 0 || j >= segments.length) return;
        const next = segments.slice();
        [next[index], next[j]] = [next[j], next[index]];
        dispatch({ type: 'commit', segments: next, select: j });
    };

    const splitSegment = (index) => {
        if (segments.length >= limits.max_segments) return;
        const seg = segments[index];
        if (seg.end - seg.start < minSeg * 2) return;
        let at = seg.start + (seg.end - seg.start) / 2;
        // Prefer the playhead when it currently maps inside this segment.
        if (!dirty && renderedSegments) {
            let offset = 0;
            for (let i = 0; i < renderedSegments.length; i += 1) {
                const r = renderedSegments[i];
                const len = r.end - r.start;
                if (i === index && playhead > offset + minSeg && playhead < offset + len - minSeg) {
                    at = r.start + (playhead - offset);
                }
                offset += len;
            }
        }
        at = snapEdge(at, 'end');
        if (at - seg.start < minSeg || seg.end - at < minSeg) at = seg.start + (seg.end - seg.start) / 2;
        const next = segments.flatMap((s, i) => (i === index
            ? [{ start: s.start, end: Math.round(at * 1000) / 1000 }, { start: Math.round(at * 1000) / 1000, end: s.end }]
            : [s]));
        dispatch({ type: 'commit', segments: next, select: index });
    };

    // ---- drag: trim handles on the clip track -------------------------------
    const onDragMove = useCallback((e) => {
        const d = dragRef.current;
        if (!d) return;
        const dt = (e.clientX - d.startX) / d.pxPerSec;
        const seg = { ...d.base[d.idx] };
        if (d.edge === 'start') seg.start = Math.max(d.lo, Math.min(seg.start + dt, seg.end - d.minSeg));
        else seg.end = Math.min(d.hi, Math.max(seg.end + dt, seg.start + d.minSeg));
        const next = d.base.map((s, i) => (i === d.idx ? seg : s));
        d.last = { seg, next };
        dispatch({ type: 'preview', segments: next });
    }, []);

    const onDragUp = useCallback(() => {
        const d = dragRef.current;
        dragRef.current = null;
        window.removeEventListener('pointermove', onDragMove);
        window.removeEventListener('pointerup', onDragUp);
        if (!d || !d.last) return;
        const { seg, next } = d.last;
        const snapped = { ...seg };
        if (d.edge === 'start') snapped.start = Math.max(d.lo, Math.min(d.snap(seg.start, 'start'), seg.end - d.minSeg));
        else snapped.end = Math.min(d.hi, Math.max(d.snap(seg.end, 'end'), seg.start + d.minSeg));
        dispatch({ type: 'commit', segments: next.map((s, i) => (i === d.idx ? snapped : s)), select: d.idx });
    }, [onDragMove]);

    const startTrimDrag = (e, idx, edge, trackEl, secondsOnTrack) => {
        e.preventDefault();
        e.stopPropagation();
        const rect = trackEl?.getBoundingClientRect();
        if (!rect || !secondsOnTrack) return;
        dragRef.current = {
            idx, edge, startX: e.clientX, pxPerSec: rect.width / secondsOnTrack,
            base: segments.map((s) => ({ ...s })), last: null,
            lo: bounds.lo, hi: bounds.hi, minSeg, snap: snapEdge,
        };
        dispatch({ type: 'select', index: idx });
        window.addEventListener('pointermove', onDragMove);
        window.addEventListener('pointerup', onDragUp);
    };

    // ---- drag: paint a NEW segment on the source track ----------------------
    const onGhostMove = useCallback((e) => {
        const d = dragRef.current;
        if (!d || d.kind !== 'ghost') return;
        const t = Math.max(0, Math.min(d.duration, d.t0 + (e.clientX - d.startX) / d.pxPerSec));
        d.range = { start: Math.min(d.t0, t), end: Math.max(d.t0, t) };
        setGhost({ ...d.range });
    }, []);

    const onGhostUp = useCallback(() => {
        const d = dragRef.current;
        dragRef.current = null;
        window.removeEventListener('pointermove', onGhostMove);
        window.removeEventListener('pointerup', onGhostUp);
        setGhost(null);
        if (!d || !d.range) return;
        const seg = {
            start: Math.round(d.snap(d.range.start, 'start') * 1000) / 1000,
            end: Math.round(d.snap(d.range.end, 'end') * 1000) / 1000,
        };
        if (seg.end - seg.start < d.minSeg || d.count >= d.maxSegments) return;
        d.commit(seg);
    }, [onGhostMove]);

    const startGhostDrag = (e) => {
        if (!sourceAvailable || !sourceDuration || segments.length >= limits.max_segments) return;
        const rect = sourceTrackRef.current?.getBoundingClientRect();
        if (!rect) return;
        const t0 = ((e.clientX - rect.left) / rect.width) * sourceDuration;
        dragRef.current = {
            kind: 'ghost', startX: e.clientX, pxPerSec: rect.width / sourceDuration,
            t0, duration: sourceDuration, range: null, snap: snapEdge, minSeg,
            count: segments.length, maxSegments: limits.max_segments,
            commit: (seg) => dispatch({ type: 'commit', segments: [...segments, seg], select: segments.length }),
        };
        window.addEventListener('pointermove', onGhostMove);
        window.addEventListener('pointerup', onGhostUp);
    };

    // ---- keyboard -----------------------------------------------------------
    useEffect(() => {
        const onKey = (e) => {
            const tag = (e.target?.tagName || '').toLowerCase();
            const typing = tag === 'input' || tag === 'textarea' || tag === 'select';
            if (e.key === 'Escape') {
                e.preventDefault();
                if (dirty) setConfirmClose(true); else onClose();
                return;
            }
            if (typing) return;
            if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'z') {
                e.preventDefault();
                dispatch({ type: e.shiftKey ? 'redo' : 'undo' });
            } else if (e.key === ' ') {
                e.preventDefault();
                const v = videoRef.current;
                if (v) { if (v.paused) v.play().catch(() => {}); else v.pause(); }
            } else if (e.key === 'Backspace' || e.key === 'Delete') {
                e.preventDefault();
                deleteSegment(selected);
            } else if (e.key.toLowerCase() === 's') {
                e.preventDefault();
                splitSegment(selected);
            }
        };
        window.addEventListener('keydown', onKey);
        return () => window.removeEventListener('keydown', onKey);
    });

    // ---- re-render ----------------------------------------------------------
    useEffect(() => {
        if (!rendering) return undefined;
        setRenderSeconds(0);
        const t = setInterval(() => setRenderSeconds((s) => s + 1), 1000);
        return () => clearInterval(t);
    }, [rendering]);

    const doRender = async () => {
        if (!canRender) return;
        setRendering(true);
        setRenderError(null);
        try {
            const res = await apiFetch('/api/clip/rerender', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    job_id: jobId,
                    clip_index: clipIndex,
                    segments: segments.map((s) => ({ start: s.start, end: s.end })),
                    snap_to_words: false, // boundaries are already word-snapped client-side
                    reapply_captions: reapplyCaptions,
                }),
            });
            if (!res.ok) {
                let detail = `re-render failed (HTTP ${res.status})`;
                try { detail = (await res.json()).detail || detail; } catch { /* keep fallback */ }
                throw new Error(detail);
            }
            const data = await res.json();
            setRenderedSegments(data.recipe.segments.map((s) => ({ ...s })));
            dispatch({ type: 'init', segments: data.recipe.segments.map((s) => ({ ...s })) });
            setPreviewUrl(`${getApiUrl(data.new_video_url)}?t=${Date.now()}`);
            onRerendered?.(clipIndex, data);
        } catch (e) {
            if (e instanceof QuotaError) {
                setRenderError(`not enough minutes left (needs ${e.minutesRequired ?? '?'}, ${e.minutesRemaining ?? 0} remaining)`);
            } else {
                setRenderError(e.message || 're-render failed');
            }
        } finally {
            setRendering(false);
        }
    };

    // ---- transcript panel data ---------------------------------------------
    const selectedSeg = segments[selected] || null;
    const panelWords = useMemo(() => {
        if (!selectedSeg) return [];
        const lo = selectedSeg.start - WORD_CONTEXT_SECONDS;
        const hi = selectedSeg.end + WORD_CONTEXT_SECONDS;
        return words.filter((w) => w.e >= lo && w.s <= hi);
    }, [words, selectedSeg]);

    // ---- render -------------------------------------------------------------
    if (loadError) {
        return (
            <div className="fixed inset-0 z-[110] bg-black/70 flex items-center justify-center p-4 animate-fade" onMouseDown={onClose}>
                <div className="card p-6 max-w-md" onMouseDown={(e) => e.stopPropagation()}>
                    <p className="eyebrow mb-2">EDITOR · CLIP {clipIndex + 1}</p>
                    <div className="flex items-center gap-2 text-danger text-sm"><AlertCircle size={16} /> {loadError}</div>
                    <button className="btn-ghost mt-5" onClick={onClose}>close</button>
                </div>
            </div>
        );
    }

    if (!edl) {
        return (
            <div className="fixed inset-0 z-[110] bg-paper/90 flex items-center justify-center animate-fade">
                <div className="flex items-center gap-3 text-muted text-sm lowercase">
                    <Loader2 size={18} className="animate-spin text-brass" /> loading clip recipe…
                </div>
            </div>
        );
    }

    const clipTrackSeconds = Math.max(total, 0.001);
    let runningOffset = 0;
    const blocks = segments.map((s, i) => {
        const left = (runningOffset / clipTrackSeconds) * 100;
        const width = ((s.end - s.start) / clipTrackSeconds) * 100;
        runningOffset += s.end - s.start;
        return { seg: s, i, left, width };
    });
    const renderedTotal = renderedSegments ? totalOf(renderedSegments) : 0;

    return (
        <div className="fixed inset-0 z-[110] bg-paper flex flex-col animate-fade">
            {/* header */}
            <div className="px-4 sm:px-6 pt-5 pb-4 border-b border-rule flex items-start justify-between gap-4 shrink-0">
                <div className="min-w-0">
                    <p className="eyebrow mb-1">EDITOR · CLIP {clipIndex + 1}</p>
                    <h2 className="font-display lowercase text-2xl text-ink truncate">edit clip</h2>
                    {clipTitle && <p className="text-xs text-muted truncate mt-0.5">{clipTitle}</p>}
                </div>
                <div className="flex items-center gap-4 shrink-0">
                    <div className="text-right">
                        <p className="readout">DURATION · {fmt(total)}</p>
                        <p className="readout mt-1">
                            {needsSourcePath ? 'PATH · FULL RE-FRAME' : 'PATH · FAST RECUT'}
                            {edl.rerender_minutes > 0 && ` · ≈${Math.max(1, Math.ceil(total / 60))} MIN`}
                        </p>
                    </div>
                    {confirmClose ? (
                        <div className="flex items-center gap-2">
                            <span className="text-xs text-warn lowercase">discard changes?</span>
                            <button className="btn-danger text-xs py-1.5 px-3" onClick={onClose}>discard</button>
                            <button className="btn-ghost text-xs py-1.5 px-3" onClick={() => setConfirmClose(false)}>keep editing</button>
                        </div>
                    ) : (
                        <button
                            onClick={() => (dirty ? setConfirmClose(true) : onClose())}
                            className="p-2 rounded-input text-muted hover:text-ink hover:bg-paper3 transition-colors"
                            aria-label="close editor"
                        >
                            <X size={18} />
                        </button>
                    )}
                </div>
            </div>

            {/* main */}
            <div className="flex-1 min-h-0 flex flex-col md:flex-row gap-6 px-4 sm:px-6 py-5 overflow-hidden">
                {/* preview */}
                <div className="flex-1 min-w-0 flex items-center justify-center">
                    <div className="h-full max-h-full aspect-[9/16] bg-black rounded-card border border-rule overflow-hidden relative">
                        <video
                            ref={videoRef}
                            src={previewUrl}
                            controls
                            playsInline
                            className="w-full h-full object-contain"
                            onTimeUpdate={(e) => setPlayhead(e.target.currentTime)}
                        />
                        {dirty && (
                            <div className="absolute top-2 left-2 badge-warn">preview shows the last render</div>
                        )}
                    </div>
                </div>

                {/* rail */}
                <div className="w-full md:w-[24rem] shrink-0 flex flex-col min-h-0">
                    <div className="flex-1 overflow-y-auto custom-scrollbar pr-1 space-y-5">
                        {/* segments */}
                        <div>
                            <div className="flex items-center justify-between mb-2">
                                <p className="eyebrow">Segments · {segments.length}/{limits.max_segments}</p>
                                <div className="flex items-center gap-1">
                                    <button className="p-1.5 rounded-input text-muted hover:text-ink hover:bg-paper3 disabled:opacity-45" disabled={!state.past.length} onClick={() => dispatch({ type: 'undo' })} aria-label="undo"><Undo2 size={14} /></button>
                                    <button className="p-1.5 rounded-input text-muted hover:text-ink hover:bg-paper3 disabled:opacity-45" disabled={!state.future.length} onClick={() => dispatch({ type: 'redo' })} aria-label="redo"><Redo2 size={14} /></button>
                                </div>
                            </div>
                            <div className="space-y-2">
                                {segments.map((seg, i) => (
                                    <div
                                        key={i}
                                        onClick={() => dispatch({ type: 'select', index: i })}
                                        className={`rounded-input border p-2.5 cursor-pointer transition-colors ${i === selected ? 'border-[color:var(--color-accent)] bg-paper3' : 'border-rule hover:bg-paper3'} ${outOfRange(seg) ? 'border-[color:var(--color-danger)]' : ''}`}
                                    >
                                        <div className="flex items-center gap-2">
                                            <span className="w-4 h-4 rounded-full shrink-0" style={{ background: SEGMENT_COLORS[i % SEGMENT_COLORS.length] }} />
                                            <span className="readout">#{i + 1}</span>
                                            <input
                                                type="number"
                                                step="0.1"
                                                value={seg.start}
                                                onClick={(e) => e.stopPropagation()}
                                                onChange={(e) => setSegment(i, { start: parseFloat(e.target.value) || 0 }, { snap: false })}
                                                className="input-field w-20 py-1 px-1.5 text-xs text-center"
                                                aria-label={`segment ${i + 1} start`}
                                            />
                                            <span className="text-muted text-xs">→</span>
                                            <input
                                                type="number"
                                                step="0.1"
                                                value={seg.end}
                                                onClick={(e) => e.stopPropagation()}
                                                onChange={(e) => setSegment(i, { end: parseFloat(e.target.value) || 0 }, { snap: false })}
                                                className="input-field w-20 py-1 px-1.5 text-xs text-center"
                                                aria-label={`segment ${i + 1} end`}
                                            />
                                            <span className="readout ml-auto">{fmt(seg.end - seg.start)}</span>
                                        </div>
                                        <div className="flex items-center gap-1 mt-2">
                                            <button className="p-1 rounded-input text-muted hover:text-ink hover:bg-paper disabled:opacity-45" disabled={i === 0} onClick={(e) => { e.stopPropagation(); moveSegment(i, -1); }} aria-label="move up"><ChevronUp size={13} /></button>
                                            <button className="p-1 rounded-input text-muted hover:text-ink hover:bg-paper disabled:opacity-45" disabled={i === segments.length - 1} onClick={(e) => { e.stopPropagation(); moveSegment(i, 1); }} aria-label="move down"><ChevronDown size={13} /></button>
                                            <button className="p-1 rounded-input text-muted hover:text-ink hover:bg-paper disabled:opacity-45" disabled={seg.end - seg.start < minSeg * 2 || segments.length >= limits.max_segments} onClick={(e) => { e.stopPropagation(); splitSegment(i); }} aria-label="split segment"><Scissors size={13} /></button>
                                            <button className="p-1 rounded-input text-muted hover:text-danger hover:bg-paper disabled:opacity-45 ml-auto" disabled={segments.length <= 1} onClick={(e) => { e.stopPropagation(); deleteSegment(i); }} aria-label="delete segment"><Trash2 size={13} /></button>
                                        </div>
                                        {outOfRange(seg) && (
                                            <p className="text-[11px] text-danger mt-1.5 lowercase">outside the original range — the source video is gone</p>
                                        )}
                                    </div>
                                ))}
                            </div>
                            <button
                                onClick={addSegment}
                                disabled={segments.length >= limits.max_segments}
                                className="mt-2 w-full flex items-center justify-center gap-1.5 py-2 rounded-input border border-dashed border-rule2 text-xs lowercase text-ink2 hover:bg-paper3 transition-colors disabled:opacity-45"
                            >
                                <Plus size={14} /> add segment
                            </button>
                        </div>

                        {/* toggles */}
                        <div className="space-y-2.5">
                            <label className="flex items-center justify-between cursor-pointer">
                                <span className="text-xs lowercase text-ink2">snap cuts to words</span>
                                <span className="relative inline-flex items-center">
                                    <input type="checkbox" checked={snapToWords} onChange={(e) => setSnapToWords(e.target.checked)} className="sr-only peer" />
                                    <span className="w-8 h-4 rounded-full bg-paper3 peer-checked:bg-brass transition-colors after:content-[''] after:absolute after:left-0.5 after:top-0.5 after:w-3 after:h-3 after:rounded-full after:bg-ink after:transition-transform peer-checked:after:translate-x-4" />
                                </span>
                            </label>
                            <label className="flex items-center justify-between cursor-pointer">
                                <span className="text-xs lowercase text-ink2">re-apply captions after recut</span>
                                <span className="relative inline-flex items-center">
                                    <input type="checkbox" checked={reapplyCaptions} onChange={(e) => setReapplyCaptions(e.target.checked)} className="sr-only peer" />
                                    <span className="w-8 h-4 rounded-full bg-paper3 peer-checked:bg-brass transition-colors after:content-[''] after:absolute after:left-0.5 after:top-0.5 after:w-3 after:h-3 after:rounded-full after:bg-ink after:transition-transform peer-checked:after:translate-x-4" />
                                </span>
                            </label>
                        </div>

                        {/* source monitor */}
                        {sourceAvailable && (
                            <div>
                                <button onClick={() => setShowSource((v) => !v)} className="w-full flex items-center justify-between mb-2">
                                    <p className="eyebrow">Source monitor</p>
                                    <Film size={13} className={showSource ? 'text-brass' : 'text-muted'} />
                                </button>
                                {showSource && (
                                    <div className="bg-black rounded-card border border-rule overflow-hidden">
                                        <video ref={sourceRef} src={getApiUrl(edl.source.url)} controls playsInline className="w-full max-h-48 object-contain" />
                                        <div className="flex flex-wrap gap-1 p-2 bg-paper2">
                                            {segments.map((seg, i) => (
                                                <button
                                                    key={i}
                                                    onClick={() => { if (sourceRef.current) { sourceRef.current.currentTime = seg.start; sourceRef.current.play().catch(() => {}); } }}
                                                    className="px-2 py-1 rounded-input border border-rule text-[11px] text-ink2 hover:bg-paper3 lowercase"
                                                >
                                                    #{i + 1} · {fmt(seg.start)}
                                                </button>
                                            ))}
                                        </div>
                                    </div>
                                )}
                            </div>
                        )}

                        {/* transcript */}
                        <div>
                            <p className="eyebrow mb-2">Transcript · segment #{selected + 1}</p>
                            {panelWords.length === 0 ? (
                                <p className="text-xs text-muted lowercase">no words near this segment</p>
                            ) : (
                                <div className="flex flex-wrap gap-x-1 gap-y-1.5 max-h-40 overflow-y-auto custom-scrollbar pr-1">
                                    {panelWords.map((w, i) => {
                                        const inside = selectedSeg && w.e > selectedSeg.start && w.s < selectedSeg.end;
                                        const isSel = selectedWord && selectedWord.s === w.s && selectedWord.e === w.e;
                                        return (
                                            <button
                                                key={`${w.s}-${i}`}
                                                onClick={() => setSelectedWord(isSel ? null : w)}
                                                title={`${fmt(w.s)} – ${fmt(w.e)}`}
                                                className={`px-1 py-0.5 rounded text-xs transition-colors ${isSel ? 'bg-brass text-brassink' : inside ? 'text-ink hover:bg-paper3' : 'text-muted hover:bg-paper3'}`}
                                            >
                                                {w.w}
                                            </button>
                                        );
                                    })}
                                </div>
                            )}
                            {selectedWord && selectedSeg && (
                                <div className="flex items-center gap-2 mt-2">
                                    <span className="readout">“{selectedWord.w}” · {fmt(selectedWord.s)}</span>
                                    <button className="btn-quiet text-[11px] py-1 px-2" onClick={() => { setSegment(selected, { start: selectedWord.s }, { snap: false }); setSelectedWord(null); }}>start here</button>
                                    <button className="btn-quiet text-[11px] py-1 px-2" onClick={() => { setSegment(selected, { end: selectedWord.e }, { snap: false }); setSelectedWord(null); }}>end here</button>
                                </div>
                            )}
                        </div>
                    </div>

                    {/* footer actions */}
                    <div className="shrink-0 pt-4 mt-4 border-t border-rule">
                        {renderError && (
                            <div className="mb-3 px-3 py-2 rounded-input text-xs text-danger bg-[color-mix(in_oklab,var(--color-danger)_10%,transparent)] flex items-center gap-2">
                                <AlertCircle size={14} className="shrink-0" /> {renderError}
                            </div>
                        )}
                        {overCaps && (
                            <p className="mb-3 text-[11px] text-warn lowercase">
                                {total > limits.max_total_seconds ? `clip is over ${Math.round(limits.max_total_seconds)}s` : `more than ${limits.max_segments} segments`}
                            </p>
                        )}
                        <div className="flex gap-2">
                            <button className="btn-ghost" onClick={() => (dirty ? setConfirmClose(true) : onClose())}>
                                {dirty ? 'cancel' : 'close'}
                            </button>
                            <button className="btn-primary flex-1 flex items-center justify-center gap-2" disabled={!canRender || !dirty} onClick={doRender}>
                                {rendering
                                    ? (<><Loader2 size={16} className="animate-spin text-brassink" /> re-rendering… {renderSeconds}s</>)
                                    : (needsSourcePath ? 're-render from source' : 're-render clip')}
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            {/* timeline */}
            <div className="shrink-0 border-t border-rule px-4 sm:px-6 py-4 space-y-3 bg-paper2">
                {/* clip track */}
                <div>
                    <div className="flex items-center justify-between mb-1.5">
                        <p className="readout">CLIP · {fmt(total)}</p>
                        <p className="readout hidden sm:block">SPACE PLAY · S SPLIT · ⌫ DELETE · ⌘Z UNDO</p>
                    </div>
                    <div ref={clipTrackRef} className="relative h-12 rounded-input bg-paper border border-rule overflow-hidden">
                        {blocks.map(({ seg, i, left, width }) => (
                            <div
                                key={i}
                                onPointerDown={() => dispatch({ type: 'select', index: i })}
                                className={`absolute top-1 bottom-1 rounded-[6px] border ${i === selected ? 'border-[color:var(--color-accent)]' : 'border-transparent'} ${outOfRange(seg) ? 'border-[color:var(--color-danger)]' : ''}`}
                                style={{ left: `${left}%`, width: `${width}%`, background: `color-mix(in oklab, ${SEGMENT_COLORS[i % SEGMENT_COLORS.length]} 28%, transparent)` }}
                            >
                                <span className="absolute inset-0 flex items-center justify-center readout pointer-events-none select-none">
                                    #{i + 1} · {fmt(seg.end - seg.start)}
                                </span>
                                {/* trim handles */}
                                <div
                                    onPointerDown={(e) => startTrimDrag(e, i, 'start', clipTrackRef.current, clipTrackSeconds)}
                                    className="absolute left-0 top-0 bottom-0 w-2 cursor-ew-resize rounded-l-[6px]"
                                    style={{ background: SEGMENT_COLORS[i % SEGMENT_COLORS.length] }}
                                />
                                <div
                                    onPointerDown={(e) => startTrimDrag(e, i, 'end', clipTrackRef.current, clipTrackSeconds)}
                                    className="absolute right-0 top-0 bottom-0 w-2 cursor-ew-resize rounded-r-[6px]"
                                    style={{ background: SEGMENT_COLORS[i % SEGMENT_COLORS.length] }}
                                />
                            </div>
                        ))}
                        {/* playhead over the last rendered recipe */}
                        {!dirty && renderedTotal > 0 && playhead <= renderedTotal && (
                            <div className="absolute top-0 bottom-0 w-px bg-ink pointer-events-none" style={{ left: `${(playhead / renderedTotal) * 100}%` }} />
                        )}
                    </div>
                </div>

                {/* source track */}
                <div>
                    <div className="flex items-center justify-between mb-1.5">
                        <p className="readout">
                            SOURCE · {fmt(sourceDuration)}{edl.source.duration_estimated ? ' (EST.)' : ''}
                            {!sourceAvailable && ' · EXPIRED — TRIMS LIMITED TO THE ORIGINAL RANGE'}
                        </p>
                        {sourceAvailable && segments.length < limits.max_segments && (
                            <p className="readout hidden sm:block">DRAG ON EMPTY SPACE TO ADD A SEGMENT</p>
                        )}
                    </div>
                    <div
                        ref={sourceTrackRef}
                        onPointerDown={startGhostDrag}
                        className={`relative h-8 rounded-input border overflow-hidden ${sourceAvailable ? 'bg-paper border-rule cursor-crosshair' : 'bg-paper border-rule opacity-60'}`}
                    >
                        {/* canonical range marker */}
                        {sourceDuration > 0 && (
                            <div
                                className="absolute top-0 bottom-0 border-x border-rule2 bg-paper3/60 pointer-events-none"
                                style={{
                                    left: `${(canonical.start / sourceDuration) * 100}%`,
                                    width: `${((canonical.end - canonical.start) / sourceDuration) * 100}%`,
                                }}
                            />
                        )}
                        {sourceDuration > 0 && segments.map((seg, i) => (
                            <div
                                key={i}
                                onPointerDown={(e) => { e.stopPropagation(); dispatch({ type: 'select', index: i }); }}
                                className={`absolute top-1 bottom-1 rounded-[4px] cursor-pointer ${i === selected ? 'ring-1 ring-[color:var(--color-accent)]' : ''}`}
                                style={{
                                    left: `${(seg.start / sourceDuration) * 100}%`,
                                    width: `${Math.max(((seg.end - seg.start) / sourceDuration) * 100, 0.4)}%`,
                                    background: SEGMENT_COLORS[i % SEGMENT_COLORS.length],
                                }}
                                title={`#${i + 1} · ${fmt(seg.start)} → ${fmt(seg.end)}`}
                            />
                        ))}
                        {ghost && sourceDuration > 0 && (
                            <div
                                className="absolute top-1 bottom-1 rounded-[4px] bg-ink/40 border border-dashed border-ink pointer-events-none"
                                style={{
                                    left: `${(ghost.start / sourceDuration) * 100}%`,
                                    width: `${((ghost.end - ghost.start) / sourceDuration) * 100}%`,
                                }}
                            />
                        )}
                    </div>
                    <div className="flex justify-between mt-1">
                        <span className="readout">0:00</span>
                        <span className="readout">{fmt(sourceDuration / 2)}</span>
                        <span className="readout">{fmt(sourceDuration)}</span>
                    </div>
                </div>
            </div>
        </div>
    );
}
