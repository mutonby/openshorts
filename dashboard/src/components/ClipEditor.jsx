import React, { useState, useEffect, useReducer, useRef, useCallback, useMemo, useDeferredValue } from 'react';
import {
    X, Loader2, Plus, Trash2, ChevronUp, ChevronDown, Scissors,
    AlertCircle, Undo2, Redo2, ChevronsRight, ChevronsLeft,
    PanelLeft, PanelLeftClose, Film,
} from 'lucide-react';
import { getApiUrl, withParam } from '../config';
import { apiFetch, apiJson, QuotaError, openQuotaWall } from '../lib/api';
import { useAuth } from '../contexts/AuthContext';

// Full-screen clip editor: shows WHICH source segments a clip was cut from,
// lets the user trim/extend/split/reorder them (word-snapped), and re-renders
// through POST /api/clip/rerender. The recipe (EDL) comes from GET .../edl.

const MIN_SEGMENT_SECONDS = 0.5;
const SNAP_WINDOW_SECONDS = 0.35;

// Hiding the source is a working preference, not a per-clip one, so it sticks.
const HIDE_SOURCE_KEY = 'openshorts_editor_hide_source';

// Words per memoised transcript slice. See TranscriptChunk for why this exists.
const CHUNK_WORDS = 50;

// Segment edges round-trip through 3 decimals, so coverage arithmetic needs a
// hair of tolerance before it calls a millisecond sliver "not rendered".
const COVERAGE_EPSILON = 0.02;

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
    const { refreshMe } = useAuth();
    const [edl, setEdl] = useState(null);
    const [loadError, setLoadError] = useState(null);
    const [state, dispatch] = useReducer(editorReducer, { segments: [], selected: 0, past: [], future: [], pendingBase: null });
    const { segments, selected } = state;

    const [snapToWords, setSnapToWords] = useState(true);
    const [reapplyCaptions, setReapplyCaptions] = useState(true);
    // Framing override: 'auto' (classifier) | 'full' (whole frame) | 'track'.
    const [framing, setFraming] = useState('auto');
    const [renderedFraming, setRenderedFraming] = useState('auto');
    // The recipe of the currently RENDERED preview (playhead maps onto it).
    const [renderedSegments, setRenderedSegments] = useState(null);
    const [previewUrl, setPreviewUrl] = useState(null);
    const [rendering, setRendering] = useState(false);
    const [renderSeconds, setRenderSeconds] = useState(0);
    const [renderError, setRenderError] = useState(null);
    const [confirmClose, setConfirmClose] = useState(false);
    const [selectedWord, setSelectedWord] = useState(null);
    const [playhead, setPlayhead] = useState(0);
    const [showSource, setShowSource] = useState(() => {
        try { return localStorage.getItem(HIDE_SOURCE_KEY) !== '1'; } catch { return true; }
    });
    // Where the source monitor's playhead is, for the transcript to follow.
    const [sourceTime, setSourceTime] = useState(0);
    // Three-point editing, like a Premiere source monitor: mark IN and OUT on
    // the source, then send that range to the clip. Kept as two independent
    // numbers rather than a range so either end can be re-marked on its own.
    const [markIn, setMarkIn] = useState(null);
    const [markOut, setMarkOut] = useState(null);

    const videoRef = useRef(null);
    const sourceRef = useRef(null);
    const clipTrackRef = useRef(null);
    const sourceTrackRef = useRef(null);
    const dragRef = useRef(null);
    const [ghost, setGhost] = useState(null); // in-progress new segment on the source track
    const transcriptRef = useRef(null);

    useEffect(() => {
        try { localStorage.setItem(HIDE_SOURCE_KEY, showSource ? '0' : '1'); } catch { /* private mode */ }
    }, [showSource]);

    // ---- scrubbing the source monitor ---------------------------------------
    // Dragging on the source track drives this <video>, so the cut is chosen
    // against the picture instead of against numbers. rAF-throttled: a drag
    // fires dozens of pointermoves a second and assigning currentTime on each
    // one makes the element stutter. `pending` also survives a seek issued
    // before the metadata is in, which onLoadedMetadata then applies.
    const seekRef = useRef({ pending: null, raf: 0, timer: 0 });

    const applySeek = useCallback(() => {
        if (seekRef.current.raf) cancelAnimationFrame(seekRef.current.raf);
        if (seekRef.current.timer) clearTimeout(seekRef.current.timer);
        seekRef.current.raf = 0;
        seekRef.current.timer = 0;
        const v = sourceRef.current;
        const t = seekRef.current.pending;
        if (!v || t === null) return;
        if (v.readyState === 0) return;          // retried from onLoadedMetadata
        if (!v.paused) v.pause();                // a moving picture cannot be aimed
        try { v.currentTime = Math.max(0, t); } catch { /* seek refused, keep pending */ }
        seekRef.current.pending = null;
    }, []);

    const seekSource = useCallback((t) => {
        if (!Number.isFinite(t)) return;
        seekRef.current.pending = t;
        if (seekRef.current.raf || seekRef.current.timer) return;   // one per frame
        seekRef.current.raf = requestAnimationFrame(applySeek);
        // The timer is the recovery path, not a second throttle: a hidden or
        // occluded tab never runs a rAF callback, so a latch waiting only on
        // rAF would deadlock the scrub for the rest of the session. Whichever
        // fires first cancels the other.
        seekRef.current.timer = setTimeout(applySeek, 120);
    }, [applySeek]);

    useEffect(() => () => {
        if (seekRef.current.raf) cancelAnimationFrame(seekRef.current.raf);
        if (seekRef.current.timer) clearTimeout(seekRef.current.timer);
    }, []);

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
                setFraming(data.framing || 'auto');
                setRenderedFraming(data.framing || 'auto');
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
    const canonical = useMemo(() => edl?.canonical_range || { start: 0, end: 0 }, [edl]);
    const limits = edl?.limits || { max_segments: 12, min_segment_seconds: MIN_SEGMENT_SECONDS, max_total_seconds: 180 };
    const minSeg = limits.min_segment_seconds || MIN_SEGMENT_SECONDS;

    // The source panel is on screen only when there IS a source and the user
    // has not put it away. An expired source forces the same two-column layout
    // the hide button asks for, so both cases share one code path.
    const sourceOpen = sourceAvailable && showSource;

    const total = totalOf(segments);
    const dirty = useMemo(() => {
        if (!renderedSegments) return false;
        return JSON.stringify(segments) !== JSON.stringify(renderedSegments)
            || framing !== renderedFraming;
    }, [segments, renderedSegments, framing, renderedFraming]);

    // Trim bounds: with the source gone, cuts must stay inside the range the
    // canonical file was rendered from.
    const bounds = sourceAvailable
        ? { lo: 0, hi: sourceDuration || Infinity }
        : { lo: canonical.start, hi: canonical.end };

    const outOfRange = useCallback(
        (seg) => !sourceAvailable && (seg.start < canonical.start - 0.05 || seg.end > canonical.end + 0.05),
        [sourceAvailable, canonical],
    );
    const needsSourcePath = framing !== 'auto'
        || segments.some((s) => s.start < canonical.start - 0.05 || s.end > canonical.end + 0.05);
    const invalidSegments = segments.some(outOfRange);
    const overCaps = segments.length > limits.max_segments || total > limits.max_total_seconds;
    const canRender = !rendering && segments.length > 0 && !invalidSegments && !overCaps
        && segments.every((s) => s.end - s.start >= minSeg)
        && (framing === 'auto' || sourceAvailable);

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
        // Cut where the playhead is, when it sits inside this segment. It lives
        // on the current assembly now, so this no longer needs the rendered
        // recipe to still match.
        let offset = 0;
        for (let i = 0; i < segments.length; i += 1) {
            const len = segments[i].end - segments[i].start;
            if (i === index && playhead > offset + minSeg && playhead < offset + len - minSeg) {
                at = segments[i].start + (playhead - offset);
            }
            offset += len;
        }
        at = snapEdge(at, 'end');
        if (at - seg.start < minSeg || seg.end - at < minSeg) at = seg.start + (seg.end - seg.start) / 2;
        const next = segments.flatMap((s, i) => (i === index
            ? [{ start: s.start, end: Math.round(at * 1000) / 1000 }, { start: Math.round(at * 1000) / 1000, end: s.end }]
            : [s]));
        dispatch({ type: 'commit', segments: next, select: index });
    };

    // ---- drag: trim/move handles on both tracks -----------------------------
    // ``edge`` is 'start' | 'end' (trim one boundary) or 'move' (slide the whole
    // segment along the source, keeping its duration).
    const onDragMove = useCallback((e) => {
        const d = dragRef.current;
        if (!d || d.kind) return;   // a ghost or a scrub, not a trim
        const dt = (e.clientX - d.startX) / d.pxPerSec;
        const seg = { ...d.base[d.idx] };
        if (d.edge === 'move') {
            const len = seg.end - seg.start;
            seg.start = Math.max(d.lo, Math.min(seg.start + dt, d.hi - len));
            seg.end = seg.start + len;
        } else if (d.edge === 'start') {
            seg.start = Math.max(d.lo, Math.min(seg.start + dt, seg.end - d.minSeg));
        } else {
            seg.end = Math.min(d.hi, Math.max(seg.end + dt, seg.start + d.minSeg));
        }
        // Source-track drags scrub the monitor to the edge being moved, so
        // the frame on screen IS the frame the cut lands on.
        if (d.scrub) d.seek(d.edge === 'end' ? seg.end : seg.start);
        const next = d.base.map((s, i) => (i === d.idx ? seg : s));
        d.last = { seg, next };
        dispatch({ type: 'preview', segments: next });
    }, []);

    const onDragUp = useCallback(() => {
        const d = dragRef.current;
        dragRef.current = null;
        window.removeEventListener('pointermove', onDragMove);
        window.removeEventListener('pointerup', onDragUp);
        window.removeEventListener('pointercancel', onDragUp);
        if (!d || !d.last) return;
        const { seg, next } = d.last;
        const snapped = { ...seg };
        if (d.edge === 'move') {
            const len = seg.end - seg.start;
            const start = Math.max(d.lo, Math.min(d.snap(seg.start, 'start'), d.hi - len));
            snapped.start = start;
            snapped.end = start + len;
        } else if (d.edge === 'start') {
            snapped.start = Math.max(d.lo, Math.min(d.snap(seg.start, 'start'), seg.end - d.minSeg));
        } else {
            snapped.end = Math.min(d.hi, Math.max(d.snap(seg.end, 'end'), seg.start + d.minSeg));
        }
        // Round like every other edit path: the number inputs in the rail show
        // these values, and raw drag arithmetic yields 190.22000000000003.
        const clean = { start: Math.round(snapped.start * 1000) / 1000, end: Math.round(snapped.end * 1000) / 1000 };
        dispatch({ type: 'commit', segments: next.map((s, i) => (i === d.idx ? clean : s)), select: d.idx });
    }, [onDragMove]);

    const startTrimDrag = (e, idx, edge, trackEl, secondsOnTrack) => {
        // Only the primary button drags — and preventDefault is what stops
        // Chrome from turning the gesture into a text selection (and then into
        // a native drag-and-drop, which swallows every later pointer event and
        // shows the 🚫 cursor instead of dragging anything).
        if (e.button !== undefined && e.button !== 0) return;
        e.preventDefault();
        e.stopPropagation();
        const rect = trackEl?.getBoundingClientRect();
        if (!rect || !secondsOnTrack) return;
        try { e.currentTarget.setPointerCapture?.(e.pointerId); } catch { /* not captured, window listeners still fire */ }
        // Only the source track maps 1:1 onto source time; the clip track's
        // x-axis is the recut's own timeline, which the monitor knows nothing of.
        const scrub = trackEl === sourceTrackRef.current;
        if (scrub) {
            const seg = segments[idx];
            if (seg) seekSource(edge === 'end' ? seg.end : seg.start);
        }
        dragRef.current = {
            idx, edge, startX: e.clientX, pxPerSec: rect.width / secondsOnTrack,
            base: segments.map((s) => ({ ...s })), last: null,
            lo: bounds.lo, hi: bounds.hi, minSeg, snap: snapEdge,
            scrub, seek: seekSource,
        };
        dispatch({ type: 'select', index: idx });
        window.addEventListener('pointermove', onDragMove);
        window.addEventListener('pointerup', onDragUp);
        window.addEventListener('pointercancel', onDragUp);
    };

    // ---- drag: paint a NEW segment on the source track ----------------------
    const onGhostMove = useCallback((e) => {
        const d = dragRef.current;
        if (!d || d.kind !== 'ghost') return;
        const t = Math.max(0, Math.min(d.duration, d.t0 + (e.clientX - d.startX) / d.pxPerSec));
        d.seek(t);
        d.range = { start: Math.min(d.t0, t), end: Math.max(d.t0, t) };
        setGhost({ ...d.range });
    }, []);

    const onGhostUp = useCallback(() => {
        const d = dragRef.current;
        dragRef.current = null;
        window.removeEventListener('pointermove', onGhostMove);
        window.removeEventListener('pointerup', onGhostUp);
        window.removeEventListener('pointercancel', onGhostUp);
        setGhost(null);
        if (!d || !d.range) return;
        const seg = {
            start: Math.round(d.snap(d.range.start, 'start') * 1000) / 1000,
            end: Math.round(d.snap(d.range.end, 'end') * 1000) / 1000,
        };
        if (seg.end - seg.start < d.minSeg) return;
        // Marks, not a segment: dragging here proposes a range, and nothing
        // enters the clip until it is sent. One concept instead of two, and
        // the range stays adjustable before it is committed.
        d.mark(seg.start, seg.end);
    }, [onGhostMove]);

    const startGhostDrag = (e) => {
        if (e.button !== undefined && e.button !== 0) return;
        if (!sourceAvailable || !sourceDuration) return;
        const rect = sourceTrackRef.current?.getBoundingClientRect();
        if (!rect) return;
        // Same reason as startTrimDrag: without this Chrome selects the track's
        // labels and the follow-up gesture becomes a native text drag.
        e.preventDefault();
        const t0 = ((e.clientX - rect.left) / rect.width) * sourceDuration;
        // Landing anywhere on the track parks the monitor there — that alone is
        // how you find a cut point, so it happens even when the segment cap is
        // reached and no new block can be painted.
        seekSource(t0);
        try { e.currentTarget.setPointerCapture?.(e.pointerId); } catch { /* window listeners still fire */ }
        dragRef.current = {
            kind: 'ghost', startX: e.clientX, pxPerSec: rect.width / sourceDuration,
            t0, duration: sourceDuration, range: null, snap: snapEdge, minSeg,
            seek: seekSource,
            mark: (a, b) => { setMarkIn(a); setMarkOut(b); },
        };
        window.addEventListener('pointermove', onGhostMove);
        window.addEventListener('pointerup', onGhostUp);
        window.addEventListener('pointercancel', onGhostUp);
    };

    // ---- what of this edit the rendered file can already show ---------------
    // The rendered file IS renderedSegments cut and concatenated in order (see
    // perform_recut), so any instant of source still inside one of those ranges
    // exists in the file and can be played straight from it. That is why a cut
    // that only REMOVES material stays previewable: every frame it keeps was
    // already rendered. Only material the last render never saw is missing, and
    // that is what turns red.
    const coverage = useMemo(() => {
        if (!segments.length) return [];
        // A framing change invalidates the whole file even where the ranges
        // still line up: the crop baked into those frames is not what will
        // come out.
        if (!renderedSegments || framing !== renderedFraming) {
            return [{ start: 0, end: totalOf(segments), rendered: null }];
        }
        const spans = [];
        let clipAcc = 0;
        for (const seg of segments) {
            const segLen = seg.end - seg.start;
            // Where this segment overlaps the rendered ones, and at what offset
            // into the file each overlap lands.
            const pieces = [];
            let renAcc = 0;
            for (const r of renderedSegments) {
                const lo = Math.max(seg.start, r.start);
                const hi = Math.min(seg.end, r.end);
                if (hi - lo > COVERAGE_EPSILON) {
                    pieces.push({ lo, hi, rendered: renAcc + (lo - r.start) });
                }
                renAcc += r.end - r.start;
            }
            pieces.sort((a, b) => a.lo - b.lo);

            let cursor = seg.start;
            for (const piece of pieces) {
                // A source range reused twice by the render would overlap here;
                // first one wins rather than emitting the same clip time twice.
                const lo = Math.max(piece.lo, cursor);
                if (piece.hi - lo <= COVERAGE_EPSILON) continue;
                if (lo - cursor > COVERAGE_EPSILON) {
                    spans.push({
                        start: clipAcc + (cursor - seg.start),
                        end: clipAcc + (lo - seg.start),
                        rendered: null,
                    });
                }
                spans.push({
                    start: clipAcc + (lo - seg.start),
                    end: clipAcc + (piece.hi - seg.start),
                    rendered: piece.rendered + (lo - piece.lo),
                });
                cursor = piece.hi;
            }
            if (seg.end - cursor > COVERAGE_EPSILON) {
                spans.push({
                    start: clipAcc + (cursor - seg.start),
                    end: clipAcc + segLen,
                    rendered: null,
                });
            }
            clipAcc += segLen;
        }
        return spans;
    }, [segments, renderedSegments, framing, renderedFraming]);

    const missingSeconds = useMemo(() => coverage.reduce(
        (acc, sp) => (sp.rendered === null ? acc + (sp.end - sp.start) : acc), 0), [coverage]);

    // Half-open [start, end), so a position exactly on a seam belongs to the
    // span that STARTS there — otherwise the playhead reads as being on the next
    // segment while the picture is still the tail of the previous one. The
    // epsilon only absorbs rounding on the way in; the very end of the clip
    // falls back to the last span.
    const spanIndexAt = useCallback((t) => {
        const i = coverage.findIndex((sp) => t >= sp.start - COVERAGE_EPSILON && t < sp.end);
        return i >= 0 ? i : coverage.length - 1;
    }, [coverage]);

    const clipToRendered = useCallback((t) => {
        const sp = coverage[spanIndexAt(t)];
        if (!sp || sp.rendered === null) return null;
        const offset = Math.max(0, Math.min(t - sp.start, sp.end - sp.start));
        return sp.rendered + offset;
    }, [coverage, spanIndexAt]);

    // Used when the native <video> controls are dragged: the file's own time
    // has to come back onto the clip's timeline.
    const renderedToClip = useCallback((r) => {
        for (const sp of coverage) {
            if (sp.rendered === null) continue;
            const end = sp.rendered + (sp.end - sp.start);
            if (r >= sp.rendered - COVERAGE_EPSILON && r <= end + COVERAGE_EPSILON) {
                return sp.start + (r - sp.rendered);
            }
        }
        return null;
    }, [coverage]);

    const hasCovered = useMemo(
        () => coverage.some((sp) => sp.rendered !== null), [coverage]);

    // Keep the playhead out of red. Stopping at the edge is the whole point:
    // inside one there is no frame to show, so a handle that could sit there
    // would just be a handle pointing at nothing. Runs of adjacent red spans
    // (one segment ending unrendered, the next starting unrendered) are treated
    // as a single wall.
    // ``path`` distinguishes a drag from a click. Dragging is continuous motion,
    // so a wall anywhere BETWEEN the two positions stops it — checking only the
    // destination lets a fast flick tunnel clean through the red. A click is
    // "go here", so it only has to land somewhere legal.
    const clampToCovered = useCallback((t, from, { path = false } = {}) => {
        if (!hasCovered) return t;                      // nothing to stay inside
        const forward = t >= from;

        let wall = null;
        if (path) {
            wall = forward
                ? coverage.find((sp) => sp.rendered === null
                    && sp.end > from + COVERAGE_EPSILON && sp.start < t - COVERAGE_EPSILON)
                : [...coverage].reverse().find((sp) => sp.rendered === null
                    && sp.start < from - COVERAGE_EPSILON && sp.end > t + COVERAGE_EPSILON);
        }
        if (!wall) {
            const sp = coverage[spanIndexAt(t)];
            if (!sp || sp.rendered !== null) return t;
            wall = sp;
        }

        // Grow it across any neighbouring uncovered spans: one segment ending
        // unrendered next to another starting unrendered is one wall, not two.
        let lo = coverage.indexOf(wall);
        let hi = lo;
        while (lo > 0 && coverage[lo - 1].rendered === null) lo -= 1;
        while (hi < coverage.length - 1 && coverage[hi + 1].rendered === null) hi += 1;
        // A millisecond inside the green, so the position still resolves to the
        // covered span rather than to the wall it is touching.
        const near = Math.max(0, coverage[lo].start - 0.001);
        const far = coverage[hi].end;
        const hasBefore = lo > 0;
        const hasAfter = hi < coverage.length - 1;

        if (forward) return hasBefore ? near : (hasAfter ? far : t);
        return hasAfter ? far : (hasBefore ? near : t);
    }, [coverage, hasCovered, spanIndexAt]);

    // An edit can leave the handle standing where the file no longer reaches.
    useEffect(() => {
        setPlayhead((t) => {
            const clamped = clampToCovered(t, t);
            return clamped === t ? t : clamped;
        });
    }, [clampToCovered]);

    // ---- scrubbing the clip track ------------------------------------------
    // Clip time -> source time, walking the segment list. Same reasoning as the
    // backend's rebase_segments, in the other direction: the clip is the
    // segments played back to back, so a position on it lands inside whichever
    // segment's running total covers it.
    const clipToSource = useCallback((t) => {
        let acc = 0;
        for (const seg of segments) {
            const len = seg.end - seg.start;
            if (t < acc + len) return seg.start + (t - acc);
            acc += len;
        }
        const last = segments[segments.length - 1];
        return last ? last.end : 0;
    }, [segments]);

    // Which coverage span the rendered file is playing through.
    const playSpanRef = useRef(0);

    // Walk the assembly while the file plays. Everything covered 1:1 (nothing
    // pending) short-circuits to the old behaviour: no jumps, no bookkeeping.
    // Reaching a boundary the file cannot cross snaps currentTime BACK onto it:
    // the frame left on screen is the endpoint itself, not the overshoot.
    const stepPlayback = useCallback((v) => {
        if (dragRef.current?.kind === 'scrub') return;   // seeks report a frame late
        if (!dirty) { setPlayhead(v.currentTime); return; }

        let i = playSpanRef.current;
        let sp = coverage[i];
        if (!sp) return;

        const spEnd = sp.rendered !== null ? sp.rendered + (sp.end - sp.start) : null;
        if (spEnd !== null && v.currentTime < spEnd - COVERAGE_EPSILON) {
            setPlayhead(sp.start + (v.currentTime - sp.rendered));
            return;
        }

        // Past the end of this span: move on to the next one.
        i += 1;
        const next = coverage[i];
        if (!next || next.rendered === null) {
            // Nothing to show past this edge until it is rendered, so stop ON
            // the edge rather than pretending.
            v.pause();
            if (spEnd !== null) { try { v.currentTime = spEnd; } catch { /* not seekable yet */ } }
            setPlayhead(next ? next.start : totalOf(segments));
            if (next) playSpanRef.current = i;
            return;
        }
        playSpanRef.current = i;
        setPlayhead(next.start);
        try { v.currentTime = next.rendered; } catch { /* not seekable yet */ }
    }, [coverage, dirty, segments]);

    const onClipTimeUpdate = useCallback((e) => stepPlayback(e.target), [stepPlayback]);

    // timeupdate fires ~4 times a second, which let playback overshoot a
    // trimmed endpoint by up to a quarter of a second before pausing — the
    // exact boundary is what an end trim is being judged against (issue #73).
    // While the file plays, this loop enforces it every frame instead.
    const playRafRef = useRef(0);
    const stopPlayLoop = useCallback(() => {
        if (playRafRef.current) cancelAnimationFrame(playRafRef.current);
        playRafRef.current = 0;
    }, []);
    const startPlayLoop = useCallback(() => {
        stopPlayLoop();
        const tick = () => {
            const v = videoRef.current;
            if (!v || v.paused || v.ended) { playRafRef.current = 0; return; }
            stepPlayback(v);
            playRafRef.current = requestAnimationFrame(tick);
        };
        playRafRef.current = requestAnimationFrame(tick);
    }, [stepPlayback, stopPlayLoop]);
    useEffect(() => stopPlayLoop, [stopPlayLoop]);

    // The native controls scrub the FILE; bring that back onto the clip.
    const onClipSeeked = useCallback((e) => {
        if (!dirty || dragRef.current?.kind === 'scrub') return;
        const t = renderedToClip(e.target.currentTime);
        if (t === null) return;   // landed on material this edit dropped
        setPlayhead(t);
        playSpanRef.current = spanIndexAt(t);
    }, [dirty, renderedToClip, spanIndexAt]);

    // Refuse to roll from a span the file cannot show — same reason as above.
    const onClipPlay = useCallback((e) => {
        if (dirty) {
            const sp = coverage[playSpanRef.current];
            if (sp && sp.rendered === null) { e.target.pause(); return; }
        }
        startPlayLoop();
    }, [coverage, dirty, startPlayLoop]);

    const onScrubMove = useCallback((e) => {
        const d = dragRef.current;
        if (!d || d.kind !== 'scrub') return;
        d.apply(d.at(e.clientX), { path: true });
    }, []);

    const onScrubUp = useCallback(() => {
        dragRef.current = null;
        window.removeEventListener('pointermove', onScrubMove);
        window.removeEventListener('pointerup', onScrubUp);
        window.removeEventListener('pointercancel', onScrubUp);
    }, [onScrubMove]);

    const startClipScrub = (e) => {
        if (e.button !== undefined && e.button !== 0) return;
        const rect = clipTrackRef.current?.getBoundingClientRect();
        if (!rect || !clipTrackSeconds) return;
        // Same reason as the other tracks: without this Chrome turns the drag
        // into a text selection and then into a native drag.
        e.preventDefault();
        try { e.currentTarget.setPointerCapture?.(e.pointerId); } catch { /* window listeners still fire */ }

        // The track can be longer than the clip (its scale is the rendered
        // length), so positions clamp to the clip's own end, not the track's.
        const at = (clientX) => Math.max(0, Math.min(
            total, ((clientX - rect.left) / rect.width) * clipTrackSeconds));
        // Direction of travel decides which wall a red span stops you at, so it
        // has to be the last APPLIED position, not the one at pointerdown.
        let last = playhead;
        const apply = (raw, { path = false } = {}) => {
            const t = clampToCovered(raw, last, { path });
            last = t;
            setPlayhead(t);
            // Not "is the file stale" but "does the file hold THIS instant" —
            // which is what keeps a trim navigable without a re-render.
            const r = clipToRendered(t);
            playSpanRef.current = spanIndexAt(t);
            if (r !== null && videoRef.current) {
                try { videoRef.current.currentTime = r; } catch { /* not seekable yet */ }
            }
            // The source is the only picture available where the file has none.
            if (sourceOpen) seekSource(clipToSource(t));
        };

        apply(at(e.clientX));
        dragRef.current = { kind: 'scrub', at, apply };
        window.addEventListener('pointermove', onScrubMove);
        window.addEventListener('pointerup', onScrubUp);
        window.addEventListener('pointercancel', onScrubUp);
    };

    // ---- three-point editing: mark in/out, then send to the clip ------------
    const round3 = (t) => Math.round(t * 1000) / 1000;

    // The marks come off the monitor's own playhead, so "what I am looking at"
    // and "where the cut lands" are the same instant by construction.
    const markHere = (which) => {
        const v = sourceRef.current;
        if (!v || !Number.isFinite(v.currentTime)) return;
        (which === 'in' ? setMarkIn : setMarkOut)(round3(v.currentTime));
    };

    const clearMarks = () => { setMarkIn(null); setMarkOut(null); };

    // Only a usable range counts: both ends marked, and long enough to render.
    // Order is forgiving — marking OUT before IN still yields the range between.
    const markRange = useMemo(() => {
        if (markIn === null || markOut === null) return null;
        const lo = Math.min(markIn, markOut);
        const hi = Math.max(markIn, markOut);
        return hi - lo >= minSeg ? { start: round3(lo), end: round3(hi) } : null;
    }, [markIn, markOut, minSeg]);

    // 'replace' overwrites the selected segment (make the clip BE this range);
    // 'insert' drops the range in right after it, rippling the rest along.
    const sendToClip = (mode) => {
        if (!markRange) return;
        if (mode === 'replace') {
            dispatch({
                type: 'commit',
                segments: segments.map((s, i) => (i === selected ? { ...markRange } : s)),
                select: selected,
            });
            return;
        }
        if (segments.length >= limits.max_segments) return;
        const next = segments.slice();
        next.splice(selected + 1, 0, { ...markRange });
        dispatch({ type: 'commit', segments: next, select: selected + 1 });
    };

    // ---- keyboard -----------------------------------------------------------
    useEffect(() => {
        const onKey = (e) => {
            const tag = (e.target?.tagName || '').toLowerCase();
            const typing = tag === 'input' || tag === 'textarea' || tag === 'select';
            if (e.key === 'Escape') {
                e.preventDefault();
                // Leaving mid-render abandons nothing: the request keeps
                // running and onRerendered updates the card when it lands.
                if (rendering) onClose();
                else if (dirty) setConfirmClose(true);
                else onClose();
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
            } else if (sourceOpen && e.key.toLowerCase() === 'i') {
                e.preventDefault();
                markHere('in');
            } else if (sourceOpen && e.key.toLowerCase() === 'o') {
                e.preventDefault();
                markHere('out');
            } else if (sourceOpen && e.key === ',') {
                e.preventDefault();
                sendToClip('insert');
            } else if (sourceOpen && e.key === '.') {
                e.preventDefault();
                sendToClip('replace');
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
                    framing,
                }),
            });
            if (!res.ok) {
                let detail = `re-render failed (HTTP ${res.status})`;
                try { detail = (await res.json()).detail || detail; } catch { /* keep fallback */ }
                throw new Error(detail);
            }
            const data = await res.json();
            setRenderedSegments(data.recipe.segments.map((s) => ({ ...s })));
            setRenderedFraming(data.framing || 'auto');
            setFraming(data.framing || 'auto');
            dispatch({ type: 'init', segments: data.recipe.segments.map((s) => ({ ...s })) });
            setPreviewUrl(withParam(getApiUrl(data.new_video_url), 't', Date.now()));
            onRerendered?.(clipIndex, data);
            refreshMe();
        } catch (e) {
            if (e instanceof QuotaError) {
                refreshMe();
                setRenderError(`not enough minutes left (needs ${e.minutesRequired ?? '?'}, ${e.minutesRemaining ?? 0} remaining)`);
                openQuotaWall(e, 'clip_editor');
            } else {
                setRenderError(e.message || 're-render failed');
            }
        } finally {
            setRendering(false);
        }
    };

    // ---- transcript panel data ---------------------------------------------
    // The panel holds the WHOLE source transcript, not a window around the
    // segment: extending a cut means reading what is said before and after it.
    const selectedSeg = segments[selected] || null;
    // Deferred so a drag is never blocked repainting a few thousand words —
    // the highlight settles a frame or two behind the handle, the drag stays
    // at full rate.
    const highlightSeg = useDeferredValue(selectedSeg);
    const anchorIndex = useMemo(() => (
        selectedSeg ? words.findIndex((w) => w.e > selectedSeg.start) : -1
    ), [words, selectedSeg]);

    // The word the source monitor is currently on. Binary search — words are
    // already sorted by start time (recut.transcript_words does that). During
    // silence the previous word stays lit rather than blinking off: "you are
    // here" is more useful than a strictly correct nothing.
    const activeWordIndex = useMemo(() => {
        let lo = 0;
        let hi = words.length - 1;
        let best = -1;
        while (lo <= hi) {
            const mid = (lo + hi) >> 1;
            if (words[mid].s <= sourceTime) { best = mid; lo = mid + 1; } else hi = mid - 1;
        }
        return best;
    }, [words, sourceTime]);

    const selectedWordIndex = useMemo(() => (
        selectedWord ? words.findIndex((w) => w.s === selectedWord.s && w.e === selectedWord.e) : -1
    ), [words, selectedWord]);

    const chunks = useMemo(() => {
        const out = [];
        for (let i = 0; i < words.length; i += CHUNK_WORDS) {
            out.push({ offset: i, items: words.slice(i, i + CHUNK_WORDS) });
        }
        return out;
    }, [words]);

    // Direct scrollTop, not scrollIntoView: the latter also scrolls every
    // ancestor, which would yank the whole column around mid-drag.
    const scrollTranscriptTo = useCallback((selector) => {
        const box = transcriptRef.current;
        if (!box) return;
        const el = box.querySelector(selector);
        if (!el) return;
        box.scrollTop = Math.max(0, el.offsetTop - box.clientHeight / 2);
    }, []);

    // Two scrollers, one policy, so they cannot fight: picking a segment jumps
    // to it, and anything that moves the monitor (playback, a track drag, a
    // seek) takes over from there. Deliberately NOT keyed on the segment's
    // start, which changes every frame of a drag.
    useEffect(() => {
        scrollTranscriptTo('[data-anchor="1"]');
    }, [selected, words.length, scrollTranscriptTo]);

    useEffect(() => {
        scrollTranscriptTo('[data-active="1"]');
    }, [activeWordIndex, scrollTranscriptTo]);

    // Stable identity keeps every untouched chunk out of the re-render.
    const pickWord = useCallback((w) => {
        setSelectedWord((prev) => (prev && prev.s === w.s && prev.e === w.e ? null : w));
        seekSource(w.s);
    }, [seekSource]);

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

    // The track's scale is the RENDERED length (or the current total, whichever
    // is longer), not the current total: normalising to the total made a
    // single-segment clip fill 100% of the track no matter how it was trimmed,
    // so dragging its handle visibly moved nothing (issue #73). Against the
    // rendered length, shortening the clip shortens the bar.
    const clipTrackSeconds = Math.max(total, totalOf(renderedSegments || []), 0.001);
    let runningOffset = 0;
    const blocks = segments.map((s, i) => {
        const left = (runningOffset / clipTrackSeconds) * 100;
        const width = ((s.end - s.start) / clipTrackSeconds) * 100;
        runningOffset += s.end - s.start;
        return { seg: s, i, left, width };
    });

    // The source track lives under the source monitor. With no source at all
    // there is no monitor to sit under, so it moves below the clip track: it
    // still explains why trims are pinned to the original range.
    const sourceTrack = (
        <div className="shrink-0 select-none">
            <div className="flex items-center justify-between mb-1.5 gap-3">
                <p className="readout">
                    SOURCE · {fmt(sourceDuration)}{edl.source.duration_estimated ? ' (EST.)' : ''}
                    {!sourceAvailable && ' · EXPIRED — TRIMS LIMITED TO THE ORIGINAL RANGE'}
                </p>
                {sourceAvailable && (
                    <p className="readout hidden xl:block truncate">
                        DRAG EMPTY SPACE TO MARK IN/OUT · BLOCK TO MOVE · EDGES TO TRIM
                    </p>
                )}
            </div>
            <div
                ref={sourceTrackRef}
                onPointerDown={startGhostDrag}
                className={`relative h-8 rounded-input border overflow-hidden touch-none ${sourceAvailable ? 'bg-paper border-rule cursor-crosshair' : 'bg-paper border-rule opacity-60'}`}
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
                        // Body drag slides the segment along the source without
                        // changing its duration; the edges trim it.
                        onPointerDown={(e) => startTrimDrag(e, i, 'move', sourceTrackRef.current, sourceDuration)}
                        className={`absolute top-1 bottom-1 rounded-[4px] touch-none cursor-grab active:cursor-grabbing ${i === selected ? 'ring-1 ring-[color:var(--color-accent)]' : ''}`}
                        style={{
                            left: `${(seg.start / sourceDuration) * 100}%`,
                            width: `${Math.max(((seg.end - seg.start) / sourceDuration) * 100, 0.4)}%`,
                            // A short segment on a 14-minute source is a sliver;
                            // without a floor there is nothing left to grab.
                            minWidth: '14px',
                            background: SEGMENT_COLORS[i % SEGMENT_COLORS.length],
                        }}
                        title={`#${i + 1} · ${fmt(seg.start)} → ${fmt(seg.end)} — drag to move, edges to trim`}
                    >
                        {/* Capped at a third each so the middle stays grabbable
                            however narrow the block gets. */}
                        <div
                            onPointerDown={(e) => startTrimDrag(e, i, 'start', sourceTrackRef.current, sourceDuration)}
                            className="absolute left-0 top-0 bottom-0 w-1.5 max-w-[33%] touch-none cursor-ew-resize rounded-l-[4px] bg-ink/35"
                        />
                        <div
                            onPointerDown={(e) => startTrimDrag(e, i, 'end', sourceTrackRef.current, sourceDuration)}
                            className="absolute right-0 top-0 bottom-0 w-1.5 max-w-[33%] touch-none cursor-ew-resize rounded-r-[4px] bg-ink/35"
                        />
                    </div>
                ))}
                {markRange && sourceDuration > 0 && !ghost && (
                    <div
                        className="absolute inset-y-0 border-x-2 border-brass bg-brass/15 pointer-events-none"
                        style={{
                            left: `${(markRange.start / sourceDuration) * 100}%`,
                            width: `${((markRange.end - markRange.start) / sourceDuration) * 100}%`,
                        }}
                    />
                )}
                {/* A lone mark still has to be visible, or setting IN and
                    then hunting for OUT gives no feedback at all. */}
                {sourceDuration > 0 && !ghost && !markRange && [markIn, markOut].map((t, i) => (
                    t === null ? null : (
                        <div
                            key={i}
                            className="absolute inset-y-0 w-0.5 bg-brass pointer-events-none"
                            style={{ left: `${(t / sourceDuration) * 100}%` }}
                        />
                    )
                ))}
                {/* the source monitor's own playhead */}
                {sourceOpen && sourceDuration > 0 && (
                    <div
                        className="absolute top-0 bottom-0 w-px bg-ink pointer-events-none"
                        style={{ left: `${(Math.min(sourceTime, sourceDuration) / sourceDuration) * 100}%` }}
                    />
                )}
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
    );

    return (
        <div className="fixed inset-0 z-[110] bg-paper flex flex-col animate-fade">
            {/* header */}
            <div className="px-4 sm:px-6 pt-4 pb-3 border-b border-rule flex items-start justify-between gap-4 shrink-0">
                <div className="min-w-0">
                    <p className="eyebrow mb-1">EDITOR · CLIP {clipIndex + 1}</p>
                    <h2 className="font-display lowercase text-xl sm:text-2xl text-ink truncate">edit clip</h2>
                    {clipTitle && <p className="text-xs text-muted truncate mt-0.5">{clipTitle}</p>}
                    {/* Phone: the readouts move under the title — as a third
                        column they squeezed the title to two characters. */}
                    <p className="readout sm:hidden mt-1 truncate">
                        {fmt(total)} · {needsSourcePath ? 'FULL RE-FRAME' : 'FAST RECUT'}
                    </p>
                </div>
                <div className="flex items-center gap-2 sm:gap-3 shrink-0">
                    <div className="text-right hidden sm:block">
                        <p className="readout">DURATION · {fmt(total)}</p>
                        <p className="readout mt-1">
                            {needsSourcePath ? 'PATH · FULL RE-FRAME' : 'PATH · FAST RECUT'}
                            {edl.rerender_minutes > 0 && ` · ≈${Math.max(1, Math.ceil(total / 60))} MIN`}
                        </p>
                    </div>
                    {sourceAvailable && (
                        <button
                            onClick={() => setShowSource((v) => !v)}
                            title={showSource
                                ? 'put the source monitor away and edit the clip on its own'
                                : 'bring back the source monitor, its transcript and in/out marking'}
                            aria-label={showSource ? 'hide source' : 'show source'}
                            className="btn-quiet text-xs py-1.5 px-2.5 sm:px-3 flex items-center gap-1.5 lowercase"
                        >
                            {showSource ? <PanelLeftClose size={14} /> : <PanelLeft size={14} />}
                            <span className="hidden sm:inline">{showSource ? 'hide source' : 'show source'}</span>
                        </button>
                    )}
                    {confirmClose ? (
                        <div className="flex flex-wrap items-center justify-end gap-2">
                            <span className="text-xs text-warn lowercase hidden sm:inline">discard changes?</span>
                            <button className="btn-danger text-xs py-1.5 px-3" onClick={onClose}>discard</button>
                            <button className="btn-ghost text-xs py-1.5 px-3" onClick={() => setConfirmClose(false)}>keep editing</button>
                        </div>
                    ) : (
                        <button
                            onClick={() => (rendering ? onClose() : dirty ? setConfirmClose(true) : onClose())}
                            className="p-2 rounded-input text-muted hover:text-ink hover:bg-paper3 transition-colors"
                            aria-label="close editor"
                        >
                            <X size={18} />
                        </button>
                    )}
                </div>
            </div>

            {/* main — three columns above xl, stacked below. select-none/touch-none
                keep the browser from turning a drag on a track into a text
                selection or a page pan. */}
            <div className="flex-1 min-h-0 flex flex-col xl:flex-row gap-4 px-4 sm:px-6 py-4 overflow-y-auto xl:overflow-hidden select-none">

                {/* ---- column 1 · source ---- */}
                {sourceOpen && (
                    <div className="flex-1 min-w-0 flex flex-col min-h-0 gap-2">
                        <p className="eyebrow shrink-0">Source</p>
                        {/* The black hugs the picture instead of the column: a wide
                            box around a short 16:9 frame is exactly the dead space
                            this layout set out to remove. */}
                        <div className="flex-1 min-h-0 min-w-0 flex items-center justify-center">
                            <video
                                ref={sourceRef}
                                src={getApiUrl(edl.source.url)}
                                controls
                                playsInline
                                preload="metadata"
                                onLoadedMetadata={applySeek}
                                onTimeUpdate={(e) => setSourceTime(e.target.currentTime)}
                                className="h-full w-auto max-w-full max-h-full bg-black rounded-card border border-rule"
                            />
                        </div>

                        {sourceTrack}

                        {/* Two steps, shown as two: pick a range on the source, then
                            put it in the clip. Six controls in one undifferentiated
                            row read as six unrelated buttons. */}
                        <div className="shrink-0 rounded-input border border-rule bg-paper2 p-2 flex items-stretch gap-3">
                            <div className="flex items-center gap-1.5 flex-1 min-w-0">
                                <span className="readout shrink-0 text-muted">1 · MARK</span>
                                <button onClick={() => markHere('in')} className="btn-quiet text-[11px] py-1 px-2 flex items-center gap-1 shrink-0">
                                    <ChevronsRight size={12} /> in <span className="text-muted">i</span>
                                </button>
                                <button onClick={() => markHere('out')} className="btn-quiet text-[11px] py-1 px-2 flex items-center gap-1 shrink-0">
                                    <ChevronsLeft size={12} /> out <span className="text-muted">o</span>
                                </button>
                                <p className={`readout px-1 truncate ${markRange ? 'text-ink' : ''}`}>
                                    {markIn === null ? '—:——' : fmt(markIn)}
                                    {' → '}{markOut === null ? '—:——' : fmt(markOut)}
                                    {markRange && ` · ${fmt(markRange.end - markRange.start)}`}
                                    {markIn !== null && markOut !== null && !markRange
                                        && ` · UNDER ${minSeg}S`}
                                </p>
                                <button
                                    onClick={clearMarks}
                                    disabled={markIn === null && markOut === null}
                                    className="p-1 rounded-input text-muted hover:text-ink hover:bg-paper3 disabled:opacity-40 shrink-0"
                                    aria-label="clear in and out marks"
                                >
                                    <X size={13} />
                                </button>
                            </div>

                            <div className="w-px bg-[color:var(--color-rule-2)] shrink-0" />

                            <div className="flex items-center gap-1.5 shrink-0">
                                <span className="readout text-muted">2 · SEND</span>
                                <button
                                    onClick={() => sendToClip('replace')}
                                    disabled={!markRange}
                                    title="the selected segment becomes this range (.)"
                                    className="btn-primary text-[11px] py-1.5 px-2 disabled:opacity-40"
                                >
                                    replace #{selected + 1}
                                </button>
                                <button
                                    onClick={() => sendToClip('insert')}
                                    disabled={!markRange || segments.length >= limits.max_segments}
                                    title="add this range as a new segment after the selected one (,)"
                                    className="btn-quiet text-[11px] py-1.5 px-2 disabled:opacity-40"
                                >
                                    insert after #{selected + 1}
                                </button>
                            </div>
                        </div>

                        {/* transcript of the whole source */}
                        <div className="shrink-0 h-[30%] min-h-[9rem] flex flex-col">
                            <div className="flex items-center justify-between mb-1.5 gap-2 shrink-0">
                                <p className="eyebrow">Transcript · full source</p>
                                {/* Boundary actions live HERE, above the words they
                                    act on, because this is where the eye is when
                                    picking a cut point from the text (issue #73). */}
                                {selectedWord ? (
                                    <div className="flex items-center gap-1.5 shrink-0">
                                        <button
                                            onClick={() => setSegment(selected, { start: selectedWord.s }, { snap: false })}
                                            title={`segment #${selected + 1} starts at "${selectedWord.w}" (${fmt(selectedWord.s)})`}
                                            className="btn-quiet text-[11px] py-1 px-2"
                                        >
                                            #{selected + 1} starts here
                                        </button>
                                        <button
                                            onClick={() => setSegment(selected, { end: selectedWord.e }, { snap: false })}
                                            title={`segment #${selected + 1} ends after "${selectedWord.w}" (${fmt(selectedWord.e)})`}
                                            className="btn-quiet text-[11px] py-1 px-2"
                                        >
                                            #{selected + 1} ends here
                                        </button>
                                    </div>
                                ) : words.length > 0 ? (
                                    <span className="readout shrink-0">CLICK A WORD, THEN SET A BOUNDARY</span>
                                ) : null}
                            </div>
                            {words.length === 0 ? (
                                <p className="text-xs text-muted lowercase">this job kept no transcript</p>
                            ) : (
                                <div
                                    ref={transcriptRef}
                                    className="relative flex-1 min-h-0 flex flex-wrap content-start gap-x-1 gap-y-1.5 overflow-y-auto custom-scrollbar pr-1"
                                >
                                    {chunks.map((c) => {
                                        const first = c.items[0].s;
                                        const last = c.items[c.items.length - 1].e;
                                        // 'none' and 'all' are stable primitives, so only the
                                        // one or two slices straddling a segment edge repaint
                                        // while a trim handle is being dragged.
                                        let lit = 'none';
                                        if (highlightSeg && !(last <= highlightSeg.start || first >= highlightSeg.end)) {
                                            lit = (first >= highlightSeg.start && last <= highlightSeg.end)
                                                ? 'all' : highlightSeg;
                                        }
                                        const local = (idx) => (
                                            idx >= c.offset && idx < c.offset + c.items.length ? idx - c.offset : -1
                                        );
                                        return (
                                            <TranscriptChunk
                                                key={c.offset}
                                                items={c.items}
                                                offset={c.offset}
                                                lit={lit}
                                                active={local(activeWordIndex)}
                                                anchorAt={local(anchorIndex)}
                                                selectedAt={local(selectedWordIndex)}
                                                onPick={pickWord}
                                            />
                                        );
                                    })}
                                </div>
                            )}
                        </div>
                    </div>
                )}

                {/* ---- column 2 · program ---- */}
                {/* Sized to the 9:16 preview plus enough track to aim with, rather
                    than an equal share: a wider column here is just black beside a
                    height-bound video, and that width is worth more to the source.
                    With the source away it takes everything, which is what gives
                    the clip track its full precision back. */}
                <div className={`flex flex-col min-h-0 gap-2 ${sourceOpen ? 'xl:w-[26rem] 2xl:w-[30rem] xl:shrink-0' : 'flex-1'}`}>
                    <div className="flex items-center justify-between gap-2 shrink-0">
                        <p className="eyebrow">Program</p>
                        {dirty && (
                            <span className="badge-warn">
                                {missingSeconds > COVERAGE_EPSILON
                                    ? `${fmt(missingSeconds)} needs rendering`
                                    : 'previewing the edit · re-render to keep it'}
                            </span>
                        )}
                    </div>
                    <div className="flex-1 min-h-0 flex items-center justify-center">
                        <div className="h-full max-h-full aspect-[9/16] bg-black rounded-card border border-rule overflow-hidden">
                            <video
                                ref={videoRef}
                                src={previewUrl}
                                controls
                                playsInline
                                className="w-full h-full object-contain"
                                onTimeUpdate={onClipTimeUpdate}
                                onSeeked={onClipSeeked}
                                onPlay={onClipPlay}
                                onPause={stopPlayLoop}
                            />
                        </div>
                    </div>

                    {/* clip track */}
                    <div className="shrink-0 select-none">
                        <div className="flex items-center justify-between mb-1.5 gap-3">
                            <p className="readout">CLIP · {fmt(total)}</p>
                            {dirty && (
                                <p className="readout truncate">
                                    {missingSeconds > COVERAGE_EPSILON
                                        ? `RED · ${fmt(missingSeconds)} NOT RENDERED YET`
                                        : 'PREVIEWING THE EDIT'}
                                </p>
                            )}
                        </div>
                        <div
                            ref={clipTrackRef}
                            onPointerDown={startClipScrub}
                            className="relative h-12 rounded-input bg-paper border border-rule overflow-hidden touch-none cursor-pointer"
                        >
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
                                        className="absolute left-0 top-0 bottom-0 w-2 touch-none cursor-ew-resize rounded-l-[6px]"
                                        style={{ background: SEGMENT_COLORS[i % SEGMENT_COLORS.length] }}
                                    />
                                    <div
                                        onPointerDown={(e) => startTrimDrag(e, i, 'end', clipTrackRef.current, clipTrackSeconds)}
                                        className="absolute right-0 top-0 bottom-0 w-2 touch-none cursor-ew-resize rounded-r-[6px]"
                                        style={{ background: SEGMENT_COLORS[i % SEGMENT_COLORS.length] }}
                                    />
                                </div>
                            ))}
                            {/* Render status, as an NLE draws it: red is material the
                                last render never saw, so there is no frame to show
                                until it is rendered. Everything else plays straight
                                out of the existing file. */}
                            {dirty && coverage.map((sp, i) => (
                                <div
                                    key={i}
                                    className={`absolute top-0 h-1 pointer-events-none ${
                                        sp.rendered === null ? 'bg-danger' : 'bg-ok/50'}`}
                                    style={{
                                        left: `${(sp.start / clipTrackSeconds) * 100}%`,
                                        width: `${((sp.end - sp.start) / clipTrackSeconds) * 100}%`,
                                    }}
                                />
                            ))}
                            {/* A handle, not a hairline: the same solid, grabbable
                                shape the source track uses, so this reads as
                                something you drive rather than a marker that
                                happens to move. No handler of its own — the
                                pointerdown bubbles to the track, which already
                                scrubs from the pointer position. */}
                            <div
                                title="drag to move through the clip"
                                className="absolute top-1 bottom-1 w-2.5 -ml-[5px] rounded-[4px] bg-ink border border-paper cursor-grab active:cursor-grabbing"
                                style={{ left: `${(Math.min(playhead, clipTrackSeconds) / clipTrackSeconds) * 100}%` }}
                            />
                        </div>
                    </div>

                    {!sourceAvailable && sourceTrack}
                </div>

                {/* ---- column 3 · controls ---- */}
                <div className="w-full xl:w-[21rem] shrink-0 flex flex-col min-h-0">
                    <div className="flex-1 xl:overflow-y-auto custom-scrollbar pr-1 space-y-5">
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
                                            {sourceOpen && (
                                                <button className="p-1 rounded-input text-muted hover:text-ink hover:bg-paper" onClick={(e) => { e.stopPropagation(); dispatch({ type: 'select', index: i }); seekSource(seg.start); }} aria-label="show this segment in the source monitor"><Film size={13} /></button>
                                            )}
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
                            {!sourceAvailable && (
                                <p className="text-[11px] text-muted mt-2 leading-relaxed">
                                    the source video is no longer on the server, so cuts are
                                    limited to the original clip range (extending or reframing
                                    needs it; newly processed videos keep theirs)
                                </p>
                            )}
                        </div>

                        {/* framing override */}
                        <div>
                            <p className="eyebrow mb-2">Framing</p>
                            <div className="grid grid-cols-2 sm:grid-cols-4 gap-1.5">
                                {[
                                    { value: 'auto', label: 'auto', hint: 'AI decides per scene' },
                                    { value: 'gameplay', label: 'gameplay', hint: 'whole game and original facecam' },
                                    { value: 'full', label: 'full frame', hint: 'whole shot, no side-crop' },
                                    { value: 'track', label: 'track subject', hint: 'crop follows the person' },
                                ].map((f) => (
                                    <button
                                        key={f.value}
                                        type="button"
                                        title={f.hint}
                                        disabled={f.value !== 'auto' && !sourceAvailable}
                                        onClick={() => setFraming(f.value)}
                                        className={`py-1.5 px-2 rounded-input border text-xs lowercase transition-colors
                                            ${framing === f.value
                                                ? 'border-[color:var(--color-accent)] text-ink'
                                                : 'border-rule2 text-muted hover:border-[color:var(--color-accent)]'}
                                            disabled:opacity-40 disabled:cursor-not-allowed`}
                                    >
                                        {f.label}
                                    </button>
                                ))}
                            </div>
                            {!sourceAvailable && (
                                <p className="text-[11px] text-muted mt-1.5 leading-relaxed">
                                    framing changes need the source video, which is no longer on the server
                                </p>
                            )}
                            {framing !== renderedFraming && (
                                <p className="text-[11px] text-muted mt-1.5 leading-relaxed">
                                    changing the framing re-runs the reframe engine (slower than a fast recut)
                                </p>
                            )}
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

                        {/* keyboard legend — moved off the clip track, which no longer
                            has the width for it */}
                        <div>
                            <p className="eyebrow mb-2">Shortcuts</p>
                            <p className="readout leading-relaxed">
                                SPACE PLAY · S SPLIT · ⌫ DELETE · ⌘Z UNDO
                                {sourceOpen && ' · I MARK IN · O MARK OUT · , INSERT · . REPLACE'}
                            </p>
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
                            <button
                                className="btn-ghost"
                                onClick={() => (rendering ? onClose() : dirty ? setConfirmClose(true) : onClose())}
                            >
                                {rendering ? 'close' : dirty ? 'cancel' : 'close'}
                            </button>
                            <button className="btn-primary flex-1 flex items-center justify-center gap-2" disabled={!canRender || !dirty} onClick={doRender}>
                                {rendering
                                    ? (<><Loader2 size={16} className="animate-spin text-brassink" /> re-rendering… {renderSeconds}s</>)
                                    : (needsSourcePath ? 're-render from source' : 're-render clip')}
                            </button>
                        </div>
                        {rendering && (
                            <p className="text-[11px] text-muted mt-2 lowercase">
                                you can close this editor; the render keeps going and the clip card updates when it finishes
                            </p>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}


// One slice of the transcript. Memoised because the whole transcript is on
// screen: repainting a couple of thousand words costs ~66ms, and the source
// monitor fires timeupdate about four times a second while it plays. Sliced,
// a moving playhead repaints ~50 words instead of all of them.
//
// It returns a Fragment rather than a wrapper element so the words stay direct
// children of the scroll box — the flex-wrap layout and the anchor's offsetTop
// both depend on that.
const TranscriptChunk = React.memo(function TranscriptChunk({
    items, offset, lit, active, anchorAt, selectedAt, onPick,
}) {
    return (
        <>
            {items.map((w, i) => {
                const inside = lit === 'all'
                    || (lit !== 'none' && w.e > lit.start && w.s < lit.end);
                const isActive = i === active;
                const isSel = i === selectedAt;
                return (
                    <button
                        key={`${w.s}-${offset + i}`}
                        data-anchor={i === anchorAt ? '1' : undefined}
                        data-active={isActive ? '1' : undefined}
                        onClick={() => onPick(w)}
                        title={`${fmt(w.s)} – ${fmt(w.e)}`}
                        className={`px-1 py-0.5 rounded text-xs transition-colors ${
                            isSel ? 'bg-brass text-brassink'
                                : isActive ? 'bg-brass/30 text-ink'
                                    : inside ? 'text-ink hover:bg-paper3'
                                        : 'text-muted hover:bg-paper3'}`}
                    >
                        {w.w}
                    </button>
                );
            })}
        </>
    );
});
