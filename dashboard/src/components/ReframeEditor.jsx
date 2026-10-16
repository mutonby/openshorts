import React, { useState, useEffect, useCallback, useRef } from 'react';
import { X, Loader2, Crosshair, RotateCcw, AlertCircle, Play, Pause, Columns2 } from 'lucide-react';
import { getApiUrl } from '../config';
import { apiJson } from '../lib/api';

// Manual reframing: the automatic crop is right most of the time and grossly
// wrong occasionally, and until now there was no way to say "frame it here".
//
// The unit is the scene, because a podcast cuts between a fixed close camera
// and a fixed wide one and the right crop differs per camera. Scenes the user
// does not touch stay automatic, so fixing one bad shot cannot spoil the good
// ones — only adjusted scenes are sent.
//
// Two things a still frame cannot tell you, and both are handled here:
//   - WHO is talking. Each scene plays its own range of an uncropped preview,
//     with sound, and the rectangle stays overlaid while it plays.
//   - Whether one window is even enough. A scene can be split into two stacked
//     regions, positioned independently.

const fmt = (s) => {
    const m = Math.floor(s / 60);
    const r = Math.floor(s % 60);
    return `${m}:${String(r).padStart(2, '0')}`;
};

export default function ReframeEditor({ jobId, clipIndex, clipTitle, onClose, onReframed }) {
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [data, setData] = useState(null);
    const [overrides, setOverrides] = useState({});   // idx -> number | {top,bottom}
    const [playing, setPlaying] = useState(null);     // scene index being played
    const [saving, setSaving] = useState(false);

    useEffect(() => {
        let alive = true;
        (async () => {
            try {
                const res = await apiJson(`/api/clip/${jobId}/${clipIndex}/scenes`);
                if (!alive) return;
                setData(res);
                // Start from what is already applied. A re-render rebuilds
                // from source with only what it receives, so opening blank
                // would quietly discard every earlier adjustment on the next
                // save.
                const salvos = res.saved_overrides || {};
                setOverrides(Object.fromEntries(
                    Object.entries(salvos).map(([k, v]) => [Number(k), v])
                ));
            } catch (e) {
                if (alive) setError(e?.message || 'Could not read the scenes of this clip.');
            } finally {
                if (alive) setLoading(false);
            }
        })();
        return () => { alive = false; };
    }, [jobId, clipIndex]);

    const half = (data?.crop_width_fraction ?? 0.5) / 2;
    const clamp = useCallback((v) => Math.min(1 - half, Math.max(half, v)), [half]);

    // What a scene shows right now: the user's value, else the backend's
    // suggestion (the biggest face in the shot).
    const valueOf = useCallback((scene) => (
        overrides[scene.index] ?? clamp(scene.suggested_center)
    ), [overrides, clamp]);

    const setSingle = useCallback((idx, fraction) => {
        setOverrides((o) => ({ ...o, [idx]: clamp(fraction) }));
    }, [clamp]);

    const setSplitHalf = useCallback((idx, which, fraction) => {
        setOverrides((o) => {
            const cur = o[idx];
            const base = (cur && typeof cur === 'object')
                ? cur
                : { top: { x: clamp(0.3), y: 0.5 }, bottom: { x: clamp(0.7), y: 0.5 } };
            const anterior = base[which] || { y: 0.5 };
            return { ...o, [idx]: {
                ...base,
                [which]: { x: clamp(fraction), y: anterior.y ?? 0.5 },
            } };
        });
    }, [clamp]);

    const toggleTripleSplit = useCallback((idx) => {
        setOverrides((o) => {
            const cur = o[idx];
            if (cur && typeof cur === 'object' && ('custom' in cur)) {
                return { ...o, [idx]: 0.5 };
            }
            return { ...o, [idx]: {
                custom: [
                    { crop: { x: 0, y: 0, w: 1, h: 0.5 }, dest: { x: 0, y: 0, w: 1, h: 0.5 } },
                    { crop: { x: 0, y: 0.5, w: 0.5, h: 0.5 }, dest: { x: 0, y: 0.5, w: 0.5, h: 0.5 } },
                    { crop: { x: 0.5, y: 0.5, w: 0.5, h: 0.5 }, dest: { x: 0.5, y: 0.5, w: 0.5, h: 0.5 } }
                ]
            } };
        });
    }, []);

    const swapTripleLayout = useCallback((idx) => {
        setOverrides((o) => {
            const cur = o[idx];
            if (!cur || !cur.custom) return o;
            const isLayoutA = cur.custom[0].dest.w === 1;
            if (isLayoutA) {
                return { ...o, [idx]: {
                    custom: [
                        { crop: cur.custom[1].crop, dest: { x: 0, y: 0, w: 0.5, h: 0.5 } },
                        { crop: cur.custom[2].crop, dest: { x: 0.5, y: 0, w: 0.5, h: 0.5 } },
                        { crop: cur.custom[0].crop, dest: { x: 0, y: 0.5, w: 1, h: 0.5 } }
                    ]
                } };
            } else {
                return { ...o, [idx]: {
                    custom: [
                        { crop: cur.custom[2].crop, dest: { x: 0, y: 0, w: 1, h: 0.5 } },
                        { crop: cur.custom[0].crop, dest: { x: 0, y: 0.5, w: 0.5, h: 0.5 } },
                        { crop: cur.custom[1].crop, dest: { x: 0.5, y: 0.5, w: 0.5, h: 0.5 } }
                    ]
                } };
            }
        });
    }, []);

    const updateCustomCrop = useCallback((idx, boxIndex, newCrop) => {
        setOverrides((o) => {
            const cur = o[idx];
            if (!cur || !cur.custom) return o;
            const newCustom = [...cur.custom];
            newCustom[boxIndex] = { ...newCustom[boxIndex], crop: newCrop };
            return { ...o, [idx]: { custom: newCustom } };
        });
    }, []);

    const toggleSplit = useCallback((idx, scene) => {
        setOverrides((o) => {
            const cur = o[idx];
            if (cur && typeof cur === 'object') {
                // back to a single window, kept where the top half was
                return { ...o, [idx]: cur.top?.x ?? 0.5 };
            }
            const centre = typeof cur === 'number' ? cur : clamp(scene.suggested_center);
            // SPLIT halves crop vertically as well, so each carries the face
            // height the backend measured for this scene.
            const y = scene.suggested_center_y ?? 0.5;
            return { ...o, [idx]: {
                top: { x: clamp(centre - 0.2), y },
                bottom: { x: clamp(centre + 0.2), y },
            } };
        });
    }, [clamp]);

    const resetScene = useCallback((idx) => {
        setOverrides((o) => {
            const next = { ...o };
            delete next[idx];
            return next;
        });
    }, []);

    const adjusted = Object.keys(overrides).length;

    const handleSave = async () => {
        if (!adjusted || saving) return;
        setSaving(true);
        setError(null);
        try {
            const payload = Object.fromEntries(Object.entries(overrides).map(([k, v]) => {
                if (typeof v === 'object') {
                    if ('custom' in v) {
                        return [String(k), { custom: v.custom.map(p => ({
                            crop: { x: Number(p.crop.x.toFixed(4)), y: Number(p.crop.y.toFixed(4)), w: Number(p.crop.w.toFixed(4)), h: Number(p.crop.h.toFixed(4)) },
                            dest: p.dest
                        })) }];
                    }
                    return [String(k), {
                        top: { x: Number(v.top.x.toFixed(4)), y: Number(v.top.y.toFixed(4)) },
                        bottom: { x: Number(v.bottom.x.toFixed(4)), y: Number(v.bottom.y.toFixed(4)) }
                    }];
                }
                return [String(k), Number(v.toFixed(4))];
            }));
            const res = await apiJson('/api/clip/reframe', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ job_id: jobId, clip_index: clipIndex, crop_overrides: payload }),
            });
            if (onReframed) onReframed(clipIndex, res);
            onClose();
        } catch (e) {
            setError(e?.message || 'The re-render failed.');
        } finally {
            setSaving(false);
        }
    };

    return (
        /* Bottom sheet on a phone, centred dialog from sm — same shape as the
           shared Modal so the app has one overlay idiom, not two. */
        <div className="fixed inset-0 z-50 bg-black/80 flex items-end sm:items-center justify-center p-0 sm:p-4">
            <div className="card w-full max-w-3xl max-h-[92vh] sm:max-h-[90vh] flex flex-col rounded-b-none sm:rounded-card animate-sheet-up sm:animate-none">
                <div className="flex items-center justify-between p-4 border-b border-rule">
                    <div className="flex items-center gap-2.5 min-w-0">
                        <Crosshair size={18} className="text-brass shrink-0" />
                        <div className="min-w-0">
                            <h2 className="text-base font-medium text-ink lowercase truncate">reframing</h2>
                            {clipTitle && <p className="text-xs text-muted truncate">{clipTitle}</p>}
                        </div>
                    </div>
                    <button onClick={onClose} className="p-1.5 hover:bg-paper3 rounded-input transition-colors">
                        <X size={18} className="text-muted" />
                    </button>
                </div>

                <div className="flex-1 overflow-y-auto overscroll-contain custom-scrollbar p-4 space-y-5">
                    <p className="text-xs text-muted leading-relaxed">
                        Play a scene to hear who is talking, then drag the rectangle over
                        the person you want. Each scene is one camera. Scenes you leave
                        alone keep the automatic camera.
                    </p>

                    {loading && (
                        <div className="flex items-center gap-2 text-sm text-muted py-8 justify-center">
                            <Loader2 size={18} className="animate-spin text-brass" />
                            reading the scenes…
                        </div>
                    )}

                    {error && (
                        <div className="flex items-start gap-2 text-xs text-warn border border-warn/40 rounded-input p-3">
                            <AlertCircle size={14} className="shrink-0 mt-0.5" />
                            <span>{error}</span>
                        </div>
                    )}

                    {data?.scenes?.map((scene) => (
                        <SceneRow
                            key={scene.index}
                            scene={scene}
                            value={valueOf(scene)}
                            widthFraction={data.crop_width_fraction}
                            previewUrl={data.preview_url}
                            touched={scene.index in overrides}
                            playing={playing === scene.index}
                            onPlayToggle={() => setPlaying((p) => (p === scene.index ? null : scene.index))}
                            onMoveSingle={(f) => setSingle(scene.index, f)}
                            onMoveHalf={(which, f) => setSplitHalf(scene.index, which, f)}
                            onToggleSplit={() => toggleSplit(scene.index, scene)}
                            onToggleTripleSplit={() => toggleTripleSplit(scene.index)}
                            onSwapTripleLayout={() => swapTripleLayout(scene.index)}
                            onUpdateCustomCrop={(boxIdx, crop) => updateCustomCrop(scene.index, boxIdx, crop)}
                            onReset={() => resetScene(scene.index)}
                        />
                    ))}
                </div>

                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 p-4 border-t border-rule">
                    <span className="text-xs text-muted">
                        {adjusted === 0
                            ? 'nothing adjusted yet'
                            : `${adjusted} scene${adjusted > 1 ? 's' : ''} reframed by hand`}
                    </span>
                    <div className="flex items-center gap-2 [&>button]:flex-1 sm:[&>button]:flex-none">
                        <button onClick={onClose} className="btn-quiet py-2 px-4 text-sm">cancel</button>
                        <button
                            onClick={handleSave}
                            disabled={!adjusted || saving}
                            className="btn-primary py-2 px-4 text-sm disabled:opacity-40"
                        >
                            {saving ? <Loader2 size={16} className="animate-spin" /> : null}
                            {saving ? 're-rendering…' : 'apply reframing'}
                        </button>
                    </div>
                </div>
                <div className="sm:hidden safe-bottom" />
            </div>
        </div>
    );
}


// seeked to this scene, so the rectangle can be judged against moving pictures
// and sound rather than a single still.
function SceneRow({ scene, value, widthFraction, previewUrl, touched, playing,
                    onPlayToggle, onMoveSingle, onMoveHalf, onToggleSplit, onToggleTripleSplit, onSwapTripleLayout, onUpdateCustomCrop, onReset }) {
    const boxRef = useRef(null);
    const videoRef = useRef(null);
    const [dragging, setDragging] = useState(null);   // null | 'single' | 'top' | 'bottom'

    const isDoubleSplit = value && typeof value === 'object' && ('top' in value);
    const isTripleSplit = value && typeof value === 'object' && ('custom' in value);
    const isSplit = isDoubleSplit; // legacy single-axis split

    // Play only this scene's slice of the shared preview.
    useEffect(() => {
        const v = videoRef.current;
        if (!v) return;
        if (!playing) { v.pause(); return; }
        v.currentTime = scene.start;
        v.play().catch(() => {});
        const stopAtEnd = () => { if (v.currentTime >= scene.end) { v.pause(); } };
        v.addEventListener('timeupdate', stopAtEnd);
        return () => v.removeEventListener('timeupdate', stopAtEnd);
    }, [playing, scene.start, scene.end]);

    const fractionFromEvent = useCallback((clientX) => {
        const el = boxRef.current;
        if (!el) return 0.5;
        const rect = el.getBoundingClientRect();
        return (clientX - rect.left) / rect.width;
    }, []);

    useEffect(() => {
        if (!dragging) return;
        const move = (e) => {
            const x = e.touches ? e.touches[0].clientX : e.clientX;
            const f = fractionFromEvent(x);
            if (dragging === 'single') onMoveSingle(f);
            else onMoveHalf(dragging, f);
        };
        const up = () => setDragging(null);
        window.addEventListener('mousemove', move);
        window.addEventListener('mouseup', up);
        window.addEventListener('touchmove', move);
        window.addEventListener('touchend', up);
        return () => {
            window.removeEventListener('mousemove', move);
            window.removeEventListener('mouseup', up);
            window.removeEventListener('touchmove', move);
            window.removeEventListener('touchend', up);
        };
    }, [dragging, fractionFromEvent, onMoveSingle, onMoveHalf]);

    const startDrag = (which) => (e) => {
        e.stopPropagation();
        setDragging(which);
        const x = e.touches ? e.touches[0].clientX : e.clientX;
        if (which === 'single') onMoveSingle(fractionFromEvent(x));
        else onMoveHalf(which, fractionFromEvent(x));
    };

    const win = (centre, label, which) => {
        const leftPct = (centre - widthFraction / 2) * 100;
        return (
            <div
                key={which}
                onMouseDown={startDrag(which)}
                onTouchStart={startDrag(which)}
                className="absolute inset-y-0 border-2 border-brass cursor-ew-resize"
                style={{ left: `${leftPct}%`, width: `${widthFraction * 100}%` }}
            >
                {label && (
                    <span className="absolute top-1 left-1 text-[10px] px-1 rounded bg-brass text-paper lowercase">
                        {label}
                    </span>
                )}
            </div>
        );
    };

    return (
        <div className="space-y-1.5">
            <div className="flex items-center justify-between text-xs gap-2">
                <div className="flex items-center gap-2 min-w-0">
                    <button
                        onClick={onPlayToggle}
                        className="flex items-center gap-1 text-ink2 hover:text-brass transition-colors"
                        title="play this scene with sound"
                    >
                        {playing ? <Pause size={13} /> : <Play size={13} />}
                    </button>
                    <span className="readout text-muted truncate">
                        scene {scene.index + 1} · {fmt(scene.start)}–{fmt(scene.end)}
                    </span>
                </div>
                <div className="flex items-center gap-3 shrink-0">
                    <button
                        onClick={onToggleSplit}
                        className={`flex items-center gap-1 transition-colors ${
                            isDoubleSplit ? 'text-brass' : 'text-muted hover:text-ink2'}`}
                        title="stack two regions instead of one window"
                    >
                        <Columns2 size={12} /> split
                    </button>
                    <button
                        onClick={onToggleTripleSplit}
                        className={`flex items-center gap-1 transition-colors ${
                            isTripleSplit ? 'text-brass' : 'text-muted hover:text-ink2'}`}
                        title="3 boxes custom layout"
                    >
                        <Columns2 size={12} className="rotate-90" /> triple
                    </button>
                    {touched ? (
                        <button onClick={onReset} className="flex items-center gap-1 text-brass hover:underline">
                            <RotateCcw size={12} /> automatic
                        </button>
                    ) : (
                        <span className="text-muted">automatic</span>
                    )}
                </div>
            </div>

            <div
                ref={boxRef}
                className={`relative overflow-hidden rounded-input select-none border ${
                    touched ? 'border-brass' : 'border-rule'}`}
            >
                {playing && previewUrl ? (
                    <video
                        ref={videoRef}
                        src={getApiUrl(previewUrl)}
                        playsInline
                        className="w-full block"
                    />
                ) : scene.thumbnail_url ? (
                    <img
                        src={getApiUrl(scene.thumbnail_url)}
                        alt=""
                        draggable={false}
                        className="w-full block pointer-events-none"
                    />
                ) : (
                    <div className="w-full aspect-video bg-paper3" />
                )}

                {/* Everything outside the kept region is dimmed, so what survives
                    the crop is what stays bright. */}
                {(!isDoubleSplit && !isTripleSplit) && (
                    <>
                        <div className="absolute inset-y-0 left-0 bg-black/65 pointer-events-none"
                             style={{ width: `${Math.max(0, (value - widthFraction / 2) * 100)}%` }} />
                        <div className="absolute inset-y-0 right-0 bg-black/65 pointer-events-none"
                             style={{ width: `${Math.max(0, 100 - (value + widthFraction / 2) * 100)}%` }} />
                    </>
                )}

                {isDoubleSplit
                    ? [win(value.top.x, 'top', 'top'), win(value.bottom.x, 'bottom', 'bottom')]
                    : null}
                    
                {isTripleSplit
                    ? value.custom.map((panel, i) => (
                        <CustomBox 
                            key={i}
                            crop={panel.crop}
                            dest={panel.dest}
                            label={`Box ${i + 1}`}
                            parentRef={boxRef}
                            onChange={(newCrop) => onUpdateCustomCrop(i, newCrop)}
                        />
                    ))
                    : null}

                {(!isDoubleSplit && !isTripleSplit) ? win(value, null, 'single') : null}
            </div>

            {isDoubleSplit && (
                <p className="text-[11px] text-muted leading-snug">
                    Two regions stacked in the vertical frame: <strong>top</strong> above,
                    <strong> bottom</strong> below. Drag each one onto the person it should hold.
                </p>
            )}
            
            {isTripleSplit && (
                <div className="flex items-start justify-between gap-4 mt-2">
                    <p className="text-[11px] text-muted leading-snug flex-1">
                        Three regions for the vertical frame. Drag boxes to move, drag bottom-right corner to resize.
                    </p>
                    <button onClick={onSwapTripleLayout} className="text-[11px] text-brass hover:underline whitespace-nowrap">
                        Swap Layout
                    </button>
                </div>
            )}
        </div>
    );
}

function CustomBox({ crop, label, onChange, parentRef }) {
    const startRef = useRef(null);

    const handlePointerDown = (e, mode) => {
        e.stopPropagation();
        e.preventDefault();
        const parentRect = parentRef.current.getBoundingClientRect();
        const x = e.touches ? e.touches[0].clientX : e.clientX;
        const y = e.touches ? e.touches[0].clientY : e.clientY;
        startRef.current = { x, y, cropX: crop.x, cropY: crop.y, cropW: crop.w, cropH: crop.h, mode, parentRect };
        
        const move = (eMove) => {
            const currentX = eMove.touches ? eMove.touches[0].clientX : eMove.clientX;
            const currentY = eMove.touches ? eMove.touches[0].clientY : eMove.clientY;
            const dx = (currentX - startRef.current.x) / startRef.current.parentRect.width;
            const dy = (currentY - startRef.current.y) / startRef.current.parentRect.height;
            
            let newCrop = { ...crop };
            if (startRef.current.mode === 'move') {
                newCrop.x = Math.max(0, Math.min(1 - newCrop.w, startRef.current.cropX + dx));
                newCrop.y = Math.max(0, Math.min(1 - newCrop.h, startRef.current.cropY + dy));
            } else if (startRef.current.mode === 'resize') {
                let w = startRef.current.cropW + dx;
                let h = startRef.current.cropH + dy;
                w = Math.max(0.1, Math.min(1 - startRef.current.cropX, w));
                h = Math.max(0.1, Math.min(1 - startRef.current.cropY, h));
                newCrop.w = w;
                newCrop.h = h;
            }
            onChange(newCrop);
        };
        
        const up = () => {
            window.removeEventListener('mousemove', move);
            window.removeEventListener('mouseup', up);
            window.removeEventListener('touchmove', move);
            window.removeEventListener('touchmove', move); // extra listener removal not needed but ok
            window.removeEventListener('touchend', up);
        };
        window.addEventListener('mousemove', move);
        window.addEventListener('mouseup', up);
        window.addEventListener('touchmove', move);
        window.addEventListener('touchend', up);
    };

    return (
        <div 
            className="absolute border-2 border-brass bg-[color:var(--color-brass)]/20 cursor-move hover:bg-[color:var(--color-brass)]/30 transition-colors"
            style={{ 
                left: `${crop.x * 100}%`, 
                top: `${crop.y * 100}%`, 
                width: `${crop.w * 100}%`, 
                height: `${crop.h * 100}%` 
            }}
            onMouseDown={(e) => handlePointerDown(e, 'move')}
            onTouchStart={(e) => handlePointerDown(e, 'move')}
        >
            <span className="absolute top-1 left-1 text-[10px] px-1 rounded bg-brass text-paper lowercase shadow-sm">
                {label}
            </span>
            <div 
                className="absolute bottom-0 right-0 w-3 h-3 bg-brass cursor-se-resize"
                onMouseDown={(e) => handlePointerDown(e, 'resize')}
                onTouchStart={(e) => handlePointerDown(e, 'resize')}
            />
        </div>
    );
}
