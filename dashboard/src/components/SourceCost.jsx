import React, { useEffect, useState } from 'react';
import { Loader2, Scissors } from 'lucide-react';
import { apiJson } from '../lib/api';
import { track } from '../lib/analytics';
import { useAuth } from '../contexts/AuthContext';

// What the source will cost, shown before Generate: the one complaint about
// price in the cancel flow was a user who could not tell why one video took 20
// minutes and the next 57 for the same six clips (billing is per minute of
// source, not per clip). Mirrors the server: whole minutes rounded down, at
// least 1, and a picked stretch is billed alone.

// Shortest stretch the server accepts (app.MIN_RANGE_SECONDS).
const MIN_RANGE_SECONDS = 60;
const STEP_SECONDS = 5;

const billable = (seconds) => Math.max(1, Math.floor(seconds / 60 + 1e-9));

const clock = (seconds) => {
    const s = Math.max(0, Math.round(seconds));
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const ss = String(s % 60).padStart(2, '0');
    return h ? `${h}:${String(m).padStart(2, '0')}:${ss}` : `${m}:${ss}`;
};

const looksLikeUrl = (u) => /^https?:\/\/\S+\.\S+/.test((u || '').trim());

export default function SourceCost({ mode, url, file, fileSeconds, range, onRangeChange, localRanges }) {
    const { isManaged } = useAuth();
    const [estimate, setEstimate] = useState(null);
    const [loading, setLoading] = useState(false);
    const [picking, setPicking] = useState(false);

    // A new source starts from the whole video again.
    useEffect(() => {
        setEstimate(null);
        setLoading(false);
        setPicking(false);
        onRangeChange(null);
        if (!isManaged) return undefined;
        let body = null;
        if (mode === 'url' && looksLikeUrl(url)) body = { url: url.trim() };
        else if (mode === 'file' && fileSeconds > 0) body = { seconds: fileSeconds };
        if (!body) return undefined;
        let cancelled = false;
        // Debounced: typing a link must not probe every keystroke.
        const t = setTimeout(() => {
            setLoading(true);
            apiJson('/api/process/estimate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            })
                .then((d) => {
                    if (cancelled) return;
                    if (d?.outcome) {
                        setEstimate(d);
                        track('CostLineSeen', { props: { outcome: d.outcome, minutes: String(d.cost_minutes) } });
                    }
                })
                .catch(() => { /* no line: the submit still says what it costs */ })
                .finally(() => { if (!cancelled) setLoading(false); });
        }, mode === 'url' ? 700 : 0);
        return () => { cancelled = true; clearTimeout(t); };
        // onRangeChange is a setState from the parent: stable.
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [isManaged, mode, url, fileSeconds]);

    if (!isManaged) {
        const duration = mode === 'file' ? fileSeconds : null;
        if (!localRanges || (mode === 'url' ? !looksLikeUrl(url) : !file || (duration && duration <= MIN_RANGE_SECONDS))) {
            return null;
        }
        const from = Number(range?.start || 0);
        const to = Number(range?.end || 0);
        return (
            <fieldset className="mt-3 rounded-input border border-rule px-3 py-2.5">
                <label className="flex items-center gap-2 text-xs text-ink2">
                    <input type="checkbox" className="accent-brass" checked={picking}
                        onChange={(event) => {
                            const open = event.target.checked;
                            setPicking(open);
                            onRangeChange(open ? { start: '0', end: duration ? String(Math.floor(duration)) : '' } : null);
                        }} />
                    Analyze a source section
                    {duration > 0 && <span className="readout ml-auto">{clock(duration)} source</span>}
                </label>
                {picking && range && (
                    <>
                        <div className="mt-3 grid grid-cols-2 gap-2">
                            <label>
                                <span className="eyebrow block mb-1">Start (s)</span>
                                <input type="number" className="input-field" required min="0" step="1"
                                    max={duration ? Math.floor(duration) - MIN_RANGE_SECONDS : undefined}
                                    value={range.start}
                                    onChange={(event) => onRangeChange({ ...range, start: event.target.value })} />
                            </label>
                            <label>
                                <span className="eyebrow block mb-1">End (s)</span>
                                <input type="number" className="input-field" required min={from + MIN_RANGE_SECONDS} step="1"
                                    max={duration ? Math.floor(duration) : undefined}
                                    value={range.end}
                                    onChange={(event) => onRangeChange({ ...range, end: event.target.value })} />
                            </label>
                        </div>
                        <p className="text-xs text-muted mt-2">
                            {to - from >= MIN_RANGE_SECONDS
                                ? `Analyze ${clock(to - from)} of source footage.`
                                : `Choose a section of at least ${MIN_RANGE_SECONDS}s.`}
                        </p>
                    </>
                )}
            </fieldset>
        );
    }
    if (loading && !estimate) {
        return (
            <p className="mt-3 flex items-center gap-2 text-xs text-muted">
                <Loader2 size={12} className="animate-spin" /> checking the video length…
            </p>
        );
    }
    if (!estimate) return null;

    const total = estimate.source_minutes * 60;
    const remaining = estimate.remaining;
    const start = range ? range.start : 0;
    const end = range ? range.end : total;
    const rangeCost = billable(end - start);
    const short = rangeCost > remaining;

    const setStart = (v) => onRangeChange({ start: Math.min(v, end - MIN_RANGE_SECONDS), end });
    const setEnd = (v) => onRangeChange({ start, end: Math.max(v, start + MIN_RANGE_SECONDS) });

    let summary;
    if (range) {
        summary = (
            <span className={short ? 'text-warn' : 'text-ink2'}>
                {clock(start)}–{clock(end)} of {clock(total)} · uses {rangeCost} of your {remaining} min
                {short && ' (not enough left)'}
            </span>
        );
    } else if (estimate.outcome === 'whole') {
        summary = <span className="text-ink2">{clock(total)} video · uses {estimate.cost_minutes} of your {remaining} min</span>;
    } else if (estimate.outcome === 'first_video_free') {
        summary = <span className="text-ink2">{clock(total)} video · your first video is clipped whole, on us</span>;
    } else if (estimate.outcome === 'first_part') {
        summary = (
            <span className="text-ink2">
                {clock(total)} video · longer than your {remaining} min left, so we clip the first {estimate.processed_minutes} min
            </span>
        );
    } else {
        summary = (
            <span className="text-warn">
                {clock(total)} video · needs {estimate.cost_minutes} min, you have {remaining}
            </span>
        );
    }

    const canPick = total >= MIN_RANGE_SECONDS * 2;

    return (
        <div className="mt-3 text-xs leading-relaxed">
            <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
                {summary}
                {canPick && (
                    <button
                        type="button"
                        onClick={() => {
                            if (picking || range) { setPicking(false); onRangeChange(null); return; }
                            track('RangePickerOpened', { props: { minutes: String(estimate.cost_minutes) } });
                            setPicking(true);
                            onRangeChange({ start: 0, end: total });
                        }}
                        className="flex items-center gap-1 text-brass hover:underline lowercase"
                    >
                        <Scissors size={12} />
                        {picking || range ? 'clip the whole video' : 'clip only part of it'}
                    </button>
                )}
            </div>
            <p className="text-muted mt-0.5">Billed per minute of video you import. Trims, reframes, hooks and captions on the clips are free.</p>
            {(picking || range) && range && (
                <div className="mt-3 space-y-2 animate-fade">
                    <label className="block">
                        <span className="flex justify-between"><span className="eyebrow">from</span><span className="readout">{clock(start)}</span></span>
                        <input
                            type="range" min="0" max={Math.max(0, total - MIN_RANGE_SECONDS)} step={STEP_SECONDS}
                            value={start}
                            onChange={(e) => setStart(Number(e.target.value))}
                            className="w-full accent-[var(--color-accent)]"
                        />
                    </label>
                    <label className="block">
                        <span className="flex justify-between"><span className="eyebrow">to</span><span className="readout">{clock(end)}</span></span>
                        <input
                            type="range" min={MIN_RANGE_SECONDS} max={total} step={STEP_SECONDS}
                            value={end}
                            onChange={(e) => setEnd(Number(e.target.value))}
                            className="w-full accent-[var(--color-accent)]"
                        />
                    </label>
                </div>
            )}
        </div>
    );
}
