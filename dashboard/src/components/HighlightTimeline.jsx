import { describePoint, timecode } from '../lib/highlightPoints';

const COLORS = { chat: '#59c7b4', voice: '#e58f7a', audio: '#8bb5ef', candidate: '#ddb66b', viewer: '#bb9ce6' };
const NAMES = { chat: 'Chat spike', voice: 'Streamer reaction', audio: 'Audio change', candidate: 'Candidate', viewer: 'Viewer clip' };

// The hype curve (chat and streamer combined, hype_engine.scan) as a strip:
// where the stream heats up, at a glance.
function HypeCurve({ curve, duration }) {
    const values = Array.isArray(curve?.values) ? curve.values.filter(Number.isFinite) : [];
    const step = Number(curve?.step) || 5;
    if (values.length < 2) return null;
    const top = Math.max(0.3, ...values);
    const path = values.map((v, i) => `${(100 * i * step / duration).toFixed(2)},${(24 - 22 * v / top).toFixed(2)}`);
    return (
        <div className="grid grid-cols-[5.75rem_1fr] items-center gap-2">
            <span className="text-xs text-muted">Hype</span>
            <svg viewBox="0 0 100 24" preserveAspectRatio="none" className="h-8 w-full" role="img"
                aria-label="Chat and streamer hype over the recording">
                <polygon points={`0,24 ${path.join(' ')} ${Math.min(100, 100 * values.length * step / duration).toFixed(2)},24`}
                    fill="#e58f7a" fillOpacity="0.25" stroke="none" />
                <polyline points={path.join(' ')} fill="none" stroke="#e58f7a" strokeWidth="0.6"
                    vectorEffect="non-scaling-stroke" />
            </svg>
        </div>
    );
}

export default function HighlightTimeline({ signals, candidates, duration, offset, active, onPoint }) {
    if (!signals || !Number.isFinite(duration) || duration <= 0) return null;
    const points = [
        ...(signals.chat || []), ...(signals.voice || []), ...(signals.audio || []), ...(signals.viewer || []),
        ...candidates.map(c => ({ kind: 'candidate', time: c.start, id: c.id, title: c.title })),
    ].filter(point => Number.isFinite(point.time) && point.time >= 0 && point.time <= duration);
    const cues = points.filter(p => p.kind !== 'candidate').sort((a, b) => a.time - b.time);
    const lanes = Object.keys(COLORS).filter(kind => points.some(p => p.kind === kind));
    const id = point => `${point.kind}-${point.id ?? point.time}`;

    return (
        <section className="card p-4 space-y-3 mt-5" aria-label="Highlight timeline">
            <div className="flex justify-between gap-3 items-start">
                <h2 className="text-sm font-medium text-ink">Stream timeline</h2>
                <span className="text-xs text-muted">{timecode(offset + duration)}</span>
            </div>
            <HypeCurve curve={signals.hype_curve} duration={duration} />
            {lanes.map(kind => (
                <div key={kind} className="grid grid-cols-[5.75rem_1fr] items-center gap-2">
                    <span className="text-xs text-muted">{NAMES[kind]}</span>
                    <div className="h-5 relative border-b border-rule">
                        {points.filter(point => point.kind === kind).map(point => (
                            <button key={id(point)} type="button"
                                className="absolute top-0 h-5 w-3 -translate-x-1/2 flex items-center justify-center rounded-sm focus-visible:outline"
                                style={{ left: `${Math.max(0.8, Math.min(99.2, 100 * point.time / duration))}%` }}
                                aria-label={`${NAMES[kind]} at ${timecode(offset + point.time)}`}
                                aria-pressed={active && id(active) === id(point)}
                                title={`${timecode(offset + point.time)} · ${describePoint(point)}`}
                                onClick={() => onPoint(point)}>
                                <span className="h-2 w-2 rounded-full" style={{ background: COLORS[kind] }} />
                            </button>
                        ))}
                    </div>
                </div>
            ))}
            {cues.length > 0 && (
                <label className="block text-xs text-muted">
                    Jump to evidence
                    <select className="input-field block w-full mt-1"
                        value={active && active.kind !== 'candidate' ? id(active) : ''}
                        onChange={event => {
                            const point = cues.find(p => id(p) === event.target.value);
                            if (point) onPoint(point);
                        }}>
                        <option value="">Choose a timestamp</option>
                        {cues.map(point => (
                            <option key={id(point)} value={id(point)}>
                                {timecode(offset + point.time)} · {NAMES[point.kind]} · {describePoint(point)}
                            </option>
                        ))}
                    </select>
                </label>
            )}
        </section>
    );
}
