import { useEffect, useState } from 'react';
import { Check, Loader2, Play } from 'lucide-react';
import { getApiUrl } from '../config';
import { apiJson } from '../lib/api';

const CRITERIA = [
    ['hook', 'Hook'], ['visual_action', 'Visual action'], ['reaction', 'Reaction'],
    ['clarity', 'Clarity'], ['payoff', 'Payoff'],
];

function timecode(seconds) {
    const whole = Math.max(0, Math.floor(seconds));
    const time = [Math.floor(whole / 60) % 60, whole % 60]
        .map(value => String(value).padStart(2, '0')).join(':');
    return whole >= 3600 ? `${Math.floor(whole / 3600)}:${time}` : time;
}

export default function CandidateReview({ jobId, results, onApprove }) {
    const candidates = results?.candidates || [];
    const offset = results?.source_offset || 0;
    const [preview, setPreview] = useState(null);
    const [selected, setSelected] = useState([]);
    const [sourceUrl, setSourceUrl] = useState('');
    const [sourceError, setSourceError] = useState('');
    const [submitError, setSubmitError] = useState('');
    const [working, setWorking] = useState(false);
    const current = candidates.find(candidate => candidate.id === preview) || candidates[0];

    useEffect(() => {
        if (!candidates.length) return undefined;
        let cancelled = false;
        apiJson(`/api/source-url/${jobId}`).then(({ url }) => {
            if (!cancelled) setSourceUrl(getApiUrl(url));
        }).catch(() => {
            if (!cancelled) setSourceError('The retained source preview could not be loaded.');
        });
        return () => { cancelled = true; };
    }, [jobId, candidates.length]);

    const submit = async () => {
        setWorking(true);
        setSubmitError('');
        try {
            await onApprove(selected);
        } catch (error) {
            setSubmitError(error.message || 'Could not queue these clips.');
        } finally {
            setWorking(false);
        }
    };

    return (
        <section className="h-full overflow-y-auto custom-scrollbar px-4 py-6 sm:px-8">
            <div className="max-w-5xl mx-auto">
                <h1 className="font-display lowercase text-3xl text-ink">Review candidate moments</h1>
                <p className="mt-2 text-sm text-muted">Play the source ranges, then choose which moments to render.</p>
                {!candidates.length ? (
                    <p className="card p-4 mt-5 text-sm text-ink2">No moments with a clear payoff were found in this range.</p>
                ) : (
                    <div className="grid gap-4 mt-5 lg:grid-cols-[5fr_6fr] lg:items-start">
                        <div className="card p-4 lg:sticky lg:top-0">
                            <h2 className="text-sm text-ink">Source preview</h2>
                            <p className="text-xs text-muted mt-1 mb-3">Original footage for this moment. Final clips appear after rendering.</p>
                            {sourceUrl && current && !sourceError && (
                                <video
                                    key={current.id}
                                    className="w-full bg-black aspect-video rounded-input"
                                    aria-label="Source candidate preview"
                                    src={`${sourceUrl}#t=${current.start},${current.end}`}
                                    controls
                                    playsInline
                                    preload="metadata"
                                    onError={() => setSourceError('This browser could not play the retained source.')}
                                    onPlay={(event) => {
                                        const video = event.currentTarget;
                                        if (video.currentTime < current.start || video.currentTime >= current.end) {
                                            video.currentTime = current.start;
                                        }
                                    }}
                                    onTimeUpdate={(event) => {
                                        if (event.currentTarget.currentTime >= current.end) event.currentTarget.pause();
                                    }}
                                />
                            )}
                            {sourceError && <p className="text-sm text-danger" role="alert">{sourceError}</p>}
                        </div>
                        <div className="space-y-3">
                            {candidates.map((candidate) => {
                                const analysis = candidate.assessment || {};
                                return (
                                    <article className={`card p-3.5 ${current?.id === candidate.id ? 'border-brass' : ''}`} key={candidate.id}>
                                        <div className="flex flex-wrap justify-between gap-2">
                                            <span className="readout">{timecode(offset + candidate.start)}–{timecode(offset + candidate.end)}</span>
                                            <button
                                                className="btn-quiet text-xs px-2 py-1"
                                                type="button"
                                                onClick={() => setPreview(candidate.id)}
                                                aria-pressed={current?.id === candidate.id}
                                            ><Play size={12} aria-hidden="true" /> Preview</button>
                                        </div>
                                        <h3 className="mt-1 text-sm font-medium text-ink">{candidate.title}</h3>
                                        {analysis.event && <p className="text-xs text-ink2 mt-1">{analysis.event}</p>}
                                        {analysis.excerpt && <p className="text-xs text-ink2 mt-1">Transcript excerpt “{analysis.excerpt}”</p>}
                                        {analysis.verdict === 'supported' && analysis.payoff && (
                                            <p className="text-xs text-ink2 mt-1">{analysis.payoff}</p>
                                        )}
                                        {analysis.verdict !== 'supported' && (
                                            <p className="text-xs text-warn mt-1">Payoff needs a manual check.</p>
                                        )}
                                        {analysis.ratings && Object.keys(analysis.ratings).length > 0 && (
                                            <dl className="flex flex-wrap gap-x-3 gap-y-1 mt-2 text-xs text-muted">
                                                {CRITERIA.map(([key, label]) => (
                                                    <div className="flex gap-1" key={key}>
                                                        <dt>{label}</dt><dd className="text-ink2">{analysis.ratings[key]}</dd>
                                                    </div>
                                                ))}
                                            </dl>
                                        )}
                                        <label className="flex items-center gap-2 text-xs text-ink mt-3">
                                            <input
                                                type="checkbox"
                                                className="w-4 h-4 accent-brass"
                                                checked={selected.includes(candidate.id)}
                                                onChange={(event) => {
                                                    const add = event.target.checked;
                                                    setSelected((previous) => add
                                                        ? [...previous, candidate.id]
                                                        : previous.filter(id => id !== candidate.id));
                                                }}
                                            />
                                            Render this moment
                                        </label>
                                    </article>
                                );
                            })}
                            {submitError && <p className="text-sm text-danger" role="alert">{submitError}</p>}
                            <button type="button" className="btn-primary w-full" disabled={!selected.length || working} onClick={submit}>
                                {working ? <Loader2 size={16} className="animate-spin" /> : <Check size={16} />}
                                {selected.length ? `Render ${selected.length} selected` : 'Render selected moments'}
                            </button>
                        </div>
                    </div>
                )}
            </div>
        </section>
    );
}
