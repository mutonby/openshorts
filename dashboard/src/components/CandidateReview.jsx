import { useEffect, useState } from 'react';
import { Check, Loader2, Play } from 'lucide-react';
import { getApiUrl } from '../config';
import { apiJson } from '../lib/api';
import HighlightTimeline from './HighlightTimeline';
import { describePoint, hypeSummary, timecode } from '../lib/highlightPoints';

const CRITERIA = [
    ['hook', 'Hook'], ['visual_action', 'Visual action'], ['reaction', 'Reaction'],
    ['clarity', 'Clarity'], ['payoff', 'Payoff'],
];

// The candidate composed as the render will compose it: the same facecam
// detection, sizing and filtergraph (viral_edit.layout_still). Slow the first
// time (the facecam check runs), cached after.
function LayoutStill({ jobId, candidateId }) {
    const [state, setState] = useState('loading');
    return (
        <figure className="mt-4">
            <figcaption className="text-sm text-ink">Vertical layout</figcaption>
            <p className="text-xs text-muted mt-1 mb-2">The facecam, game crop and blur as the clip will be rendered (captions and hook not shown).</p>
            {state === 'loading' && (
                <p className="text-xs text-muted flex items-center gap-1.5"><Loader2 size={12} className="animate-spin" aria-hidden="true" /> Composing…</p>
            )}
            {state === 'error' && <p className="text-xs text-muted">No layout preview for this candidate.</p>}
            <img
                src={getApiUrl(`/api/jobs/${jobId}/candidate-layout/${candidateId}`)}
                alt="The candidate's vertical layout"
                className={`w-40 sm:w-48 rounded-input border border-rule ${state === 'ready' ? '' : 'hidden'}`}
                onLoad={() => setState('ready')}
                onError={() => setState('error')}
            />
        </figure>
    );
}

export default function CandidateReview({ jobId, results, onApprove }) {
    const candidates = results?.candidates || [];
    const offset = results?.source_offset || 0;
    const [preview, setPreview] = useState(null);
    const [point, setPoint] = useState(null);
    const signals = results?.signals;
    const deferred = results?.deferred;
    const source = signals?.source;
    const duration = results?.source_duration;
    const [selected, setSelected] = useState([]);
    const [sourceUrl, setSourceUrl] = useState('');
    const [sourceError, setSourceError] = useState('');
    const [submitError, setSubmitError] = useState('');
    const [working, setWorking] = useState(false);
    const selectedCandidate = candidates.find(candidate => candidate.id === preview) || (!point && candidates[0]);
    const cueRange = !deferred && point && !selectedCandidate
        ? { id: `cue-${point.time}`, start: point.start, end: Math.min(duration, point.end + 8) } : null;
    const current = selectedCandidate || cueRange;
    const directPreview = deferred && current?.preview
        ? getApiUrl(`/api/jobs/${jobId}/candidate-preview/${current.id}`) : '';
    // A judged clip can start after the window its preview file begins at.
    const previewOffset = directPreview && Number.isFinite(current?.preview_offset) ? current.preview_offset : 0;
    const start = directPreview ? previewOffset : current?.start;
    const end = directPreview ? previewOffset + current.end - current.start : current?.end;
    const videoUrl = directPreview || (sourceUrl && current ? `${sourceUrl}#t=${start},${end}` : '');

    const selectCandidate = id => {
        setSourceError('');
        setPoint(null);
        setPreview(id);
    };
    const selectPoint = cue => {
        if (cue.kind === 'candidate') {
            selectCandidate(cue.id);
            return;
        }
        setSourceError('');
        setPoint(cue);
        setPreview(candidates.find(candidate => candidate.start < cue.end && candidate.end > cue.start)?.id || null);
    };

    useEffect(() => {
        if (deferred || (!candidates.length && !signals)) return undefined;
        let cancelled = false;
        apiJson(`/api/source-url/${jobId}`).then(({ url }) => {
            if (!cancelled) setSourceUrl(getApiUrl(url));
        }).catch(() => {
            if (!cancelled) setSourceError('The retained source preview could not be loaded.');
        });
        return () => { cancelled = true; };
    }, [jobId, candidates.length, deferred, signals]);

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
                <p className="mt-2 text-sm text-muted">{results?.engine === 'hype'
                    ? 'Found from chat and streamer hype across the whole recording, then watched by the clip judge. Play them, then choose which to render.'
                    : 'Play the source ranges, then choose which moments to render.'}</p>
                {signals && (
                    <dl className="card p-4 mt-5 grid gap-x-5 gap-y-3 sm:grid-cols-2 text-xs">
                        <div><dt className="text-muted">Source</dt>
                            <dd className="text-ink">{source?.title || 'Uploaded recording'}{source?.channel && ` · ${source.channel}`}</dd></div>
                        <div><dt className="text-muted">Chat replay</dt>
                            <dd className="text-ink">{signals.chat_status === 'available'
                                ? `${signals.chat?.length || 0} detected spikes`
                                : signals.chat_status === 'invalid' ? 'Could not read or align replay' : 'Chat replay unavailable'}</dd></div>
                        <div><dt className="text-muted">Transcription</dt>
                            <dd className="text-ink">{results.transcript_segments ? `${results.transcript_segments.toLocaleString()} segments` : 'No usable speech'}</dd></div>
                        {Number.isFinite(results.cost_analysis?.total_cost) && (
                            <div><dt className="text-muted">Estimated scoring and selection cost</dt>
                                <dd className="text-ink">${results.cost_analysis.total_cost.toFixed(4)}</dd></div>
                        )}
                        {results.performance && (
                            <div><dt className="text-muted">Discovery time</dt>
                                <dd className="text-ink">{timecode(results.performance.wall_seconds)}</dd></div>
                        )}
                    </dl>
                )}
                <HighlightTimeline signals={signals} candidates={candidates} duration={duration}
                    offset={offset} active={point} onPoint={selectPoint} />
                {!candidates.length && (
                    <p className="card p-4 mt-5 text-sm text-ink2">No moments with a clear payoff were found in this range.</p>
                )}
                {(candidates.length > 0 || point) && (
                    <div className="grid gap-4 mt-5 lg:grid-cols-[5fr_6fr] lg:items-start">
                        <div className="card p-4 lg:sticky lg:top-0">
                            <h2 className="text-sm text-ink">Source preview</h2>
                            <p className="text-xs text-muted mt-1 mb-3">{deferred
                                ? 'Source preview. Approved ranges render from the full-quality track.'
                                : 'Original footage for this moment.'}</p>
                            {point && <p className="text-xs text-ink2 mb-3">{timecode(offset + point.time)} · {describePoint(point)}</p>}
                            {deferred && point && !current && /^[A-Za-z0-9_-]{11}$/.test(source?.id || '') && (
                                <a className="btn-quiet text-sm" href={`https://www.youtube.com/watch?v=${source.id}&t=${Math.floor(point.start)}s`}
                                    target="_blank" rel="noreferrer">Open this moment on YouTube</a>
                            )}
                            {deferred && point && !current && /^v\d+$/.test(source?.id || '') && (
                                <a className="btn-quiet text-sm" href={`https://www.twitch.tv/videos/${source.id.slice(1)}?t=${Math.floor(point.start / 3600)}h${Math.floor(point.start / 60) % 60}m${Math.floor(point.start) % 60}s`}
                                    target="_blank" rel="noreferrer">Open this moment on Twitch</a>
                            )}
                            {videoUrl && current && !sourceError && (
                                <video
                                    key={current.id}
                                    className="w-full bg-black aspect-video rounded-input"
                                    aria-label="Source candidate preview"
                                    src={videoUrl}
                                    controls
                                    playsInline
                                    preload="metadata"
                                    onError={() => setSourceError('This browser could not play the retained source.')}
                                    onPlay={(event) => {
                                        const video = event.currentTarget;
                                        if (video.currentTime < start || video.currentTime >= end) {
                                            video.currentTime = start;
                                        }
                                    }}
                                    onTimeUpdate={(event) => {
                                        if (event.currentTarget.currentTime >= end) event.currentTarget.pause();
                                    }}
                                />
                            )}
                            {sourceError && <p className="text-sm text-danger" role="alert">{sourceError}</p>}
                            {results?.layout_preview && selectedCandidate && (
                                <LayoutStill key={selectedCandidate.id} jobId={jobId} candidateId={selectedCandidate.id} />
                            )}
                        </div>
                        <div className="space-y-3">
                            {candidates.map((candidate) => {
                                const analysis = candidate.assessment || {};
                                return (
                                    <article className={`card p-3.5 ${current?.id === candidate.id ? 'border-brass' : ''}`} key={candidate.id}>
                                        <div className="flex flex-wrap justify-between gap-2">
                                            <span className="readout">{timecode(offset + candidate.start)}–{timecode(offset + candidate.end)} · {Math.round(candidate.end - candidate.start)}s</span>
                                            <button
                                                className="btn-quiet text-xs px-2 py-1"
                                                type="button"
                                                onClick={() => selectCandidate(candidate.id)}
                                                aria-pressed={current?.id === candidate.id}
                                            ><Play size={12} aria-hidden="true" /> Preview</button>
                                        </div>
                                        <h3 className="mt-1 text-sm font-medium text-ink">{candidate.title}</h3>
                                        {hypeSummary(candidate).length > 0 && (
                                            <ul className="flex flex-wrap gap-1.5 mt-1.5" aria-label="Why this moment ranked here">
                                                {hypeSummary(candidate).map(reason => (
                                                    <li key={reason} className="text-[11px] px-1.5 py-0.5 rounded-sm border border-rule text-ink2">{reason}</li>
                                                ))}
                                            </ul>
                                        )}
                                        {analysis.reaction && <p className="text-xs text-ink2 mt-1">Reaction: {analysis.reaction}</p>}
                                        {analysis.event && <p className="text-xs text-ink2 mt-1">{analysis.event}</p>}
                                        {analysis.excerpt && <p className="text-xs text-ink2 mt-1">Transcript excerpt “{analysis.excerpt}”</p>}
                                        {analysis.verdict === 'supported' && analysis.payoff && (
                                            <p className="text-xs text-ink2 mt-1">{analysis.payoff}</p>
                                        )}
                                        {analysis.verdict !== 'supported' && (
                                            <p className="text-xs text-warn mt-1">Payoff needs a manual check.</p>
                                        )}
                                        {candidate.signals?.chat && (
                                            <p className="text-xs text-muted mt-1">{describePoint(candidate.signals.chat)}</p>
                                        )}
                                        {candidate.signals?.audio && (
                                            <p className="text-xs text-muted mt-1">{describePoint(candidate.signals.audio)}</p>
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
                            {candidates.length > 0 && <button type="button" className="btn-primary w-full" disabled={!selected.length || working} onClick={submit}>
                                {working ? <Loader2 size={16} className="animate-spin" /> : <Check size={16} />}
                                {selected.length ? `Render ${selected.length} selected` : 'Render selected moments'}
                            </button>}
                        </div>
                    </div>
                )}
            </div>
        </section>
    );
}
