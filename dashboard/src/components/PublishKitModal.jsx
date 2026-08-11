import React, { useState, useEffect, useCallback } from 'react';
import { Megaphone, Loader2, Copy, Check, Download, RefreshCw, X } from 'lucide-react';
import { apiFetch } from '../lib/api';
import Modal from './ui/Modal';

export default function PublishKitModal({ isOpen, onClose, jobId, clipIndex, clip }) {
    const [kit, setKit] = useState(null);
    const [loading, setLoading] = useState(false);
    const [region, setRegion] = useState('US');
    const [regions, setRegions] = useState({ US: 'United States' });
    const [title, setTitle] = useState('');
    const [description, setDescription] = useState('');
    const [selectedTags, setSelectedTags] = useState(new Set());
    const [error, setError] = useState('');
    const [copied, setCopied] = useState('');

    useEffect(() => {
        apiFetch('/api/publish-kit/regions')
            .then((r) => (r.ok ? r.json() : null))
            .then((d) => { if (d?.regions) setRegions(d.regions); })
            .catch(() => {});
    }, []);

    const generate = useCallback(async (useRegion) => {
        if (!jobId || clipIndex === undefined) return;
        setLoading(true);
        setError('');
        try {
            const res = await apiFetch('/api/publish-kit', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ job_id: jobId, clip_index: clipIndex, language: 'auto', region: useRegion }),
            });
            const data = await res.json().catch(() => ({}));
            if (!res.ok) throw new Error(data.detail || 'publish kit failed');
            const k = data.kit;
            setKit(k);
            setTitle(k.title || '');
            setDescription(k.description || '');
            setSelectedTags(new Set((k.hashtags || []).map((h) => h.tag)));
        } catch (e) {
            setError(String(e.message || e));
        } finally {
            setLoading(false);
        }
    }, [jobId, clipIndex]);

    useEffect(() => {
        if (isOpen && jobId && clipIndex !== undefined) {
            setKit(null); setError(''); generate(region);
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [isOpen, jobId, clipIndex]);

    const toggleTag = (tag) => {
        setSelectedTags((prev) => {
            const next = new Set(prev);
            if (next.has(tag)) next.delete(tag);
            else next.add(tag);
            return next;
        });
    };

    const finalCopy = () => {
        const tags = (kit?.hashtags || [])
            .filter((h) => selectedTags.has(h.tag))
            .map((h) => h.tag);
        const parts = [title];
        if (description) parts.push('', description);
        if (tags.length) parts.push('', tags.join(' '));
        return parts.join('\n');
    };

    const copy = async (what) => {
        const text = what === 'tags'
            ? (kit?.hashtags || []).filter((h) => selectedTags.has(h.tag)).map((h) => h.tag).join(' ')
            : finalCopy();
        try {
            await navigator.clipboard.writeText(text);
            setCopied(what);
            setTimeout(() => setCopied(''), 2000);
        } catch (_) { /* clipboard unavailable */ }
    };

    const download = () => {
        const blob = new Blob([finalCopy()], { type: 'text/plain' });
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = `publish_kit_${(title || 'clip').toLowerCase().replace(/[^a-z0-9]+/g, '_').slice(0, 30)}.txt`;
        a.click();
        URL.revokeObjectURL(a.href);
    };

    const toggleAll = () => {
        const all = (kit?.hashtags || []).map((h) => h.tag);
        setSelectedTags(selectedTags.size === all.length && all.length ? new Set() : new Set(all));
    };

    return (
        <Modal isOpen={isOpen} onClose={onClose} size="xl" eyebrow="PUBLISH · MANUAL" title="publish kit">
            <div className="flex flex-col gap-4">
                <div className="flex flex-wrap items-center justify-between gap-3">
                    <p className="text-xs text-muted max-w-2xl leading-relaxed">
                        Viral title, description and <strong>today's trending hashtags</strong> for this clip —
                        refreshed automatically every day per region. Nothing is posted: review the clip, then
                        copy this kit into YouTube / TikTok / Instagram yourself.
                    </p>
                    <div className="flex items-center gap-2">
                        <label className="text-xs text-muted">Region</label>
                        <select value={region} onChange={(e) => { setRegion(e.target.value); generate(e.target.value); }} className="input-field text-sm px-2.5 py-1.5">
                            {Object.entries(regions).map(([code, name]) => (
                                <option key={code} value={code}>{name}</option>
                            ))}
                        </select>
                        <button onClick={() => generate(region)} className="btn-ghost text-xs flex items-center gap-1.5" title="Regenerate (also refreshes today's trends if stale)">
                            <RefreshCw size={13} /> refresh
                        </button>
                    </div>
                </div>

                {loading && (
                    <div className="flex items-center gap-2 text-muted text-sm py-10 justify-center">
                        <Loader2 size={18} className="animate-spin" /> Writing your publish kit…
                    </div>
                )}
                {error && !loading && (
                    <div className="text-sm text-warn py-6 text-center">
                        {error}
                        <div className="mt-2"><button onClick={() => generate(region)} className="btn-quiet text-xs">retry</button></div>
                    </div>
                )}

                {kit && !loading && (
                    <>
                        {/* Title */}
                        <div>
                            <label className="eyebrow block mb-1.5">TITLE (≤100 chars)</label>
                            <input
                                type="text"
                                value={title}
                                onChange={(e) => setTitle(e.target.value)}
                                maxLength={100}
                                className="input-field"
                            />
                            <div className="text-right readout mt-0.5">{title.length}/100</div>
                        </div>

                        {/* Description */}
                        <div>
                            <label className="eyebrow block mb-1.5">DESCRIPTION</label>
                            <textarea
                                value={description}
                                onChange={(e) => setDescription(e.target.value)}
                                rows={5}
                                className="input-field resize-none leading-relaxed"
                            />
                        </div>

                        {/* Hashtags */}
                        <div>
                            <div className="flex items-center justify-between mb-1.5">
                                <p className="eyebrow">
                                    HASHTAGS <span className="readout">({selectedTags.size}/{kit.hashtags?.length || 0} selected)</span>
                                </p>
                                <button onClick={toggleAll} className="text-xs text-brass underline underline-offset-2">
                                    {selectedTags.size === (kit.hashtags?.length || 0) && kit.hashtags?.length ? 'select none' : 'select all'}
                                </button>
                            </div>
                            <div className="flex flex-wrap gap-1.5">
                                {(kit.hashtags || []).map((h) => {
                                    const on = selectedTags.has(h.tag);
                                    return (
                                        <button
                                            key={h.tag}
                                            onClick={() => toggleTag(h.tag)}
                                            title={h.why || h.source}
                                            className={`px-2.5 py-1.5 rounded-full border text-xs transition-colors flex items-center gap-1.5
                                                ${on ? 'border-[color:var(--color-accent)] text-ink bg-paper3' : 'border-rule2 text-muted hover:border-[color:var(--color-accent)]'}`}
                                        >
                                            {h.source === 'trending' && <span className="w-1.5 h-1.5 rounded-full bg-danger shrink-0" title="trending today" />}
                                            {h.tag}
                                        </button>
                                    );
                                })}
                            </div>
                            {kit.trending_topics?.length > 0 && (
                                <p className="text-[11px] text-muted mt-2">
                                    🔥 Today's trends in {regions[kit.region] || kit.region}
                                    {kit.trend_source === 'trends24' ? ' (live trends)' : kit.trend_source === 'ai' ? ' (AI) ' : ''}:{' '}
                                    {kit.trending_topics.slice(0, 8).join(' · ')}
                                </p>
                            )}
                        </div>

                        {/* Actions */}
                        <div className="flex flex-wrap gap-2 pt-2 border-t border-rule">
                            <button onClick={() => copy('all')} className="btn-primary">
                                {copied === 'all' ? <Check size={16} /> : <Copy size={16} />}
                                {copied === 'all' ? 'copied!' : 'copy title + description + hashtags'}
                            </button>
                            <button onClick={() => copy('tags')} className="btn-ghost">
                                {copied === 'tags' ? <Check size={16} /> : <Copy size={16} />}
                                {copied === 'tags' ? 'hashtags copied' : 'copy hashtags'}
                            </button>
                            <button onClick={download} className="btn-quiet">
                                <Download size={16} /> download .txt
                            </button>
                        </div>
                        <p className="text-[11px] text-muted flex items-center gap-1.5">
                            <Megaphone size={12} /> Tip: play the clip once more, then paste into YouTube's upload page — hashtags go at the end of the description.
                        </p>
                    </>
                )}
            </div>
        </Modal>
    );
}
