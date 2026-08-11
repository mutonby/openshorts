import React, { useState, useEffect, useCallback } from 'react';
import { FileText, Loader2, Copy, Check, Download, X } from 'lucide-react';
import { apiFetch } from '../lib/api';
import Modal from './ui/Modal';

function fmtTime(seconds) {
    const s = Math.max(0, Math.floor(Number(seconds) || 0));
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const sec = s % 60;
    const mm = String(m).padStart(2, '0');
    const ss = String(sec).padStart(2, '0');
    return h > 0 ? `${h}:${mm}:${ss}` : `${m}:${ss}`;
}

function toMarkdown(summary) {
    const lines = [];
    lines.push(`# ${summary.title || 'Video summary'}`);
    lines.push('');
    if (summary.overview) lines.push(summary.overview);
    lines.push('');
    if (Array.isArray(summary.chapters)) {
        lines.push('## Chapters');
        lines.push('');
        summary.chapters.forEach((c) => {
            lines.push(`**${fmtTime(c.start)} – ${fmtTime(c.end)} — ${c.title || ''}**`);
            (c.points || []).forEach((p) => lines.push(`- ${p}`));
            lines.push('');
        });
    }
    if (Array.isArray(summary.key_takeaways) && summary.key_takeaways.length) {
        lines.push('## Key takeaways');
        lines.push('');
        summary.key_takeaways.forEach((k) => lines.push(`- ${k}`));
        lines.push('');
    }
    if (Array.isArray(summary.quotes) && summary.quotes.length) {
        lines.push('## Best quotes');
        lines.push('');
        summary.quotes.forEach((q) => lines.push(`> ${q.quote} — [${fmtTime(q.time)}]`));
        lines.push('');
    }
    if (Array.isArray(summary.hooks) && summary.hooks.length) {
        lines.push('## Clip hooks');
        lines.push('');
        summary.hooks.forEach((h) => lines.push(`- ${h}`));
    }
    return lines.join('\n');
}

export default function SummaryModal({ isOpen, onClose, jobId }) {
    const [summary, setSummary] = useState(null);
    const [generating, setGenerating] = useState(false);
    const [error, setError] = useState('');
    const [copied, setCopied] = useState(false);
    const [markdown, setMarkdown] = useState('');

    const generate = useCallback(async () => {
        if (!jobId) return;
        setGenerating(true);
        setError('');
        try {
            const res = await apiFetch('/api/summary', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ job_id: jobId, language: 'auto' }),
            });
            if (!res.ok) {
                const data = await res.json().catch(() => ({}));
                throw new Error(data.detail || 'summary failed');
            }
            const data = await res.json();
            setSummary(data.summary);
            setMarkdown(toMarkdown(data.summary));
        } catch (e) {
            setError(String(e.message || e));
        } finally {
            setGenerating(false);
        }
    }, [jobId]);

    useEffect(() => {
        if (isOpen && jobId) { setSummary(null); setError(''); generate(); }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [isOpen, jobId]);

    const copy = async () => {
        try {
            await navigator.clipboard.writeText(markdown);
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        } catch (_) { /* clipboard unavailable */ }
    };

    const download = () => {
        const blob = new Blob([markdown], { type: 'text/markdown' });
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = `summary_${jobId.slice(0, 8)}.md`;
        a.click();
        URL.revokeObjectURL(a.href);
    };

    return (
        <Modal isOpen={isOpen} onClose={onClose} size="xl" eyebrow="REPURPOSE · TEXT" title="video summary">
            <div className="flex flex-col gap-4">
                <div className="flex flex-wrap items-center justify-between gap-3">
                    <p className="text-xs text-muted max-w-xl leading-relaxed">
                        Turns this video's transcript into a chaptered written digest — timestamps,
                        key points, best quotes and clip hooks. Built with the free AI providers you configured.
                        Great for show notes, newsletters, LinkedIn threads or blog posts.
                    </p>
                    {summary && (
                        <div className="flex gap-2">
                            <button onClick={copy} className="btn-ghost text-xs flex items-center gap-1.5">
                                {copied ? <Check size={13} /> : <Copy size={13} />} {copied ? 'copied' : 'copy'}
                            </button>
                            <button onClick={download} className="btn-quiet text-xs flex items-center gap-1.5">
                                <Download size={13} /> .md
                            </button>
                        </div>
                    )}
                </div>

                {generating && (
                    <div className="flex items-center gap-2 text-muted text-sm py-10 justify-center">
                        <Loader2 size={18} className="animate-spin" /> Summarizing the transcript…
                    </div>
                )}
                {error && !generating && (
                    <div className="text-sm text-warn py-6 text-center">
                        {error}
                        <div className="mt-2">
                            <button onClick={generate} className="btn-quiet text-xs">retry</button>
                        </div>
                    </div>
                )}
                {summary && !generating && (
                    <div className="max-h-[55vh] overflow-y-auto custom-scrollbar space-y-4 pr-1">
                        <h2 className="font-display text-xl text-ink">{summary.title}</h2>
                        {summary.overview && <p className="text-sm text-ink2 leading-relaxed">{summary.overview}</p>}

                        {Array.isArray(summary.chapters) && summary.chapters.length > 0 && (
                            <div className="space-y-3">
                                <p className="eyebrow">Chapters</p>
                                {summary.chapters.map((c, i) => (
                                    <div key={i} className="bg-paper3 rounded-input p-3">
                                        <div className="flex items-center gap-2 mb-1">
                                            <span className="readout">{fmtTime(c.start)} – {fmtTime(c.end)}</span>
                                            <span className="text-sm font-medium text-ink">{c.title}</span>
                                        </div>
                                        {(c.points || []).length > 0 && (
                                            <ul className="list-disc list-inside space-y-0.5 text-xs text-muted">
                                                {(c.points || []).map((p, j) => <li key={j}>{p}</li>)}
                                            </ul>
                                        )}
                                    </div>
                                ))}
                            </div>
                        )}

                        {Array.isArray(summary.key_takeaways) && summary.key_takeaways.length > 0 && (
                            <div>
                                <p className="eyebrow mb-2">Key takeaways</p>
                                <ul className="list-disc list-inside space-y-1 text-sm text-ink2">
                                    {summary.key_takeaways.map((k, i) => <li key={i}>{k}</li>)}
                                </ul>
                            </div>
                        )}

                        {Array.isArray(summary.quotes) && summary.quotes.length > 0 && (
                            <div>
                                <p className="eyebrow mb-2">Best quotes</p>
                                <div className="space-y-2">
                                    {summary.quotes.map((q, i) => (
                                        <blockquote key={i} className="border-l-2 border-brass pl-3 text-sm text-ink2">
                                            “{q.quote}” <span className="readout">[{fmtTime(q.time)}]</span>
                                        </blockquote>
                                    ))}
                                </div>
                            </div>
                        )}

                        {Array.isArray(summary.hooks) && summary.hooks.length > 0 && (
                            <div>
                                <p className="eyebrow mb-2">Clip hooks</p>
                                <ul className="space-y-1 text-sm text-ink2">
                                    {summary.hooks.map((h, i) => <li key={i}>🔥 {h}</li>)}
                                </ul>
                            </div>
                        )}
                    </div>
                )}
            </div>
        </Modal>
    );
}
