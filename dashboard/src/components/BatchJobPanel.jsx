import React, { useState } from 'react';
import { ChevronDown, Loader2, CheckCircle2, AlertCircle, Clock, FileVideo, Youtube, Sparkles } from 'lucide-react';
import ResultCard from './ResultCard';

function StatusBadge({ status }) {
    const map = {
        queued:     'bg-zinc-500/10 border-zinc-500/20 text-zinc-400',
        submitting: 'bg-blue-500/10  border-blue-500/20  text-blue-400',
        processing: 'bg-primary/10   border-primary/20   text-primary',
        complete:   'bg-green-500/10 border-green-500/20 text-green-400',
        error:      'bg-red-500/10   border-red-500/20   text-red-400',
    };
    const icons = {
        queued:     <Clock size={11} />,
        submitting: <Loader2 size={11} className="animate-spin" />,
        processing: <Loader2 size={11} className="animate-spin" />,
        complete:   <CheckCircle2 size={11} />,
        error:      <AlertCircle size={11} />,
    };
    return (
        <span className={`inline-flex items-center gap-1 text-[10px] font-bold px-2 py-0.5 rounded-full border uppercase tracking-wide ${map[status] || map.queued}`}>
            {icons[status]}
            {status}
        </span>
    );
}

export default function BatchJobPanel({ job, uploadPostKey, uploadUserId, geminiApiKey, elevenLabsKey, onDeleteClip }) {
    const [expanded, setExpanded] = useState(false);
    const [logsExpanded, setLogsExpanded] = useState(false);

    const clipCount = job.results?.clips?.length ?? 0;
    const lastLog = job.logs?.length > 0 ? job.logs[job.logs.length - 1] : null;
    const mediaIcon = job.media?.type === 'url'
        ? <Youtube size={14} className="text-red-400 shrink-0" />
        : <FileVideo size={14} className="text-blue-400 shrink-0" />;

    return (
        <div className="bg-surface border border-white/5 rounded-2xl overflow-hidden animate-[fadeIn_0.4s_ease-out]">
            {/* Header row */}
            <div
                className="flex items-center gap-3 p-4 cursor-pointer hover:bg-white/3 transition-colors select-none"
                onClick={() => job.status === 'complete' && setExpanded(e => !e)}
            >
                {mediaIcon}
                <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-white truncate" title={job.media?.name}>
                        {job.media?.name || 'Video'}
                    </p>
                    {lastLog && job.status === 'processing' && (
                        <p className="text-[10px] text-zinc-500 truncate mt-0.5">{lastLog}</p>
                    )}
                    {job.error && (
                        <p className="text-[10px] text-red-400 truncate mt-0.5">{job.error}</p>
                    )}
                </div>

                <StatusBadge status={job.status} />

                {job.status === 'complete' && (
                    <span className="text-[10px] bg-white/10 text-white px-1.5 py-0.5 rounded-full shrink-0">
                        {clipCount} clips
                    </span>
                )}

                {job.status === 'complete' && (
                    <ChevronDown
                        size={16}
                        className={`text-zinc-500 shrink-0 transition-transform duration-200 ${expanded ? 'rotate-180' : ''}`}
                    />
                )}

                {/* Inline progress ring for processing */}
                {job.status === 'processing' && (
                    <div className="w-4 h-4 border-2 border-zinc-800 border-t-primary rounded-full animate-spin shrink-0" />
                )}
            </div>

            {/* Expandable logs strip (visible during processing) */}
            {job.status === 'processing' && job.logs?.length > 0 && (
                <div className="px-4 pb-3">
                    <button
                        onClick={() => setLogsExpanded(e => !e)}
                        className="text-[10px] text-zinc-600 hover:text-zinc-400 flex items-center gap-1 transition-colors"
                    >
                        <ChevronDown size={10} className={`transition-transform ${logsExpanded ? 'rotate-180' : ''}`} />
                        {logsExpanded ? 'Hide logs' : 'Show logs'}
                    </button>
                    {logsExpanded && (
                        <div className="mt-2 bg-black/40 rounded-lg p-2 max-h-28 overflow-y-auto custom-scrollbar font-mono text-[10px] text-zinc-400 space-y-0.5">
                            {job.logs.map((l, i) => <div key={i}>{l}</div>)}
                        </div>
                    )}
                </div>
            )}

            {/* Clips grid */}
            {expanded && job.status === 'complete' && clipCount > 0 && (
                <div className="border-t border-white/5 p-4 animate-[fadeIn_0.25s_ease-out]">
                    <div className="flex items-center gap-2 mb-4">
                        <Sparkles size={14} className="text-yellow-400" />
                        <span className="text-xs font-bold text-white">Generated Clips</span>
                    </div>
                    <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
                        {job.results.clips.map((clip, i) => (
                            <ResultCard
                                key={i}
                                clip={clip}
                                index={i}
                                jobId={job.jobId}
                                uploadPostKey={uploadPostKey}
                                uploadUserId={uploadUserId}
                                geminiApiKey={geminiApiKey}
                                elevenLabsKey={elevenLabsKey}
                                onDelete={(deletedIndex) => onDeleteClip(job.id, deletedIndex)}
                            />
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}
