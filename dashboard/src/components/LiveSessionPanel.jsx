import React, { useState } from 'react';
import { Pause, Play, Square, Loader2, FolderOpen, AlertTriangle, Film, Terminal } from 'lucide-react';
import { getApiUrl } from '../config';
import { LIVE_ACTIVE_STATES, fmtWhen } from '../lib/live';

const STATE_TEXT = {
  scheduled: 'scheduled',
  waiting: 'waiting for the stream',
  recording: 'recording',
  paused: 'paused',
  ended: 'stream ended · clipping the rest',
  finished: 'finished',
  stopped: 'stopped',
  error: 'error',
};

const STATUS_TEXT = {
  recorded: 'waiting to be clipped',
  submitting: 'starting',
  clipping: 'clipping…',
  done: 'clipped',
  failed: 'failed',
  skipped: 'skipped',
  kept: 'kept, not clipped',
};

const fmtDuration = (secs) => {
  const s = Math.max(0, Math.round(secs || 0));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return h ? `${h}h ${String(m).padStart(2, '0')}m` : `${m}m ${String(s % 60).padStart(2, '0')}s`;
};

export default function LiveSessionPanel({ session: s, busy, onAct }) {
  const [selected, setSelected] = useState(null);
  const chunks = (s.chunks || []).slice().reverse();
  const playable = chunks.filter((c) => c.available);
  const current = playable.find((c) => c.index === selected) || playable[0] || null;
  const active = LIVE_ACTIVE_STATES.includes(s.state);

  return (
    <div className="flex flex-col gap-4 min-h-0">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <p className={`eyebrow ${s.state === 'recording' ? 'text-ok' : s.state === 'error' ? 'text-danger' : s.state === 'paused' ? 'text-warn' : ''}`}>
            {s.state === 'recording' && <span className="inline-block w-2 h-2 rounded-full bg-danger animate-pulse mr-1.5 align-middle" />}
            {STATE_TEXT[s.state] || s.state}
            {s.state === 'scheduled' && s.startAt ? ` · starts ${fmtWhen(s.startAt * 1000)}` : ''}
            {` · ${fmtDuration(s.recordedSeconds)} recorded`}
          </p>
          <p className="text-ink text-sm truncate">{s.title || s.url}</p>
          <p className="text-muted text-[11px] flex items-center gap-1.5 mt-0.5 break-all"><FolderOpen size={11} className="shrink-0" /> {s.dir}</p>
        </div>
        {active && (
          <div className="flex gap-2">
            {s.state === 'recording' && (
              <button className="btn-ghost px-3 py-1.5 text-xs" onClick={() => onAct('pause')} disabled={!!busy}>
                {busy === 'pause' ? <Loader2 size={13} className="animate-spin" /> : <Pause size={13} />} pause
              </button>
            )}
            {s.state === 'paused' && (
              <button className="btn-primary px-3 py-1.5 text-xs" onClick={() => onAct('resume')} disabled={!!busy}>
                {busy === 'resume' ? <Loader2 size={13} className="animate-spin" /> : <Play size={13} />} resume
              </button>
            )}
            <button className="btn-ghost px-3 py-1.5 text-xs" onClick={() => onAct('stop')} disabled={!!busy}>
              {busy === 'stop' ? <Loader2 size={13} className="animate-spin" /> : <Square size={13} />} stop
            </button>
          </div>
        )}
      </div>

      {s.error && <p className="text-warn text-sm flex items-center gap-2"><AlertTriangle size={14} /> {s.error}</p>}

      <div className="rounded-card border border-rule bg-black overflow-hidden aspect-video flex items-center justify-center">
        {current ? (
          <video
            key={current.index}
            src={getApiUrl(`/api/live/${s.id}/recording/${current.index}`)}
            controls
            preload="metadata"
            className="w-full h-full object-contain"
          />
        ) : (
          <div className="text-center text-muted text-sm px-6">
            {active ? <Loader2 size={20} className="animate-spin text-brass mx-auto mb-2" /> : null}
            <p className="lowercase">
              {!active
                ? 'no recordings left on disk'
                : chunks.length
                  ? `clipped recordings are deleted · the next one appears within ${s.chunkMinutes} minutes`
                  : `the first recording appears here after ${s.chunkMinutes} minutes`}
            </p>
          </div>
        )}
      </div>

      <div>
        <p className="eyebrow mb-2">recordings · {s.chunkMinutes} min each</p>
        {chunks.length === 0 ? (
          <p className="text-xs text-muted">none yet</p>
        ) : (
          <div className="space-y-1">
            {chunks.map((c) => (
              <button
                key={c.index}
                type="button"
                disabled={!c.available}
                onClick={() => setSelected(c.index)}
                className={`w-full flex items-center justify-between gap-3 text-xs px-2 py-1.5 rounded-input border transition-colors text-left
                  ${current && current.index === c.index ? 'border-[color:var(--color-accent)]' : 'border-rule'}
                  ${c.available ? 'hover:border-[color:var(--color-accent)]' : 'opacity-60 cursor-default'}`}
              >
                <span className="flex items-center gap-2 text-ink2 shrink-0">
                  <Film size={12} /> part {c.index} · {fmtDuration(c.duration)}
                </span>
                <span className={`truncate ${c.status === 'failed' ? 'text-danger' : c.status === 'clipping' ? 'text-brass' : 'text-muted'}`}>
                  {STATUS_TEXT[c.status] || c.status}
                  {c.status === 'done' ? ` · ${c.clips} clips` : ''}
                  {c.error ? ` · ${c.error}` : ''}
                </span>
              </button>
            ))}
          </div>
        )}
      </div>

      {(s.log || []).length > 0 && (
        <div className="bg-paper rounded-card border border-rule overflow-hidden">
          <p className="readout flex items-center gap-2 px-3.5 py-2 border-b border-rule bg-paper2"><Terminal size={12} /> session log</p>
          <pre className="text-[11px] text-muted whitespace-pre-wrap max-h-40 overflow-y-auto custom-scrollbar p-3.5 font-mono">{s.log.join('\n')}</pre>
        </div>
      )}
    </div>
  );
}
