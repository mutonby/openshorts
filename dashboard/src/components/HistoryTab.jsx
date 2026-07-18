import React, { useState, useEffect, useMemo } from 'react';
import { Loader2, Download, Film, FolderOpen } from 'lucide-react';
import { apiJson } from '../lib/api';

// The signed-in user's saved video library (stored in R2). Private, signed links.
// Videos are grouped by project (job); re-openable projects get a "reopen"
// action that restores the whole job for further editing in the Clip Generator.
export default function HistoryTab({ onReopenProject }) {
  const [videos, setVideos] = useState(null);
  const [projects, setProjects] = useState({});
  const [reopening, setReopening] = useState(null);
  const [reopenError, setReopenError] = useState('');
  const [error, setError] = useState('');

  useEffect(() => {
    apiJson('/api/history')
      .then((d) => setVideos(d.videos || []))
      .catch(() => setError('Could not load your library.'));
    apiJson('/api/projects')
      .then((d) => {
        const map = {};
        for (const p of d.projects || []) map[p.job_id] = p;
        setProjects(map);
      })
      .catch(() => {});
  }, []);

  // Group videos by job, preserving the newest-first order of /api/history.
  const groups = useMemo(() => {
    const byJob = new Map();
    for (const v of videos || []) {
      const key = v.job_id || v.id;
      if (!byJob.has(key)) byJob.set(key, []);
      byJob.get(key).push(v);
    }
    return [...byJob.entries()];
  }, [videos]);

  const handleReopen = async (jobId) => {
    if (!onReopenProject || reopening) return;
    setReopening(jobId);
    setReopenError('');
    try {
      await onReopenProject(jobId);
    } catch (e) {
      setReopenError('Could not reopen this project. Please try again.');
      setReopening(null);
    }
  };

  const fmtDate = (iso) => (iso ? new Date(iso).toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' }) : '');

  if (videos === null && !error) {
    return <div className="flex justify-center py-20"><Loader2 className="animate-spin text-brass" /></div>;
  }

  return (
    <div className="h-full overflow-y-auto p-8 max-w-5xl mx-auto animate-fade">
      <p className="eyebrow mb-1.5">06 · HISTORY</p>
      <h1 className="font-display lowercase text-2xl text-ink mb-2">Your library</h1>
      <p className="text-muted text-sm mb-8 lowercase">
        All the shorts you've generated, saved while your plan is active. Kept for 7 days after your plan ends. Reopen a project to keep editing its clips.
      </p>

      {error && <p className="text-danger text-sm">{error}</p>}
      {reopenError && <p className="text-danger text-sm mb-4">{reopenError}</p>}

      {videos && videos.length === 0 && (
        <div className="text-center py-20 text-muted">
          <Film size={40} className="mx-auto mb-4 text-muted" />
          <p className="lowercase">No videos yet. Generate your first short from the Clip Generator.</p>
        </div>
      )}

      <div className="space-y-10">
        {groups.map(([jobId, vids]) => {
          const project = projects[jobId];
          return (
            <section key={jobId}>
              <div className="flex flex-wrap items-center justify-between gap-3 mb-4 pb-2 border-b border-rule">
                <div className="min-w-0">
                  <p className="text-sm text-ink font-medium truncate" title={project?.title || vids[0]?.title}>
                    {project?.title || vids[0]?.title || 'Project'}
                  </p>
                  <p className="readout mt-0.5">
                    {fmtDate(vids[0]?.created_at)} · {vids.length} clip{vids.length === 1 ? '' : 's'}
                  </p>
                </div>
                {project && onReopenProject && (
                  <button
                    onClick={() => handleReopen(jobId)}
                    disabled={!!reopening}
                    className="btn-ghost px-3 py-2 text-xs shrink-0"
                    title="Restore this project in the Clip Generator to keep editing subtitles, hooks, effects and dubbing"
                  >
                    {reopening === jobId
                      ? <><Loader2 size={14} className="animate-spin" /> reopening…</>
                      : <><FolderOpen size={14} /> reopen project</>}
                  </button>
                )}
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-5">
                {vids.map((v) => (
                  <div key={v.id} className="card card-hover overflow-hidden group">
                    <div className="aspect-[9/16] bg-black">
                      <video src={v.view_url} controls preload="metadata" className="w-full h-full object-contain" />
                    </div>
                    <div className="p-3">
                      <p className="text-sm text-ink font-medium line-clamp-2 mb-1" title={v.title}>{v.title || 'Short'}</p>
                      <div className="flex items-center justify-between">
                        <span className="readout">{fmtDate(v.created_at)}</span>
                        <a href={v.download_url} className="text-micro font-mono uppercase text-brass hover:text-ink flex items-center gap-1 transition-colors" title="Download">
                          <Download size={14} /> Download
                        </a>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </section>
          );
        })}
      </div>
    </div>
  );
}
