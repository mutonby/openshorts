// Clip Generator — the existing /api/process workflow. Phase 1 carves this
// out of the old App.jsx tab body. Phase 2 will refactor to consume the
// extracted job + keys stores instead of local state.

import { useEffect, useState } from 'react';
import {
  Activity, AlertTriangle, Calendar, ChevronDown, Instagram, KeyRound,
  PlusCircle, RotateCcw, Sparkles, Terminal, X, Youtube,
} from 'lucide-react';
import MediaInput from '../components/MediaInput';
import ResultCard from '../components/ResultCard';
import ProcessingAnimation from '../components/ProcessingAnimation';
import ScheduleWeekModal from '../components/ScheduleWeekModal';
import { getApiUrl } from '../config';
import { useKeys } from '../state/keysStore.js';
import {
  getJob, recoverJob, resetJob, triggerSyncedPause, triggerSyncedPlay, updateJob, useJob,
} from '../state/jobStore.js';

const TikTokIcon = ({ size = 16, className = '' }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor" className={className}>
    <path d="M19.589 6.686a4.793 4.793 0 0 1-3.77-4.245V2h-3.445v13.672a2.896 2.896 0 0 1-5.201 1.743l-.002-.001.002.001a2.895 2.895 0 0 1 3.183-4.51v-3.5a6.329 6.329 0 0 0-5.394 10.692 6.33 6.33 0 0 0 10.857-4.424V8.687a8.182 8.182 0 0 0 4.773 1.526V6.79a4.831 4.831 0 0 1-1.003-.104z" />
  </svg>
);

export default function ClipGenerator() {
  const keys = useKeys();
  const job = useJob();
  const {
    jobId, status, results, logs, processingMedia,
    syncedTime, isSyncedPlaying, syncTrigger, sessionRecovered,
  } = job;

  const [logsVisible, setLogsVisible] = useState(true);
  const [showScheduleWeek, setShowScheduleWeek] = useState(false);
  const [showKeyModal, setShowKeyModal] = useState(false);

  useEffect(() => {
    if (status === 'idle') recoverJob();
  }, []);

  const handleProcess = async (data) => {
    if (!keys.gemini || !keys.uploadPost) {
      setShowKeyModal(true);
      return;
    }
    updateJob({
      status: 'processing',
      logs: ['Starting process...'],
      results: null,
      processingMedia: data,
    });
    try {
      let body;
      const headers = { 'X-Gemini-Key': keys.gemini };
      if (data.type === 'url') {
        headers['Content-Type'] = 'application/json';
        body = JSON.stringify({ url: data.payload, acknowledged: !!data.acknowledged });
      } else {
        const formData = new FormData();
        formData.append('file', data.payload);
        formData.append('acknowledged', data.acknowledged ? 'true' : 'false');
        body = formData;
      }
      const res = await fetch(getApiUrl('/api/process'), {
        method: 'POST',
        headers: data.type === 'url' ? headers : { 'X-Gemini-Key': keys.gemini },
        body,
      });
      if (!res.ok) throw new Error(await res.text());
      const resData = await res.json();
      updateJob({ jobId: resData.job_id });
    } catch (e) {
      updateJob({
        status: 'error',
        logs: [...(getJob().logs || []), `Error starting job: ${e.message}`],
      });
    }
  };

  return (
    <div className="h-full flex flex-col">
      {/* Reset / new project button row */}
      {status !== 'idle' && (
        <div className="px-6 pt-4 flex items-center gap-3">
          <button
            onClick={resetJob}
            className="flex items-center gap-2 text-[13px] text-zinc-400 hover:text-white transition-colors"
          >
            <PlusCircle size={14} />
            New project
          </button>
          <span className={`text-[11px] px-2 py-0.5 rounded-full border ${
            status === 'processing' ? 'bg-primary/10 border-primary/30 text-primary'
              : status === 'complete' ? 'bg-success/10 border-success/30 text-success'
              : status === 'error' ? 'bg-red-500/10 border-red-500/30 text-red-400'
              : 'bg-white/5 border-border text-zinc-500'
          }`}>
            {String(status).toUpperCase()}
          </span>
        </div>
      )}

      {/* Missing keys banner */}
      {(!keys.gemini || !keys.uploadPost) && (
        <div className="mx-6 mt-3 p-3 bg-amber-500/10 border border-amber-500/30 rounded-lg flex items-center justify-between gap-4 shrink-0">
          <div className="flex items-center gap-3 text-[13px] text-amber-200">
            <KeyRound size={14} className="shrink-0 text-amber-400" />
            <div>
              <span className="font-semibold">API keys missing.</span>{' '}
              <span className="text-amber-200/80">
                {!keys.gemini && !keys.uploadPost ? 'Set Gemini and Upload-Post keys.'
                  : !keys.gemini ? 'Set your Gemini API key.'
                  : 'Set your Upload-Post API key.'}
              </span>
            </div>
          </div>
          <a
            href="#/settings"
            className="shrink-0 text-[11px] font-medium px-3 py-1.5 rounded-md bg-amber-500 hover:bg-amber-400 text-black transition-colors"
          >
            Go to Settings
          </a>
        </div>
      )}

      {sessionRecovered && (
        <div className="mx-6 mt-2 p-3 bg-primary/10 border border-primary/30 rounded-lg flex items-center justify-between shrink-0">
          <div className="flex items-center gap-2 text-[13px] text-primary">
            <RotateCcw size={14} />
            <span className="font-medium">Session recovered</span>
            <span className="text-zinc-400 text-[11px]">Your previous work has been restored.</span>
          </div>
          <button onClick={() => updateJob({ sessionRecovered: false })} className="text-zinc-500 hover:text-white transition-colors">
            <X size={14} />
          </button>
        </div>
      )}

      <div className="flex-1 overflow-hidden">
        {status === 'idle' && (
          <div className="h-full flex flex-col items-center justify-center p-6">
            <div className="max-w-xl w-full text-center space-y-8">
              <div className="space-y-3">
                <h1 className="text-3xl md:text-4xl font-bold text-white">
                  Create Viral Shorts
                </h1>
                <p className="text-zinc-400 text-[14px]">
                  Drop your long-form video below to generate viral clips with AI.
                </p>
              </div>
              <MediaInput onProcess={handleProcess} isProcessing={status === 'processing'} />
              <div className="flex items-center justify-center gap-6 text-zinc-500 text-[12px]">
                <span className="flex items-center gap-2"><Youtube size={14} /> YouTube</span>
                <span className="flex items-center gap-2"><Instagram size={14} /> Instagram</span>
                <span className="flex items-center gap-2"><TikTokIcon size={14} /> TikTok</span>
              </div>
            </div>
          </div>
        )}

        {(status === 'processing' || status === 'complete' || status === 'error') && (
          <div className="h-full flex flex-col md:flex-row">
            <div className={`${status === 'complete' ? 'md:w-[30%] lg:w-[25%]' : 'md:w-[55%] lg:w-[60%]'} h-full flex flex-col border-r border-border bg-black/20 p-6 overflow-y-auto custom-scrollbar transition-all duration-700`}>
              <div className="mb-6 flex items-center justify-between">
                <h2 className="text-[15px] font-semibold flex items-center gap-2 text-white">
                  <Activity className={`text-primary ${status === 'processing' ? 'animate-pulse' : ''}`} size={18} />
                  Live Analysis
                </h2>
              </div>

              {processingMedia && (
                <ProcessingAnimation
                  media={processingMedia}
                  isComplete={status === 'complete'}
                  syncedTime={syncedTime}
                  isSyncedPlaying={isSyncedPlaying}
                  syncTrigger={syncTrigger}
                />
              )}

              <div className={`bg-[#0c0c0e] rounded-lg border border-border overflow-hidden flex flex-col transition-all duration-500 ${status === 'complete' ? 'h-32 opacity-50 hover:opacity-100' : 'flex-1 min-h-[200px]'}`}>
                <div className="px-4 py-2 border-b border-border flex items-center justify-between bg-white/5 shrink-0">
                  <span className="text-[11px] font-mono text-zinc-400 flex items-center gap-2">
                    <Terminal size={12} /> System Logs
                  </span>
                  <button onClick={() => setLogsVisible(!logsVisible)} className="text-zinc-500 hover:text-white transition-colors">
                    <ChevronDown size={14} className={logsVisible ? '' : 'rotate-180'} />
                  </button>
                </div>
                {logsVisible && (
                  <div className="flex-1 p-4 overflow-y-auto font-mono text-[11px] space-y-1.5 custom-scrollbar text-zinc-400">
                    {logs.map((log, i) => (
                      <div key={i} className={`flex gap-2 ${log.toLowerCase().includes('error') ? 'text-red-400' : 'text-zinc-400'}`}>
                        <span className="text-zinc-700 shrink-0">{new Date().toLocaleTimeString()}</span>
                        <span>{log}</span>
                      </div>
                    ))}
                    {status === 'processing' && <div className="animate-pulse text-primary/70">_</div>}
                  </div>
                )}
              </div>
            </div>

            <div className={`${status === 'complete' ? 'md:w-[70%] lg:w-[75%]' : 'md:w-[45%] lg:w-[40%]'} h-full flex flex-col bg-background p-6 transition-all duration-700`}>
              <h2 className="text-[15px] font-semibold mb-6 flex items-center gap-2 shrink-0 text-white">
                <Sparkles className="text-yellow-400" size={18} />
                Generated Shorts
                {results?.clips?.length > 0 && (
                  <span className="text-[11px] bg-white/10 text-white px-2 py-0.5 rounded-full ml-2">
                    {results.clips.length} clips
                  </span>
                )}
                {results?.cost_analysis && (
                  <span className="text-[11px] bg-success/10 border border-success/30 text-success px-2 py-0.5 rounded-full ml-2">
                    ${results.cost_analysis.total_cost.toFixed(5)}
                  </span>
                )}
                {results?.clips?.length > 1 && status === 'complete' && (
                  <button
                    onClick={() => setShowScheduleWeek(true)}
                    className="ml-auto flex items-center gap-1.5 px-3 py-1.5 bg-primary/10 hover:bg-primary/20 border border-primary/30 text-primary rounded-md text-[11px] font-medium transition-colors"
                  >
                    <Calendar size={12} />
                    Schedule week
                  </button>
                )}
              </h2>

              <div className="flex-1 overflow-y-auto custom-scrollbar p-1">
                {results?.clips?.length > 0 ? (
                  <div className={`grid gap-4 pb-10 ${status === 'complete' ? 'grid-cols-1 xl:grid-cols-2' : 'grid-cols-1'}`}>
                    {results.clips.map((clip, i) => (
                      <ResultCard
                        key={i}
                        clip={clip}
                        index={i}
                        jobId={jobId}
                        uploadPostKey={keys.uploadPost}
                        uploadUserId={keys.uploadUserId}
                        geminiApiKey={keys.gemini}
                        elevenLabsKey={keys.elevenLabs}
                        onPlay={(t) => triggerSyncedPlay(t)}
                        onPause={triggerSyncedPause}
                      />
                    ))}
                  </div>
                ) : status === 'processing' ? (
                  <div className="h-full flex flex-col items-center justify-center text-zinc-500 space-y-3 opacity-60">
                    <div className="w-10 h-10 rounded-full border-2 border-zinc-800 border-t-primary animate-spin" />
                    <p className="text-[13px]">Waiting for clips...</p>
                  </div>
                ) : status === 'error' ? (
                  <div className="h-full flex flex-col items-center justify-center text-red-400">
                    <p>Generation failed.</p>
                  </div>
                ) : null}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Missing-keys modal */}
      {showKeyModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm" onClick={() => setShowKeyModal(false)}>
          <div className="bg-surface border border-border rounded-xl p-6 max-w-md w-full mx-4 space-y-4 shadow-2xl" onClick={(e) => e.stopPropagation()}>
            <h2 className="text-[15px] font-semibold text-white">API keys required</h2>
            <p className="text-[13px] text-zinc-400">
              Gemini and Upload-Post keys are required to generate clips. Both have free tiers.
            </p>
            <div className="flex gap-2 text-[11px] text-zinc-300">
              <AlertTriangle size={14} className="text-amber-400 shrink-0" />
              <span>Set them in Settings → API Keys.</span>
            </div>
            <div className="flex gap-3">
              <button onClick={() => setShowKeyModal(false)} className="flex-1 text-[13px] text-zinc-400 py-2 rounded-md border border-border hover:bg-white/5">Cancel</button>
              <a href="#/settings" onClick={() => setShowKeyModal(false)} className="flex-1 text-[13px] text-white py-2 rounded-md bg-primary hover:bg-primary/90 text-center font-medium">Go to Settings</a>
            </div>
          </div>
        </div>
      )}

      <ScheduleWeekModal
        isOpen={showScheduleWeek}
        onClose={() => setShowScheduleWeek(false)}
        clips={results?.clips || []}
        jobId={jobId}
        uploadPostKey={keys.uploadPost}
        uploadUserId={keys.uploadUserId}
      />
    </div>
  );
}
