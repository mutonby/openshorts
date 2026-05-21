// Job store. Holds the active Clip Generator job state — jobId, status,
// results, logs, the source media being processed, and the synced-playback
// timing the Result cards use to scrub the source video.
//
// Same custom-event pattern as keysStore + brandKit. Single global slot
// because at any moment there's one active job in the existing pipeline.

import { useEffect, useState } from 'react';

const SESSION_KEY = 'openshorts_session';
const SESSION_MAX_AGE = 3_600_000; // 1 hour — matches backend job retention
const EVENT = 'openshorts:job-changed';

const INITIAL_STATE = {
  jobId: null,
  status: 'idle',          // idle | processing | complete | error
  results: null,
  logs: [],
  processingMedia: null,   // { type: 'url'|'file', payload, acknowledged }
  syncedTime: 0,
  isSyncedPlaying: false,
  syncTrigger: 0,
  sessionRecovered: false,
};

let _state = { ...INITIAL_STATE };

function emit() {
  window.dispatchEvent(new CustomEvent(EVENT, { detail: _state }));
}

export function getJob() {
  return _state;
}

export function updateJob(patch) {
  _state = { ..._state, ...patch };
  emit();
  // Persist non-trivial state to localStorage so reloads can recover.
  if (_state.status === 'idle') {
    localStorage.removeItem(SESSION_KEY);
    return;
  }
  try {
    localStorage.setItem(SESSION_KEY, JSON.stringify({
      jobId: _state.jobId,
      status: _state.status,
      results: _state.results,
      processingMedia: _state.processingMedia?.type === 'url' ? _state.processingMedia : null,
      timestamp: Date.now(),
    }));
  } catch {
    // localStorage full — ignore
  }
}

export function resetJob() {
  _state = { ...INITIAL_STATE };
  localStorage.removeItem(SESSION_KEY);
  emit();
}

export function recoverJob() {
  try {
    const saved = localStorage.getItem(SESSION_KEY);
    if (!saved) return false;
    const session = JSON.parse(saved);
    if (Date.now() - session.timestamp > SESSION_MAX_AGE) {
      localStorage.removeItem(SESSION_KEY);
      return false;
    }
    if (!session.jobId || !session.status || session.status === 'idle') return false;
    _state = {
      ...INITIAL_STATE,
      jobId: session.jobId,
      status: session.status === 'processing' ? 'processing' : session.status,
      results: session.results || null,
      processingMedia: session.processingMedia || null,
      sessionRecovered: true,
    };
    emit();
    setTimeout(() => updateJob({ sessionRecovered: false }), 5000);
    return true;
  } catch {
    localStorage.removeItem(SESSION_KEY);
    return false;
  }
}

export function triggerSyncedPlay(startTime) {
  updateJob({
    syncedTime: startTime,
    isSyncedPlaying: true,
    syncTrigger: _state.syncTrigger + 1,
  });
}

export function triggerSyncedPause() {
  updateJob({ isSyncedPlaying: false });
}

export function useJob() {
  const [state, setState] = useState(() => getJob());
  useEffect(() => {
    const onChange = (e) => setState(e.detail || getJob());
    window.addEventListener(EVENT, onChange);
    return () => window.removeEventListener(EVENT, onChange);
  }, []);
  return state;
}
