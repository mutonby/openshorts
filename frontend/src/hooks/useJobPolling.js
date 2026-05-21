// Polls /api/status/{job_id} every 2s while a job is processing.
// Drives the Clip Generator UI — extracted verbatim from the App.jsx
// polling effect that lived around lines 267-298.

import { useEffect } from 'react';
import { getApiUrl } from '../config.js';
import { getJob, updateJob, useJob } from '../state/jobStore.js';

async function pollJob(jobId) {
  const res = await fetch(getApiUrl(`/api/status/${jobId}`));
  if (!res.ok) throw new Error('Status check failed');
  return res.json();
}

export function useJobPolling() {
  const { status, jobId } = useJob();

  useEffect(() => {
    if (!(status === 'processing' || status === 'completed') || !jobId) return;
    let cancelled = false;
    const interval = setInterval(async () => {
      try {
        const data = await pollJob(jobId);
        if (cancelled) return;
        const patch = {};
        if (data.result) patch.results = data.result;
        if (data.status === 'completed') {
          patch.status = 'complete';
          updateJob(patch);
          clearInterval(interval);
          return;
        }
        if (data.status === 'failed') {
          const errorMsg = data.error
            || (data.logs && data.logs.length ? data.logs[data.logs.length - 1] : 'Process failed');
          patch.status = 'error';
          patch.logs = [...(getJob().logs || []), `Error: ${errorMsg}`];
          updateJob(patch);
          clearInterval(interval);
          return;
        }
        if (data.logs) patch.logs = data.logs;
        if (Object.keys(patch).length) updateJob(patch);
      } catch (e) {
        console.error('Polling error', e);
      }
    }, 2000);

    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [status, jobId]);
}
