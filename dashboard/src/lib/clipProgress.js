// How far a running job is, read from its own log: the pipeline says how many
// clips it found, then names each one as it finishes. Both lines read the same
// in the curated cloud log and in the raw self-hosted one.
const FOUND = /Found (\d+) (?:viral )?clips/;
const READY = /Clip (\d+) ready/;
const DOWNLOADING = /Downloading video|Download attempt|\[download\]\s+\d/i;
const TRANSCRIBING = /Audio ready ahead|Transcribing(?:…| audio| video)/i;
const SELECTING = /Finding the best moments|finding moments in the picture|Analyzing with (?:Gemini|local LLM)|Silent video.*analyzing/i;
const VIDEO_DOWNLOADED = /Video downloaded|Download succeeded/i;

// What each plan gets out of one video (the server enforces it; this is copy).
// The free plan: its best clips, fewer while the free queue is busy.
export const FREE_CLIPS_PER_VIDEO = 2;
export const FREE_CLIPS_QUIET = 4;
export const MAX_CLIPS_PER_VIDEO = 15;

// Works with raw self-hosted logs and the curated Cloud progress lines.
export function clipProgress(logs) {
  let total = 0;
  let stage = 'Starting job';
  const ready = new Set();
  for (const line of logs || []) {
    const text = String(line);
    const found = FOUND.exec(text);
    if (found) {
      total = Number(found[1]);
      continue;
    }
    const done = READY.exec(text);
    if (done) ready.add(done[1]);
    if (SELECTING.test(text)) stage = 'Finding clip moments';
    else if (TRANSCRIBING.test(text) && stage !== 'Finding clip moments') stage = 'Transcribing audio';
    else if (DOWNLOADING.test(text) && stage === 'Starting job') stage = 'Downloading source';
    else if (VIDEO_DOWNLOADED.test(text) && stage === 'Downloading source') stage = 'Preparing source';
  }
  const rendered = total ? Math.min(ready.size, total) : 0;
  if (total) stage = rendered === total ? 'Finalizing clips' : 'Rendering clips';
  return { total, ready: rendered, stage };
}

export function lastJobError(logs = []) {
  return [...logs].reverse().find((line) => /❌|(?:^|\s)(?:\w*Error|Exception|FATAL):/i.test(line))
    || 'See the processing log for details.';
}
