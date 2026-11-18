// How far a running job is, read from its own log: the pipeline says how many
// clips it found, then names each one as it finishes. Both lines read the same
// in the curated cloud log and in the raw self-hosted one.
const FOUND = /Found (\d+) (?:viral )?clips/;
const READY = /Clip (\d+) ready/;

// What each plan gets out of one video (the server enforces it; this is copy).
// The free plan: its best clips, fewer while the free queue is busy.
export const FREE_CLIPS_PER_VIDEO = 2;
export const FREE_CLIPS_QUIET = 4;
export const MAX_CLIPS_PER_VIDEO = 15;

// {total, ready}; total is 0 until the job has picked its clips.
export function clipProgress(logs) {
  let total = 0;
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
  }
  return { total, ready: total ? Math.min(ready.size, total) : 0 };
}
