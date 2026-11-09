// What the dashboard does when a /api/status poll fails.
//
// A session restored from localStorage can point at a job the server no longer
// has: the Cloud sweep drops finished jobs after JOB_RETENTION_SECONDS (its
// clips live on in History), and a job the caller does not own answers 404 by
// design. Retrying those forever left the screen on "processing" with empty
// logs, so a gone job ends the poll at once. Any other failure (network drop,
// 5xx during a deploy) is retried, up to a ceiling of consecutive misses.

// 404 = purged or not yours, 403 = not yours, 410 = gone.
export const JOB_GONE_STATUSES = [403, 404, 410];

// Consecutive non-"gone" failures before giving up: ~2 minutes at 2 s a poll,
// longer than a deploy handover takes to answer again.
export const MAX_POLL_FAILURES = 60;

export class JobPollError extends Error {
  constructor(status) {
    super(`Status check failed (${status})`);
    this.name = 'JobPollError';
    this.status = status;
  }
}

// 'gone'  → stop polling, clear the saved session, point the user at History.
// 'retry' → keep polling.
export function pollFailureAction(error, consecutiveFailures) {
  const status = error && typeof error.status === 'number' ? error.status : null;
  if (status !== null && JOB_GONE_STATUSES.includes(status)) return 'gone';
  if (consecutiveFailures >= MAX_POLL_FAILURES) return 'gone';
  return 'retry';
}
