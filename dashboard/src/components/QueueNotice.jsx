import React from 'react';
import { Clock } from 'lucide-react';
import { getApiUrl } from '../config';
import { track } from '../lib/analytics';

// Shown only while the API says the free queue is saturated (/api/config,
// freeQueueBusy): free videos then wait behind paid ones, sometimes for hours,
// and a visitor should know before signing up. It states what is true right
// now and goes away by itself when the queue recovers; nothing here is a
// countdown or a made-up limit. Rendered in the browser only, so the static
// SEO copy of the page never carries a notice that may no longer hold.
export default function QueueNotice({ className = '' }) {
  const [busy, setBusy] = React.useState(false);

  React.useEffect(() => {
    let alive = true;
    fetch(getApiUrl('/api/config'))
      .then((r) => (r.ok ? r.json() : null))
      .then((cfg) => {
        if (alive && cfg && cfg.freeQueueBusy) {
          setBusy(true);
          track('QueueNoticeSeen');
        }
      })
      .catch(() => {});
    return () => { alive = false; };
  }, []);

  if (!busy) return null;
  return (
    <a
      href="#pricing"
      onClick={() => track('QueueNoticeClick')}
      className={`flex items-start gap-2 px-3 py-2 rounded-input bg-paper3 border border-rule text-sm text-muted hover:text-ink transition-colors ${className}`}
    >
      <Clock size={14} className="shrink-0 mt-0.5 text-brass" />
      <span>
        High demand right now: free videos wait in a queue and can take hours. We are adding capacity.{' '}
        <span className="text-brass">Paid plans skip the line and are ready in minutes.</span>
      </span>
    </a>
  );
}
