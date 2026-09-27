import { useRef, useState } from 'react';
import { apiJson } from '../lib/api';

const REASONS = [
  ['too_expensive', 'Too expensive'], ['not_using', 'Not using it enough'],
  ['quality', 'Clip quality'], ['missing_features', 'Missing features'],
  ['technical_issues', 'Technical issues'], ['switching', 'Switching to another service'],
  ['other', 'Other'],
];

export default function CancellationFeedback() {
  const dialog = useRef(null);
  const trigger = useRef(null);
  const [reason, setReason] = useState('');
  const [comment, setComment] = useState('');
  const [rating, setRating] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  async function proceed(skip) {
    setBusy(true);
    setError('');
    try {
      let result;
      try {
        result = await apiJson('/api/billing/cancel', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(skip ? {} : {
            reason: reason || null, comment: comment || null,
            rating: rating ? Number(rating) : null,
          }),
          signal: AbortSignal.timeout(10000),
        });
      } catch (_) {
        // Feedback/validation/network failures must not prevent billing access.
        result = await apiJson('/api/billing/portal', {
          method: 'POST', signal: AbortSignal.timeout(10000),
        });
      }
      window.location.href = result.url;
    } catch (_) {
      setError('Could not open Stripe. Please try again, or use Manage billing.');
    } finally {
      setBusy(false);
    }
  }

  return <>
    <button ref={trigger} className="btn-ghost mt-4" onClick={() => {
      setError(''); dialog.current.showModal();
    }}>Cancel subscription</button>
    <dialog ref={dialog} aria-labelledby="cancel-title" aria-describedby="cancel-description"
      onClose={() => trigger.current?.focus()}
      className="card p-6 w-[calc(100%-2rem)] max-w-lg max-h-[90vh] overflow-y-auto text-ink backdrop:bg-black/70">
      <h2 id="cancel-title" className="font-display text-xl">Cancel subscription</h2>
      <p id="cancel-description" className="text-sm text-muted my-3">
        Feedback is optional and shared privately with the OpenShorts team via Telegram.
        Please do not include personal or sensitive information.
        Nothing is cancelled yet. Continue to Stripe to review and confirm cancellation.
      </p>
      <form onSubmit={(event) => { event.preventDefault(); proceed(false); }} className="space-y-4">
        <label className="block text-sm">Reason (optional)
          <select value={reason} onChange={(e) => setReason(e.target.value)} className="block w-full mt-1 bg-paper2 border border-rule rounded p-2">
            <option value="">Prefer not to say</option>
            {REASONS.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
          </select>
        </label>
        <label className="block text-sm">Your experience (optional)
          <textarea value={comment} onChange={(e) => setComment(e.target.value)} rows={3}
            maxLength={2000} className="block w-full mt-1 bg-paper2 border border-rule rounded p-2"
            placeholder="What could we improve? Please avoid sensitive information." />
        </label>
        <label className="block text-sm">Overall rating (optional)
          <select value={rating} onChange={(e) => setRating(e.target.value)} className="block w-full mt-1 bg-paper2 border border-rule rounded p-2">
            <option value="">No rating</option>
            {[1, 2, 3, 4, 5].map((n) => <option key={n} value={n}>{n} / 5{n === 1 ? ' — poor' : n === 5 ? ' — excellent' : ''}</option>)}
          </select>
        </label>
        {error && <p role="alert" className="text-sm text-warn">{error}</p>}
        {busy && <p role="status" className="text-sm">Opening Stripe…</p>}
        <div className="grid gap-2 sm:grid-cols-2">
          <button type="submit" disabled={busy} className="btn-ghost justify-center">Send feedback and continue</button>
          <button type="button" disabled={busy} className="btn-ghost justify-center" onClick={() => proceed(true)}>Skip feedback and continue</button>
        </div>
        <button type="button" className="btn-quiet" onClick={() => dialog.current.close()}>Go back</button>
      </form>
    </dialog>
  </>;
}
