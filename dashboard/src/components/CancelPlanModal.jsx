import React, { useState, useRef, useEffect } from 'react';
import { Loader2, Star } from 'lucide-react';
import { apiJson } from '../lib/api';
import { track } from '../lib/analytics';
import Modal from './ui/Modal';

// Mirrors cloud/cancellation.CANCEL_REASONS. A closed list so churn can be
// counted; the text box under it is where the specifics go.
const REASONS = [
  ['too_expensive', 'Too expensive'],
  ['not_using_it', "I'm not using it enough"],
  ['one_off_project', 'I only needed it for one project'],
  ['clip_quality', "The clips weren't good enough"],
  ['missing_feature', 'Missing a feature I need'],
  ['too_complex', 'Too hard to use'],
  ['found_alternative', 'I found something better'],
  ['support', 'Bad experience with support'],
  ['other', 'Something else'],
];

const DETAIL_PROMPT = {
  too_expensive: 'What price would have worked for you?',
  clip_quality: 'What was wrong with the clips?',
  missing_feature: 'Which feature were you missing?',
  found_alternative: 'Which tool are you moving to, and what does it do better?',
  too_complex: 'Where did you get stuck?',
  support: 'What happened?',
};

const fmtDate = (iso) => {
  if (!iso) return 'the end of your billing period';
  try {
    return new Date(iso).toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' });
  } catch (_) { return 'the end of your billing period'; }
};

// Cancel flow: reason -> review -> confirm. The cancel only happens on the
// last step, and at period end (backend: cloud/cancellation.py), so every
// "keep my plan" exit leaves the subscription untouched. The last step leads
// with the retention offer when the server says this subscription gets one.
// The offer is stated in money and dates (what each renewal costs, from which
// day, and that nothing moves today): read as a bare percentage it was taken
// for an immediate refund, and accepted by people who then cancelled anyway.
// The confirm step also says what happens to the library once the plan ends.
export default function CancelPlanModal({ plan, periodEnd, onClose, onCanceled }) {
  const [step, setStep] = useState(0);
  const [reason, setReason] = useState('');
  const [details, setDetails] = useState('');
  const [rating, setRating] = useState(0);
  const [hover, setHover] = useState(0);
  const [review, setReview] = useState('');
  const [publicOk, setPublicOk] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [offer, setOffer] = useState(null);
  const [saved, setSaved] = useState(null);
  const sending = useRef(false);

  // Asked up front so the last step never waits on it; no answer means no offer.
  // The same answer says whether a month on hold is on the table, and for
  // which reasons (cloud/cancellation.PAUSE_REASONS).
  const [pauseInfo, setPauseInfo] = useState(null);
  const [paused, setPaused] = useState(null);
  useEffect(() => {
    apiJson('/api/billing/retention-offer')
      .then((o) => {
        if (o?.eligible) setOffer(o);
        if (o?.pause_eligible) setPauseInfo(o);
      })
      .catch(() => {});
  }, []);
  const pause = pauseInfo && (pauseInfo.pause_reasons || []).includes(reason) ? pauseInfo : null;

  // What the plan costs a month, to put the discount in money. The offer only
  // exists for monthly plans; without a price the copy falls back to the percentage.
  const [price, setPrice] = useState(null);
  useEffect(() => {
    apiJson('/api/billing/plans')
      .then((d) => setPrice((d.plans || []).find((p) => p.plan === plan && p.interval === 'month') || null))
      .catch(() => {});
  }, [plan]);
  const money = (cents) => new Intl.NumberFormat('en-US', {
    style: 'currency', currency: (price?.currency || 'usd').toUpperCase(),
    minimumFractionDigits: cents % 100 ? 2 : 0, maximumFractionDigits: 2,
  }).format(cents / 100);
  // { was, now } per renewal with the offer's discount, or null.
  const deal = (percentOff) => (price?.amount && percentOff
    ? { was: money(price.amount), now: money(Math.round(price.amount * (100 - percentOff) / 100)) }
    : null);

  useEffect(() => {
    if (step === 2 && offer) track('RetentionOfferShown', { props: { plan, reason } });
    if (step === 2 && pause) track('PauseOfferShown', { props: { plan, reason } });
  }, [step, offer, pause]); // eslint-disable-line react-hooks/exhaustive-deps

  const feedback = () => JSON.stringify({
    reason,
    details: details.trim() || undefined,
    rating: rating || undefined,
    review: review.trim() || undefined,
    review_public_ok: publicOk,
  });

  const acceptOffer = async () => {
    if (sending.current) return;
    sending.current = true;
    setBusy(true);
    setError('');
    try {
      const out = await apiJson('/api/billing/retention-offer/accept', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: feedback(),
      });
      track('RetentionOfferAccepted', { props: { plan, reason, priced: deal(offer?.percent_off) ? '1' : '0' } });
      setSaved(out);
    } catch (e) {
      setError(e?.detail || 'Could not apply the discount. Please try again or email info@openshorts.app.');
    }
    sending.current = false;
    setBusy(false);
  };

  const acceptPause = async () => {
    if (sending.current) return;
    sending.current = true;
    setBusy(true);
    setError('');
    try {
      const out = await apiJson('/api/billing/pause', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: feedback(),
      });
      track('PauseOfferAccepted', { props: { plan, reason } });
      setPaused(out);
    } catch (e) {
      setError(e?.detail || 'Could not pause your plan. Please try again or email info@openshorts.app.');
    }
    sending.current = false;
    setBusy(false);
  };

  const keep = () => {
    track('CancelFlowAbandoned', { props: { step, reason: reason || undefined } });
    onClose();
  };

  const submit = async () => {
    if (sending.current) return;
    sending.current = true;
    setBusy(true);
    setError('');
    try {
      await apiJson('/api/billing/cancel', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: feedback(),
      });
      track('SubscriptionCanceled', { props: { plan, reason, rating: rating || undefined, offered: !!offer } });
      onCanceled();
    } catch (e) {
      setError(e?.detail || 'Could not cancel. Please try again or email info@openshorts.app.');
      sending.current = false;
      setBusy(false);
    }
  };

  const shown = hover || rating;
  const months = offer?.months ? ` for the next ${offer.months} months` : '';
  const offerDeal = deal(offer?.percent_off);
  const savedDeal = deal(saved?.percent_off);
  const renewals = (n) => (n ? `next ${n} renewal${n > 1 ? 's' : ''}` : 'renewals');

  if (paused) {
    return (
      <Modal isOpen onClose={onCanceled} eyebrow="PLAN PAUSED" title="Next month is on us">
        <div className="space-y-4 text-sm text-ink2">
          <p>
            Your plan runs as usual until <b className="text-ink">{fmtDate(paused.pause_from)}</b>.
            Then it skips one month: <b className="text-ink">no charge</b>, your clips stay
            where they are, and you still get the free minutes. After that it picks up again on its own.
          </p>
          <p>Need it sooner? Restart it from your account page any time.</p>
          <div className="flex justify-end pt-1">
            <button onClick={onCanceled} className="btn-primary px-4 py-2 text-xs">Done</button>
          </div>
        </div>
      </Modal>
    );
  }

  if (saved) {
    return (
      <Modal isOpen onClose={onCanceled} eyebrow="YOU'RE STAYING" title="Discount applied">
        <div className="space-y-4 text-sm text-ink2">
          {savedDeal ? (
            <p>
              Thanks for giving us another go. Your plan stays as it is, and your {renewals(saved.months)} cost{' '}
              <b className="text-ink">{savedDeal.now} instead of {savedDeal.was}</b>, starting{' '}
              {fmtDate(periodEnd)}. Nothing was charged or refunded today.
            </p>
          ) : (
            <p>
              Thanks for giving us another go. Your plan stays as it is, and your next
              {saved.months ? ` ${saved.months}` : ''} invoice{saved.months > 1 ? 's are' : ' is'}{' '}
              <b className="text-ink">{saved.percent_off}% off</b>. Nothing was charged or refunded today.
            </p>
          )}
          <p>We read every answer you gave us and will work on it.</p>
          <div className="flex justify-end pt-1">
            <button onClick={onCanceled} className="btn-primary px-4 py-2 text-xs">Done</button>
          </div>
        </div>
      </Modal>
    );
  }

  return (
    <Modal isOpen onClose={busy ? undefined : keep} eyebrow={`CANCEL PLAN · ${step + 1}/3`}
           title={['Why are you cancelling?', 'How was OpenShorts?',
                   offer || pause ? 'Before you go' : 'Confirm cancellation'][step]}>
      {step === 0 && (
        <div className="space-y-4">
          <p className="text-sm text-muted">Your answer goes straight to the people building OpenShorts.</p>
          <div className="space-y-1.5">
            {REASONS.map(([value, label]) => (
              <label key={value}
                     className={`flex items-center gap-2.5 rounded-card border px-3 py-2 text-sm cursor-pointer transition-colors
                       ${reason === value ? 'border-brass text-ink' : 'border-rule text-ink2 hover:border-brass/60'}`}>
                <input type="radio" name="cancel-reason" value={value} checked={reason === value}
                       onChange={() => setReason(value)} />
                {label}
              </label>
            ))}
          </div>
          {reason && (
            <label className="block">
              <span className="text-sm text-muted">{DETAIL_PROMPT[reason] || 'Anything else we should know?'} (optional)</span>
              <textarea value={details} onChange={(e) => setDetails(e.target.value)} maxLength={2000} rows={3}
                        className="input-field w-full text-sm mt-1" />
            </label>
          )}
          <div className="flex items-center justify-between gap-2 pt-1">
            <button onClick={keep} className="btn-quiet">Keep my plan</button>
            <button onClick={() => setStep(1)} disabled={!reason} className="btn-ghost px-4 py-2">Continue</button>
          </div>
        </div>
      )}

      {step === 1 && (
        <div className="space-y-4">
          <p className="text-sm text-muted">Rate your experience and leave a short review. It takes ten seconds and helps us more than anything else.</p>
          <div className="flex items-center gap-1" onMouseLeave={() => setHover(0)}>
            {[1, 2, 3, 4, 5].map((n) => (
              <button key={n} type="button" aria-label={`${n} star${n > 1 ? 's' : ''}`}
                      onClick={() => setRating(n)} onMouseEnter={() => setHover(n)} className="p-1">
                <Star size={26} className={n <= shown ? 'text-brass fill-current' : 'text-muted'} />
              </button>
            ))}
          </div>
          <label className="block">
            <span className="text-sm text-muted">Your review</span>
            <textarea value={review} onChange={(e) => setReview(e.target.value)} maxLength={2000} rows={4}
                      placeholder="What worked, what didn't, who you'd recommend it to…"
                      className="input-field w-full text-sm mt-1" />
          </label>
          {review.trim() && (
            <label className="flex items-start gap-2 text-sm text-ink2 cursor-pointer">
              <input type="checkbox" checked={publicOk} onChange={(e) => setPublicOk(e.target.checked)} className="mt-0.5" />
              OpenShorts may quote this review publicly (first name only).
            </label>
          )}
          <div className="flex items-center justify-between gap-2 pt-1">
            <button onClick={() => setStep(0)} className="btn-quiet">Back</button>
            <button onClick={() => setStep(2)} disabled={!rating} className="btn-ghost px-4 py-2">Continue</button>
          </div>
        </div>
      )}

      {step === 2 && (
        <div className="space-y-4 text-sm text-ink2">
          {pause && (
            <div className="rounded-card border border-brass bg-brass/5 p-4 space-y-3">
              <p className="text-ink">
                Not using it right now? <b>Skip a month instead.</b> No charge from{' '}
                {fmtDate(pause.pause_from)} for one month, your clips stay, and the plan
                comes back by itself.
              </p>
              <button onClick={acceptPause} disabled={busy} className="btn-primary px-4 py-2 text-xs">
                {busy && <Loader2 size={16} className="animate-spin" />}
                Pause for a month
              </button>
            </div>
          )}
          {offer && (
            <div className="rounded-card border border-brass bg-brass/5 p-4 space-y-3">
              {offerDeal ? (
                <p className="text-ink">
                  Stay and your {renewals(offer.months)} cost <b>{offerDeal.now} instead of {offerDeal.was}</b>.
                  Same plan, same minutes. Nothing is charged or refunded today: the lower
                  price starts with your renewal on {fmtDate(periodEnd)}.
                </p>
              ) : (
                <p className="text-ink">
                  Stay and get <b>{offer.percent_off}% off{months}</b>. Same plan, same
                  minutes. Nothing is charged or refunded today: the discount starts with your next renewal.
                </p>
              )}
              <button onClick={acceptOffer} disabled={busy} className="btn-primary px-4 py-2 text-xs">
                {busy && <Loader2 size={16} className="animate-spin" />}
                {offerDeal ? `Keep my plan at ${offerDeal.now}/mo` : `Keep my plan with ${offer.percent_off}% off`}
              </button>
            </div>
          )}
          <p>
            {offer || pause ? 'If you still want to leave, your' : 'Your'}
            {' '}{plan ? <b className="text-ink capitalize">{plan}</b> : null} plan stays active until{' '}
            <b className="text-ink">{fmtDate(periodEnd)}</b>, with every minute you have left.
            You won't be charged again.
          </p>
          <p>
            After that your library is no longer kept for good: your clips are{' '}
            <b className="text-ink">deleted 7 days after your plan ends</b> (later ones, 7 days after
            they are made), so download the ones you want to keep.
            {pause ? ' Pausing keeps them.' : ''}
          </p>
          <p>You can undo this from your account page any time before then.</p>
          {error && <p className="text-danger">{error}</p>}
          <div className="flex items-center justify-between gap-2 pt-1">
            <button onClick={keep} disabled={busy}
                    className={offer || pause ? 'btn-quiet' : 'btn-primary px-4 py-2 text-xs'}>
              {offer ? 'Keep my plan at full price' : 'Keep my plan'}
            </button>
            <button onClick={submit} disabled={busy} className="btn-danger px-4 py-2">
              {busy && <Loader2 size={16} className="animate-spin" />}
              {busy ? 'Cancelling…' : 'Cancel subscription'}
            </button>
          </div>
        </div>
      )}
    </Modal>
  );
}
