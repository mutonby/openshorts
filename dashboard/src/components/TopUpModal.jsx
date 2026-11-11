import React, { useState, useEffect } from 'react';
import { Loader2, Check, Zap, Clock } from 'lucide-react';
import { apiJson } from '../lib/api';
import { track } from '../lib/analytics';
import { useBackFromStripe } from '../lib/backFromStripe';
import Modal from './ui/Modal';

// Opens when a job hits a 402 (quota exceeded). This is the highest-intent
// moment in the product — the user WANTS to clip a video and can't — so it
// sells the subscription first (recurring value: minutes, no watermark,
// permanent clips) and keeps one-time top-ups as a secondary escape hatch.
//
// Conversion techniques used (all honest — no fake scarcity or countdowns):
// - Goal-gradient: copy centers on finishing THIS video, not abstract quota.
// - Compromise effect: all three tiers shown, creator highlighted in the
//   middle — starter floors the price, pro anchors it, creator reads as the
//   sensible pick (and matches the pricing page's "most popular").
// - Outcome framing: every line says what the user GETS (clips ready to post,
//   a library that stays), with the free-plan fact stated second. The earlier
//   copy led with what free costs them, which reads as "pay to undo the
//   restrictions we added" — the losing side of an argument against our own
//   MIT repo, whose headline is literally "no watermarks, no limits".
// - Risk reversal: "cancel anytime" on the CTA.
//
// Deliberately NOT sold here: GPU rendering and the included Gemini key. Both
// are real and both are why the cloud beats self-hosting, but the free plan
// already gets them (the pricing page says "Same GPU rendering" on the free
// tier) — so they belong on the landing/pricing surface facing repo visitors,
// not in a modal shown to someone who is already using them.
const PLAN_BLURBS = {
  starter: 'for getting started',
  creator: 'for daily posting',
  pro: 'for power users',
};

// context: 'wall' (default) = user hit the 402 quota wall mid-task.
//          'upsell' = user opened it voluntarily (results banner, header meter)
//          — different framing: they still HAVE minutes, sell the watermark
//          removal + permanence instead of "you ran out".
// partialMinutes / onPartial: the server's offer to clip only the first N
// minutes of the blocked video on the minutes the user has (see
// app.partial_offer). Most walls open on an account that has not spent a
// single free minute and pasted a 21-90 min video (93 of 99 sampled,
// 16-sep-2026): they are being asked to pay before seeing one clip. The offer
// turns that wall into a first run; the plans stay the way to the whole video.
// source: the surface that opened it, when it is not the wall or the generic
// upsell (WatermarkModal passes 'watermark'), so CheckoutStarted can be split
// by origin. highlight: the plan put forward; the watermark notice asked about
// the cheapest way out, so it opens on starter instead of creator.
// context 'queue_full': the free queue's brake refused the submission (402
// free_queue_full; nothing was reserved or spent). Its own events
// (QueueFullSeen / QueueFullCheckout / QueueFullDismissed, plus the shared
// CheckoutStarted/Redirected with source 'queue_full') keep it out of the quota
// wall's numbers. The copy never says why the queue is full for this user.
// partialJob: { processed_minutes, total_minutes } of the job on screen when the
// "clip the whole video" banner opened it, so the copy talks about the rest of
// that video instead of the watermark.
export default function TopUpModal({ onClose, required, remaining, partialMinutes = 0, onPartial = null,
                                     context = 'wall', source: sourceOverride = null,
                                     highlight: highlightPlan = 'creator', partialJob = null }) {
  const [plans, setPlans] = useState([]);
  const [topups, setTopups] = useState([]);
  const [showTopups, setShowTopups] = useState(false);
  const [busyPrice, setBusyPrice] = useState(null);
  useBackFromStripe(() => setBusyPrice(null));
  const isUpsell = context === 'upsell';
  const isQueueFull = context === 'queue_full';
  const source = sourceOverride || (isUpsell ? 'upsell' : isQueueFull ? 'queue_full' : 'wall');
  const seenEvent = isUpsell ? 'UpsellModalSeen' : isQueueFull ? 'QueueFullSeen' : 'QuotaWallSeen';
  const checkoutEvent = isUpsell ? 'UpsellModalCheckout' : isQueueFull ? 'QueueFullCheckout' : 'QuotaWallCheckout';

  useEffect(() => {
    track(seenEvent,
          { props: { required: required ?? null, remaining: remaining ?? null, source } });
    apiJson('/api/billing/plans')
      .then((d) => {
        const monthly = (d.plans || []).filter((p) => p.interval === 'month');
        setPlans(['starter', 'creator', 'pro']
          .map((name) => monthly.find((p) => p.plan === name))
          .filter(Boolean));
        setTopups(d.topups || []);
      })
      .catch(() => {});
    // Plans are fetched once per open; the props only shape the copy.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Instrumented in three steps on purpose. Between 25-jul and 1-ago the modals
  // logged 35 checkout clicks and Stripe recorded ONE new subscription, and
  // nothing in between was measured — this modal fired only its own
  // *ModalCheckout event, so "clicked but never reached Stripe" and "reached
  // Stripe and abandoned" were indistinguishable. CheckoutStarted (click, same
  // meaning as PricingSection's) → CheckoutRedirected (we hold a Stripe URL) →
  // CheckoutFailed (we don't) separates them.

  const buy = async (entry, kind) => {
    setBusyPrice(entry.price_id);
    const props = { kind, plan: entry.plan || null, minutes: entry.minutes, source };
    track(checkoutEvent, { props });
    track('CheckoutStarted', { props });
    // Same stash PricingSection sets, so a plan bought from the modal also
    // carries its price into the Subscribed goal (AccountPage reads it back).
    // Top-ups deliberately don't set it — they never count as a subscription.
    if (kind === 'subscription') {
      try {
        localStorage.setItem('os_pending_checkout', JSON.stringify({
          plan: entry.plan, interval: entry.interval, amount: entry.amount,
          currency: entry.currency,
        }));
      } catch (_) { /* ignore storage errors */ }
    }
    try {
      const { url } = await apiJson('/api/billing/checkout', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ price_id: entry.price_id, source }),
      });
      track('CheckoutRedirected', { props });
      window.location.href = url;
    } catch (e) {
      track('CheckoutFailed', { props: { ...props, reason: String(e?.detail || e?.message || 'unknown').slice(0, 120) } });
      setBusyPrice(null);
      alert(e?.detail || 'Could not start checkout.');
    }
  };

  const fmt = (a, c) => new Intl.NumberFormat('en-US', {
    style: 'currency', currency: (c || 'usd').toUpperCase(), maximumFractionDigits: 0,
  }).format((a || 0) / 100);

  // A video longer than the minutes left. With under a minute left the 402
  // comes before the probe (no duration yet, minutes_required is a floor of 1),
  // so that case reads as "out of minutes", not "this video is 1 min long".
  const blockedByLength = typeof required === 'number' && typeof remaining === 'number'
    && remaining >= 1 && required > remaining;
  const partialOffer = !isUpsell && !isQueueFull && partialMinutes > 0 && typeof onPartial === 'function';
  const remainingShown = Math.max(0, Math.round((remaining || 0) * 10) / 10);

  return (
    <Modal isOpen onClose={onClose} eyebrow="UPGRADE"
           title={isQueueFull ? 'The free queue is full right now'
             : isUpsell
             ? (partialJob ? 'Clip the whole video' : 'Keep your clips forever')
             : (blockedByLength ? 'Your video is ready to clip' : "You've used this month's minutes")}
           size="xl">
      {/* Goal-gradient framing: they're one step from the thing they came for. */}
      <p className="text-muted text-sm mb-5">
        {isQueueFull
          ? <>Come back later, or skip the line with a plan: paid videos go to the{' '}
              <b className="text-ink font-medium">front of the queue</b>. Nothing was charged and your
              free minutes are still yours.</>
          : isUpsell
          ? (partialJob
            ? <>These clips come from the first <b className="text-ink font-medium">{partialJob.processed_minutes} min</b>{' '}
                of a {partialJob.total_minutes}-min video. A plan gives you the minutes to clip{' '}
                <b className="text-ink font-medium">all of it</b>, with no watermark, and the clips you already
                made lose the mark on the spot.</>
            : <>Every clip comes out <b className="text-ink font-medium">ready to post</b> and stays in your
                library for good. On the free plan they carry a watermark and are deleted after 7 days;
                upgrade now and the clips you already made <b className="text-ink font-medium">lose the mark on the spot</b>.</>)
          : blockedByLength
            ? <>This video is <b className="text-ink font-medium">{required} min</b> long and you have{' '}
                <b className="text-ink font-medium">{remainingShown} min</b> left this month. Pick a plan and
                clip the whole video.</>
            : <>Your minutes renew next month. Pick a plan to keep clipping now.</>}
      </p>

      {isQueueFull && (
        <div className="mb-5 flex justify-end">
          <button onClick={() => { track('QueueFullDismissed', { props: { source } }); onClose(); }}
                  disabled={busyPrice !== null} className="btn-ghost whitespace-nowrap">
            I'll come back later
          </button>
        </div>
      )}

      {partialOffer && (
        <div className="mb-5 card p-4 flex flex-col sm:flex-row sm:items-center gap-3">
          <div className="flex-1 text-sm text-ink2">
            <b className="text-ink font-medium">Or see it first:</b> clip the first{' '}
            <b className="text-ink font-medium">{partialMinutes} min</b> now with the minutes you have,
            and decide with the clips in front of you.
          </div>
          <button onClick={onPartial} disabled={busyPrice !== null}
                  className="btn-ghost whitespace-nowrap">
            Clip the first {partialMinutes} min
          </button>
        </div>
      )}

      <div className="grid sm:grid-cols-3 gap-3">
        {plans.map((entry) => {
          const highlight = entry.plan === highlightPlan;
          return (
            <div key={entry.price_id}
                 className={`relative card p-5 flex flex-col ${highlight ? 'border-brass' : ''}`}>
              {highlight && (
                <span className="absolute -top-3 left-1/2 -translate-x-1/2 badge-float">
                  {entry.plan === 'creator' ? 'Most popular' : 'Removes the mark'}
                </span>
              )}
              <h3 className="font-display lowercase text-lg text-ink">{entry.plan}</h3>
              <p className="text-muted text-xs mb-3 lowercase">{PLAN_BLURBS[entry.plan] || ''}</p>
              <div className="mb-3 flex items-baseline gap-1.5">
                <span className="font-display text-3xl text-ink tabular-nums">{fmt(entry.amount, entry.currency)}</span>
                <span className="readout">/mo</span>
              </div>
              <ul className="space-y-1.5 text-sm text-ink2 mb-4 flex-1">
                <li className="flex items-start gap-2">
                  <Check size={15} className="text-ok shrink-0 mt-0.5" />
                  <span><b>{entry.minutes} min</b> every month ({Math.round(entry.minutes / 20)}× your free quota)</span>
                </li>
                <li className="flex items-start gap-2">
                  <Check size={15} className="text-ok shrink-0 mt-0.5" />
                  <span>Clips <b>ready to post</b>, with no watermark</span>
                </li>
                <li className="flex items-start gap-2">
                  <Check size={15} className="text-ok shrink-0 mt-0.5" />
                  <span>Your library <b>stays</b> (free clips delete after 7 days)</span>
                </li>
                <li className="flex items-start gap-2">
                  <Zap size={15} className="text-brass shrink-0 mt-0.5" />
                  <span>Skips the free queue</span>
                </li>
              </ul>
              <button onClick={() => buy(entry, 'subscription')} disabled={busyPrice !== null}
                      className={`w-full ${highlight ? 'btn-primary' : 'btn-ghost'}`}>
                {busyPrice === entry.price_id
                  ? <Loader2 size={18} className="animate-spin" />
                  : `Get ${entry.plan}`}
              </button>
              <p className="text-center text-xs text-muted mt-2 lowercase">cancel anytime.</p>
            </div>
          );
        })}
        {plans.length === 0 && (
          <div className="sm:col-span-3 flex justify-center py-8">
            <Loader2 className="animate-spin text-brass" />
          </div>
        )}
      </div>

      {/* Secondary escape hatch: one-time packs, deliberately de-emphasized. */}
      <div className="mt-4 text-center">
        {!showTopups ? (
          <button onClick={() => setShowTopups(true)}
                  className="text-xs text-muted underline underline-offset-2 lowercase hover:text-ink2">
            just need a few extra minutes? one-time packs
          </button>
        ) : (
          <div className="grid grid-cols-2 gap-3 mt-2">
            {topups.map((t) => (
              <button key={t.price_id} onClick={() => buy(t, 'topup')} disabled={busyPrice !== null}
                      className="border border-rule hover:border-brass rounded-card p-3 text-left transition-colors disabled:opacity-50">
                <div className="text-ink text-sm font-medium flex items-center gap-1.5">
                  <Clock size={14} className="text-muted" />+{t.minutes} min
                </div>
                <div className="readout mt-0.5">{fmt(t.amount, t.currency)} · one-time</div>
              </button>
            ))}
            {topups.length === 0 && (
              <div className="col-span-2 flex justify-center py-3">
                <Loader2 className="animate-spin text-brass" size={18} />
              </div>
            )}
          </div>
        )}
      </div>
    </Modal>
  );
}
