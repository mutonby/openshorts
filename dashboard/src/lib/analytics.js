import { allows } from './consent';
import { firstTouchProps } from './attribution';

// Lightweight custom-event helper (OpenPanel).
//
// The hosted openshorts.app build loads OpenPanel (see index.html), which
// exposes `window.op`. Self-hosted builds, ad-blockers or offline dev simply
// won't have it — every call here is a safe no-op in that case, so analytics
// can never break the app or leak into the open-source experience.
//
// Events (name — meaning):
//   - Signup             — account created / signed in
//   - QuotaWallSeen      — the 402 wall modal opened
//   - UpsellModalSeen    — the voluntary upgrade modal opened
//   - QuotaWallCheckout  — a plan/top-up clicked inside the wall modal
//   - UpsellModalCheckout— a plan/top-up clicked inside the upsell modal
//   - QueueFullSeen      — the "free queue is full" wall opened (402
//                          free_queue_full; its checkouts carry source 'queue_full')
//   - QueueFullCheckout  — a plan/top-up clicked inside that wall
//   - QueueFullDismissed — "I'll come back later" clicked
//   - CheckoutStarted    — checkout clicked, on ANY surface (see `source` prop:
//                          'wall' | 'upsell' | 'pricing' | 'watermark')
//   - CheckoutRedirected — Stripe returned a URL and we are sending them there
//   - CheckoutFailed     — /api/billing/checkout errored, `reason` says why
//   - CheckoutCanceled   — back from Stripe unpaid (`back_to`: app | pricing)
//   - PartialClipChosen  — the wall's "clip the first N min" taken instead of a
//                          plan (`required` / `partial` minutes)
//   - FirstVideoGrant    — free account's first video (<= 60 min) clipped whole
//                          past its balance, no wall shown
//   - AutoPartial        — free source past the balance clipped to the first N
//                          minutes by the server, no wall shown
//   - Subscribed         — plan activated after checkout
//   - SocialNudgeSeen    — post-generation "connect socials" banner rendered
//   - SocialNudgeConnect — its connect button clicked (opens hosted connect page)
//   - SocialNudgeDismissed — its X clicked (persisted, never shown again)
//   - ClipTutorialStarted  — first-login tutorial: user hit Start
//   - ClipTutorialSkipped  — first-login tutorial dismissed (intro or coach)
//   - ClipTutorialCompleted— first Clip Generator job finished with clips
//   - JobResumedAfterSignin— a job started signed-out was replayed after the
//                          sign-in (App.jsx), instead of being dropped
// The Started → Redirected → Subscribed chain is what separates "never reached
// Stripe" from "reached Stripe and abandoned"; before 2-ago-2026 the modals
// emitted only their own *Checkout event and the difference was invisible.
// Prices ride along as ordinary props (e.g. value_usd) for breakdowns.
// Conversion events carry the visitor's first touch (landing page, referrer,
// campaign). A signup happens after the Google/magic-link redirect, in a new
// OpenPanel session whose entry page is always "/", so without these props
// every signup looked like it came from the homepage and the SEO pages showed
// zero conversions (7,481 of 7,481 signup_attribution rows in the 30 days to
// 23-sep-2026 said landing_path "/").
const FIRST_TOUCH_EVENTS = new Set(['Signup', 'CheckoutStarted', 'Subscribed']);

export function track(event, options) {
  try {
    // `op` is a queueing stub until consent loads op1.js, so a call made before
    // the visitor accepted would be flushed the moment they did. Check first.
    if (!allows('analytics')) return;
    if (typeof window !== 'undefined' && typeof window.op === 'function') {
      const props = (options && options.props) || {};
      window.op('track', event,
        FIRST_TOUCH_EVENTS.has(event) ? { ...firstTouchProps(), ...props } : props);
    }
  } catch (_) {
    /* analytics must never throw into the app */
  }
}

/**
 * Bind this browser to the signed-in account. `profileId` is the user's uuid —
 * the same id the backend uses for its server-side events (ClipsDelivered,
 * JobFailed and the `revenue` mirror of the Stripe webhook) — so a sale lands
 * on the profile that carries the first visit's referrer and campaign.
 * OpenPanel keeps the profile in memory only, so this runs on every boot.
 */
export function identify(user, props) {
  try {
    if (!allows('analytics')) return;
    if (!user || !user.id) return;
    if (typeof window !== 'undefined' && typeof window.op === 'function') {
      // profileId only. The email used to ride along here, which put a direct
      // identifier of every signed-in user into the analytics store and made
      // both the privacy policy ("no third-party trackers", "aggregate
      // measurement") and the deletion notice ("OpenPanel never received the
      // address") untrue. The uuid is enough to join a sale to a first visit,
      // and it is the same id the server-side events already use.
      window.op('identify', { profileId: String(user.id), ...(props || {}) });
    }
  } catch (_) {
    /* analytics must never throw into the app */
  }
}

/** Forget the current profile (sign-out). */
export function reset() {
  try {
    if (typeof window !== 'undefined' && typeof window.op === 'function') {
      window.op('clear');
    }
  } catch (_) {
    /* ignore */
  }
}
