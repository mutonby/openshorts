// Lightweight Plausible custom-event helper.
//
// The hosted openshorts.app build loads the Plausible script (see index.html),
// which exposes `window.plausible`. Self-hosted builds, ad-blockers or offline
// dev simply won't have it — every call here is a safe no-op in that case, so
// analytics can never break the app or leak into the open-source experience.
//
// This targets a Plausible Community Edition instance, which has NO revenue
// goals (that's a Cloud-only feature) — so prices ride along as ordinary props
// (e.g. value_usd) that CE can break a goal down by. Goals to create (by name):
//   - Signup           (custom event)  — account created / signed in
//   - CheckoutStarted  (custom event)  — redirected to Stripe Checkout
//   - Subscribed       (custom event)  — plan activated after checkout
export function track(event, options) {
  try {
    if (typeof window !== 'undefined' && typeof window.plausible === 'function') {
      window.plausible(event, options);
    }
  } catch (_) {
    /* analytics must never throw into the app */
  }
}
