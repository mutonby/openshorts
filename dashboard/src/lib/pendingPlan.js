// A plan picked while signed out survives the sign-in: PricingSection stashes
// it, AuthContext sends the fresh session back to #/pricing, and the section
// opens that checkout on its own instead of dropping the visitor in the app
// with the choice forgotten. Stale picks are ignored.
const PENDING_PLAN_KEY = 'os_pending_plan';
const PENDING_PLAN_TTL_MS = 2 * 60 * 60 * 1000;

// source: the checkout-event source to keep through the sign-in (e.g. a link an
// API error handed out), so the resumed checkout is not counted as the page's.
export function stashPendingPlan(entry, source = null) {
  try {
    localStorage.setItem(PENDING_PLAN_KEY, JSON.stringify({
      price_id: entry.price_id, plan: entry.plan, interval: entry.interval,
      amount: entry.amount, currency: entry.currency, source, ts: Date.now(),
    }));
  } catch (_) { /* ignore storage errors */ }
}

export function readPendingPlan() {
  try {
    const raw = JSON.parse(localStorage.getItem(PENDING_PLAN_KEY) || 'null');
    if (raw?.price_id && Date.now() - (raw.ts || 0) < PENDING_PLAN_TTL_MS) return raw;
  } catch (_) { /* ignore */ }
  return null;
}

export function clearPendingPlan() {
  try { localStorage.removeItem(PENDING_PLAN_KEY); } catch (_) { /* ignore */ }
}
