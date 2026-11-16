import React, { useState, useEffect, useCallback } from 'react';
import { Loader2, CreditCard, LogOut, Plus } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';
import { apiJson } from '../lib/api';
import { track } from '../lib/analytics';
import { useBackFromStripe } from '../lib/backFromStripe';
import ApiKeysCard from './ApiKeysCard';
import McpConnectCard from './McpConnectCard';
import DeleteAccountCard from './DeleteAccountCard';
import SocialAnalyticsCard from './SocialAnalyticsCard';
import InvoicesCard from './InvoicesCard';
import CancelPlanModal from './CancelPlanModal';

const fmt1 = (n) => Math.round((n || 0) * 10) / 10;

// Mirrors cloud/config.BILLING_ATTENTION_STATES: states where the customer has
// to do something (fix a card, finish a payment) before minutes come back.
const PAYMENT_ISSUE_STATES = ['past_due', 'unpaid', 'incomplete', 'paused'];
// Mirrors cloud/cancellation.CANCELLABLE_STATES.
const CANCELLABLE_STATES = ['active', 'trialing', 'past_due', 'on_hold'];
const fmtDay = (iso) => (iso ? new Date(iso).toLocaleDateString('en-US', { month: 'long', day: 'numeric' }) : 'the end of the period');

// Account/billing page: plan, usage meter, top-ups, manage billing, logout.
export default function AccountPage() {
  const { me, refreshMe, logout, plan, minutes } = useAuth();
  const [busy, setBusy] = useState(false);
  const [topups, setTopups] = useState([]);
  const [activating, setActivating] = useState(false);
  const [cancelOpen, setCancelOpen] = useState(false);

  // After returning from Checkout the webhook may lag — poll /api/me briefly.
  useEffect(() => {
    const hash = window.location.hash || '';
    if (!hash.includes('checkout=success')) return;
    setActivating(true);
    let tries = 0;
    const t = setInterval(async () => {
      tries += 1;
      const data = await refreshMe();
      // 'free' is the default plan for any signed-in account, so it does NOT
      // mean the checkout landed — keep polling until the webhook writes the
      // paid subscription.
      const paidPlan = data?.plan && data.plan !== 'free';
      if (paidPlan || tries > 15) {
        clearInterval(t);
        setActivating(false);
        // Fire the Subscribed conversion goal. A pending-checkout stash (set in
        // PricingSection) carries the plan price, so we can attach real revenue;
        // top-ups don't set it, so they never count as a subscription.
        if (paidPlan) {
          let pending = null;
          try { pending = JSON.parse(localStorage.getItem('os_pending_checkout') || 'null'); } catch (_) { /* ignore */ }
          if (pending) {
            // This Plausible is Community Edition, which has no revenue goals —
            // so the price rides along as plain props (value_usd / plan) that CE
            // can break the goal down by. The exact MRR still lives in Stripe.
            track('Subscribed', {
              props: {
                plan: pending.plan,
                interval: pending.interval,
                value_usd: Math.round((pending.amount || 0) / 100),
              },
            });
            try { localStorage.removeItem('os_pending_checkout'); } catch (_) { /* ignore */ }
          }
        }
        // First time a plan activates, take the user straight to connect their
        // socials. Guard with a flag so top-up checkouts don't re-trigger it.
        if (paidPlan && !localStorage.getItem('os_socials_prompted')) {
          localStorage.setItem('os_socials_prompted', '1');
          try {
            const { access_url } = await apiJson('/api/social/connect', { method: 'POST' });
            if (access_url) { window.location.href = access_url; return; }
          } catch (_) { /* fall through to the account page */ }
        }
        window.location.hash = '#/account';
      }
    }, 2000);
    return () => clearInterval(t);
  }, [refreshMe]);

  useEffect(() => {
    apiJson('/api/billing/plans').then((d) => setTopups(d.topups || [])).catch(() => {});
  }, []);

  const openPortal = useCallback(async () => {
    setBusy(true);
    try {
      const { url } = await apiJson('/api/billing/portal', { method: 'POST' });
      window.location.href = url;
    } catch (e) { setBusy(false); alert('Could not open billing portal.'); }
  }, []);

  // The portal no longer cancels, and its "renew" went with that, so undoing
  // a scheduled cancel happens here (cloud/cancellation.py).
  const resume = useCallback(async () => {
    setBusy(true);
    try {
      await apiJson('/api/billing/resume', { method: 'POST' });
      track('SubscriptionResumed', { props: { plan } });
      await refreshMe();
    } catch (e) { alert(e?.detail || 'Could not resume your subscription.'); }
    setBusy(false);
  }, [plan, refreshMe]);

  const unpause = useCallback(async () => {
    setBusy(true);
    try {
      const out = await apiJson('/api/billing/unpause', { method: 'POST' });
      track('SubscriptionUnpaused', { props: { plan, charged: !!out?.charged } });
      // A restart charges a new period; the plan comes back with the webhook.
      if (out?.charged) await new Promise((r) => setTimeout(r, 2500));
      await refreshMe();
    } catch (e) { alert(e?.detail || 'Could not restart your plan.'); }
    setBusy(false);
  }, [plan, refreshMe]);

  const buyTopup = useCallback(async (price_id) => {
    setBusy(true);
    const props = { kind: 'topup', source: 'account_topup' };
    track('CheckoutStarted', { props });
    try {
      const { url } = await apiJson('/api/billing/checkout', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ price_id, source: props.source }),
      });
      track('CheckoutRedirected', { props });
      window.location.href = url;
    } catch (e) {
      track('CheckoutFailed', { props: { ...props, reason: String(e?.detail || e?.message || 'unknown').slice(0, 120) } });
      setBusy(false);
      alert(e?.detail || 'Could not start checkout.');
    }
  }, []);
  useBackFromStripe(() => setBusy(false));

  if (!me) return <div className="flex justify-center py-16"><Loader2 className="animate-spin text-brass" /></div>;

  const m = minutes || {};
  const total = (m.plan_allowance || 0) + (m.topup_remaining || 0) + (m.plan_used || 0);
  const usedPct = total > 0 ? Math.min(100, ((m.plan_used || 0) / (m.plan_allowance || 1)) * 100) : 0;
  const low = total > 0 && (m.remaining || 0) <= total * 0.2;

  return (
    <div className="max-w-2xl mx-auto space-y-6">
      <div className="flex items-end justify-between gap-4">
        <div>
          <p className="eyebrow mb-1.5">ACCOUNT</p>
          <h2 className="font-display lowercase text-2xl text-ink leading-tight">Your account</h2>
          <p className="text-muted text-sm mt-1">{me.user?.email}</p>
        </div>
        <button onClick={logout} className="btn-quiet shrink-0">
          <LogOut size={16} /> Sign out
        </button>
      </div>

      {activating && (
        <div className="card px-4 py-3 text-sm text-ink2 flex items-center gap-2 lowercase">
          <Loader2 size={16} className="animate-spin text-brass" /> Activating your plan…
        </div>
      )}

      <div className="card p-6">
        <div className="flex items-center justify-between gap-4 mb-4">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-ink font-medium capitalize">{plan ? `${plan} plan` : 'No active plan'}</span>
            {/* Payment-issue states get the explanatory banner below instead —
                showing raw Stripe jargon ("past_due") twice explains nothing. */}
            {me.status && me.status !== 'active' && me.status !== 'on_hold'
              && !PAYMENT_ISSUE_STATES.includes(me.status) && (
              <span className="badge-warn">{me.status}</span>
            )}
            {me.cancel_at_period_end && (
              <span className="badge-warn">cancels at period end</span>
            )}
          </div>
          {/* Any Stripe relationship — live OR broken (past_due, unpaid) — must
              keep the portal reachable. A declined card demotes `plan` to free,
              so keying this off the plan alone hid the one button that fixes it. */}
          {me.has_billing_account ? (
            <button onClick={openPortal} disabled={busy} className="btn-ghost px-4 py-2 shrink-0">
              <CreditCard size={16} /> Manage billing
            </button>
          ) : (
            <button onClick={() => { window.location.hash = '#/pricing'; }} className="btn-primary px-4 py-2 shrink-0 text-xs">
              Upgrade
            </button>
          )}
        </div>

        {/* A bare "past_due" chip explains nothing. Say what happened and what
            fixes it — this is the whole reason the account reads as free. */}
        {PAYMENT_ISSUE_STATES.includes(me.status) && (
          <div className="mb-4 rounded-card border border-warn/40 bg-warn/5 p-3 text-sm text-ink2">
            <b className="text-ink">We couldn't charge your card.</b>{' '}
            {me.status === 'incomplete'
              ? 'Your payment was never completed, so the plan never started.'
              : 'Your plan is paused and you\'re on free minutes until it goes through.'}{' '}
            <button onClick={openPortal} disabled={busy}
                    className="underline underline-offset-2 hover:text-ink">
              Update your card
            </button>{' '}
            and it resumes right away.
          </div>
        )}

        {/* Month on hold from the cancel flow (cloud/cancellation). */}
        {(me.status === 'on_hold' || me.hold_scheduled) && (
          <div className="mb-4 rounded-card border border-brass/40 bg-brass/5 p-3 text-sm text-ink2 flex items-center justify-between gap-3">
            <span>
              {me.status === 'on_hold'
                ? <>Your plan is paused until <b className="text-ink">{fmtDay(me.hold_until)}</b>. You're on free minutes and your clips are kept.</>
                : <>Your plan skips next month from <b className="text-ink">{fmtDay(me.period_end)}</b>, with no charge.</>}
            </span>
            <button onClick={unpause} disabled={busy} className="btn-ghost px-4 py-2 text-xs shrink-0">
              {me.status === 'on_hold' ? 'Restart now' : 'Undo pause'}
            </button>
          </div>
        )}

        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-muted lowercase">Plan minutes</span>
            <span className="text-ink2">{fmt1(Math.max(0, (m.plan_used || 0) - (m.plan_held || 0)))} / {fmt1(m.plan_allowance)} used</span>
          </div>
          <div className="h-1.5 bg-paper3 rounded-full overflow-hidden">
            <div className={`h-full transition-all ${low ? 'bg-warn' : 'bg-brass'}`} style={{ width: `${usedPct}%` }} />
          </div>
          {/* A queued or running video holds its minutes until it ends: spent
              only if it completes, back otherwise. Not "used" yet. */}
          {(m.held || 0) > 0 && (
            <div className="flex justify-between gap-3 text-xs text-muted">
              <span>Held for videos still in progress</span>
              <span className="text-right">{fmt1(m.held)} min · back if they don't finish</span>
            </div>
          )}
          <div className="flex justify-between text-sm pt-1">
            <span className="text-muted lowercase">Top-up minutes</span>
            <span className="text-ink2">{fmt1(m.topup_remaining)} remaining</span>
          </div>
          <div className="flex justify-between text-sm pt-2 border-t border-rule">
            <span className="text-ink font-medium lowercase">Total remaining</span>
            <span className="text-brass font-medium">{fmt1(m.remaining)} min</span>
          </div>
        </div>

        {/* Our own cancel flow (reason + review + retention offer). The portal
            cannot cancel (turned off in Stripe), so this is the only way out. */}
        {me.has_billing_account && CANCELLABLE_STATES.includes(me.status) && !me.cancel_at_period_end && (
          <div className="pt-4 mt-4 border-t border-rule text-right">
            <button onClick={() => { track('CancelFlowOpened', { props: { plan } }); setCancelOpen(true); }}
                    className="text-xs text-muted underline underline-offset-2 hover:text-ink">
              Cancel subscription
            </button>
          </div>
        )}
        {me.has_billing_account && CANCELLABLE_STATES.includes(me.status) && me.cancel_at_period_end && (
          <div className="pt-4 mt-4 border-t border-rule flex items-center justify-between gap-3 text-sm text-ink2">
            <span>Your plan ends on {me.period_end ? new Date(me.period_end).toLocaleDateString('en-US', { month: 'long', day: 'numeric' }) : 'the end of the period'}.</span>
            <button onClick={resume} disabled={busy} className="btn-ghost px-4 py-2 text-xs shrink-0">
              Keep my subscription
            </button>
          </div>
        )}
      </div>

      {cancelOpen && (
        <CancelPlanModal
          plan={plan}
          periodEnd={me.period_end}
          onClose={() => setCancelOpen(false)}
          onCanceled={() => { setCancelOpen(false); refreshMe(); }}
        />
      )}

      {/* Only accounts that ever had a Stripe relationship can have invoices. */}
      {me.has_billing_account && <InvoicesCard />}

      <SocialAnalyticsCard />

      {topups.length > 0 && (
        <div className="card p-6">
          <h3 className="font-display lowercase text-lg text-ink mb-1 flex items-center gap-2"><Plus size={16} className="text-brass" /> Buy more minutes</h3>
          <p className="text-muted text-sm mb-4 lowercase">Top-ups never expire while your plan is active.</p>
          <div className="grid grid-cols-2 gap-3">
            {topups.map((t) => (
              <button key={t.price_id} onClick={() => buyTopup(t.price_id)} disabled={busy}
                className="border border-rule hover:border-brass rounded-card p-4 text-left transition-colors disabled:opacity-50">
                <div className="text-ink font-medium">+{t.minutes} min</div>
                <div className="readout mt-1">
                  {new Intl.NumberFormat('en-US', { style: 'currency', currency: (t.currency || 'usd').toUpperCase(), maximumFractionDigits: 0 }).format((t.amount || 0) / 100)}
                </div>
              </button>
            ))}
          </div>
        </div>
      )}

      <McpConnectCard cloud />

      <ApiKeysCard />

      <DeleteAccountCard />
    </div>
  );
}
