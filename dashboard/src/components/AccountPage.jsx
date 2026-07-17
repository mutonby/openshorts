import React, { useState, useEffect, useCallback } from 'react';
import { Loader2, CreditCard, LogOut, Plus } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';
import { apiJson } from '../lib/api';

const fmt1 = (n) => Math.round((n || 0) * 10) / 10;

// Account/billing page: plan, usage meter, top-ups, manage billing, logout.
export default function AccountPage() {
  const { me, refreshMe, logout, plan, minutes } = useAuth();
  const [busy, setBusy] = useState(false);
  const [topups, setTopups] = useState([]);
  const [activating, setActivating] = useState(false);

  // After returning from Checkout the webhook may lag — poll /api/me briefly.
  useEffect(() => {
    const hash = window.location.hash || '';
    if (!hash.includes('checkout=success')) return;
    setActivating(true);
    let tries = 0;
    const t = setInterval(async () => {
      tries += 1;
      const data = await refreshMe();
      if ((data && data.plan) || tries > 15) {
        clearInterval(t);
        setActivating(false);
        // First time a plan activates, take the user straight to connect their
        // socials. Guard with a flag so top-up checkouts don't re-trigger it.
        if (data && data.plan && !localStorage.getItem('os_socials_prompted')) {
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

  const buyTopup = useCallback(async (price_id) => {
    setBusy(true);
    try {
      const { url } = await apiJson('/api/billing/checkout', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ price_id }),
      });
      window.location.href = url;
    } catch (e) { setBusy(false); alert('Could not start checkout.'); }
  }, []);

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
            {me.status && me.status !== 'active' && (
              <span className="badge-warn">{me.status}</span>
            )}
            {me.cancel_at_period_end && (
              <span className="badge-warn">cancels at period end</span>
            )}
          </div>
          <button onClick={openPortal} disabled={busy} className="btn-ghost px-4 py-2 shrink-0">
            <CreditCard size={16} /> Manage billing
          </button>
        </div>

        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-muted lowercase">Plan minutes</span>
            <span className="text-ink2">{fmt1(m.plan_used)} / {fmt1(m.plan_allowance)} used</span>
          </div>
          <div className="h-1.5 bg-paper3 rounded-full overflow-hidden">
            <div className={`h-full transition-all ${low ? 'bg-warn' : 'bg-brass'}`} style={{ width: `${usedPct}%` }} />
          </div>
          <div className="flex justify-between text-sm pt-1">
            <span className="text-muted lowercase">Top-up minutes</span>
            <span className="text-ink2">{fmt1(m.topup_remaining)} remaining</span>
          </div>
          <div className="flex justify-between text-sm pt-2 border-t border-rule">
            <span className="text-ink font-medium lowercase">Total remaining</span>
            <span className="text-brass font-medium">{fmt1(m.remaining)} min</span>
          </div>
        </div>
      </div>

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
    </div>
  );
}
