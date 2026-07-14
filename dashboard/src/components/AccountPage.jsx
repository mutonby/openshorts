import React, { useState, useEffect, useCallback } from 'react';
import { Loader2, CreditCard, LogOut, Zap, Plus } from 'lucide-react';
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

  if (!me) return <div className="flex justify-center py-16"><Loader2 className="animate-spin text-primary" /></div>;

  const m = minutes || {};
  const total = (m.plan_allowance || 0) + (m.topup_remaining || 0) + (m.plan_used || 0);
  const usedPct = total > 0 ? Math.min(100, ((m.plan_used || 0) / (m.plan_allowance || 1)) * 100) : 0;

  return (
    <div className="max-w-2xl mx-auto space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Your account</h2>
          <p className="text-zinc-400 text-sm">{me.user?.email}</p>
        </div>
        <button onClick={logout} className="text-zinc-400 hover:text-white flex items-center gap-2 text-sm">
          <LogOut size={16} /> Sign out
        </button>
      </div>

      {activating && (
        <div className="bg-primary/10 border border-primary/30 rounded-xl p-4 text-sm flex items-center gap-2">
          <Loader2 size={16} className="animate-spin" /> Activating your plan…
        </div>
      )}

      <div className="bg-surface border border-white/10 rounded-2xl p-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <span className="text-lg font-semibold capitalize">{plan ? `${plan} plan` : 'No active plan'}</span>
            {me.status && me.status !== 'active' && (
              <span className="ml-2 text-xs px-2 py-0.5 rounded-full bg-yellow-500/20 text-yellow-400">{me.status}</span>
            )}
            {me.cancel_at_period_end && (
              <span className="ml-2 text-xs px-2 py-0.5 rounded-full bg-zinc-500/20 text-zinc-300">cancels at period end</span>
            )}
          </div>
          <button onClick={openPortal} disabled={busy}
            className="text-sm bg-white/10 hover:bg-white/20 px-4 py-2 rounded-lg flex items-center gap-2">
            <CreditCard size={16} /> Manage billing
          </button>
        </div>

        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-zinc-400">Plan minutes</span>
            <span>{fmt1(m.plan_used)} / {fmt1(m.plan_allowance)} used</span>
          </div>
          <div className="h-2 bg-white/10 rounded-full overflow-hidden">
            <div className="h-full bg-primary transition-all" style={{ width: `${usedPct}%` }} />
          </div>
          <div className="flex justify-between text-sm pt-1">
            <span className="text-zinc-400">Top-up minutes</span>
            <span>{fmt1(m.topup_remaining)} remaining</span>
          </div>
          <div className="flex justify-between text-base font-semibold pt-2 border-t border-white/5">
            <span>Total remaining</span>
            <span className="text-primary">{fmt1(m.remaining)} min</span>
          </div>
        </div>
      </div>

      {topups.length > 0 && (
        <div className="bg-surface border border-white/10 rounded-2xl p-6">
          <h3 className="font-semibold mb-1 flex items-center gap-2"><Plus size={16} /> Buy more minutes</h3>
          <p className="text-zinc-400 text-sm mb-4">Top-ups never expire while your plan is active.</p>
          <div className="grid grid-cols-2 gap-3">
            {topups.map((t) => (
              <button key={t.price_id} onClick={() => buyTopup(t.price_id)} disabled={busy}
                className="border border-white/10 hover:border-primary rounded-xl p-4 text-left transition-all">
                <div className="text-lg font-bold">+{t.minutes} min</div>
                <div className="text-zinc-400 text-sm">
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
