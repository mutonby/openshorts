import React, { useState, useEffect } from 'react';
import { Check, Loader2, Zap, ArrowRight } from 'lucide-react';
import { apiJson } from '../lib/api';
import Modal from './ui/Modal';

const PLAN_ORDER = ['starter', 'creator', 'pro'];
const FREE_MINUTES = 20;
const fmt = (a, c) => new Intl.NumberFormat('en-US', { style: 'currency', currency: (c || 'usd').toUpperCase(), maximumFractionDigits: 0 }).format((a || 0) / 100);

// Welcome popup after sign-up, instead of dumping the user on the pricing page.
// Free is the default (any Google or permanent-email account qualifies), paid
// plans check out inline.
export default function PlanChoiceModal({ onClose }) {
  const [plans, setPlans] = useState([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(null);

  useEffect(() => {
    apiJson('/api/billing/plans')
      .then((d) => setPlans(d.plans || []))
      .catch(() => setPlans([]))
      .finally(() => setLoading(false));
  }, []);

  const byPlan = (p) => plans.find((x) => x.plan === p && x.interval === 'month');

  const checkout = async (price_id) => {
    setBusy(price_id);
    try {
      const { url } = await apiJson('/api/billing/checkout', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ price_id }),
      });
      window.location.href = url;
    } catch (e) { setBusy(null); alert(e?.detail || 'Could not start checkout. Please try again.'); }
  };

  const startFree = () => {
    // A signed-in user with a valid account is already on Free — just start.
    onClose();
  };

  return (
    <Modal isOpen onClose={onClose} eyebrow="WELCOME" title="Pick how you want to start" size="lg">
      {loading ? (
        <div className="flex justify-center py-10"><Loader2 className="animate-spin text-brass" /></div>
      ) : (
        <div className="grid sm:grid-cols-2 gap-4">
          {/* Free — the default */}
          <div className="card p-5 flex flex-col border-brass">
            <div className="flex items-center justify-between mb-1">
              <h3 className="font-display lowercase text-lg text-ink">free</h3>
              <span className="badge-ok">$0</span>
            </div>
            <p className="text-muted text-xs mb-3 lowercase">try it on your own videos</p>
            <ul className="space-y-1.5 text-sm text-ink2 mb-4 flex-1">
              <li className="flex items-start gap-2"><Check size={15} className="text-ok shrink-0 mt-0.5" /> <span><b>{FREE_MINUTES} min</b> / month</span></li>
              <li className="flex items-start gap-2"><Check size={15} className="text-ok shrink-0 mt-0.5" /> <span>No credit card</span></li>
              <li className="flex items-start gap-2"><Check size={15} className="text-muted shrink-0 mt-0.5" /> <span className="text-muted">Clips kept 7 days</span></li>
            </ul>
            <button onClick={startFree} className="w-full btn-primary text-sm">
              Start free <ArrowRight size={15} />
            </button>
            <p className="text-center text-[11px] text-muted mt-2 lowercase">no card · start clipping now</p>
          </div>

          {/* Paid — compact list */}
          <div className="card p-5 flex flex-col">
            <div className="flex items-center justify-between mb-1">
              <h3 className="font-display lowercase text-lg text-ink">paid plans</h3>
              <Zap size={16} className="text-brass" />
            </div>
            <p className="text-muted text-xs mb-3 lowercase">more minutes · durable library</p>
            <div className="space-y-2 flex-1">
              {PLAN_ORDER.map((p) => {
                const e = byPlan(p);
                if (!e) return null;
                return (
                  <button key={p} onClick={() => checkout(e.price_id)} disabled={busy === e.price_id}
                    className="w-full flex items-center justify-between border border-rule hover:border-brass rounded-input px-3 py-2 text-left transition-colors disabled:opacity-50">
                    <span className="text-sm text-ink capitalize">{p} <span className="text-muted">· {e.minutes} min</span></span>
                    <span className="readout">{busy === e.price_id ? '…' : `${fmt(e.amount, e.currency)}/mo`}</span>
                  </button>
                );
              })}
            </div>
            <button onClick={() => { onClose(); window.location.hash = '#/pricing'; }}
              className="text-center text-[11px] text-muted mt-3 lowercase hover:text-ink transition-colors">
              see full pricing & yearly →
            </button>
          </div>
        </div>
      )}
    </Modal>
  );
}
