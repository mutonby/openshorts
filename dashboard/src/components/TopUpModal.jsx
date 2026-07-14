import React, { useState, useEffect } from 'react';
import { X, Loader2, Zap } from 'lucide-react';
import { apiJson } from '../lib/api';

// Opens when a job hits a 402 (quota exceeded). Lets the user buy more minutes.
export default function TopUpModal({ onClose, required, remaining }) {
  const [topups, setTopups] = useState([]);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    apiJson('/api/billing/plans').then((d) => setTopups(d.topups || [])).catch(() => {});
  }, []);

  const buy = async (price_id) => {
    setBusy(true);
    try {
      const { url } = await apiJson('/api/billing/checkout', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ price_id }),
      });
      window.location.href = url;
    } catch (e) { setBusy(false); alert('Could not start checkout.'); }
  };

  const fmt = (a, c) => new Intl.NumberFormat('en-US', { style: 'currency', currency: (c || 'usd').toUpperCase(), maximumFractionDigits: 0 }).format((a || 0) / 100);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
      <div className="bg-surface border border-white/10 rounded-2xl p-8 w-full max-w-md relative">
        <button onClick={onClose} className="absolute right-4 top-4 text-zinc-400 hover:text-white"><X size={20} /></button>
        <div className="inline-flex p-3 bg-amber-500/20 rounded-full text-amber-400 mb-4"><Zap size={24} /></div>
        <h2 className="text-xl font-bold mb-1">You're out of minutes</h2>
        <p className="text-zinc-400 text-sm mb-6">
          {typeof required === 'number'
            ? <>This video needs <b>{required} min</b> but you have <b>{Math.round((remaining || 0) * 10) / 10} min</b> left.</>
            : 'Add more minutes to keep generating.'}
        </p>
        <div className="grid grid-cols-2 gap-3">
          {topups.map((t) => (
            <button key={t.price_id} onClick={() => buy(t.price_id)} disabled={busy}
              className="border border-white/10 hover:border-primary rounded-xl p-4 text-left transition-all disabled:opacity-50">
              <div className="text-lg font-bold">+{t.minutes} min</div>
              <div className="text-zinc-400 text-sm">{fmt(t.amount, t.currency)}</div>
            </button>
          ))}
          {topups.length === 0 && <div className="col-span-2 flex justify-center py-4"><Loader2 className="animate-spin text-primary" /></div>}
        </div>
        <p className="text-zinc-500 text-xs mt-4 text-center">Or upgrade your plan for more monthly minutes.</p>
      </div>
    </div>
  );
}
