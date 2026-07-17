import React, { useState, useEffect } from 'react';
import { Loader2, Zap } from 'lucide-react';
import { apiJson } from '../lib/api';
import Modal from './ui/Modal';

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
    <Modal isOpen onClose={onClose} eyebrow="QUOTA" title="You're out of minutes" size="md">
      <div className="flex items-start gap-3 mb-5">
        <div className="p-2 bg-paper3 rounded-input text-warn shrink-0"><Zap size={18} /></div>
        <p className="text-muted text-sm pt-1">
          {typeof required === 'number'
            ? <>This video needs <b className="text-ink font-medium">{required} min</b> but you have <b className="text-ink font-medium">{Math.round((remaining || 0) * 10) / 10} min</b> left.</>
            : 'Add more minutes to keep generating.'}
        </p>
      </div>
      <div className="grid grid-cols-2 gap-3">
        {topups.map((t) => (
          <button key={t.price_id} onClick={() => buy(t.price_id)} disabled={busy}
            className="border border-rule hover:border-brass rounded-card p-4 text-left transition-colors disabled:opacity-50">
            <div className="text-ink font-medium">+{t.minutes} min</div>
            <div className="readout mt-1">{fmt(t.amount, t.currency)}</div>
          </button>
        ))}
        {topups.length === 0 && <div className="col-span-2 flex justify-center py-4"><Loader2 className="animate-spin text-brass" /></div>}
      </div>
      <p className="text-muted text-xs mt-4 text-center lowercase">Or upgrade your plan for more monthly minutes.</p>
    </Modal>
  );
}
