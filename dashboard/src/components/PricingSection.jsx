import React, { useState, useEffect } from 'react';
import { Check, Loader2, Zap, Github, Server } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';
import { apiJson } from '../lib/api';

const PLAN_ORDER = ['starter', 'creator', 'pro'];
const PLAN_BLURB = {
  starter: 'For getting started',
  creator: 'For regular creators',
  pro: 'For power users & teams',
};
const HIGHLIGHT = 'creator';

const fmt = (amount, currency) =>
  new Intl.NumberFormat('en-US', { style: 'currency', currency: (currency || 'usd').toUpperCase(), maximumFractionDigits: 0 }).format((amount || 0) / 100);

// 3-tier pricing with monthly/annual toggle. Checkout requires sign-in.
export default function PricingSection({ onRequireLogin }) {
  const { isSignedIn } = useAuth();
  const [plans, setPlans] = useState([]);
  const [interval, setInterval] = useState('month');
  const [loading, setLoading] = useState(true);
  const [busyPrice, setBusyPrice] = useState(null);

  useEffect(() => {
    apiJson('/api/billing/plans')
      .then((d) => setPlans(d.plans || []))
      .catch(() => setPlans([]))
      .finally(() => setLoading(false));
  }, []);

  const checkout = async (price_id) => {
    if (!isSignedIn) { onRequireLogin?.(price_id); return; }
    setBusyPrice(price_id);
    try {
      const { url } = await apiJson('/api/billing/checkout', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ price_id }),
      });
      window.location.href = url;
    } catch (e) {
      setBusyPrice(null);
      alert('Could not start checkout. Please try again.');
    }
  };

  if (loading) {
    return <div className="flex justify-center py-16"><Loader2 className="animate-spin text-primary" /></div>;
  }

  const byPlan = (plan) => plans.find((p) => p.plan === plan && p.interval === interval);

  return (
    <div className="max-w-5xl mx-auto">
      <div className="flex items-center justify-center gap-3 mb-10">
        <span className={interval === 'month' ? 'text-white' : 'text-zinc-500'}>Monthly</span>
        <button
          onClick={() => setInterval(interval === 'month' ? 'year' : 'month')}
          className="relative w-14 h-7 rounded-full bg-white/10 transition-colors"
        >
          <span className={`absolute top-1 w-5 h-5 rounded-full bg-primary transition-all ${interval === 'year' ? 'left-8' : 'left-1'}`} />
        </button>
        <span className={interval === 'year' ? 'text-white' : 'text-zinc-500'}>
          Yearly <span className="text-green-400 text-sm">· 2 months free</span>
        </span>
      </div>

      <div className="grid md:grid-cols-3 gap-6">
        {PLAN_ORDER.map((plan) => {
          const entry = byPlan(plan);
          if (!entry) return null;
          const highlight = plan === HIGHLIGHT;
          return (
            <div
              key={plan}
              className={`relative rounded-2xl p-6 border flex flex-col ${highlight ? 'border-primary bg-primary/5 shadow-xl shadow-primary/10' : 'border-white/10 bg-surface'}`}
            >
              {highlight && (
                <span className="absolute -top-3 left-1/2 -translate-x-1/2 bg-primary text-white text-xs font-semibold px-3 py-1 rounded-full">
                  Most popular
                </span>
              )}
              <h3 className="text-xl font-bold capitalize">{plan}</h3>
              <p className="text-zinc-400 text-sm mb-4">{PLAN_BLURB[plan]}</p>
              <div className="mb-4">
                <span className="text-4xl font-bold">{fmt(entry.amount, entry.currency)}</span>
                <span className="text-zinc-400">/{interval === 'month' ? 'mo' : 'yr'}</span>
              </div>
              <ul className="space-y-2 text-sm mb-6 flex-1">
                <li className="flex items-center gap-2"><Check size={16} className="text-green-400" /> <b>{entry.minutes} min</b> of video / month</li>
                <li className="flex items-center gap-2"><Check size={16} className="text-green-400" /> No API keys needed</li>
                <li className="flex items-center gap-2"><Check size={16} className="text-green-400" /> Gemini + auto-posting included</li>
                {plan === 'pro' && <li className="flex items-center gap-2"><Zap size={16} className="text-yellow-400" /> Priority processing queue</li>}
              </ul>
              <button
                onClick={() => checkout(entry.price_id)}
                disabled={busyPrice === entry.price_id}
                className={`w-full py-3 rounded-xl font-medium transition-all flex items-center justify-center gap-2 ${highlight ? 'bg-primary hover:bg-blue-600 text-white' : 'bg-white/10 hover:bg-white/20 text-white'}`}
              >
                {busyPrice === entry.price_id ? <Loader2 size={18} className="animate-spin" /> : 'Start 3-day free trial'}
              </button>
              <p className="text-center text-[11px] text-zinc-500 mt-2">3 days free, then billed monthly. Cancel anytime.</p>
            </div>
          );
        })}
      </div>

      {/* What every plan includes vs what's bring-your-own-key */}
      <div className="mt-10 grid md:grid-cols-2 gap-4">
        <div className="rounded-2xl border border-green-500/20 bg-green-500/5 p-6">
          <h3 className="font-semibold mb-3 flex items-center gap-2">
            <Check size={18} className="text-green-400" /> Included in every plan
          </h3>
          <ul className="space-y-2 text-sm text-zinc-300">
            <li className="flex items-center gap-2"><Check size={15} className="text-green-400 shrink-0" /> <b>Clip Generator</b> — fully managed, no API keys</li>
            <li className="flex items-center gap-2"><Check size={15} className="text-green-400 shrink-0" /> <b>YouTube Studio</b> — titles, thumbnails, descriptions</li>
            <li className="flex items-center gap-2"><Check size={15} className="text-green-400 shrink-0" /> Auto-posting to TikTok, Reels &amp; Shorts</li>
            <li className="flex items-center gap-2"><Check size={15} className="text-green-400 shrink-0" /> All the AI &amp; compute run on our servers</li>
          </ul>
          <p className="text-[11px] text-zinc-500 mt-3 pt-3 border-t border-white/5">
            Your monthly minutes cover video processing. Titles &amp; descriptions are free;
            AI <b>thumbnail image generation</b> uses ~3 min of your quota per batch.
          </p>
        </div>
        <div className="rounded-2xl border border-amber-500/20 bg-amber-500/5 p-6">
          <h3 className="font-semibold mb-3 flex items-center gap-2">
            <Zap size={18} className="text-amber-400" /> Bring your own key
          </h3>
          <p className="text-sm text-zinc-400 mb-3 leading-relaxed">
            <b>AI Shorts</b> (AI-actor UGC videos) and <b>voice dubbing</b> use premium generation from
            <b> fal.ai</b> and <b>ElevenLabs</b>. Connect your own keys for those — you're billed by those
            providers directly (typically ~$0.65-2 per video). Your plan still covers the script &amp; orchestration.
          </p>
          <p className="text-[11px] text-zinc-500">Managed credits for these are coming later — no keys needed.</p>
        </div>
      </div>

      {/* Free self-hosted path — the honest "free" option */}
      <div className="mt-10 rounded-2xl border border-white/10 bg-surface/60 p-6 md:p-8">
        <div className="flex flex-col md:flex-row md:items-center gap-6 justify-between">
          <div className="flex items-start gap-4">
            <div className="p-3 rounded-xl bg-white/5 text-zinc-300"><Server size={22} /></div>
            <div>
              <h3 className="text-lg font-bold flex items-center gap-2">
                Free forever — self-hosted
                <span className="text-[10px] bg-green-500/10 border border-green-500/30 px-2 py-0.5 rounded text-green-400 uppercase tracking-wider">$0</span>
              </h3>
              <p className="text-zinc-400 text-sm mt-1 max-w-xl leading-relaxed">
                OpenShorts is open source. Run it on your own machine with Docker and use it <b>completely free</b> —
                you just bring your own API keys and your own hardware. The plans above are for the
                <b> hosted version on this site</b>: zero setup, no keys, we run everything for you.
              </p>
            </div>
          </div>
          <a
            href="https://github.com/mutonby/openshorts"
            target="_blank" rel="noopener noreferrer"
            className="shrink-0 inline-flex items-center gap-2 bg-white/10 hover:bg-white/20 text-white px-5 py-3 rounded-xl font-medium transition-all"
          >
            <Github size={18} /> Self-host free
          </a>
        </div>
      </div>

      <p className="text-center text-zinc-500 text-xs mt-6">
        Use it free on your own computer, or skip the setup and use it here — 3 days free, then a plan.
      </p>
    </div>
  );
}
