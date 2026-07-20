import React, { useState, useEffect } from 'react';
import { Check, Loader2, Zap, Github, Server } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';
import { apiJson } from '../lib/api';
import { track } from '../lib/analytics';
import SegmentedControl from './ui/SegmentedControl';

const PLAN_ORDER = ['starter', 'creator', 'pro'];
const PLAN_BLURB = {
  starter: 'For getting started',
  creator: 'For regular creators',
  pro: 'For power users & teams',
};
const HIGHLIGHT = 'creator';
const FREE_MINUTES = 20;

const fmt = (amount, currency) =>
  new Intl.NumberFormat('en-US', { style: 'currency', currency: (currency || 'usd').toUpperCase(), maximumFractionDigits: 0 }).format((amount || 0) / 100);

// Free tier + 3 paid tiers with monthly/annual toggle. Checkout requires sign-in.
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

  const checkout = async (entry) => {
    if (!isSignedIn) { onRequireLogin?.(entry.price_id); return; }
    setBusyPrice(entry.price_id);
    track('CheckoutStarted', { props: { plan: entry.plan, interval: entry.interval } });
    // Stash the price so we can attach real revenue to the Subscribed goal when
    // the user returns from Stripe (see AccountPage's checkout=success handler).
    try {
      localStorage.setItem('os_pending_checkout', JSON.stringify({
        plan: entry.plan, interval: entry.interval, amount: entry.amount, currency: entry.currency,
      }));
    } catch (_) { /* ignore storage errors */ }
    try {
      const { url } = await apiJson('/api/billing/checkout', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ price_id: entry.price_id }),
      });
      window.location.href = url;
    } catch (e) {
      setBusyPrice(null);
      alert('Could not start checkout. Please try again.');
    }
  };

  if (loading) {
    return <div className="flex justify-center py-16"><Loader2 className="animate-spin text-brass" /></div>;
  }

  const byPlan = (plan) => plans.find((p) => p.plan === plan && p.interval === interval);

  return (
    <div className="max-w-5xl mx-auto">
      <div className="max-w-xs mx-auto mb-10">
        <SegmentedControl
          size="sm"
          value={interval}
          onChange={setInterval}
          options={[
            { value: 'month', label: 'Monthly' },
            { value: 'year', label: 'Yearly', hint: '2 months free' },
          ]}
        />
      </div>

      <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
        {/* Free tier — no card, Google sign-in only */}
        <div className="relative card p-6 flex flex-col">
          <h3 className="font-display lowercase text-xl text-ink">free</h3>
          <p className="text-muted text-sm mb-4 lowercase">Try it on your own videos</p>
          <div className="mb-4 flex items-baseline gap-1.5">
            <span className="font-display text-4xl text-ink tabular-nums">$0</span>
            <span className="readout">/mo</span>
          </div>
          <ul className="space-y-2 text-sm text-ink2 mb-6 flex-1">
            <li className="flex items-start gap-2"><Check size={16} className="text-ok shrink-0 mt-0.5" /> <span><b>{FREE_MINUTES} min</b> of video / month</span></li>
            <li className="flex items-start gap-2"><Check size={16} className="text-ok shrink-0 mt-0.5" /> <span>YouTube URL or upload</span></li>
            <li className="flex items-start gap-2"><Check size={16} className="text-ok shrink-0 mt-0.5" /> <span>No credit card — Google sign-in</span></li>
            <li className="flex items-start gap-2"><Check size={16} className="text-muted shrink-0 mt-0.5" /> <span className="text-muted">Watermark · clips kept 7 days</span></li>
          </ul>
          <button
            onClick={() => { if (!isSignedIn) { onRequireLogin?.(null); } else { window.location.hash = ''; } }}
            className="w-full btn-ghost"
          >
            Start free
          </button>
          <p className="text-center text-xs text-muted mt-2 lowercase">free minutes reset monthly.</p>
        </div>

        {PLAN_ORDER.map((plan) => {
          const entry = byPlan(plan);
          if (!entry) return null;
          const highlight = plan === HIGHLIGHT;
          return (
            <div
              key={plan}
              className={`relative card p-6 flex flex-col ${highlight ? 'border-brass' : ''}`}
            >
              {highlight && (
                <span className="absolute -top-3 left-1/2 -translate-x-1/2 badge-brass">
                  Most popular
                </span>
              )}
              <h3 className="font-display lowercase text-xl text-ink">{plan}</h3>
              <p className="text-muted text-sm mb-4 lowercase">{PLAN_BLURB[plan]}</p>
              <div className="mb-4 flex items-baseline gap-1.5">
                <span className="font-display text-4xl text-ink tabular-nums">{fmt(entry.amount, entry.currency)}</span>
                <span className="readout">/{interval === 'month' ? 'mo' : 'yr'}</span>
              </div>
              <ul className="space-y-2 text-sm text-ink2 mb-6 flex-1">
                <li className="flex items-start gap-2"><Check size={16} className="text-ok shrink-0 mt-0.5" /> <span><b>{entry.minutes} min</b> of video / month</span></li>
                <li className="flex items-start gap-2"><Check size={16} className="text-ok shrink-0 mt-0.5" /> <span>No API keys needed</span></li>
                <li className="flex items-start gap-2"><Check size={16} className="text-ok shrink-0 mt-0.5" /> <span>Gemini + auto-posting included</span></li>
                {plan === 'pro' && <li className="flex items-start gap-2"><Zap size={16} className="text-brass shrink-0 mt-0.5" /> <span>Priority processing queue</span></li>}
              </ul>
              <button
                onClick={() => checkout(entry)}
                disabled={busyPrice === entry.price_id}
                className={`w-full ${highlight ? 'btn-primary' : 'btn-ghost'}`}
              >
                {busyPrice === entry.price_id ? <Loader2 size={18} className="animate-spin" /> : `Get ${plan}`}
              </button>
              <p className="text-center text-xs text-muted mt-2 lowercase">billed {interval === 'month' ? 'monthly' : 'yearly'}. cancel anytime.</p>
            </div>
          );
        })}
      </div>

      {/* What every plan includes vs what's bring-your-own-key */}
      <div className="mt-10 grid md:grid-cols-2 gap-4">
        <div className="card p-6">
          <div className="mb-4">
            <span className="badge-ok"><Check size={12} /> Included in every plan</span>
          </div>
          <ul className="space-y-2 text-sm text-ink2">
            <li className="flex items-start gap-2"><Check size={15} className="text-ok shrink-0 mt-0.5" /> <span><b>Clip Generator</b> — fully managed, no API keys</span></li>
            <li className="flex items-start gap-2"><Check size={15} className="text-ok shrink-0 mt-0.5" /> <span><b>YouTube Studio</b> — titles, thumbnails, descriptions</span></li>
            <li className="flex items-start gap-2"><Check size={15} className="text-ok shrink-0 mt-0.5" /> <span>Auto-posting to TikTok, Reels &amp; Shorts</span></li>
            <li className="flex items-start gap-2"><Check size={15} className="text-ok shrink-0 mt-0.5" /> <span>All the AI &amp; compute run on our servers</span></li>
          </ul>
          <p className="text-xs text-muted mt-3 pt-3 border-t border-rule">
            Your monthly minutes cover video processing. Titles &amp; descriptions are free;
            AI <b>thumbnail image generation</b> uses ~3 min of your quota per batch.
          </p>
        </div>
        <div className="card p-6">
          <div className="mb-4">
            <span className="badge-warn"><Zap size={12} /> Bring your own key</span>
          </div>
          <p className="text-sm text-muted mb-3 leading-relaxed">
            <b className="text-ink2">AI Shorts</b> (AI-actor UGC videos) and <b className="text-ink2">voice dubbing</b> use premium generation from
            <b className="text-ink2"> fal.ai</b> and <b className="text-ink2">ElevenLabs</b>. Connect your own keys for those — you're billed by those
            providers directly (typically ~$0.65-2 per video). Your plan still covers the script &amp; orchestration.
          </p>
          <p className="text-xs text-muted">Managed credits for these are coming later — no keys needed.</p>
        </div>
      </div>

      {/* Free self-hosted path — the honest "free" option */}
      <div className="mt-10 card p-6 md:p-8">
        <div className="flex flex-col md:flex-row md:items-center gap-6 justify-between">
          <div className="flex flex-col sm:flex-row items-start gap-4 min-w-0">
            <div className="p-3 rounded-input bg-paper3 text-muted shrink-0"><Server size={18} /></div>
            <div className="min-w-0">
              <h3 className="font-display lowercase text-lg text-ink flex flex-wrap items-center gap-2">
                Free forever — self-hosted
                <span className="badge-ok">$0</span>
              </h3>
              <p className="text-muted text-sm mt-1 max-w-xl leading-relaxed">
                OpenShorts is open source. Run it on your own machine with Docker and use it <b className="text-ink2">completely free</b> —
                you just bring your own API keys and your own hardware. The plans above are for the
                <b className="text-ink2"> hosted version on this site</b>: zero setup, no keys, we run everything for you.
              </p>
            </div>
          </div>
          <a
            href="https://github.com/mutonby/openshorts"
            target="_blank" rel="noopener noreferrer"
            className="shrink-0 btn-ghost whitespace-nowrap"
          >
            <Github size={16} /> Self-host free
          </a>
        </div>
      </div>

      <p className="text-center text-muted text-xs mt-6 lowercase">
        Use it free on your own computer, start free right here, or upgrade for more minutes and no watermark.
      </p>
    </div>
  );
}
