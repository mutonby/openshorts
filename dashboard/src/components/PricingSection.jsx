import React, { useState, useEffect, useRef } from 'react';
import { Check, Loader2, Zap, Cpu, KeyRound, Send, HardDrive, Bot, Repeat } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';
import { apiJson } from '../lib/api';
import { track } from '../lib/analytics';
import { stashPendingPlan, readPendingPlan, clearPendingPlan } from '../lib/pendingPlan';
import { useBackFromStripe } from '../lib/backFromStripe';
import SegmentedControl from './ui/SegmentedControl';
import { FREE_CLIPS_PER_VIDEO, FREE_CLIPS_QUIET, MAX_CLIPS_PER_VIDEO } from '../lib/clipProgress';

const PLAN_ORDER = ['starter', 'creator', 'pro'];
const PLAN_BLURB = {
  starter: 'For getting started',
  creator: 'For regular creators',
  pro: 'For power users & teams',
};
const HIGHLIGHT = 'creator';
const FREE_MINUTES = 20;


// Emails link to #/pricing?src=email_<kind>: keep that origin on the checkout
// events so a click from an email is not counted as the pricing page itself.
const pricingSource = () => {
  const src = new URLSearchParams((window.location.hash.split('?')[1]) || '').get('src') || '';
  return /^[a-z_]{1,32}$/.test(src) ? `pricing_${src}` : 'pricing';
};

const fmt = (amount, currency) =>
  new Intl.NumberFormat('en-US', { style: 'currency', currency: (currency || 'usd').toUpperCase(), maximumFractionDigits: 0 }).format((amount || 0) / 100);

// Free tier + 3 paid tiers with monthly/annual toggle. Checkout requires sign-in.
export default function PricingSection({ onRequireLogin }) {
  const { isSignedIn } = useAuth();
  const [plans, setPlans] = useState([]);
  const [interval, setInterval] = useState('month');
  const [loading, setLoading] = useState(true);
  const [busyPrice, setBusyPrice] = useState(null);
  // Set in the same tick as the click: a double click (or monthly then yearly)
  // before React re-renders the disabled buttons must not open two checkouts.
  const inFlightRef = useRef(false);
  useBackFromStripe(() => { inFlightRef.current = false; setBusyPrice(null); });
  // One automatic checkout per page load (resumed plan or ?plan= link).
  const autoCheckoutRef = useRef(false);

  useEffect(() => {
    apiJson('/api/billing/plans')
      .then((d) => setPlans(d.plans || []))
      .catch(() => setPlans([]))
      .finally(() => setLoading(false));
  }, []);

  const checkout = async (entry, { resumed = false, source = null } = {}) => {
    if (!isSignedIn) {
      stashPendingPlan(entry, source || pricingSource());
      track('CheckoutLoginRequired', { props: { plan: entry.plan, interval: entry.interval } });
      onRequireLogin?.(entry.price_id);
      return;
    }
    if (inFlightRef.current) return;
    inFlightRef.current = true;
    setBusyPrice(entry.price_id);
    // `source` segments this surface from the two modals (see TopUpModal), which
    // now emit the same three events. `resumed`: the plan was picked before
    // signing in and the checkout opened on its own after it.
    const props = { plan: entry.plan, interval: entry.interval, source: source || pricingSource(), resumed: resumed ? '1' : '0' };
    track('CheckoutStarted', { props });
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
        body: JSON.stringify({ price_id: entry.price_id, source: props.source }),
      });
      track('CheckoutRedirected', { props });
      window.location.href = url;
    } catch (e) {
      track('CheckoutFailed', { props: { ...props, reason: String(e?.detail || e?.message || 'unknown').slice(0, 120) } });
      inFlightRef.current = false;
      setBusyPrice(null);
      alert(e?.detail || 'Could not start checkout. Please try again.');
    }
  };

  // Back from sign-in with a plan picked before it: open that checkout.
  useEffect(() => {
    if (!isSignedIn || plans.length === 0) return;
    const pending = readPendingPlan();
    if (!pending) return;
    clearPendingPlan();
    const entry = plans.find((p) => p.price_id === pending.price_id);
    if (entry && !autoCheckoutRef.current) {
      autoCheckoutRef.current = true;
      checkout(entry, { resumed: true, source: pending.source });
    }
    // checkout is recreated every render; the stash makes this run once.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isSignedIn, plans]);

  // #/pricing?plan=starter[&interval=year]: a link that opens that plan's
  // checkout (API/MCP errors hand it to agents). Signed out, it goes through the
  // sign-in first like any plan click. The param is dropped so a reload or the
  // back button from Stripe does not reopen the checkout.
  useEffect(() => {
    if (loading || plans.length === 0) return;
    const [path, query = ''] = window.location.hash.split('?');
    const params = new URLSearchParams(query);
    const planName = params.get('plan');
    if (!planName) return;
    const entry = plans.find((p) => p.plan === planName && p.interval === (params.get('interval') || 'month'));
    const source = pricingSource();
    params.delete('plan');
    params.delete('interval');
    const rest = params.toString();
    try { window.history.replaceState(null, '', `${window.location.pathname}${window.location.search}${path}${rest ? `?${rest}` : ''}`); } catch (_) { /* ignore */ }
    if (entry && !autoCheckoutRef.current) {
      autoCheckoutRef.current = true;
      checkout(entry, { source });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loading, plans]);

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
            <li className="flex items-start gap-2"><Check size={16} className="text-ok shrink-0 mt-0.5" /> <span>The <b>{FREE_CLIPS_PER_VIDEO} to {FREE_CLIPS_QUIET} best clips</b> of each video</span></li>
            <li className="flex items-start gap-2"><Check size={16} className="text-ok shrink-0 mt-0.5" /> <span>YouTube URL or upload</span></li>
            <li className="flex items-start gap-2"><Cpu size={16} className="text-ok shrink-0 mt-0.5" /> <span>Same <b>GPU rendering</b>, about 50s per 8-min video</span></li>
            <li className="flex items-start gap-2"><KeyRound size={16} className="text-ok shrink-0 mt-0.5" /> <span>Gemini key included, no setup</span></li>
            <li className="flex items-start gap-2"><Check size={16} className="text-ok shrink-0 mt-0.5" /> <span>No credit card — Google sign-in</span></li>
            <li className="flex items-start gap-2"><Check size={16} className="text-muted shrink-0 mt-0.5" /> <span className="text-muted">Watermark · clips kept 7 days</span></li>
            <li className="flex items-start gap-2"><Check size={16} className="text-muted shrink-0 mt-0.5" /> <span className="text-muted">Download only: no auto-posting or Autopilot</span></li>
          </ul>
          <button
            onClick={() => { window.location.hash = '#app'; }}
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
                <span className="absolute -top-3 left-1/2 -translate-x-1/2 badge-float">
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
                <li className="flex items-start gap-2"><Check size={16} className="text-ok shrink-0 mt-0.5" /> <span><b>Every clip</b> of each video, up to {MAX_CLIPS_PER_VIDEO}</span></li>
                <li className="flex items-start gap-2"><Check size={16} className="text-ok shrink-0 mt-0.5" /> <span><b>No watermark</b>, no 7-day clip expiry</span></li>
                <li className="flex items-start gap-2"><Cpu size={16} className="text-ok shrink-0 mt-0.5" /> <span><b>GPU rendering</b>, about 50s per 8-min video</span></li>
                <li className="flex items-start gap-2"><Send size={16} className="text-ok shrink-0 mt-0.5" /> <span><b>Auto-post every clip</b> to TikTok, Instagram, YouTube, Facebook, LinkedIn, Threads &amp; Bluesky</span></li>
                <li className="flex items-start gap-2"><Repeat size={16} className="text-ok shrink-0 mt-0.5" /> <span><b>Autopilot</b>: new channel videos become scheduled shorts</span></li>
                <li className="flex items-start gap-2"><KeyRound size={16} className="text-ok shrink-0 mt-0.5" /> <span>Gemini key included, no setup</span></li>
                <li className="flex items-start gap-2"><Bot size={16} className="text-ok shrink-0 mt-0.5" /> <span><b>MCP + API access</b> for AI agents &amp; automations</span></li>
                {plan === 'pro' && <li className="flex items-start gap-2"><Zap size={16} className="text-brass shrink-0 mt-0.5" /> <span>Priority processing queue</span></li>}
              </ul>
              <button
                onClick={() => checkout(entry)}
                disabled={busyPrice !== null}
                className={`w-full ${highlight ? 'btn-primary' : 'btn-ghost'}`}
              >
                {busyPrice === entry.price_id ? <Loader2 size={18} className="animate-spin" /> : `Get ${plan}`}
              </button>
              <p className="text-center text-xs text-muted mt-2 lowercase">billed {interval === 'month' ? 'monthly' : 'yearly'}. cancel anytime.</p>
            </div>
          );
        })}
      </div>

      {/* The two things only a paid plan does: post the clips, and make them unattended */}
      <div className="mt-10 card p-6 border-brass">
        <div className="mb-4">
          <span className="badge-ok"><Zap size={12} /> On every paid plan: your clips post themselves</span>
        </div>
        <div className="grid md:grid-cols-2 gap-6 text-sm text-ink2">
          <div className="flex items-start gap-3">
            <Send size={18} className="text-ok shrink-0 mt-0.5" />
            <p className="leading-relaxed"><b className="text-ink">Auto-posting to your socials.</b> Connect TikTok, Instagram, YouTube, Facebook, LinkedIn, Threads and Bluesky once
              from your account, then send every clip straight to all of them, or schedule it, without downloading a file.
              Posting is built on Upload-Post, already wired up for you.</p>
          </div>
          <div className="flex items-start gap-3">
            <Repeat size={18} className="text-ok shrink-0 mt-0.5" />
            <p className="leading-relaxed"><b className="text-ink">Autopilot.</b> Connect your YouTube channel and every new video is clipped on
              its own; the best clips are scheduled to your socials, one a day, at the hour you pick. You upload as usual,
              the shorts come out by themselves. <a href="/auto-clip" className="underline underline-offset-2 hover:text-ink" target="_blank" rel="noopener">How it works</a>.</p>
          </div>
        </div>
        <p className="text-xs text-muted mt-4 pt-3 border-t border-rule">
          Only want to post videos you already have, without clipping? Use{' '}
          <a href="https://www.upload-post.com" className="underline underline-offset-2 hover:text-ink" target="_blank" rel="noopener">Upload-Post</a>{' '}
          directly: it posts to TikTok, Instagram, YouTube, LinkedIn, X and more, with a free plan.
        </p>
      </div>

      {/* What every plan includes vs what's bring-your-own-key */}
      <div className="mt-4 grid md:grid-cols-2 gap-4">
        <div className="card p-6">
          <div className="mb-4">
            <span className="badge-ok"><Check size={12} /> Included in every plan</span>
          </div>
          <ul className="space-y-2 text-sm text-ink2">
            <li className="flex items-start gap-2"><Cpu size={15} className="text-ok shrink-0 mt-0.5" /> <span><b>Our NVIDIA GPU does the rendering.</b> An 8-minute video is clipped in about 50 seconds, instead of the 5 to 8 minutes it takes on a typical CPU.</span></li>
            <li className="flex items-start gap-2"><KeyRound size={15} className="text-ok shrink-0 mt-0.5" /> <span><b>The Gemini API key is included.</b> Nothing to sign up for, nothing to paste, no per-request quota of your own to babysit.</span></li>
            <li className="flex items-start gap-2"><HardDrive size={15} className="text-ok shrink-0 mt-0.5" /> <span><b>Clips are stored and served for you</b>, ready to re-open and re-edit from any browser.</span></li>
            <li className="flex items-start gap-2"><Check size={15} className="text-ok shrink-0 mt-0.5" /> <span>Full <b>YouTube Studio</b>: titles, thumbnails and descriptions.</span></li>
            <li className="flex items-start gap-2"><Bot size={15} className="text-ok shrink-0 mt-0.5" /> <span><b>MCP server &amp; API for agents.</b> Connect Claude, ChatGPT or n8n to an always-on endpoint and automate clipping end to end. API calls use the same minutes, nothing extra to buy. <a href="/mcp" className="underline underline-offset-2 hover:text-ink" target="_blank" rel="noopener">Guide</a>.</span></li>
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

      <p className="text-center text-muted text-xs mt-8 lowercase">
        start free right here, upgrade for more minutes and no watermark, or run it yourself on your own hardware.
      </p>
    </div>
  );
}
