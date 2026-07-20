import React from 'react';
import { Sparkles, ArrowRight } from 'lucide-react';

// Slim, non-blocking banner shown above a tool when a hosted user has no
// entitlement. Google-authed users get the free plan automatically, so this
// only fires for signed-out or magic-link-only accounts.
export default function TrialGate({ toolName = 'this' }) {
  return (
    <div className="card mx-6 mt-3 px-4 py-3 flex items-center justify-between gap-4 shrink-0 animate-fade">
      <div className="flex items-center gap-3 text-sm">
        <Sparkles size={16} className="shrink-0 text-brass" />
        <div className="text-ink2 lowercase">
          <span className="font-medium text-ink">Preview mode.</span>{' '}
          Sign in with <span className="font-medium text-ink">Google</span> to use {toolName} free —
          20 min/month, no credit card. <span className="text-muted">Or run it free by self-hosting.</span>
        </div>
      </div>
      <button
        onClick={() => { window.location.hash = '#/pricing'; }}
        className="btn-primary shrink-0 text-xs px-4 py-2"
      >
        Start free <ArrowRight size={14} />
      </button>
    </div>
  );
}
