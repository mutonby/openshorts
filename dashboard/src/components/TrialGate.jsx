import React from 'react';
import { Sparkles, ArrowRight } from 'lucide-react';

// Slim, non-blocking banner shown above a tool when a hosted user has no active
// plan/trial. The tool stays visible below; actions still route to pricing.
export default function TrialGate({ toolName = 'this' }) {
  return (
    <div className="card mx-6 mt-3 px-4 py-3 flex items-center justify-between gap-4 shrink-0 animate-fade">
      <div className="flex items-center gap-3 text-sm">
        <Sparkles size={16} className="shrink-0 text-brass" />
        <div className="text-ink2 lowercase">
          <span className="font-medium text-ink">Preview mode.</span>{' '}
          Start your <span className="font-medium text-ink">3-day free trial</span> to generate with {toolName} —
          no API keys, from $12/mo. <span className="text-muted">Or run it free by self-hosting.</span>
        </div>
      </div>
      <button
        onClick={() => { window.location.hash = '#/pricing'; }}
        className="btn-primary shrink-0 text-xs px-4 py-2"
      >
        See plans <ArrowRight size={14} />
      </button>
    </div>
  );
}
