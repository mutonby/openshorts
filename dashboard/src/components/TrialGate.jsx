import React from 'react';
import { Sparkles, ArrowRight } from 'lucide-react';

// Slim, non-blocking banner shown above a tool when a hosted user has no active
// plan/trial. The tool stays visible below; actions still route to pricing.
export default function TrialGate({ toolName = 'this' }) {
  return (
    <div className="mx-6 mt-3 p-3 bg-primary/10 border border-primary/30 rounded-xl flex items-center justify-between gap-4 shrink-0 animate-[fadeIn_0.3s_ease-out]">
      <div className="flex items-center gap-3 text-sm">
        <Sparkles size={16} className="shrink-0 text-primary" />
        <div className="text-primary/90">
          <span className="font-semibold text-white">Preview mode.</span>{' '}
          Start your <span className="font-semibold text-white">3-day free trial</span> to generate with {toolName} —
          no API keys, from $12/mo. <span className="text-zinc-400">Or run it free by self-hosting.</span>
        </div>
      </div>
      <button
        onClick={() => { window.location.hash = '#/pricing'; }}
        className="shrink-0 inline-flex items-center gap-1.5 text-xs font-medium px-3 py-1.5 rounded-lg bg-primary hover:bg-blue-600 text-white transition-colors"
      >
        See plans <ArrowRight size={14} />
      </button>
    </div>
  );
}
