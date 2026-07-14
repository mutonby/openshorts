import React from 'react';
import { KeyRound, ArrowRight } from 'lucide-react';

// Banner for advanced tools (AI Shorts, AI Agent) that use fal.ai + ElevenLabs.
// These are BYOK: the plan covers the script/orchestration, the user brings their
// own keys for the premium generation. If they have no plan yet, also nudge trial.
export default function AdvancedBanner({ needsPlan, onKeys }) {
  return (
    <div className="mx-6 mt-3 p-3 bg-amber-500/10 border border-amber-500/30 rounded-xl flex items-center justify-between gap-4 shrink-0 animate-[fadeIn_0.3s_ease-out]">
      <div className="flex items-center gap-3 text-sm">
        <KeyRound size={16} className="shrink-0 text-amber-400" />
        <div className="text-amber-100/90">
          <span className="font-semibold text-white">Advanced tool.</span>{' '}
          {needsPlan
            ? <>Start your <span className="font-semibold text-white">3-day free trial</span> to unlock. This tool also uses your own <span className="font-semibold text-white">fal.ai + ElevenLabs</span> keys (you pay those providers).</>
            : <>AI video &amp; voice generation use your own <span className="font-semibold text-white">fal.ai + ElevenLabs</span> keys — add them in Settings. Script &amp; orchestration are included in your plan.</>}
        </div>
      </div>
      <button
        onClick={needsPlan ? () => { window.location.hash = '#/pricing'; } : onKeys}
        className="shrink-0 inline-flex items-center gap-1.5 text-xs font-medium px-3 py-1.5 rounded-lg bg-amber-500 hover:bg-amber-400 text-black transition-colors"
      >
        {needsPlan ? <>See plans <ArrowRight size={14} /></> : 'Add keys'}
      </button>
    </div>
  );
}
