import React from 'react';
import { KeyRound, ArrowRight } from 'lucide-react';

// Banner for advanced tools (AI Shorts, AI Agent) that use fal.ai + ElevenLabs.
// These are BYOK: the plan covers the script/orchestration, the user brings their
// own keys for the premium generation. If they have no plan yet, also nudge trial.
export default function AdvancedBanner({ needsPlan, onKeys }) {
  return (
    <div className="card mx-6 mt-3 px-4 py-3 flex items-center justify-between gap-4 shrink-0 animate-fade">
      <div className="flex items-center gap-3 text-sm">
        <KeyRound size={16} className="shrink-0 text-warn" />
        <div className="text-ink2 lowercase">
          <span className="font-medium text-ink">Advanced tool.</span>{' '}
          {needsPlan
            ? <>Start your <span className="font-medium text-ink">3-day free trial</span> to unlock. This tool also uses your own <span className="font-medium text-ink normal-case">fal.ai + ElevenLabs</span> keys (you pay those providers).</>
            : <>AI video &amp; voice generation use your own <span className="font-medium text-ink normal-case">fal.ai + ElevenLabs</span> keys — add them in Settings. Script &amp; orchestration are included in your plan.</>}
        </div>
      </div>
      <button
        onClick={needsPlan ? () => { window.location.hash = '#/pricing'; } : onKeys}
        className="btn-quiet shrink-0 text-xs px-3 py-1.5"
      >
        {needsPlan ? <>See plans <ArrowRight size={14} /></> : 'Add keys'}
      </button>
    </div>
  );
}
