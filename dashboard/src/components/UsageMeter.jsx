import React from 'react';
import { Zap } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';

// Compact minutes-remaining pill for the header (managed users only).
export default function UsageMeter({ onClick }) {
  const { isManaged, minutes } = useAuth();
  if (!isManaged || !minutes) return null;

  const remaining = Math.round((minutes.remaining || 0) * 10) / 10;
  const allowance = (minutes.plan_allowance || 0) + (minutes.topup_remaining || 0) + (minutes.plan_used || 0);
  const pct = allowance > 0 ? Math.max(4, Math.min(100, (remaining / allowance) * 100)) : 0;
  const low = remaining <= (allowance * 0.2);

  return (
    <button
      onClick={onClick}
      title="Manage your plan"
      className={`flex items-center gap-2 px-3 py-1.5 rounded-full border text-xs transition-colors ${low ? 'border-amber-500/40 bg-amber-500/10 text-amber-300 hover:bg-amber-500/20' : 'border-white/10 bg-white/5 text-zinc-300 hover:bg-white/10'}`}
    >
      <Zap size={13} className={low ? 'text-amber-400' : 'text-primary'} />
      <span className="font-medium">{remaining} min</span>
      <span className="w-12 h-1.5 rounded-full bg-white/10 overflow-hidden hidden sm:inline-block">
        <span className={`block h-full ${low ? 'bg-amber-400' : 'bg-primary'}`} style={{ width: `${pct}%` }} />
      </span>
    </button>
  );
}
