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
      className="flex items-center gap-2 px-3 py-1.5 rounded-full border border-rule2 bg-paper2 hover:bg-paper3 transition-colors"
    >
      <Zap size={14} className={low ? 'text-warn' : 'text-brass'} />
      <span className={`readout ${low ? 'text-warn' : ''}`}>{remaining} min</span>
      <span className="w-12 h-1.5 rounded-full bg-paper3 overflow-hidden hidden sm:inline-block">
        <span className={`block h-full ${low ? 'bg-warn' : 'bg-brass'}`} style={{ width: `${pct}%` }} />
      </span>
    </button>
  );
}
