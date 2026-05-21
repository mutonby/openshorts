// StatCard — single-stat panel for the Dashboard. Lucide icon optional.

export default function StatCard({ label, value, delta, tone = 'default', icon: Icon }) {
  const tones = {
    default: 'text-white',
    accent:  'text-primary',
    success: 'text-success',
  };
  return (
    <div className="rounded-xl border border-border bg-surface p-5">
      <div className="flex items-center justify-between mb-3">
        <span className="text-[11px] uppercase tracking-wider text-zinc-500">{label}</span>
        {Icon && <Icon size={16} className="text-zinc-600" />}
      </div>
      <div className={`text-2xl font-semibold ${tones[tone] || tones.default}`}>{value}</div>
      {delta && <div className="text-[11px] text-zinc-500 mt-1">{delta}</div>}
    </div>
  );
}
