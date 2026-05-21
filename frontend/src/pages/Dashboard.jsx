// Dashboard. At-a-glance counters + scheduled-uploads list + recent
// activity. All stats are derived locally from the notifications store +
// short/long-form histories. Once the backend ships GET /api/clips/recent
// (plan TODO #10) we can swap the counters to a live feed.

import { useEffect, useMemo, useState } from 'react';
import { Calendar, CheckCircle2, Clock, Film, ScrollText } from 'lucide-react';
import { Link } from 'react-router-dom';
import StatCard from '../components/ui/StatCard.jsx';
import PlatformBadge from '../components/ui/PlatformBadge.jsx';
import { useNotifications } from '../state/notificationsStore.js';

const SHORT_HISTORY_KEY = 'openshorts.shortForm.history';
const LONG_HISTORY_KEY  = 'openshorts.longForm.history';

function loadHistory(key) {
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export default function Dashboard() {
  const { items: notifications } = useNotifications();
  const [shortHistory, setShortHistory] = useState([]);
  const [longHistory,  setLongHistory]  = useState([]);

  useEffect(() => {
    setShortHistory(loadHistory(SHORT_HISTORY_KEY));
    setLongHistory(loadHistory(LONG_HISTORY_KEY));
  }, []);

  const clipsProcessed = useMemo(
    () => shortHistory.reduce((sum, h) => sum + (h.clipCount || 0), 0)
        + longHistory.length,
    [shortHistory, longHistory],
  );

  const scheduled = useMemo(
    () => notifications.filter((n) => n.status === 'scheduled'),
    [notifications],
  );

  const published = useMemo(
    () => notifications.filter((n) => n.status === 'submitted' || n.status === 'published'),
    [notifications],
  );

  const recent = notifications.slice(0, 8);

  return (
    <div className="h-full overflow-y-auto custom-scrollbar">
      <div className="p-6 max-w-6xl mx-auto space-y-6">
        <header>
          <h1 className="text-[20px] font-semibold text-white">Dashboard</h1>
          <p className="text-[13px] text-zinc-500 mt-1">
            At-a-glance view of your pipeline. Live backend feed lands with plan TODO #10.
          </p>
        </header>

        <section className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <StatCard
            label="Clips processed"
            value={clipsProcessed}
            tone="accent"
            icon={Film}
            delta={shortHistory.length > 0 ? `${shortHistory.length} short-form batch${shortHistory.length === 1 ? '' : 'es'}` : 'No batches yet'}
          />
          <StatCard
            label="Scheduled"
            value={scheduled.length}
            tone="default"
            icon={Calendar}
            delta={scheduled[0] ? `Next: ${scheduled[0].platform || 'unknown'}` : 'Nothing on deck'}
          />
          <StatCard
            label="Published"
            value={published.length}
            tone="success"
            icon={CheckCircle2}
            delta={published[0] ? `Latest: ${published[0].platform || 'unknown'}` : 'No publishes yet'}
          />
        </section>

        <section className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <div className="rounded-xl border border-border bg-surface p-5">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-[14px] font-semibold text-white flex items-center gap-2">
                <Calendar size={14} className="text-primary" />
                Upcoming uploads
              </h2>
              <Link to="/short-form" className="text-[11px] text-primary hover:underline">
                Schedule new
              </Link>
            </div>
            {scheduled.length === 0 ? (
              <p className="text-[12px] text-zinc-500">No scheduled uploads. Schedule a clip from the Review step of the short-form wizard.</p>
            ) : (
              <ul className="space-y-2">
                {scheduled.slice(0, 6).map((n) => (
                  <li key={n.id} className="flex items-center gap-3 rounded-lg border border-border bg-background/40 p-3">
                    {n.platform && <PlatformBadge platform={n.platform} withLabel={false} size="sm" />}
                    <div className="flex-1 min-w-0">
                      <div className="text-[12px] text-white truncate">{n.message || 'Scheduled clip'}</div>
                      <div className="text-[10px] text-zinc-500 mt-0.5 flex items-center gap-1">
                        <Clock size={10} /> {new Date(n.ts).toLocaleString()}
                      </div>
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </div>

          <div className="rounded-xl border border-border bg-surface p-5">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-[14px] font-semibold text-white flex items-center gap-2">
                <ScrollText size={14} className="text-primary" />
                Recent activity
              </h2>
            </div>
            {recent.length === 0 ? (
              <p className="text-[12px] text-zinc-500">No activity yet — publish your first clip and it’ll appear here.</p>
            ) : (
              <ul className="space-y-2">
                {recent.map((n) => (
                  <li key={n.id} className="flex items-center gap-3 rounded-lg border border-border bg-background/40 p-3">
                    {n.platform && <PlatformBadge platform={n.platform} withLabel={false} size="sm" />}
                    <div className="flex-1 min-w-0">
                      <div className="text-[12px] text-white truncate">{n.message || `${n.type} event`}</div>
                      <div className="text-[10px] text-zinc-500 mt-0.5">
                        {n.status} · {new Date(n.ts).toLocaleString()}
                      </div>
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </section>
      </div>
    </div>
  );
}
