// Header notification bell. Shows unread badge + dropdown panel listing
// the latest publish/render events. Reads from notificationsStore.

import { useEffect, useRef, useState } from 'react';
import { Bell, Check, X } from 'lucide-react';
import { useNotifications } from '../../state/notificationsStore.js';

const PLATFORM_DOT = {
  youtube:   'bg-platform-youtube',
  tiktok:    'bg-platform-tiktok',
  instagram: 'bg-platform-instagram',
  snapchat:  'bg-platform-snapchat',
  facebook:  'bg-platform-facebook',
};

const STATUS_LABEL = {
  submitted: 'Submitted',
  scheduled: 'Scheduled',
  published: 'Published',
  failed:    'Failed',
};

function formatTime(ts) {
  const date = new Date(ts);
  const now = new Date();
  const sameDay = date.toDateString() === now.toDateString();
  return sameDay
    ? date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    : date.toLocaleDateString([], { month: 'short', day: 'numeric' });
}

export default function NotificationBell() {
  const { items, unread, markRead, markAllRead, clearNotifications } = useNotifications();
  const [open, setOpen] = useState(false);
  const ref = useRef(null);

  useEffect(() => {
    if (!open) return;
    function handleClick(e) {
      if (ref.current && !ref.current.contains(e.target)) setOpen(false);
    }
    window.addEventListener('mousedown', handleClick);
    return () => window.removeEventListener('mousedown', handleClick);
  }, [open]);

  return (
    <div className="relative" ref={ref}>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="relative w-8 h-8 flex items-center justify-center rounded-md text-zinc-400 hover:text-white hover:bg-white/5 transition-colors"
        aria-label={unread > 0 ? `Notifications (${unread} unread)` : 'Notifications'}
      >
        <Bell size={16} />
        {unread > 0 && (
          <span className="absolute top-1.5 right-1.5 w-2 h-2 rounded-full bg-primary" />
        )}
      </button>

      {open && (
        <div className="absolute right-0 mt-2 w-80 bg-surface border border-border rounded-lg shadow-2xl z-50 overflow-hidden">
          <div className="flex items-center justify-between px-4 py-2 border-b border-border bg-black/30">
            <span className="text-[12px] font-medium text-white">Notifications</span>
            <div className="flex items-center gap-2">
              {unread > 0 && (
                <button
                  onClick={markAllRead}
                  className="text-[10px] uppercase tracking-wider text-zinc-400 hover:text-white"
                >
                  Mark all read
                </button>
              )}
              <button
                onClick={() => setOpen(false)}
                className="text-zinc-500 hover:text-white"
                aria-label="Close"
              >
                <X size={14} />
              </button>
            </div>
          </div>

          <div className="max-h-96 overflow-y-auto custom-scrollbar">
            {items.length === 0 ? (
              <div className="px-4 py-8 text-center text-[12px] text-zinc-500">
                No notifications yet.
              </div>
            ) : (
              items.map((n) => (
                <button
                  key={n.id}
                  type="button"
                  onClick={() => markRead(n.id)}
                  className={`w-full px-4 py-3 border-b border-border last:border-0 flex items-start gap-3 text-left transition-colors ${
                    n.read ? 'hover:bg-white/5' : 'bg-primary/[0.04] hover:bg-primary/[0.08]'
                  }`}
                >
                  <span
                    className={`mt-1.5 w-1.5 h-1.5 rounded-full shrink-0 ${
                      n.platform ? PLATFORM_DOT[n.platform] || 'bg-zinc-500' : 'bg-zinc-500'
                    }`}
                  />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 text-[12px] text-white">
                      <span className="font-medium truncate">{n.message || 'Event'}</span>
                      {n.status === 'failed' && (
                        <span className="text-[10px] uppercase tracking-wider text-red-400">{n.status}</span>
                      )}
                    </div>
                    <div className="flex items-center gap-2 mt-0.5">
                      {n.platform && (
                        <span className="text-[10px] uppercase tracking-wider text-zinc-500">{n.platform}</span>
                      )}
                      <span className="text-[10px] text-zinc-600">{STATUS_LABEL[n.status] || n.status}</span>
                      <span className="text-[10px] text-zinc-700 ml-auto">{formatTime(n.ts)}</span>
                    </div>
                  </div>
                  {!n.read && <Check size={12} className="text-primary shrink-0 mt-1" />}
                </button>
              ))
            )}
          </div>

          {items.length > 0 && (
            <div className="px-4 py-2 border-t border-border bg-black/30 flex justify-end">
              <button
                onClick={clearNotifications}
                className="text-[10px] uppercase tracking-wider text-zinc-500 hover:text-zinc-300"
              >
                Clear all
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
