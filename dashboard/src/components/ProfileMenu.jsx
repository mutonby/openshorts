import React, { useState, useRef, useEffect } from 'react';
import { CreditCard, LogOut, Sparkles } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';

// Header avatar + dropdown for signed-in cloud users: shows the email and gives
// access to Account & billing (manage subscription, top-ups) and Sign out.
export default function ProfileMenu() {
  const { user, isManaged, logout } = useAuth();
  const [open, setOpen] = useState(false);
  const ref = useRef(null);

  useEffect(() => {
    if (!open) return;
    const onClick = (e) => { if (ref.current && !ref.current.contains(e.target)) setOpen(false); };
    document.addEventListener('mousedown', onClick);
    return () => document.removeEventListener('mousedown', onClick);
  }, [open]);

  if (!user) return null;
  const initial = (user.email || '?').trim().charAt(0).toUpperCase();

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen((v) => !v)}
        aria-label="Account menu"
        className="w-8 h-8 rounded-full bg-paper3 border border-rule hover:border-rule2 text-brass flex items-center justify-center text-sm font-medium transition-colors"
      >
        {initial}
      </button>

      {open && (
        <div className="card absolute right-0 top-full mt-2 w-56 z-30 shadow-none overflow-hidden animate-fade">
          <div className="px-4 py-3 border-b border-rule">
            <p className="eyebrow">Signed in as</p>
            <p className="text-sm text-ink truncate mt-0.5" title={user.email}>{user.email}</p>
          </div>
          {!isManaged && (
            <button
              onClick={() => { setOpen(false); window.location.hash = '#/pricing'; }}
              className="w-full flex items-center gap-3 px-4 py-2.5 text-sm lowercase text-brass hover:bg-paper3 transition-colors"
            >
              <Sparkles size={16} /> Start free trial
            </button>
          )}
          <button
            onClick={() => { setOpen(false); window.location.hash = '#/account'; }}
            className="w-full flex items-center gap-3 px-4 py-2.5 text-sm lowercase text-ink2 hover:bg-paper3 transition-colors"
          >
            <CreditCard size={16} className="text-muted" /> Account &amp; billing
          </button>
          <button
            onClick={() => { setOpen(false); logout(); }}
            className="w-full flex items-center gap-3 px-4 py-2.5 text-sm lowercase text-ink2 hover:bg-paper3 transition-colors border-t border-rule"
          >
            <LogOut size={16} className="text-muted" /> Sign out
          </button>
        </div>
      )}
    </div>
  );
}
