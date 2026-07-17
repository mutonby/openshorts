import React, { useState } from 'react';
import { Mail, Check, Loader2 } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';
import Modal from './ui/Modal';

// Sign-in modal: magic link (email) + Google OAuth.
export default function LoginModal({ onClose }) {
  const { requestMagicLink, loginWithGoogle, googleAuthEnabled } = useAuth();
  const [email, setEmail] = useState('');
  const [sent, setSent] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  const submit = async (e) => {
    e.preventDefault();
    if (!email.trim()) return;
    setBusy(true);
    setError('');
    try {
      await requestMagicLink(email.trim());
      setSent(true);
    } catch (err) {
      setError(err.message || 'Something went wrong.');
    } finally {
      setBusy(false);
    }
  };

  return (
    <Modal isOpen onClose={onClose} eyebrow="ACCOUNT" title="Sign in to OpenShorts" size="md">
      <p className="text-muted text-sm mb-6 lowercase">Access your plan and generate shorts with no API keys.</p>

      {sent ? (
        <div className="text-center py-6">
          <span className="badge-ok px-3 py-1"><Check size={14} /> sent</span>
          <p className="text-ink font-medium lowercase mt-4">Check your inbox</p>
          <p className="text-muted text-sm mt-1">We sent a sign-in link to <b className="text-ink font-medium">{email}</b>. It expires in 15 minutes.</p>
        </div>
      ) : (
        <>
          <form onSubmit={submit} className="space-y-3">
            <div className="relative">
              <Mail size={18} className="absolute left-3 top-1/2 -translate-y-1/2 text-muted" />
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@email.com"
                className="input-field pl-10"
                autoFocus
              />
            </div>
            {error && <p className="text-danger text-sm">{error}</p>}
            <button
              type="submit"
              disabled={busy || !email.trim()}
              className="btn-primary w-full"
            >
              {busy ? <Loader2 size={18} className="animate-spin" /> : 'Email me a sign-in link'}
            </button>
          </form>

          {googleAuthEnabled && (
            <>
              <div className="flex items-center gap-3 my-5">
                <div className="flex-1 border-t border-rule" />
                <span className="readout">or</span>
                <div className="flex-1 border-t border-rule" />
              </div>
              <button
                onClick={loginWithGoogle}
                className="btn-ghost w-full"
              >
                <svg width="18" height="18" viewBox="0 0 24 24"><path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/><path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/><path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l3.66-2.84z"/><path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/></svg>
                Continue with Google
              </button>
            </>
          )}
        </>
      )}
    </Modal>
  );
}
