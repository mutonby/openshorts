// Auth + billing session state for cloud mode.
// - Reads /api/config to learn whether billing is enabled at all.
// - Handles the magic-link and Google OAuth redirect hashes.
// - Exposes the current user, plan and minute balance to the app.
// When billingEnabled is false the provider is inert and the app behaves as the
// classic BYOK dashboard.
import { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { getApiUrl } from '../config';
import { apiFetch, apiJson, getToken, setToken, clearToken } from '../lib/api';
import { track, identify, reset as resetAnalytics } from '../lib/analytics';
import { report as reportAttribution } from '../lib/attribution';
import { readPendingPlan } from '../lib/pendingPlan';
import { ensureMediaToken, clearMediaToken } from '../lib/mediaToken';

// Google sign-in: the verifier for the code the callback returns. Kept in
// sessionStorage (this tab only) so a sign-in link opened anywhere else — a
// code or token someone sent you — signs nobody in.
const GOOGLE_LOGIN_KEY = 'os_google_login';
const GOOGLE_LOGIN_MAX_AGE_MS = 15 * 60 * 1000;

const b64url = (bytes) => btoa(String.fromCharCode(...new Uint8Array(bytes)))
  .replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');

const readGoogleLogin = () => {
  try {
    const pending = JSON.parse(sessionStorage.getItem(GOOGLE_LOGIN_KEY) || 'null');
    sessionStorage.removeItem(GOOGLE_LOGIN_KEY);
    if (pending && Date.now() - pending.at < GOOGLE_LOGIN_MAX_AGE_MS) return pending;
  } catch (_) { /* ignore */ }
  return null;
};

const AuthContext = createContext(null);
// eslint-disable-next-line react-refresh/only-export-components
export const useAuth = () => useContext(AuthContext);

export function AuthProvider({ children }) {
  const [config, setConfig] = useState({ billingEnabled: false, googleAuthEnabled: false });
  const [me, setMe] = useState(null);           // /api/me payload, or null when signed out
  const [loading, setLoading] = useState(true);
  const [signingIn, setSigningIn] = useState(false);

  const refreshMe = useCallback(async () => {
    if (!getToken()) { setMe(null); return null; }
    try {
      const data = await apiJson('/api/me');
      // Before the user is set: the first players rendered need the token.
      await ensureMediaToken(data?.user?.id);
      setMe(data);
      // Same profileId (user uuid) the server-side events use.
      identify(data?.user, { plan: data?.plan || 'free' });
      return data;
    } catch (e) {
      // Stale/invalid token: drop it and fall back to anonymous BYOK.
      clearToken();
      clearMediaToken();
      setMe(null);
      return null;
    }
  }, []);

  // Handle auth redirect hashes: #/auth/verify?ml=... and #/auth/callback?code=...
  // (?token=... from an API older than the code exchange, briefly).
  const handleAuthHash = useCallback(async () => {
    const hash = window.location.hash || '';
    const match = hash.match(/^#\/auth\/(verify|callback)\??(.*)$/);
    if (!match) return false;
    const [, kind, query] = match;
    const params = new URLSearchParams(query);

    setSigningIn(true);
    let destination = '#app';
    try {
      if (kind === 'callback') {
        // Scrub the code/token from the URL immediately (replaceState, no new
        // history entry) so it isn't left reachable via Back or analytics.
        try {
          window.history.replaceState(null, document.title,
            window.location.pathname + window.location.search);
        } catch (_) { /* ignore */ }
        const pending = readGoogleLogin();
        const code = params.get('code');
        const token = params.get('token');
        if (code && pending?.verifier) {
          const data = await apiJson('/api/auth/exchange', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ code, verifier: pending.verifier }),
          });
          if (data.token) setToken(data.token);
        } else if (token && pending) {
          // Only for a sign-in this tab started (see GOOGLE_LOGIN_KEY).
          setToken(token);
        }
      } else if (kind === 'verify') {
        const ml = params.get('ml');
        if (ml) {
          const data = await apiJson('/api/auth/magic-link/verify', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ token: ml }),
          });
          if (data.token) setToken(data.token);
        }
      }
      const signedInMe = await refreshMe();
      // This handler only runs on the auth redirect, so a resolved user here is
      // a fresh sign-in / sign-up — the top of the conversion funnel.
      if (signedInMe?.user) {
        track('Signup', { props: { method: kind === 'verify' ? 'magic_link' : 'google' } });
        // Server-side twin of the Signup event: the one place we learn which
        // channel produced an account. Awaited but never allowed to throw.
        await reportAttribution(apiJson);
      }
      // Everyone lands in the app. First-time (unpaid) sign-in gets the Clip
      // Generator tutorial, not a pricing dump — they need one successful job
      // before the rest of the tools unlock. os_welcomed keeps this once-only
      // even if they skip the tutorial.
      const paid = ['starter', 'creator', 'pro'].includes(signedInMe?.plan);
      // A plan picked before signing in: back to pricing, which opens its checkout.
      if (signedInMe?.user && !paid && readPendingPlan()) destination = '#/pricing';
      let welcomed = false;
      try { welcomed = localStorage.getItem('os_welcomed') === '1'; } catch (_) { /* ignore */ }
      if (signedInMe?.user && !paid && !welcomed) {
        try {
          localStorage.setItem('os_show_clip_tutorial', '1');
          localStorage.setItem('os_welcomed', '1');
        } catch (_) { /* ignore */ }
      }
    } catch (e) {
      // fall through — user lands signed-out
    } finally {
      setSigningIn(false);
      // Clear the sensitive hash, land wherever we resolved above.
      window.location.hash = destination;
    }
    return true;
  }, [refreshMe]);

  useEffect(() => {
    (async () => {
      try {
        const cfg = await (await fetch(getApiUrl('/api/config'))).json();
        setConfig(cfg);
        if (cfg.billingEnabled) {
          const handled = await handleAuthHash();
          if (!handled) await refreshMe();
        }
      } catch (_) { /* config fetch failed — stay in BYOK */ }
      setLoading(false);
    })();
  }, [handleAuthHash, refreshMe]);

  const requestMagicLink = useCallback(async (email) => {
    const res = await apiFetch('/api/auth/magic-link', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email }),
    });
    if (res.status === 429) throw new Error('Too many attempts. Try again in a few minutes.');
    if (!res.ok) throw new Error('Could not send sign-in link.');
    // The app-directory review account signs in with a password instead.
    const data = await res.json().catch(() => ({}));
    return { passwordRequired: !!data.password_required };
  }, []);

  const loginWithPassword = useCallback(async (email, password) => {
    const res = await apiFetch('/api/auth/review-login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    });
    if (!res.ok) throw new Error('Wrong email or password.');
    const data = await res.json();
    setToken(data.token);
    await refreshMe();
  }, [refreshMe]);

  const loginWithGoogle = useCallback(async () => {
    let query = '';
    try {
      const raw = new Uint8Array(32);
      crypto.getRandomValues(raw);
      const verifier = b64url(raw);
      const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(verifier));
      sessionStorage.setItem(GOOGLE_LOGIN_KEY, JSON.stringify({ verifier, at: Date.now() }));
      query = `?cc=${b64url(digest)}`;
    } catch (_) {
      // No WebCrypto (insecure context): mark the sign-in as ours anyway.
      try { sessionStorage.setItem(GOOGLE_LOGIN_KEY, JSON.stringify({ at: Date.now() })); } catch (__) { /* ignore */ }
    }
    window.location.href = getApiUrl(`/api/auth/google${query}`);
  }, []);

  const logout = useCallback(() => {
    clearToken();
    clearMediaToken();
    setMe(null);
    resetAnalytics();
  }, []);

  const value = {
    billingEnabled: config.billingEnabled,
    localLlm: config.localLlm || null,
    serverGeminiKey: !!config.serverGeminiKey,
    googleAuthEnabled: config.googleAuthEnabled,
    jobRetentionSeconds: config.jobRetentionSeconds || null,
    loading,
    signingIn,
    user: me?.user || null,
    me,
    plan: me?.plan || null,
    entitled: !!me?.entitled,
    minutes: me?.minutes || null,
    isSignedIn: !!me?.user,
    // Managed = signed-in AND entitled (active plan or top-up credit).
    isManaged: !!(config.billingEnabled && me?.entitled),
    refreshMe,
    requestMagicLink,
    loginWithPassword,
    loginWithGoogle,
    logout,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}
