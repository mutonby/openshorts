// A deep link into a tab that needs a session (#app?tab=history, the link of
// the "clips ready" email) survives the sign-in: App stashes the tab while the
// visitor is signed out, and AuthContext sends the fresh session to
// #app?tab=<tab> instead of plain #app, which would open the Clip Generator
// (and the first-run tutorial on a new device) with the clips nowhere in sight.
const TAB_AFTER_LOGIN_KEY = 'os_tab_after_login';
const TAB_AFTER_LOGIN_TTL_MS = 2 * 60 * 60 * 1000;
const TABS = ['history', 'autopilot'];

// Any other tab clears it: a visitor who moved on to the generator signs in
// from there and should stay there.
export function rememberTabAfterLogin(tab) {
  try {
    if (TABS.includes(tab)) {
      localStorage.setItem(TAB_AFTER_LOGIN_KEY, JSON.stringify({ tab, ts: Date.now() }));
    } else {
      localStorage.removeItem(TAB_AFTER_LOGIN_KEY);
    }
  } catch (_) { /* ignore storage errors */ }
}

// Reads and clears: the tab is honoured once.
export function takeTabAfterLogin() {
  try {
    const raw = JSON.parse(localStorage.getItem(TAB_AFTER_LOGIN_KEY) || 'null');
    localStorage.removeItem(TAB_AFTER_LOGIN_KEY);
    if (raw && TABS.includes(raw.tab) && Date.now() - (raw.ts || 0) < TAB_AFTER_LOGIN_TTL_MS) return raw.tab;
  } catch (_) { /* ignore */ }
  return null;
}
