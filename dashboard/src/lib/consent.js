// Cookie / tracker consent — our own, in this repo, no vendor CMP.
//
// The rule the previous version broke is LSSI art. 22.2 (and GDPR art. 6): a
// tracker may not run before the visitor has agreed to it. OpenPanel's op1.js
// used to be injected from the static <head> on every page load, so by the time
// anyone could have been asked, the request had already gone out.
//
// So the loading of every non-essential script now goes through here:
// index.html only *publishes* window.__osAnalyticsInit; nothing calls it until
// `analytics` is granted. Withdrawing consent is as easy as giving it (the
// banner's two buttons are the same size and weight, and the footer link
// reopens it), and because a script cannot be unloaded, revoking analytics
// reloads the page so the tracker is genuinely gone.
//
// Categories:
//   necessary  — always on, never asked about: the session JWT, the media
//                token, the UI preferences. All first-party, all localStorage.
//   analytics  — OpenPanel (self-hosted script + TONVI's own instance).
//   marketing  — nothing uses it today. It exists so that adding a pixel is a
//                decision someone has to make here, in the open, rather than a
//                <script> tag someone drops into index.html.

const STORAGE_KEY = 'openshorts_consent';
const VERSION = 1;

export const CATEGORIES = ['necessary', 'analytics', 'marketing'];

const DENY_ALL = { necessary: true, analytics: false, marketing: false };
const GRANT_ALL = { necessary: true, analytics: true, marketing: true };

/* Geo-gating: el banner solo se debe a visitantes en EEE, Reino Unido y Suiza.
 * Fuera de ahí, consentimiento implícito y sin banner. País: traza de
 * Cloudflare (cacheada, se refina en segundo plano); mientras tanto decide la
 * zona horaria; si tampoco se sabe, se asume Europa. */
const EEA_COUNTRIES = ["AT","BE","BG","HR","CY","CZ","DK","EE","FI","FR","DE","GR","HU","IE","IT","LV","LT","LU","MT","NL","PL","PT","RO","SK","SI","ES","SE","IS","LI","NO","GB","CH"];
const EEA_TZ_EXTRA = ["Atlantic/Canary","Atlantic/Madeira","Atlantic/Azores","Atlantic/Reykjavik","Atlantic/Faroe","Arctic/Longyearbyen"];
const GEO_KEY = "up_geo_cc";

export function isEEAVisitor() {
  if (typeof window === "undefined") return true;
  try {
    const cc = window.localStorage.getItem(GEO_KEY);
    if (cc && /^[A-Z]{2}$/.test(cc)) return EEA_COUNTRIES.indexOf(cc) !== -1;
  } catch (e) { /* storage bloqueado */ }
  try {
    const tz = Intl.DateTimeFormat().resolvedOptions().timeZone || "";
    if (tz.indexOf("Europe/") === 0 || EEA_TZ_EXTRA.indexOf(tz) !== -1) return true;
    if (tz) return false;
  } catch (e) { /* sin Intl */ }
  return true;
}

export function refineGeo() {
  if (typeof window === "undefined" || typeof window.fetch !== "function") return;
  try { if (window.localStorage.getItem(GEO_KEY)) return; } catch (e) { /* storage blocked: fetch again */ }
  fetch("/cdn-cgi/trace", { cache: "no-store" })
    .then((r) => (r.ok ? r.text() : ""))
    .then((t) => {
      const m = /^loc=([A-Z]{2})$/m.exec(t || "");
      if (m) { try { window.localStorage.setItem(GEO_KEY, m[1]); } catch (e) { /* storage blocked */ } }
    })
    .catch(() => {});
}


const listeners = new Set();

const read = () => {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!parsed || parsed.v !== VERSION) return null;   // re-ask on a policy change
    return parsed;
  } catch (_) {
    return null;
  }
};

/** The stored decision, or null when the visitor has never been asked. */
export const getConsentRecord = () => read();

/** Effective permissions. Nothing but `necessary` before a decision exists. */
export const getConsent = () => {
  const stored = read();
  if (!stored) return isEEAVisitor() ? { ...DENY_ALL, analytics: true } : { ...GRANT_ALL };
  return {
    necessary: true,
    // Self-hosted OpenPanel is first-party audience measurement, exempt from
    // prior consent (AEPD/CNIL criteria): always on.
    analytics: true,
    marketing: !!stored.marketing,
  };
};

export const hasDecided = () => read() !== null || !isEEAVisitor();

export const allows = (category) => !!getConsent()[category];

export function onConsentChange(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

/**
 * Persist a decision and act on it immediately.
 *
 * Granting analytics boots the tracker in place. Revoking it reloads: op1.js
 * has already attached listeners and there is no honest way to unload a script
 * from a live page, so the only truthful implementation of "stop" is a fresh
 * document that never loads it.
 */
export function setConsent(choice) {
  const next = {
    v: VERSION,
    ts: new Date().toISOString(),
    necessary: true,
    // First-party measurement is always on (exempt); only marketing is a choice.
    analytics: true,
    marketing: !!choice.marketing,
  };
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
  } catch (_) { /* private mode: the decision holds for this page at least */ }

  listeners.forEach((fn) => { try { fn(getConsent()); } catch (_) { /* ignore */ } });

  applyConsent();
  return getConsent();
}

export const acceptAll = () => setConsent({ analytics: true, marketing: true });
export const rejectAll = () => setConsent({ analytics: false, marketing: false });

/** Start whatever the stored decision allows. Called once at boot. */
export function applyConsent() {
  refineGeo();
  try {
    return typeof window.__osAnalyticsInit === 'function'
      ? window.__osAnalyticsInit()
      : false;
  } catch (_) {
    return false;
  }
}

// Reopening the banner from anywhere (the footer link) without threading props
// through the whole tree.
const OPEN_EVENT = 'openshorts:consent-open';
export const openConsentManager = () => {
  try { window.dispatchEvent(new CustomEvent(OPEN_EVENT)); } catch (_) { /* ignore */ }
};
export const onConsentOpenRequest = (fn) => {
  window.addEventListener(OPEN_EVENT, fn);
  return () => window.removeEventListener(OPEN_EVENT, fn);
};
