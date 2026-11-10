// Centralized API client for cloud mode.
// Adds the Authorization: Bearer header from the stored session token, and turns
// a 402 (quota exceeded) into a typed QuotaError the UI can catch to prompt a top-up.
import { getApiUrl } from '../config';

export const AUTH_TOKEN_KEY = 'openshorts_auth';

export const getToken = () => localStorage.getItem(AUTH_TOKEN_KEY) || '';
export const setToken = (t) => localStorage.setItem(AUTH_TOKEN_KEY, t);
export const clearToken = () => localStorage.removeItem(AUTH_TOKEN_KEY);

export class QuotaError extends Error {
  constructor(detail) {
    super('quota_exceeded');
    this.name = 'QuotaError';
    this.minutesRequired = detail?.minutes_required;
    this.minutesRemaining = detail?.minutes_remaining;
    // Minutes of the source the server offers to clip instead (0 = no offer).
    this.partialMinutes = Number(detail?.partial_minutes) || 0;
    // The server's reason: 'quota_exceeded', 'no_plan', or 'free_queue_full'
    // (the free queue's brake refused the submission; nothing was spent).
    this.code = detail?.error || 'quota_exceeded';
  }
}

// A 402 caught outside the main clip flow (thumbnails, clip editor, autopilot):
// ask the app shell to open the quota wall instead of a dead-end error. `source`
// tags CheckoutStarted so each surface can be measured.
export const QUOTA_WALL_EVENT = 'os:quota-wall';
export function openQuotaWall(err, source) {
  try {
    window.dispatchEvent(new CustomEvent(QUOTA_WALL_EVENT, {
      detail: { required: err?.minutesRequired, remaining: err?.minutesRemaining, source },
    }));
  } catch (_) { /* ignore */ }
}

// Drop-in fetch wrapper. Always attaches the bearer token when present.
export async function apiFetch(path, options = {}) {
  const headers = new Headers(options.headers || {});
  const token = getToken();
  if (token) headers.set('Authorization', `Bearer ${token}`);

  const res = await fetch(getApiUrl(path), { ...options, headers });

  if (res.status === 402) {
    let detail = {};
    try {
      const body = await res.clone().json();
      detail = body.detail || body;
    } catch (_) { /* ignore */ }
    throw new QuotaError(detail);
  }
  return res;
}

// A failed request, carrying the server's own explanation. Callers that show an
// alert should prefer `detail` — the generic "something went wrong" strings hid
// real, actionable messages (e.g. a declined card blocking checkout).
export class ApiError extends Error {
  constructor(status, detail, raw) {
    super(`${status}: ${raw}`);   // message kept verbatim for existing callers
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
  }
}

// Convenience JSON helper.
export async function apiJson(path, options = {}) {
  const res = await apiFetch(path, options);
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    let detail = '';
    try {
      const d = JSON.parse(text).detail;
      // FastAPI details are strings here, but validation errors arrive as objects.
      if (typeof d === 'string') detail = d;
    } catch (_) { /* not JSON — leave detail empty and fall back to the caller's copy */ }
    throw new ApiError(res.status, detail, text);
  }
  return res.json();
}
