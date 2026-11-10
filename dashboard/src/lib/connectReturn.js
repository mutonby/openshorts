// Coming back from the Upload-Post connect page. A connect started there
// returns to our redirectUrl with the outcome in the query string, before the
// hash: ?connect_status=success|cancelled|error&platform=<p>&error_code=<CODE>.
// Pressing the page's own "Back to OpenShorts" button carries no outcome.

const CONNECT_PARAMS = ['connect_status', 'platform', 'error_code', 'error_description'];
const STATUSES = ['success', 'cancelled', 'error'];

const PLATFORM_NAMES = {
  tiktok: 'TikTok',
  instagram: 'Instagram',
  youtube: 'YouTube',
  twitch: 'Twitch',
};

// { status, platform, errorCode } from a location.search, or null without an outcome.
export function readConnectReturn(search) {
  const params = new URLSearchParams(search || '');
  const status = params.get('connect_status');
  if (!STATUSES.includes(status)) return null;
  return {
    status,
    platform: params.get('platform') || null,
    errorCode: params.get('error_code') || null,
  };
}

// The same search without the connect outcome, so a reload does not show it again.
export function stripConnectReturn(search) {
  const params = new URLSearchParams(search || '');
  CONNECT_PARAMS.forEach((k) => params.delete(k));
  const rest = params.toString();
  return rest ? `?${rest}` : '';
}

// The notice for an outcome, or '' when there is nothing to say.
export function connectNotice(result) {
  if (!result) return '';
  const name = PLATFORM_NAMES[result.platform] || '';
  if (result.status === 'success') return name ? `${name} connected.` : 'Account connected.';
  if (result.status === 'cancelled') {
    return name ? `${name} was not connected: the connection was cancelled.` : 'The connection was cancelled.';
  }
  if (result.errorCode === 'INVALID_STATE') return 'The connection link expired. Please connect again.';
  return name
    ? `${name} could not be connected. Please try again.`
    : 'The account could not be connected. Please try again.';
}
