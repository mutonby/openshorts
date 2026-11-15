export const LIVE_ACTIVE_STATES = ['scheduled', 'waiting', 'recording', 'paused', 'ended'];

export const liveStartEpoch = (hhmm) => {
  if (!hhmm) return null;
  const [h, m] = hhmm.split(':').map(Number);
  const t = new Date();
  t.setHours(h, m, 0, 0);
  if (t.getTime() < Date.now() - 5 * 60 * 1000) t.setDate(t.getDate() + 1);
  return t.getTime();
};

export const fmtWhen = (ms) => new Date(ms).toLocaleString(undefined, { weekday: 'short', hour: '2-digit', minute: '2-digit' });
