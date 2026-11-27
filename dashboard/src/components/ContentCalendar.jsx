import { useEffect, useMemo, useState } from 'react';
import { Calendar, Loader2, RefreshCw } from 'lucide-react';
import { apiJson, QuotaError } from '../lib/api';
import { platformLabel } from '../lib/socialPlatforms';
import Modal from './ui/Modal';

const SCHEDULE_PATH = '/api/social/scheduled';
const dateFormat = new Intl.DateTimeFormat(undefined, { dateStyle: 'full' });
const timeFormat = new Intl.DateTimeFormat(undefined, {
  hour: 'numeric',
  minute: '2-digit',
  timeZoneName: 'short',
});
const localTimezone = timeFormat.resolvedOptions().timeZone;

function localInputValue(date) {
  const offset = date.getTimezoneOffset() * 60_000;
  return new Date(date.getTime() - offset).toISOString().slice(0, 16);
}

function groupByDate(posts) {
  const scheduled = posts
    .filter((post) => typeof post?.job_id === 'string' && post.job_id)
    .map((post) => {
      const timestamp = Date.parse(post.scheduled_date);
      return { post, date: Number.isFinite(timestamp) ? new Date(timestamp) : null };
    })
    .sort((a, b) => (a.date?.getTime() ?? Number.MAX_SAFE_INTEGER) - (b.date?.getTime() ?? Number.MAX_SAFE_INTEGER));

  const groups = new Map();
  for (const item of scheduled) {
    const day = item.date ? dateFormat.format(item.date) : 'Date unavailable';
    if (!groups.has(day)) groups.set(day, []);
    groups.get(day).push(item);
  }
  return Array.from(groups);
}

export default function ContentCalendar({ isManaged, uploadPostKey, uploadUserId, onConnect, onClose }) {
  const [calendar, setCalendar] = useState({ loading: true, posts: [], error: '', needsConnection: false });
  const [revision, setRevision] = useState(0);
  const [action, setAction] = useState(null);
  const [busy, setBusy] = useState(false);
  const [actionError, setActionError] = useState('');
  const [notice, setNotice] = useState('');
  const key = isManaged ? '' : uploadPostKey;
  const headers = useMemo(() => key ? { 'X-Upload-Post-Key': key } : {}, [key]);
  const profileQuery = isManaged ? '' : `?${new URLSearchParams({ user: uploadUserId })}`;

  useEffect(() => {
    if (!isManaged && (!key || !uploadUserId)) return undefined;
    const controller = new AbortController();
    apiJson(`${SCHEDULE_PATH}${profileQuery}`, { headers, signal: controller.signal })
      .then((data) => {
        if (controller.signal.aborted) return;
        setCalendar({
          loading: false,
          posts: Array.isArray(data.scheduled_posts) ? data.scheduled_posts : [],
          error: '',
          needsConnection: false,
        });
      })
      .catch((error) => {
        if (controller.signal.aborted) return;
        setCalendar({
          loading: false,
          posts: [],
          error: error instanceof QuotaError
            ? 'An active plan is required to view the content calendar.'
            : error.detail || 'Could not load scheduled posts.',
          needsConnection: isManaged && error.status === 409,
        });
      });
    return () => controller.abort();
  }, [headers, isManaged, key, profileQuery, revision, uploadUserId]);

  const days = useMemo(() => groupByDate(calendar.posts), [calendar.posts]);
  const canLoad = isManaged || (key && uploadUserId);

  function refresh() {
    setAction(null);
    setActionError('');
    setNotice('');
    setCalendar((current) => ({ ...current, loading: true, error: '', needsConnection: false }));
    setRevision((current) => current + 1);
  }

  function startAction(post, mode, date) {
    setNotice('');
    setActionError('');
    setAction({ jobId: post.job_id, mode, time: date ? localInputValue(date) : '' });
  }

  async function saveAction(event) {
    event.preventDefault();
    if (busy) return;
    const { jobId, mode, time } = action;
    const cancel = mode === 'cancel';
    const date = new Date(time);
    if (!cancel && !(date.getTime() > Date.now() && localInputValue(date) === time)) {
      setActionError('Choose a valid future time.');
      return;
    }
    setBusy(true);
    setActionError('');
    try {
      const result = await apiJson(`${SCHEDULE_PATH}/${encodeURIComponent(jobId)}${profileQuery}`, cancel
        ? { method: 'DELETE', headers }
        : {
            method: 'PATCH',
            headers: { ...headers, 'Content-Type': 'application/json' },
            body: JSON.stringify({ scheduled_date: date.toISOString() }),
          });
      setCalendar((current) => ({
        ...current,
        posts: cancel
          ? current.posts.filter((post) => post.job_id !== jobId)
          : current.posts.map((post) => post.job_id === jobId
            ? { ...post, scheduled_date: result.scheduled_date }
            : post),
      }));
      setNotice(cancel ? 'Scheduled post canceled.' : 'Publish time updated.');
      setAction(null);
    } catch (error) {
      setActionError(error.detail || 'Could not update the scheduled post.');
    } finally {
      setBusy(false);
    }
  }

  return (
    <Modal isOpen onClose={busy ? undefined : onClose} title="Content calendar" size="lg">
      {!canLoad ? (
        <p className="text-sm text-muted">Connect an Upload-Post profile to view its scheduled posts.</p>
      ) : (
        <div className="space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <p className="text-xs text-muted">
              {!isManaged && <>{uploadUserId} · </>}Times shown in {localTimezone}
            </p>
            <button type="button" className="btn-quiet px-3 py-1.5 text-xs" onClick={refresh}
              disabled={calendar.loading || busy}>
              <RefreshCw size={14} className={calendar.loading ? 'animate-spin' : ''} aria-hidden="true" />
              refresh
            </button>
          </div>
          {notice && <p className="text-sm text-ok" role="status">{notice}</p>}
          {calendar.error && (
            <div className="flex flex-wrap items-center gap-2">
              <p className="text-sm text-danger" role="alert">{calendar.error}</p>
              {calendar.needsConnection && onConnect && (
                <button type="button" className="btn-quiet px-3 py-1.5 text-xs" onClick={onConnect}>
                  connect social accounts
                </button>
              )}
            </div>
          )}
          {calendar.loading && days.length === 0 && (
            <p className="flex items-center gap-2 py-3 text-sm text-muted" role="status">
              <Loader2 size={16} className="animate-spin" aria-hidden="true" /> Loading scheduled posts…
            </p>
          )}
          {!calendar.loading && !calendar.error && days.length === 0 && (
            <p className="flex items-center gap-2 py-3 text-sm text-muted">
              <Calendar size={16} aria-hidden="true" /> No upcoming posts.
            </p>
          )}
          {days.map(([day, posts]) => (
            <section key={day}>
              <h3 className="eyebrow mb-2">{day}</h3>
              <ul className="divide-y divide-rule rounded-input border border-rule">
                {posts.map(({ post, date }) => {
                  const title = [post.title, post.source_filename].find((text) => typeof text === 'string' && text.trim())
                    || 'Untitled post';
                  const platforms = Array.isArray(post.platforms)
                    ? post.platforms.filter((platform) => typeof platform === 'string').map(platformLabel).join(', ')
                    : '';
                  const active = action?.jobId === post.job_id;
                  return (
                    <li key={post.job_id} className="p-3">
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <div className="min-w-0">
                          <p className="break-words text-sm text-ink">{title}</p>
                          <p className="flex flex-wrap gap-x-2 text-xs text-muted">
                            {date && <time dateTime={date.toISOString()}>{timeFormat.format(date)}</time>}
                            {platforms && <span>{platforms}</span>}
                          </p>
                        </div>
                        {!active && (
                          <div className="flex flex-wrap gap-2">
                            <button type="button" className="btn-quiet px-2.5 py-1.5 text-xs"
                              onClick={() => startAction(post, 'move', date)}
                              disabled={calendar.loading || busy} aria-label={`Reschedule ${title}`}>
                              reschedule
                            </button>
                            <button type="button" className="btn-ghost px-2.5 py-1.5 text-xs"
                              onClick={() => startAction(post, 'cancel')}
                              disabled={calendar.loading || busy} aria-label={`Cancel scheduled post ${title}`}>
                              cancel post
                            </button>
                          </div>
                        )}
                      </div>
                      {active && (
                        <form className="mt-3 space-y-2 border-t border-rule pt-3" onSubmit={saveAction}>
                          {action.mode === 'move' ? (
                            <label className="block max-w-xs text-xs text-muted">
                              Publish date and time
                              <input type="datetime-local" className="input-field mt-1 [color-scheme:dark]"
                                required value={action.time} disabled={busy}
                                onChange={(event) => setAction({ ...action, time: event.target.value })} />
                            </label>
                          ) : (
                            <p className="text-sm text-ink2">Cancel this post on every scheduled platform?</p>
                          )}
                          {actionError && <p className="text-sm text-danger" role="alert">{actionError}</p>}
                          <div className="flex flex-wrap gap-2">
                            <button type="button" className="btn-ghost px-3 py-1.5 text-xs"
                              onClick={() => setAction(null)} disabled={busy}>
                              {action.mode === 'move' ? 'discard' : 'keep post'}
                            </button>
                            <button type="submit" className={`${action.mode === 'move' ? 'btn-primary' : 'btn-danger'} px-3 py-1.5 text-xs`}
                              disabled={busy || (action.mode === 'move' && date && action.time === localInputValue(date))}>
                              {busy && <Loader2 size={14} className="animate-spin" aria-hidden="true" />}
                              {action.mode === 'move' ? 'save time' : 'confirm cancellation'}
                            </button>
                          </div>
                        </form>
                      )}
                    </li>
                  );
                })}
              </ul>
            </section>
          ))}
        </div>
      )}
    </Modal>
  );
}
