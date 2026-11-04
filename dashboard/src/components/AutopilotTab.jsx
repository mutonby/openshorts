import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  Rocket, Youtube, Twitch, Loader2, RefreshCw, Play, CheckCircle2, AlertTriangle,
  Clock, Scissors, Send, ExternalLink, FolderOpen, Lock,
} from 'lucide-react';
import { apiJson, openQuotaWall } from '../lib/api';
import { track } from '../lib/analytics';

// Autopilot: every new video on the user's connected YouTube channel is turned
// into shorts automatically, and optionally the best ones are scheduled on
// their socials, one a day. The channel comes from the Upload-Post connection
// the app already uses for posting (GET /api/uploadposts/media under the hood),
// so there is nothing to paste: connect once, switch it on.

const PLATFORM_LABELS = { tiktok: 'TikTok', instagram: 'Instagram', youtube: 'YouTube Shorts' };

const ERROR_TEXT = {
  plan_required: 'Autopilot is paused: it needs an active Starter, Creator or Pro plan.',
  youtube_not_connected: 'Connect your YouTube channel so Autopilot can see your new videos.',
  youtube_reauth_required: 'YouTube asked to reconnect your channel. Reconnect it to keep Autopilot running.',
  twitch_not_connected: 'Connect your Twitch account so Autopilot can see your past broadcasts.',
  twitch_reauth_required: 'Twitch asked to reconnect your account. Reconnect it to keep Autopilot running.',
  channel_unavailable: 'We could not read your channel on the last check. We will try again shortly.',
};

const REASON_TEXT = {
  youtube_short: 'already a short',
  private_video: 'private on YouTube — set it to Unlisted to clip it',
  out_of_minutes: 'out of minutes',
  too_short: 'too short to clip',
  unavailable: 'not available yet (private or processing)',
  low_quality: 'only low quality available',
  timeout: 'did not finish',
  processing_failed: 'processing failed',
  publish_partial: 'some clips could not be scheduled',
  url_ingest_disabled: 'links are disabled on this server',
  still_live: 'still live: clipped once the broadcast ends',
};

const fmtDate = (iso) => (iso
  ? new Date(iso).toLocaleDateString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
  : '');

// ApiError.detail is a string or {error, message}; never hand an object to JSX.
const errText = (e, fallback) => {
  const d = e?.detail;
  if (typeof d === 'string' && d) return d;
  if (d && typeof d.message === 'string') return d.message;
  return fallback;
};

const browserTimezone = () => {
  try { return Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC'; } catch { return 'UTC'; }
};

function Switch({ checked, onChange, disabled, label }) {
  return (
    <span className="relative inline-flex items-center shrink-0">
      <input
        type="checkbox"
        role="switch"
        aria-label={label}
        checked={checked}
        disabled={disabled}
        onChange={(e) => onChange(e.target.checked)}
        className="sr-only peer"
      />
      <span className="w-11 h-6 rounded-full bg-paper3 border border-rule2 peer-checked:bg-brass peer-disabled:opacity-40 transition-colors after:content-[''] after:absolute after:left-1 after:top-1 after:w-4 after:h-4 after:rounded-full after:bg-ink after:transition-transform peer-checked:after:translate-x-5" />
    </span>
  );
}

function StatusBadge({ run }) {
  if (!run) return <span className="readout">not clipped</span>;
  const { status } = run;
  if (status === 'completed') {
    return (
      <span className="badge-ok inline-flex items-center gap-1">
        <CheckCircle2 size={11} /> {run.clips_count || 0} clips
        {run.posted_count ? ` · ${run.posted_count} scheduled` : ''}
      </span>
    );
  }
  if (status === 'processing' || status === 'queued') {
    return <span className="badge-brass inline-flex items-center gap-1"><Loader2 size={11} className="animate-spin" /> clipping</span>;
  }
  if (status === 'skipped') return <span className="badge-warn">{REASON_TEXT[run.reason] || 'skipped'}</span>;
  return <span className="badge-danger">{REASON_TEXT[run.reason] || 'failed'}</span>;
}

export default function AutopilotTab({ onOpenProject, onUpgrade, justConnected }) {
  const [data, setData] = useState(null);          // GET /api/autopilot
  const [videos, setVideos] = useState(null);      // GET /api/autopilot/videos
  const [videosError, setVideosError] = useState(null);
  const [loadError, setLoadError] = useState('');
  const [saving, setSaving] = useState(false);
  const [running, setRunning] = useState(null);    // video id being submitted
  const [notice, setNotice] = useState('');
  // Feedback for "clip it", shown next to the list: the page-level notice sits
  // at the top of the tab, off screen from the button, so a click looked dead.
  const [listNotice, setListNotice] = useState('');
  const [opening, setOpening] = useState(null);
  const pollRef = useRef(null);

  const load = useCallback(async () => {
    try {
      const d = await apiJson('/api/autopilot');
      setData(d);
      setLoadError('');
      if (d.youtube || d.twitch) {
        const v = await apiJson('/api/autopilot/videos');
        setVideos(v.videos || []);
        setVideosError(v.error || null);
      } else {
        setVideos([]);
      }
    } catch (e) {
      setLoadError(errText(e, 'Could not load Autopilot. Please try again.'));
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  // Coming back from the connect page (#app?tab=autopilot&connected=1).
  useEffect(() => {
    if (justConnected) {
      setNotice('Accounts connected.');
      track('AutopilotChannelConnected');
    }
  }, [justConnected]);

  // Refresh while something is being clipped, so the badge flips on its own.
  const busy = (data?.runs || []).some((r) => r.status === 'processing' || r.status === 'queued');
  useEffect(() => {
    clearInterval(pollRef.current);
    if (busy) pollRef.current = setInterval(load, 20000);
    return () => clearInterval(pollRef.current);
  }, [busy, load]);

  const save = useCallback(async (patch) => {
    setSaving(true);
    setNotice('');
    try {
      const d = await apiJson('/api/autopilot', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ timezone: browserTimezone(), ...patch }),
      });
      setData((prev) => ({ ...prev, settings: d.settings }));
      return d.settings;
    } catch (e) {
      setNotice(errText(e, 'Could not save. Please try again.'));
      return null;
    } finally {
      setSaving(false);
    }
  }, []);

  const connect = useCallback(async () => {
    track('AutopilotConnectClick');
    try {
      const { access_url } = await apiJson('/api/social/connect?return_to=autopilot', { method: 'POST' });
      if (access_url) window.location.href = access_url;
    } catch (e) {
      setNotice(errText(e, 'Could not open the connection page. Please try again.'));
    }
  }, []);

  const toggleEnabled = useCallback(async (on) => {
    const s = await save(on ? { enabled: true, rights_ack: true } : { enabled: false });
    if (s) {
      track(on ? 'AutopilotEnabled' : 'AutopilotDisabled', { props: { autopublish: String(!!s.autopublish) } });
      setNotice(on ? 'Autopilot is on. Your next video will be clipped automatically.' : 'Autopilot paused.');
    }
  }, [save]);

  const togglePlatform = useCallback((p) => {
    const cur = data?.settings?.publish_platforms || [];
    const next = cur.includes(p) ? cur.filter((x) => x !== p) : [...cur, p];
    save({ publish_platforms: next });
  }, [data, save]);

  const runNow = useCallback(async (video) => {
    setRunning(video.id);
    setListNotice('');
    try {
      const r = await apiJson('/api/autopilot/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_id: video.id }),
      });
      track('AutopilotManualRun', { props: { status: r.status } });
      if (r.status === 'processing') setListNotice(`Clipping “${video.title || 'your video'}”. We will email you when the clips are ready.`);
      else setListNotice(`Not clipped: ${REASON_TEXT[r.reason] || r.reason || r.status}.`);
    } catch (e) {
      if (e?.name === 'QuotaError') {
        setListNotice('You are out of minutes for this period.');
        openQuotaWall(e, 'autopilot_run');
      }
      else setListNotice(errText(e, 'Could not start the job.'));
    } finally {
      setRunning(null);
      load();
    }
  }, [load]);

  const openProject = useCallback(async (jobId) => {
    if (!onOpenProject || opening) return;
    setOpening(jobId);
    try { await onOpenProject(jobId); } catch { setNotice('Could not open this project yet. Find it in History.'); }
    setOpening(null);
  }, [onOpenProject, opening]);

  if (!data && !loadError) {
    return <div className="flex justify-center py-20"><Loader2 className="animate-spin text-brass" /></div>;
  }

  const s = data?.settings || {};
  const yt = data?.youtube;
  const tw = data?.twitch;
  const twitchAvailable = !!data?.twitch_available;
  const accounts = data?.accounts || {};
  const eligible = !!data?.eligible;
  const limits = data?.limits || { max_minutes_choices: [10, 20, 30, 45, 60, 90], max_clips_to_publish: 5, platforms: ['tiktok', 'instagram', 'youtube'] };
  const runs = data?.runs || [];
  const errorText = s.enabled && s.last_error ? ERROR_TEXT[s.last_error] : null;

  return (
    <div className="max-w-4xl mx-auto animate-fade">
      <p className="eyebrow mb-1.5">02 · AUTOPILOT</p>
      <h1 className="font-display lowercase text-3xl text-ink mb-2">your channel, clipped on its own</h1>
      <p className="text-muted text-sm mb-8 max-w-2xl">
        Publish on YouTube as usual. Autopilot notices every new video, turns it into shorts
        and emails you when they are ready. Switch on autopublish and the best ones go out on
        TikTok, Instagram and YouTube Shorts, one a day.
      </p>

      {loadError && <p className="text-danger text-sm mb-4">{loadError}</p>}
      {notice && (
        <div className="mb-6 rounded-card border border-brass/40 bg-brass/5 px-4 py-3 text-sm text-ink2">{notice}</div>
      )}

      {/* Plan gate: Autopilot spends minutes on its own, so it is a paid feature. */}
      {!eligible && (
        <div className="card p-6 mb-6">
          <h3 className="font-display lowercase text-lg text-ink mb-1 flex items-center gap-2">
            <Lock size={16} className="text-brass" /> included in paid plans
          </h3>
          <p className="text-muted text-sm mb-4">
            Autopilot runs on your plan&apos;s minutes, so it comes with Starter, Creator and Pro.
            Every new upload turns into shorts without you opening OpenShorts.
          </p>
          <button onClick={() => { track('AutopilotUpgradeClick'); onUpgrade?.(); }} className="btn-primary">
            <Rocket size={16} /> choose a plan
          </button>
        </div>
      )}

      {/* 1 · channel */}
      <div className="card p-6 mb-6">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3 min-w-0">
            <div className="w-10 h-10 rounded-full bg-paper3 border border-rule flex items-center justify-center shrink-0">
              <Youtube size={18} className={yt ? 'text-brass' : 'text-muted'} />
            </div>
            <div className="min-w-0">
              <p className="eyebrow">01 · CHANNEL</p>
              {yt ? (
                <p className="text-ink truncate">{yt.display_name || yt.handle} <span className="readout ml-1">{yt.handle}</span></p>
              ) : (
                <p className="text-muted text-sm">no YouTube channel connected yet</p>
              )}
            </div>
          </div>
          {(!yt || yt.reauth_required) && (
            <button onClick={connect} className="btn-primary" disabled={!eligible}>
              <Youtube size={16} /> {yt ? 'reconnect youtube' : 'connect youtube'}
            </button>
          )}
          {yt && !yt.reauth_required && (
            <button onClick={connect} className="btn-quiet text-xs">manage accounts</button>
          )}
        </div>
        {yt?.reauth_required && (
          <p className="text-warn text-sm mt-3 flex items-center gap-2"><AlertTriangle size={14} /> YouTube needs you to reconnect this channel.</p>
        )}
        {/* Twitch: past broadcasts are a second source. Read-only: nothing is
            ever posted to Twitch. The stretch viewers clipped most is clipped. */}
        {twitchAvailable && (
          <div className="flex flex-wrap items-center justify-between gap-4 mt-5 pt-5 border-t border-rule">
            <div className="flex items-center gap-3 min-w-0">
              <div className="w-10 h-10 rounded-full bg-paper3 border border-rule flex items-center justify-center shrink-0">
                <Twitch size={18} className={tw ? 'text-brass' : 'text-muted'} />
              </div>
              <div className="min-w-0">
                {tw ? (
                  <p className="text-ink truncate">{tw.display_name || tw.handle} <span className="readout ml-1">{tw.handle}</span></p>
                ) : (
                  <p className="text-muted text-sm">stream on Twitch? connect it and every past broadcast is clipped once it ends, starting from the part your viewers clipped</p>
                )}
              </div>
            </div>
            {(!tw || tw.reauth_required) && (
              <button onClick={connect} className="btn-ghost px-4 py-2 text-sm" disabled={!eligible}>
                <Twitch size={16} /> {tw ? 'reconnect twitch' : 'connect twitch'}
              </button>
            )}
          </div>
        )}
        {tw?.reauth_required && (
          <p className="text-warn text-sm mt-3 flex items-center gap-2"><AlertTriangle size={14} /> Twitch needs you to reconnect this account.</p>
        )}
      </div>

      {/* 2 · switch */}
      <div className="card p-6 mb-6">
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="eyebrow mb-1">02 · AUTOPILOT</p>
            <p className="text-ink">{s.enabled ? 'on: new videos are clipped automatically' : 'off'}</p>
            <p className="text-muted text-sm mt-1">
              Only videos you publish from now on. Up to {limits.daily_auto || 1} video a day, the first
              {' '}{s.max_minutes || 30} minutes of each. Shorts are skipped.
            </p>
            {s.enabled && s.last_checked_at && (
              <p className="readout mt-2 flex items-center gap-1"><Clock size={11} /> last check {fmtDate(s.last_checked_at)} · every hour</p>
            )}
          </div>
          <Switch
            label="autopilot"
            checked={!!s.enabled}
            disabled={saving || !eligible || !(yt || tw)}
            onChange={toggleEnabled}
          />
        </div>

        {errorText && (
          <p className="text-warn text-sm mt-4 flex items-center gap-2"><AlertTriangle size={14} /> {errorText}</p>
        )}

        <div className="grid sm:grid-cols-2 gap-4 mt-6">
          <label className="block">
            <span className="eyebrow block mb-2">minutes per video</span>
            <select
              className="input-field"
              value={s.max_minutes || 30}
              disabled={saving || !eligible}
              onChange={(e) => save({ max_minutes: Number(e.target.value) })}
            >
              {limits.max_minutes_choices.map((m) => <option key={m} value={m}>first {m} min</option>)}
            </select>
          </label>
        </div>
      </div>

      {/* 3 · autopublish */}
      <div className="card p-6 mb-6">
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="eyebrow mb-1">03 · AUTOPUBLISH</p>
            <p className="text-ink">{s.autopublish ? 'the best clips are scheduled for you' : 'off: you review and post the clips yourself'}</p>
            <p className="text-muted text-sm mt-1">
              Picks the {s.clips_to_publish || 3} highest-scoring clips of each video and schedules
              them one a day at {String(s.publish_hour ?? 17).padStart(2, '0')}:00 ({s.timezone || browserTimezone()}).
            </p>
          </div>
          <Switch
            label="autopublish"
            checked={!!s.autopublish}
            disabled={saving || !eligible}
            onChange={(on) => { track('AutopilotAutopublish', { props: { on: String(on) } }); save({ autopublish: on }); }}
          />
        </div>

        {s.autopublish && (
          <>
            <div className="mt-5">
              <span className="eyebrow block mb-2">publish to</span>
              <div className="flex flex-wrap gap-2">
                {limits.platforms.map((p) => {
                  const connected = !!accounts[p];
                  const on = (s.publish_platforms || []).includes(p);
                  return connected ? (
                    <button
                      key={p}
                      onClick={() => togglePlatform(p)}
                      disabled={saving}
                      aria-pressed={on}
                      className={`px-4 py-2 rounded-full text-sm lowercase border transition-colors ${on ? 'border-brass bg-brass/10 text-ink' : 'border-rule2 text-muted hover:text-ink2'}`}
                    >
                      {on && <CheckCircle2 size={13} className="inline mr-1.5 -mt-0.5 text-brass" />}{PLATFORM_LABELS[p]}
                    </button>
                  ) : (
                    <button key={p} onClick={connect} className="px-4 py-2 rounded-full text-sm lowercase border border-dashed border-rule2 text-muted hover:text-ink2">
                      + connect {PLATFORM_LABELS[p]}
                    </button>
                  );
                })}
              </div>
              {(s.publish_platforms || []).length === 0 && (
                <p className="text-warn text-sm mt-3">Pick at least one account, or nothing gets published.</p>
              )}
              {(s.publish_platforms || []).includes('tiktok') && (
                <p className="text-muted text-xs mt-3">TikTok clips arrive as drafts in your TikTok inbox: open the app to post them.</p>
              )}
            </div>
            <div className="grid sm:grid-cols-2 gap-4 mt-5">
              <label className="block">
                <span className="eyebrow block mb-2">clips per video</span>
                <select
                  className="input-field"
                  value={s.clips_to_publish || 3}
                  disabled={saving}
                  onChange={(e) => save({ clips_to_publish: Number(e.target.value) })}
                >
                  {Array.from({ length: limits.max_clips_to_publish || 5 }, (_, i) => i + 1).map((n) => (
                    <option key={n} value={n}>{n} {n === 1 ? 'clip' : 'clips'} (one a day)</option>
                  ))}
                </select>
              </label>
              <label className="block">
                <span className="eyebrow block mb-2">time of day</span>
                <select
                  className="input-field"
                  value={s.publish_hour ?? 17}
                  disabled={saving}
                  onChange={(e) => save({ publish_hour: Number(e.target.value) })}
                >
                  {Array.from({ length: 24 }, (_, h) => h).map((h) => (
                    <option key={h} value={h}>{String(h).padStart(2, '0')}:00</option>
                  ))}
                </select>
              </label>
            </div>
          </>
        )}
      </div>

      {/* 4 · channel videos */}
      {(yt || tw) && (
        <div className="card p-6 mb-6">
          <div className="flex items-center justify-between gap-3 mb-4">
            <div>
              <p className="eyebrow mb-1">04 · YOUR LATEST VIDEOS</p>
              <p className="text-muted text-sm">Clip any of them now, or let Autopilot handle the next one.</p>
            </div>
            <button onClick={load} className="btn-quiet text-xs" aria-label="refresh"><RefreshCw size={13} /> refresh</button>
          </div>
          {listNotice && (
            <div className="mb-4 rounded-card border border-brass/40 bg-brass/5 px-4 py-3 text-sm text-ink2" role="status">{listNotice}</div>
          )}
          {videosError && <p className="text-warn text-sm mb-3">{ERROR_TEXT[videosError] || 'Could not read your channel.'}</p>}
          {videos === null && <div className="flex justify-center py-6"><Loader2 className="animate-spin text-brass" size={18} /></div>}
          {videos && videos.length === 0 && !videosError && (
            <p className="text-muted text-sm">No videos on this channel yet.</p>
          )}

          <div className="divide-y divide-rule">
            {(videos || []).map((v) => {
              const run = v.run;
              // Skipped runs can be retried too: private videos used to be
              // misread as Shorts, and a real Short just skips again.
              const canRun = eligible && !v.is_live && (!run || run.status === 'failed' || run.status === 'skipped');
              return (
                <div key={v.id} className="flex items-center gap-3 py-3">
                  {v.thumbnail_url ? (
                    <img src={v.thumbnail_url} alt="" className="w-20 sm:w-24 h-[45px] sm:h-[54px] object-cover rounded-input border border-rule shrink-0" loading="lazy" />
                  ) : (
                    <div className="w-20 sm:w-24 h-[45px] sm:h-[54px] rounded-input bg-paper3 shrink-0" />
                  )}
                  <div className="min-w-0 flex-1">
                    <a href={v.url} target="_blank" rel="noopener noreferrer" className="text-sm text-ink2 hover:text-ink line-clamp-2">
                      {v.title || v.id} <ExternalLink size={11} className="inline -mt-0.5 text-muted" />
                    </a>
                    <div className="flex flex-wrap items-center gap-2 mt-1">
                      {v.source === 'twitch' && <Twitch size={12} className="text-muted" aria-label="Twitch broadcast" />}
                      <span className="readout">{fmtDate(v.published_at)}</span>
                      {v.is_live && !run ? <span className="badge-warn">live now</span> : <StatusBadge run={run} />}
                    </div>
                  </div>
                  {run?.status === 'completed' && run.job_id ? (
                    <button onClick={() => openProject(run.job_id)} className="btn-quiet text-xs shrink-0" disabled={opening === run.job_id}>
                      {opening === run.job_id ? <Loader2 size={13} className="animate-spin" /> : <FolderOpen size={13} />} open
                    </button>
                  ) : canRun ? (
                    <button onClick={() => runNow(v)} className="btn-ghost px-4 py-2 text-xs shrink-0" disabled={!!running}>
                      {running === v.id ? <Loader2 size={13} className="animate-spin" /> : <Scissors size={13} />} clip it
                    </button>
                  ) : null}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* 5 · activity */}
      {runs.length > 0 && (
        <div className="card p-6 mb-10">
          <p className="eyebrow mb-4">05 · ACTIVITY</p>
          <div className="space-y-3">
            {runs.map((r) => (
              <div key={r.id} className="flex items-center gap-3 text-sm">
                {r.status === 'completed' ? <CheckCircle2 size={15} className="text-ok shrink-0" />
                  : (r.status === 'processing' || r.status === 'queued') ? <Loader2 size={15} className="animate-spin text-brass shrink-0" />
                    : r.status === 'skipped' ? <Play size={15} className="text-muted shrink-0" />
                      : <AlertTriangle size={15} className="text-danger shrink-0" />}
                <span className="text-ink2 truncate flex-1">{r.video_title || r.video_id}</span>
                {r.posted_count ? <span className="readout hidden sm:inline-flex items-center gap-1"><Send size={11} /> {r.posted_count} scheduled</span> : null}
                <span className="readout shrink-0">{r.trigger === 'manual' ? 'manual' : 'auto'} · {fmtDate(r.created_at)}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
