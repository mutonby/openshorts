import React, { useEffect, useMemo, useState } from 'react';
import { Instagram, Youtube, Loader2 } from 'lucide-react';
import Modal from './ui/Modal';
import SegmentedControl from './ui/SegmentedControl';
import { apiFetch } from '../lib/api';
import { useAuth } from '../contexts/AuthContext';
import { TIMEZONES, detectTimezone } from './ScheduleWeekModal';

const MODES = [
  { value: 'now', label: 'now' },
  { value: 'at', label: 'pick a time' },
  { value: 'interval', label: 'every n' },
  { value: 'mixed', label: 'mix' },
];

function defaultStart() {
  const date = new Date();
  date.setDate(date.getDate() + 1);
  date.setHours(18, 0, 0, 0);
  const pad = (n) => String(n).padStart(2, '0');
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}T18:00`;
}

function splitNames(value) {
  return String(value || '')
    .split(',')
    .map((name) => name.trim().replace(/^@/, ''))
    .filter(Boolean);
}

function errorText(body, fallback) {
  if (!body) return fallback;
  if (typeof body.detail === 'string') return body.detail;
  if (Array.isArray(body.detail)) return body.detail.map((item) => item.msg || String(item)).join(' ');
  return fallback;
}

export default function LocalPostPanel({ isOpen, onClose, clips, jobId, focusIndex = null }) {
  const { dispatch } = useAuth();
  const flags = dispatch || { youtube: false, instagram: false, r2: false };
  const youtubeReady = !!flags.youtube;
  const instagramReady = !!flags.instagram && !!flags.r2;

  const [rows, setRows] = useState([]);
  const [platforms, setPlatforms] = useState([]);
  const [timezone, setTimezone] = useState(detectTimezone);
  const [mode, setMode] = useState('interval');
  const [start, setStart] = useState(defaultStart);
  const [every, setEvery] = useState(1);
  const [everyUnit, setEveryUnit] = useState('days');
  const [preview, setPreview] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [queued, setQueued] = useState(null);

  useEffect(() => {
    if (!isOpen) return;
    const initial = (clips || []).map((clip, index) => ({
      index,
      checked: focusIndex == null || focusIndex === index,
      title: clip.video_title_for_youtube_short || clip.title || 'Short',
      caption: clip.video_description_for_instagram || clip.video_description_for_tiktok || '',
      collaborators: '',
      when: '',
    }));
    setRows(initial);
    setPlatforms([...(youtubeReady ? ['youtube'] : []), ...(instagramReady ? ['instagram'] : [])]);
    setPreview(null);
    setQueued(null);
    setError('');
    setMode(focusIndex == null ? 'interval' : 'now');
  }, [isOpen, clips, focusIndex, youtubeReady, instagramReady]);

  const platformOptions = useMemo(() => ([
    {
      value: 'youtube',
      label: 'youtube',
      icon: <Youtube size={16} />,
      disabled: !youtubeReady,
      hint: youtubeReady ? '' : 'set YOUTUBE_REFRESH_TOKEN',
    },
    {
      value: 'instagram',
      label: 'instagram',
      icon: <Instagram size={16} />,
      disabled: !instagramReady,
      hint: !flags.instagram ? 'set IG_TOKEN' : (!flags.r2 ? 'set R2_PUBLIC_BASE' : ''),
    },
  ]), [youtubeReady, instagramReady, flags.instagram, flags.r2]);

  const checked = rows.filter((row) => row.checked);
  const needsStart = mode === 'interval' || mode === 'mixed';
  const everyMinutes = everyUnit === 'days' ? Number(every) * 1440 : Number(every) * 60;
  const formReady = checked.length > 0 && platforms.length > 0 && (!needsStart || (start && everyMinutes > 0));

  function updateRow(index, patch) {
    setPreview(null);
    setRows((current) => current.map((row) => (row.index === index ? { ...row, ...patch } : row)));
  }

  function payload() {
    return {
      job_id: jobId,
      timezone,
      mode,
      start: needsStart ? start : null,
      every_minutes: needsStart ? everyMinutes : null,
      clips: checked.map((row) => ({
        index: row.index,
        platforms,
        title: row.title,
        caption: row.caption,
        collaborators: platforms.includes('instagram') ? splitNames(row.collaborators) : [],
        when: mode === 'at' || (mode === 'mixed' && row.when) ? row.when : null,
      })),
    };
  }

  async function runPreview() {
    setBusy(true);
    setError('');
    try {
      const res = await apiFetch('/api/dispatch/preview', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload()),
      });
      const body = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(errorText(body, 'Could not preview that schedule.'));
      setPreview(body.planned || []);
    } catch (err) {
      setPreview(null);
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function submit() {
    setBusy(true);
    setError('');
    try {
      const res = await apiFetch('/api/dispatch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload()),
      });
      const body = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(errorText(body, 'Could not queue those posts.'));
      setQueued(true);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    if (!queued || !isOpen) return undefined;
    let stop = false;
    const tick = async () => {
      try {
        const res = await apiFetch(`/api/dispatch?job_id=${encodeURIComponent(jobId)}`);
        if (!res.ok || stop) return;
        const body = await res.json();
        setPreview((body.rows || []).map((row) => ({
          clip_index: row.clip_index,
          platform: row.platform,
          local_time: row.local_time,
          status: row.status,
          last_error: row.last_error,
          id: row.id,
        })));
      } catch {
        /* the next tick retries */
      }
    };
    tick();
    const timer = setInterval(tick, 2000);
    return () => { stop = true; clearInterval(timer); };
  }, [queued, isOpen, jobId]);

  async function cancelRow(id) {
    await apiFetch(`/api/dispatch/${id}/cancel`, { method: 'POST' });
    setPreview((current) => (current || []).map((row) => (
      row.id === id ? { ...row, status: 'canceled' } : row
    )));
  }

  const footer = (
    <div className="flex flex-col sm:flex-row gap-2">
      <button type="button" className="btn-ghost flex-1" onClick={onClose} disabled={busy}>close</button>
      {!queued && (
        <button type="button" className="btn-ghost flex-1" onClick={runPreview} disabled={!formReady || busy}>
          {busy ? <Loader2 size={14} className="animate-spin" /> : null} preview
        </button>
      )}
      <button type="button" className="btn-primary flex-1" onClick={submit} disabled={!formReady || !preview || busy || queued}>
        {busy ? <Loader2 size={14} className="animate-spin" /> : null}
        {mode === 'now' ? 'post' : 'schedule'}
      </button>
    </div>
  );

  return (
    <Modal isOpen={isOpen} onClose={onClose} eyebrow="PUBLISH" title="post clips" size="lg" footer={footer}>
      <div className="space-y-4">
        <SegmentedControl multi options={platformOptions} value={platforms} onChange={(next) => { setPlatforms(next); setPreview(null); }} />
        <SegmentedControl options={MODES} value={mode} onChange={(next) => { setMode(next); setPreview(null); }} size="sm" />

        <label className="block">
          <span className="eyebrow">timezone</span>
          <select className="input-field mt-1" value={timezone} onChange={(e) => { setTimezone(e.target.value); setPreview(null); }}>
            {TIMEZONES.map((zone) => <option key={zone.value} value={zone.value}>{zone.label}</option>)}
          </select>
        </label>

        {needsStart && (
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
            <label className="block sm:col-span-2">
              <span className="eyebrow">starts</span>
              <input type="datetime-local" className="input-field mt-1" value={start} onChange={(e) => { setStart(e.target.value); setPreview(null); }} />
            </label>
            <label className="block">
              <span className="eyebrow">then every</span>
              <span className="flex gap-1 mt-1">
                <input type="number" min="1" className="input-field w-16" value={every} onChange={(e) => { setEvery(e.target.value); setPreview(null); }} />
                <select className="input-field" value={everyUnit} onChange={(e) => { setEveryUnit(e.target.value); setPreview(null); }}>
                  <option value="hours">hours</option>
                  <option value="days">days</option>
                </select>
              </span>
            </label>
          </div>
        )}

        <ul className="space-y-3">
          {rows.map((row) => (
            <li key={row.index} className="border border-rule rounded-input p-3 space-y-2">
              <label className="flex items-center gap-2 text-sm text-ink">
                <input type="checkbox" checked={row.checked} onChange={(e) => updateRow(row.index, { checked: e.target.checked })} />
                clip {row.index + 1}
              </label>
              {row.checked && (
                <>
                  <input className="input-field" aria-label={`title ${row.index + 1}`} value={row.title} onChange={(e) => updateRow(row.index, { title: e.target.value })} />
                  <textarea className="input-field min-h-[4rem]" aria-label={`caption ${row.index + 1}`} value={row.caption} onChange={(e) => updateRow(row.index, { caption: e.target.value })} />
                  {platforms.includes('instagram') && (
                    <input className="input-field" aria-label={`collaborators ${row.index + 1}`} placeholder="instagram collaborators, comma separated" value={row.collaborators} onChange={(e) => updateRow(row.index, { collaborators: e.target.value })} />
                  )}
                  {(mode === 'at' || mode === 'mixed') && (
                    <input type="datetime-local" aria-label={`time ${row.index + 1}`} className="input-field" value={row.when} onChange={(e) => updateRow(row.index, { when: e.target.value })} />
                  )}
                </>
              )}
            </li>
          ))}
        </ul>

        {error && <p className="text-sm text-warn">{error}</p>}

        {preview && (
          <ul className="space-y-1 text-sm">
            {preview.map((item, i) => (
              <li key={item.id || `${item.clip_index}-${item.platform}-${i}`} className="flex items-center justify-between gap-2">
                <span className="text-ink2">
                  clip {Number(item.clip_index) + 1} · {item.platform} · {item.local_time}
                  {item.status ? ` · ${item.status}` : ''}
                  {item.last_error ? ` · ${item.last_error}` : ''}
                </span>
                {queued && item.status === 'queued' && item.id && (
                  <button type="button" className="btn-ghost text-xs" onClick={() => cancelRow(item.id)}>cancel</button>
                )}
              </li>
            ))}
          </ul>
        )}
      </div>
    </Modal>
  );
}
