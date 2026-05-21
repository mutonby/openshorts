import { useEffect, useState } from 'react';
import KeyInput from '../../../components/KeyInput';
import SectionHeader from './SectionHeader.jsx';
import InfoIcon from '../../../components/ui/InfoIcon.jsx';
import { fetchUploadProfiles, setKey, useKeys } from '../../../state/keysStore.js';

export default function ApiKeysSection() {
  const keys = useKeys();
  const [profiles, setProfiles] = useState([]);
  const [connectStatus, setConnectStatus] = useState('idle'); // idle | loading | error

  useEffect(() => {
    if (keys.uploadPost && profiles.length === 0) {
      handleFetchProfiles();
    }
  }, [keys.uploadPost]);

  async function handleFetchProfiles() {
    if (!keys.uploadPost) return;
    setConnectStatus('loading');
    try {
      const data = await fetchUploadProfiles(keys.uploadPost);
      if (data.profiles?.length) {
        setProfiles(data.profiles);
        if (!keys.uploadUserId) setKey('uploadUser', data.profiles[0].username);
        setConnectStatus('idle');
      } else {
        setConnectStatus('error');
      }
    } catch {
      setConnectStatus('error');
    }
  }

  return (
    <div className="space-y-6">
      <SectionHeader
        title="API Keys"
        description="All keys are encrypted in localStorage and sent per-request only to the OpenShorts backend. They are never stored server-side."
      />

      <Panel
        title="Gemini"
        badge="Required"
        badgeTone="amber"
        info="Powers viral-moment detection (Short-form), effect prompts, title and description generation, thumbnails, and the Nano-Banana relight step inside AI Restyle."
        description="Google's Gemini 2.5 / 3.x — the only universally required key. Free tier covers personal use."
      >
        <KeyInput onKeySet={(v) => setKey('gemini', v)} savedKey={keys.gemini} />
      </Panel>

      <Panel
        title="Upload-Post"
        badge="Required for publishing"
        badgeTone="amber"
        info="Required to publish clips to TikTok, Instagram Reels, YouTube Shorts, Snapchat, and Facebook. Includes a free tier."
      >
        <div className="space-y-3">
          <label className="block text-[12px] text-zinc-400">API Key</label>
          <div className="flex gap-2">
            <input
              type="password"
              value={keys.uploadPost}
              onChange={(e) => setKey('uploadPost', e.target.value)}
              className="input-field"
              placeholder="ey..."
            />
            <button
              onClick={handleFetchProfiles}
              disabled={!keys.uploadPost || connectStatus === 'loading'}
              className="btn-primary py-2 px-4 text-sm disabled:opacity-50"
            >
              {connectStatus === 'loading' ? 'Connecting...' : 'Connect'}
            </button>
          </div>
          {connectStatus === 'error' && (
            <p className="text-[12px] text-red-400">No profiles found. Check the key and try again.</p>
          )}
          {profiles.length > 0 && (
            <div className="flex items-center gap-3 text-[12px] text-zinc-400 pt-1">
              <span>Profile:</span>
              <select
                value={keys.uploadUserId || profiles[0].username}
                onChange={(e) => setKey('uploadUser', e.target.value)}
                className="bg-surface border border-border rounded-md px-2 py-1 text-[12px] text-white"
              >
                {profiles.map((p) => <option key={p.username} value={p.username}>{p.username}</option>)}
              </select>
            </div>
          )}
        </div>
      </Panel>

      <Panel
        title="ElevenLabs"
        badge="Optional"
        info="Powers AI voice dubbing across 30+ languages on a per-clip basis."
        description="Translate clips into other languages while preserving the speaker's voice."
      >
        <input
          type="password"
          value={keys.elevenLabs}
          onChange={(e) => setKey('elevenLabs', e.target.value)}
          className="input-field"
          placeholder="sk_..."
        />
      </Panel>

      <Panel
        title="fal.ai"
        badge="Required for AI Restyle"
        badgeTone="amber"
        info="Powers AI Restyle's video-to-video model (Wan 2.5 Turbo) AND the legacy SaaS UGC pipeline (Flux Pro + Kling for actor portraits and B-roll). Without this key, AI Restyle and SaaSShorts are disabled; Short-form / Long-form still work."
        description="Sign up at fal.ai/dashboard/keys. Pay-as-you-go; budget roughly $1.50 per 30-second AI Restyle clip."
      >
        <input
          type="password"
          value={keys.fal}
          onChange={(e) => setKey('fal', e.target.value)}
          className="input-field"
          placeholder="fal_..."
        />
      </Panel>
    </div>
  );
}

function Panel({ title, badge, badgeTone, info, description, children }) {
  const toneClass = badgeTone === 'amber'
    ? 'bg-amber-500/10 border-amber-500/30 text-amber-400'
    : 'bg-white/5 border-border text-zinc-500';
  return (
    <div className="rounded-xl border border-border bg-surface p-6">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <h2 className="text-[14px] font-semibold text-white">{title}</h2>
          {info && <InfoIcon label={info} side="right" />}
        </div>
        {badge && (
          <span className={`text-[10px] px-2 py-0.5 rounded uppercase tracking-wider border ${toneClass}`}>
            {badge}
          </span>
        )}
      </div>
      {description && <p className="text-[12px] text-zinc-500 mb-4 leading-relaxed">{description}</p>}
      {children}
    </div>
  );
}
