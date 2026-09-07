import React, { useState, useEffect, useRef, useMemo } from 'react';
import { Upload, Sparkles, Youtube, Instagram, Share2, ChevronDown, Check, Activity, LayoutDashboard, Settings, Plus, History, X, Terminal, Shield, LayoutGrid, Image, Globe, RotateCcw, Calendar, AlertTriangle, KeyRound, Bot, Users, Smartphone, ExternalLink, Copy, CheckCircle2, Mail, Loader2, Download, Menu, Lock } from 'lucide-react';
import KeyInput from './components/KeyInput';
import MediaInput from './components/MediaInput';
import McpConnectCard from './components/McpConnectCard';
import ResultCard from './components/ResultCard';
import ProcessingAnimation from './components/ProcessingAnimation';
// import Gallery from './components/Gallery';
import ThumbnailStudio from './components/ThumbnailStudio';
import SaaShortsTab from './components/SaaShortsTab';
import UGCGallery from './components/UGCGallery';
import ScheduleWeekModal from './components/ScheduleWeekModal';
import ClipEditor from './components/ClipEditor';
import ReframeEditor from './components/ReframeEditor';
import UsageMeter from './components/UsageMeter';
import TopUpModal from './components/TopUpModal';
import StarBanner from './components/StarBanner';
import PlanChoiceModal from './components/PlanChoiceModal';
import ClipTutorial from './components/ClipTutorial';
import TrialUpgradeModal from './components/TrialUpgradeModal';
import LoginModal from './components/LoginModal';
import TrialGate from './components/TrialGate';
import AdvancedBanner from './components/AdvancedBanner';
import HistoryTab from './components/HistoryTab';
import ProfileMenu from './components/ProfileMenu';
import Modal from './components/ui/Modal';
import { useAuth } from './contexts/AuthContext';
import { apiFetch, apiJson, QuotaError } from './lib/api';
import { track } from './lib/analytics';

// Enhanced "Encryption" using XOR + Base64 with a Salt
// This is better than plain Base64 but still client-side.
const SECRET_KEY = import.meta.env.VITE_ENCRYPTION_KEY || "OpenShorts-Static-Salt-Change-Me";
const ENCRYPTION_PREFIX = "ENC:";

const encrypt = (text) => {
  if (!text) return '';
  try {
    const xor = text.split('').map((c, i) =>
      String.fromCharCode(c.charCodeAt(0) ^ SECRET_KEY.charCodeAt(i % SECRET_KEY.length))
    ).join('');
    return ENCRYPTION_PREFIX + btoa(xor);
  } catch (e) {
    console.error("Encryption failed", e);
    return text;
  }
};

const decrypt = (text) => {
  if (!text) return '';
  if (text.startsWith(ENCRYPTION_PREFIX)) {
    try {
      const raw = text.slice(ENCRYPTION_PREFIX.length);
      // Check if it's plain base64 or our custom XOR (simple try)
      const xor = atob(raw);
      const result = xor.split('').map((c, i) =>
        String.fromCharCode(c.charCodeAt(0) ^ SECRET_KEY.charCodeAt(i % SECRET_KEY.length))
      ).join('');
      return result;
    } catch (e) {
      // Fallback if decryption fails (might be old plain text)
      return '';
    }
  }
  // Backward compatibility: If no prefix, assume old plain text (or return empty if you want to force re-login)
  // For migration: Return text as is, so it populates the field, and next save will encrypt it.
  return text;
};

// Simple TikTok icon sine Lucide might not have it or it varies
const TikTokIcon = ({ size = 16, className = "" }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor" className={className}>
    <path d="M19.589 6.686a4.793 4.793 0 0 1-3.77-4.245V2h-3.445v13.672a2.896 2.896 0 0 1-5.201 1.743l-.002-.001.002.001a2.895 2.895 0 0 1 3.183-4.51v-3.5a6.329 6.329 0 0 0-5.394 10.692 6.33 6.33 0 0 0 10.857-4.424V8.687a8.182 8.182 0 0 0 4.773 1.526V6.79a4.831 4.831 0 0 1-1.003-.104z" />
  </svg>
);

// Cloud accounts get an auto-generated opaque id (os_<hash>) as username —
// meaningless to the user, so the selector shows connected networks instead.
const isAutoProfileId = (username) => /^os_[0-9a-f]/i.test(username || "");

const formatRetention = (seconds) => {
  if (seconds >= 86400) return `${Math.round(seconds / 86400)} day${seconds >= 172800 ? 's' : ''}`;
  if (seconds >= 3600) return `${Math.round(seconds / 3600)} hour${seconds >= 7200 ? 's' : ''}`;
  return `${Math.max(1, Math.round(seconds / 60))} min`;
};

const ProfileNetworkIcons = ({ profile, size = 12 }) => (
  <span className="flex items-center gap-1.5">
    <span className={profile?.connected?.includes('tiktok') ? 'text-ink' : 'text-muted opacity-40'}>
      <TikTokIcon size={size} />
    </span>
    <span className={profile?.connected?.includes('instagram') ? 'text-ink' : 'text-muted opacity-40'}>
      <Instagram size={size} />
    </span>
    <span className={profile?.connected?.includes('youtube') ? 'text-ink' : 'text-muted opacity-40'}>
      <Youtube size={size} />
    </span>
  </span>
);

const UserProfileSelector = ({ profiles, selectedUserId, onSelect, onConnect }) => {
  const [isOpen, setIsOpen] = useState(false);

  if (!profiles || profiles.length === 0) return null;

  const selectedProfile = profiles.find(p => p.username === selectedUserId) || profiles[0];
  const autoId = isAutoProfileId(selectedProfile?.username);

  return (
    <div className="relative z-50">
      <button
        onClick={() => setIsOpen(!isOpen)}
        aria-label="social profile"
        /* Phone: avatar + chevron only. A 180px pill next to the menu button,
           the section title and the minutes meter overflowed a 360px header. */
        className="flex items-center justify-between gap-1 bg-paper2 border border-rule2 rounded-input px-2 sm:px-3 py-2 text-sm text-ink2 hover:bg-paper3 transition-colors sm:min-w-[180px]"
      >
        <span className="flex items-center gap-2">
          <div className="w-5 h-5 rounded-full bg-paper3 border border-rule flex items-center justify-center font-mono text-micro text-brass shrink-0">
            {autoId ? "S" : (selectedProfile?.username?.substring(0, 1).toUpperCase() || "U")}
          </div>
          {autoId ? (
            <span className="hidden sm:flex"><ProfileNetworkIcons profile={selectedProfile} size={13} /></span>
          ) : (
            <span className="hidden sm:block font-medium text-ink truncate max-w-[100px]">{selectedProfile?.username || "Select User"}</span>
          )}
        </span>
        <ChevronDown size={14} className={`text-muted transition-transform ${isOpen ? 'rotate-180' : ''}`} />
      </button>

      {isOpen && (
        <div className="absolute top-full mt-2 right-0 w-64 card overflow-hidden">
          <div className="max-h-60 overflow-y-auto custom-scrollbar">
            {profiles.map((profile) => (
              <button
                key={profile.username}
                onClick={() => {
                  onSelect(profile.username);
                  setIsOpen(false);
                }}
                className="w-full flex items-center justify-between px-4 py-3 hover:bg-paper3 transition-colors text-left group border-b border-rule last:border-0"
              >
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 rounded-full bg-paper3 flex items-center justify-center font-mono text-micro text-ink border border-rule shrink-0">
                    {isAutoProfileId(profile.username) ? "S" : profile.username.substring(0, 2).toUpperCase()}
                  </div>
                  <div className="min-w-0">
                    <div className="text-sm font-medium text-ink2 group-hover:text-ink transition-colors truncate">
                      {isAutoProfileId(profile.username)
                        ? `Social profile ${profiles.indexOf(profile) + 1}`
                        : profile.username}
                    </div>
                    <div className="flex gap-2 mt-0.5">
                      {/* Status indicators */}
                      <div className={`flex items-center gap-1 ${profile.connected.includes('tiktok') ? 'text-ink2' : 'text-muted opacity-40'}`}>
                        <TikTokIcon size={10} />
                      </div>
                      <div className={`flex items-center gap-1 ${profile.connected.includes('instagram') ? 'text-ink2' : 'text-muted opacity-40'}`}>
                        <Instagram size={10} />
                      </div>
                      <div className={`flex items-center gap-1 ${profile.connected.includes('youtube') ? 'text-ink2' : 'text-muted opacity-40'}`}>
                        <Youtube size={10} />
                      </div>
                    </div>
                  </div>
                </div>
                {selectedUserId === profile.username && <Check size={14} className="text-brass shrink-0" />}
              </button>
            ))}
          </div>
          {/* For managed users this dropdown otherwise does nothing (one profile,
              nothing to switch) — its real job is being the door to connecting
              the greyed-out networks it displays. */}
          {onConnect && (
            <button
              onClick={() => { setIsOpen(false); onConnect(); }}
              className="w-full flex items-center gap-2 px-4 py-3 text-sm text-brass hover:bg-paper3 transition-colors text-left border-t border-rule"
            >
              <Share2 size={14} /> Connect / manage accounts
            </button>
          )}
        </div>
      )}
    </div>
  );
};

const SESSION_KEY = 'openshorts_session';
// Matches the self-host JOB_RETENTION_SECONDS default. A restore whose job was
// already purged server-side fails gracefully and clears the saved session.
const SESSION_MAX_AGE = 86400000; // 24 hours

// Mock polling function
const pollJob = async (jobId) => {
  const res = await apiFetch(`/api/status/${jobId}`);
  if (!res.ok) throw new Error('Status check failed');
  return res.json();
};

function App() {
  // Cloud auth/billing session (inert when billing is disabled).
  const { billingEnabled, isManaged, isSignedIn, me, plan, refreshMe, jobRetentionSeconds, localLlm } = useAuth();
  const [showLogin, setShowLogin] = useState(false);
  const [showTopUp, setShowTopUp] = useState(false);
  const [showPlanChoice, setShowPlanChoice] = useState(false);
  const [tutorialPhase, setTutorialPhase] = useState(null); // null | intro | coach | celebrate
  const [showTrialUpgrade, setShowTrialUpgrade] = useState(false);
  const [topUpInfo, setTopUpInfo] = useState({});
  // Durable R2 URLs (per clip index) for the current job — used as a fallback when
  // the ephemeral local /videos/ files have been cleaned up (e.g. after a reload).
  const [durableClips, setDurableClips] = useState({});

  const [apiKey, setApiKey] = useState(localStorage.getItem('gemini_key') || '');
  // Social API State - Load encrypted or plain
  const [uploadPostKey, setUploadPostKey] = useState(() => {
    const stored = localStorage.getItem('uploadPostKey_v3');
    if (stored) return decrypt(stored);
    return '';
  });
  // ElevenLabs API State - Load encrypted
  const [elevenLabsKey, setElevenLabsKey] = useState(() => {
    const stored = localStorage.getItem('elevenLabsKey_v1');
    if (stored) return decrypt(stored);
    return '';
  });

  // fal.ai API State - Load encrypted
  const [falKey, setFalKey] = useState(() => {
    const stored = localStorage.getItem('falKey_v1');
    if (stored) return decrypt(stored);
    return '';
  });

  const [uploadUserId, setUploadUserId] = useState(() => localStorage.getItem('uploadUserId') || '');
  const [userProfiles, setUserProfiles] = useState([]); // List of {username, connected: []}
  // Post-generation social nudge: shown at the results peak until the user
  // either connects a network or dismisses it. Only 2.7% of cloud users who
  // reach the social flow ever connect an account — this is the moment (clips
  // just appeared) with the best odds of moving that number.
  const [socialNudgeDismissed, setSocialNudgeDismissed] = useState(() => {
    try { return localStorage.getItem('os_social_nudge_dismissed') === '1'; } catch (_) { return false; }
  });
  const [showKeyModal, setShowKeyModal] = useState(false);
  const [jobId, setJobId] = useState(null);
  const [status, setStatus] = useState('idle'); // idle, processing, complete, error
  const [results, setResults] = useState(null);
  // Best clips first. The backend hands them back in transcript order, which
  // buries the strongest one wherever it happens to fall in the video — and
  // the first card is the one people actually watch and publish.
  //
  // The ORIGINAL array position travels with each clip and is what gets passed
  // down as `index`: it is the clip's identity everywhere else (clip_index on
  // /api/subtitle, /api/edit and publishing, the clip-N.mp4 download name, the
  // saved per-clip project state). Sorting the array itself would silently
  // repoint all of that at the wrong clip.
  const rankedClips = useMemo(() => {
    const clips = results?.clips;
    if (!Array.isArray(clips)) return [];
    return clips
      .map((clip, index) => ({ clip, index }))
      .sort((a, b) => {
        const sa = Number.isFinite(a.clip?.predicted_score) ? a.clip.predicted_score : -1;
        const sb = Number.isFinite(b.clip?.predicted_score) ? b.clip.predicted_score : -1;
        // Ties (and clips with no score at all) keep transcript order.
        return sb - sa || a.index - b.index;
      });
  }, [results]);
  // Bulk subtitles: apply one style to every clip of the job (triggered from
  // within a clip's subtitle modal via "apply to all").
  const [bulkSub, setBulkSub] = useState({ running: false, current: 0, total: 0, errors: 0 });
  const [downloadingAll, setDownloadingAll] = useState(false);
  // Pre-flight quality gate: { info: {max_height, min_height, cookies_invalid}, data }
  const [qualityGate, setQualityGate] = useState(null);
  const [logs, setLogs] = useState([]);
  // Collapsed on phones: the log tail is the least useful thing on a 360px
  // screen and it was pushing the actual clips a full scroll down.
  const [logsVisible, setLogsVisible] = useState(() => {
    try { return window.innerWidth >= 768; } catch { return true; }
  });
  const [processingMedia, setProcessingMedia] = useState(null);
  const [activeTab, setActiveTab] = useState('dashboard'); // dashboard, settings
  // Mobile only: the full nav lives in a drawer behind the header's menu button.
  const [navOpen, setNavOpen] = useState(false);
  // Reopened-project state (paid mode): per-clip {index, server_file, active_layers}
  // restored from the backend so ResultCards resume editing where they left off.
  const [projectState, setProjectState] = useState(null);
  // True when the current job was reopened from the library: its source video
  // was never persisted, so the session must not fall back to /api/source.
  const [noSource, setNoSource] = useState(false);

  const [sessionRecovered, setSessionRecovered] = useState(false);
  const [showScheduleWeek, setShowScheduleWeek] = useState(false);
  // Clip editor overlay: index of the clip being edited, or null.
  const [editingClip, setEditingClip] = useState(null);
  const [reframingClip, setReframingClip] = useState(null);

  // Silent-success "saved" states for the settings key inputs (design.md: no alert popups)
  const [elevenLabsSaved, setElevenLabsSaved] = useState(false);
  const [falSaved, setFalSaved] = useState(false);

  // Sync state for original video playback
  const [syncedTime, setSyncedTime] = useState(0);
  const [isSyncedPlaying, setIsSyncedPlaying] = useState(false);
  const [syncTrigger, setSyncTrigger] = useState(0);

  const handleClipPlay = (startTime) => {
    setSyncedTime(startTime);
    setIsSyncedPlaying(true);
    setSyncTrigger(prev => prev + 1);
  };

  const handleClipPause = () => {
    setIsSyncedPlaying(false);
  };

  // --- Project persistence (paid mode) ---
  // Debounced sync of each clip's browser-only edit state (Remotion layers +
  // current server file) to the backend, so a reopened project resumes intact.
  const clipStateSync = useRef({ jobId: null, pending: {}, files: {}, timer: null });
  // Read by in-flight async chases to notice that the user moved on to another job.
  const jobIdRef = useRef(jobId);
  useEffect(() => { jobIdRef.current = jobId; }, [jobId]);

  const flushClipState = () => {
    const s = clipStateSync.current;
    if (s.timer) { clearTimeout(s.timer); s.timer = null; }
    const entries = Object.entries(s.pending);
    if (!s.jobId || entries.length === 0) return;
    const clips = entries.map(([i, v]) => ({
      index: Number(i),
      active_layers: v.activeLayers,
      server_file: v.serverVideoFile,
    }));
    s.pending = {};
    apiFetch(`/api/projects/${s.jobId}/state`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ clips }),
    }).catch(() => {});
  };

  // One /api/history read, reduced to this job's clips.
  const fetchDurableMap = async () => {
    const d = await apiJson('/api/history');
    const map = {};
    for (const v of (d.videos || [])) {
      if (v.job_id === jobId && v.clip_index != null) map[v.clip_index] = { url: v.view_url, filename: v.filename };
    }
    return map;
  };

  // A server-side edit rewrites the clip's file while the R2 re-archive behind it
  // is still in flight (_archive_clip_edit_bg is fire-and-forget), so the durable
  // map goes stale and the card falls back to streaming from the API. Re-read the
  // map on a backoff until the archived name matches the clip's new file, then it
  // can play from R2 again. Gives up quietly: staying on /videos is correct, just
  // slower, and is exactly what happens for self-hosted users all the time.
  const chaseDurableFile = async (index, expectedFile) => {
    const forJob = jobId;
    for (const delay of [2500, 6000, 15000, 30000]) {
      await new Promise((r) => setTimeout(r, delay));
      if (jobIdRef.current !== forJob) return;
      let map;
      try { map = await fetchDurableMap(); } catch { return; }
      // A new job started mid-chase: this map describes the old one, so dropping
      // it here keeps it from overwriting the new job's URLs.
      if (jobIdRef.current !== forJob) return;
      setDurableClips(map);
      if (map[index]?.filename === expectedFile) return;
    }
  };

  const handleClipStateChange = (index, state) => {
    if (!isManaged || !jobId) return;
    const s = clipStateSync.current;
    if (s.jobId !== jobId) { s.pending = {}; s.files = {}; s.jobId = jobId; }
    s.pending[index] = state;
    // Cards report on mount too, so only an actual change of server file means an
    // edit just landed. The first report per clip is the mount, never a chase.
    const files = s.files || (s.files = {});
    const file = state?.serverVideoFile;
    if (file && files[index] !== file) {
      const isMount = files[index] === undefined;
      files[index] = file;
      if (!isMount) chaseDurableFile(index, file);
    }
    if (s.timer) clearTimeout(s.timer);
    s.timer = setTimeout(flushClipState, 2000);
  };

  // A recut replaced the clip's server file with a fresh render (burned layers
  // reset), so update the results, the reopened-project state and the synced
  // per-clip edit state, and let the ResultCard remount from the new file.
  const handleClipRerendered = (index, data) => {
    const newFile = (data.new_video_url || '').split('/').pop();
    setResults((prev) => {
      if (!prev?.clips?.[index]) return prev;
      const clips = prev.clips.slice();
      clips[index] = {
        ...clips[index],
        video_url: data.new_video_url,
        start: data.start,
        end: data.end,
        recipe: data.recipe,
      };
      return { ...prev, clips };
    });
    setProjectState((prev) => {
      if (!prev?.clips) return prev;
      return {
        ...prev,
        clips: prev.clips.map((c) => (c.index === index
          ? { ...c, server_file: newFile, active_layers: null }
          : c)),
      };
    });
    // The old durable R2 object is deleted when the recut is archived, so the
    // stale URL would 404 as a fallback; drop it until the next refresh.
    setDurableClips((prev) => {
      if (!(index in prev)) return prev;
      const next = { ...prev };
      delete next[index];
      return next;
    });
    handleClipStateChange(index, { activeLayers: null, serverVideoFile: newFile });
  };

  // Reopen an archived project from the History tab: the backend re-downloads
  // its files from R2 into the server's working dir and returns the full state.
  const restoreProject = async (projectJobId) => {
    const data = await apiJson(`/api/projects/${projectJobId}/restore`, { method: 'POST' });
    flushClipState();
    setProjectState(data.project_state || null);
    setNoSource(true);
    setJobId(data.job_id);
    setResults(data.result || null);
    setLogs(['♻️ Project restored from your library.']);
    setProcessingMedia(null);
    setQualityGate(null);
    setStatus('complete');
    setActiveTab('dashboard');
  };

  // Apply one subtitle style to every clip of the job, sequentially.
  const handleBulkSubtitles = async (options) => {
    const clips = results?.clips || [];
    const total = clips.length;
    if (!total) return;
    setBulkSub({ running: true, current: 0, total, errors: 0 });
    let errors = 0;
    for (let i = 0; i < total; i++) {
      setBulkSub({ running: true, current: i + 1, total, errors });
      try {
        const res = await apiFetch('/api/subtitle', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            job_id: jobId,
            clip_index: i,
            position: options.position,
            font_size: options.fontSize,
            font_name: options.fontName,
            font_color: options.fontColor,
            border_color: options.borderColor,
            border_width: options.borderWidth,
            bg_color: options.bgColor,
            bg_opacity: options.bgOpacity,
            style: options.style || 'classic',
            highlight_color: options.highlightColor || '#FFD700',
            effect: options.effect || 'none',
            base_opacity: options.baseOpacity ?? 1.0,
            uppercase: options.uppercase || false,
            // Chain from the clip's current server file (its video_url basename).
            input_filename: (clips[i].video_url || '').split('/').pop(),
          }),
        });
        if (!res.ok) errors++;
      } catch {
        errors++;
      }
    }
    setBulkSub({ running: false, current: total, total, errors });
    refreshMe();
    // Refresh results so each ResultCard picks up its new subtitled video_url.
    try {
      const data = await pollJob(jobId);
      if (data.result) setResults(data.result);
    } catch { /* keep current results */ }
  };

  const handleDownloadAll = async () => {
    if (!jobId) return;
    setDownloadingAll(true);
    try {
      const res = await apiFetch(`/api/jobs/${jobId}/download-all`);
      if (!res.ok) throw new Error(await res.text());
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `openshorts_clips_${(jobId || '').slice(0, 8)}.zip`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch (e) {
      alert(`Download failed: ${e.message}`);
    } finally {
      setDownloadingAll(false);
    }
  };

  // Session Recovery: Restore on mount
  useEffect(() => {
    try {
      const saved = localStorage.getItem(SESSION_KEY);
      if (!saved) return;
      const session = JSON.parse(saved);
      if (Date.now() - session.timestamp > SESSION_MAX_AGE) {
        localStorage.removeItem(SESSION_KEY);
        return;
      }
      if (session.jobId && session.status && session.status !== 'idle') {
        setJobId(session.jobId);
        setResults(session.results || null);
        // Restore the source preview. Older sessions (or uploads) saved no
        // media, so fall back to the backend-served source for this job —
        // except for reopened projects, whose source was never persisted.
        if (session.processingMedia) setProcessingMedia(session.processingMedia);
        else if (!session.noSource) setProcessingMedia({ type: 'server', payload: `/api/source/${session.jobId}` });
        if (session.noSource) setNoSource(true);
        if (session.projectState) setProjectState(session.projectState);
        if (session.activeTab) setActiveTab(session.activeTab);
        // If was processing, resume polling; if complete/error, just show results
        setStatus(session.status === 'processing' ? 'processing' : session.status);
        setSessionRecovered(true);
        setTimeout(() => setSessionRecovered(false), 5000);
      }
    } catch (e) {
      localStorage.removeItem(SESSION_KEY);
    }
  }, []);

  // Session Recovery: Save state changes
  useEffect(() => {
    if (status === 'idle') {
      localStorage.removeItem(SESSION_KEY);
      return;
    }
    try {
      // URL (YouTube) media serializes as-is. Uploaded 'file' media is a blob
      // that can't be persisted, so point the recovered preview at the source
      // served by the backend instead of dropping it.
      let persistMedia = null;
      if (processingMedia?.type === 'url') persistMedia = processingMedia;
      else if (processingMedia && jobId) persistMedia = { type: 'server', payload: `/api/source/${jobId}` };
      const sessionData = {
        jobId,
        status,
        results,
        processingMedia: persistMedia,
        activeTab,
        noSource,
        projectState,
        timestamp: Date.now()
      };
      localStorage.setItem(SESSION_KEY, JSON.stringify(sessionData));
    } catch (e) {
      // localStorage full or serialization error - ignore
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId, status, results, activeTab, noSource, projectState]);

  useEffect(() => {
    // Encrypt Gemini Key too for consistency if desired, but user asked specifically about Social integration not saving well.
    // For now keeping gemini plain for compatibility unless requested.
    if (apiKey) localStorage.setItem('gemini_key', apiKey);
  }, [apiKey]);

  useEffect(() => {
    if (uploadPostKey) {
      localStorage.setItem('uploadPostKey_v3', encrypt(uploadPostKey));
    }
    if (uploadUserId) {
      localStorage.setItem('uploadUserId', uploadUserId);
    }
  }, [uploadPostKey, uploadUserId]);

  useEffect(() => {
    if (elevenLabsKey) {
      localStorage.setItem('elevenLabsKey_v1', encrypt(elevenLabsKey));
    }
  }, [elevenLabsKey]);

  useEffect(() => {
    if (falKey) {
      localStorage.setItem('falKey_v1', encrypt(falKey));
    }
  }, [falKey]);

  useEffect(() => {
    if ((uploadPostKey || isManaged) && userProfiles.length === 0) {
      fetchUserProfiles({ silent: true });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [uploadPostKey, isManaged]);

  // For managed users, fetch the durable R2 URLs of the current job's clips. The
  // preview player prefers them (free egress, edge-served, and not competing with
  // the renders for the API process), and falls back to /videos when the local
  // file is newer than the archived one or the signed link fails.
  // Kept fresh by chaseDurableFile after each edit and by the completion chase
  // below; until either lands, the card just streams from /videos.
  useEffect(() => {
    if (!isManaged || !jobId || !(results?.clips?.length)) { setDurableClips({}); return; }
    let cancelled = false;
    fetchDurableMap()
      .then((map) => { if (!cancelled) setDurableClips(map); })
      .catch(() => {});
    return () => { cancelled = true; };
    // Keyed on the clip COUNT, not on results: the status poll hands back a new
    // results object every couple of seconds while the job runs, and this used to
    // re-read the history on every one of them.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isManaged, jobId, results?.clips?.length]);

  // A job is marked complete BEFORE _archive_managed_job has finished uploading
  // (app.py:878), so the read above can land while R2 still has nothing for it and
  // every card would stream from the API for the rest of the session. Chase until
  // all clips have a durable copy.
  useEffect(() => {
    const count = results?.clips?.length || 0;
    if (!isManaged || !jobId || status !== 'complete' || !count) return;
    let cancelled = false;
    (async () => {
      for (const delay of [0, 3000, 8000, 20000, 40000]) {
        if (delay) await new Promise((r) => setTimeout(r, delay));
        if (cancelled) return;
        let map;
        try { map = await fetchDurableMap(); } catch { return; }
        if (cancelled) return;
        setDurableClips(map);
        if (Object.keys(map).length >= count) return;
      }
    })();
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isManaged, jobId, status, results?.clips?.length]);

  useEffect(() => {
    let interval;
    if ((status === 'processing' || status === 'completed') && jobId) {
      interval = setInterval(async () => {
        try {
          const data = await pollJob(jobId);
          console.log("Job status:", data);

          // Update results if available (real-time)
          if (data.result) {
            setResults(data.result);
          }

          if (data.status === 'completed') {
            setStatus('complete');
            clearInterval(interval);
            refreshMe();
          } else if (data.status === 'failed') {
            setStatus('error');
            const errorMsg = data.error || (data.logs && data.logs.length > 0 ? data.logs[data.logs.length - 1] : "Process failed");
            setLogs(prev => [...prev, "Error: " + errorMsg]);
            clearInterval(interval);
            refreshMe();
          } else {
            // Update logs if available
            if (data.logs) setLogs(data.logs);
          }
        } catch (e) {
          console.error("Polling error", e);
        }
      }, 2000);
    }
    return () => clearInterval(interval);
  }, [status, jobId, refreshMe]);


  // silent: background auto-fetch — never alert(), just log. Managed users need
  // no local key (the server resolves its own); BYOK sends the header.
  const fetchUserProfiles = async ({ silent = false } = {}) => {
    if (!uploadPostKey && !isManaged) return;
    try {
      const res = await apiFetch('/api/social/user', {
        headers: uploadPostKey ? { 'X-Upload-Post-Key': uploadPostKey } : {}
      });
      if (!res.ok) throw new Error("Failed to fetch");
      const data = await res.json();
      if (data.profiles && data.profiles.length > 0) {
        setUserProfiles(data.profiles);
        // Auto select first if none selected
        if (!uploadUserId) {
          setUploadUserId(data.profiles[0].username);
        }
      } else if (!silent) {
        alert("No profiles found for this API Key.");
      }
    } catch (e) {
      if (!silent) alert("Error fetching User Profiles. Please check key.");
      console.error(e);
    }
  };

  // Hosted is paid-only (no BYOK core). Self-host uses BYOK keys.
  // `keysMissing` now means "self-host BYOK keys missing" — it never fires on hosted.
  // A self-hosted server running the moment picker on a local LLM
  // (LLM_BASE_URL) does not need a Gemini key for the core pipeline.
  const geminiOk = !!apiKey || !!localLlm;
  const keysMissing = !billingEnabled && (!geminiOk || !uploadPostKey);
  const needsPlan = billingEnabled && !isManaged;   // hosted, signed-out or no active plan/trial

  // Fresh sign-up: Clip Generator tutorial (AuthContext set os_show_clip_tutorial
  // after the auth redirect). QA: #app?tutorial=1. Resume coach if they refreshed
  // mid-job. Runs once on mount so a later isSignedIn flip cannot reset intro→coach.
  useEffect(() => {
    let showTutorial = false;
    let resumeCoach = false;
    try {
      const q = new URLSearchParams((window.location.hash.split('?')[1] || ''));
      const qa = q.get('tutorial');
      if (qa === '1') showTutorial = true;
      if (qa === 'coach') resumeCoach = true;
      if (qa === 'celebrate') { setTutorialPhase('celebrate'); return; }
      if (localStorage.getItem('os_show_clip_tutorial') === '1') showTutorial = true;
      if (localStorage.getItem('os_clip_tutorial') === 'coach') resumeCoach = true;
    } catch (_) { /* ignore */ }
    if (showTutorial) {
      setTutorialPhase('intro');
      setActiveTab('dashboard');
    } else if (resumeCoach) {
      setTutorialPhase('coach');
      setActiveTab('dashboard');
    }
  }, []);

  // Legacy: an older build may still have set os_show_plan_choice. Don't open it
  // on top of the tutorial.
  useEffect(() => {
    if (tutorialPhase) return;
    if (!(billingEnabled && isSignedIn)) return;
    let showPlans = false;
    try { showPlans = localStorage.getItem('os_show_plan_choice') === '1'; } catch (_) { /* ignore */ }
    if (showPlans) {
      setShowPlanChoice(true);
      try { localStorage.removeItem('os_show_plan_choice'); } catch (_) { /* ignore */ }
    }
  }, [billingEnabled, isSignedIn, tutorialPhase]);

  const tutorialLock = tutorialPhase === 'intro' || tutorialPhase === 'coach' || tutorialPhase === 'celebrate';

  useEffect(() => {
    if (tutorialLock && activeTab !== 'dashboard') setActiveTab('dashboard');
  }, [tutorialLock, activeTab]);

  useEffect(() => {
    if (tutorialPhase === 'coach' && status === 'complete' && (results?.clips?.length > 0)) {
      setTutorialPhase('celebrate');
      track('ClipTutorialCompleted', { props: { clips: results.clips.length } });
    }
  }, [tutorialPhase, status, results]);

  const finishTutorial = () => {
    try { localStorage.setItem('os_clip_tutorial', 'done'); } catch (_) { /* ignore */ }
    try { localStorage.removeItem('os_show_clip_tutorial'); } catch (_) { /* ignore */ }
    setTutorialPhase(null);
  };
  const startTutorial = () => {
    try { localStorage.setItem('os_clip_tutorial', 'coach'); } catch (_) { /* ignore */ }
    try { localStorage.removeItem('os_show_clip_tutorial'); } catch (_) { /* ignore */ }
    track('ClipTutorialStarted');
    setTutorialPhase('coach');
    setActiveTab('dashboard');
  };
  const skipTutorial = () => {
    track('ClipTutorialSkipped', { props: { phase: tutorialPhase } });
    finishTutorial();
  };
  // Included in the plan (fully managed, no keys): Clip Generator + YouTube Studio.
  // Advanced (bring your own fal.ai + ElevenLabs keys): AI Shorts + AI Agent.
  const INCLUDED_TOOL_TABS = ['dashboard', 'thumbnails'];
  const ADVANCED_TOOL_TABS = ['saasshorts', 'ai-agent'];
  const TOOL_NAMES = { dashboard: 'the Clip Generator', thumbnails: 'the YouTube Studio' };
  const gateThisTab = needsPlan && INCLUDED_TOOL_TABS.includes(activeTab);      // included tool, no plan yet
  const advancedThisTab = billingEnabled && ADVANCED_TOOL_TABS.includes(activeTab); // BYOK-notice tools

  // Social nudge visibility: managed users with clips on screen and no network
  // connected yet. userProfiles being empty (not yet fetched / none created)
  // also counts as "not connected" — that is the 97% case.
  const connectedSocials = ((userProfiles.find((p) => p.username === uploadUserId) || userProfiles[0])?.connected) || [];
  const showSocialNudge = isManaged && !socialNudgeDismissed && connectedSocials.length === 0 && !tutorialLock;

  // One Seen event per job, only when the banner actually rendered.
  const socialNudgeSeenRef = useRef(null);
  useEffect(() => {
    if (status === 'complete' && (results?.clips?.length > 0) && showSocialNudge && socialNudgeSeenRef.current !== jobId) {
      socialNudgeSeenRef.current = jobId;
      track('SocialNudgeSeen', { props: { clips: results.clips.length } });
    }
  }, [status, results, showSocialNudge, jobId]);

  // Managed users connect their socials via Upload-Post's branded hosted page.
  const handleConnectSocials = async () => {
    try {
      const { access_url } = await apiJson('/api/social/connect', { method: 'POST' });
      // Same tab so the connect page's redirectUrl brings the user back into the app.
      if (access_url) window.location.href = access_url;
    } catch (e) {
      alert('Could not open the connection page. Please try again.');
    }
  };

  // Open the Upload-Post white-label page (which includes the scheduling calendar)
  // in a new tab, for consulting/managing scheduled posts from the dashboard.
  const handleOpenCalendar = async () => {
    try {
      const { access_url } = await apiJson('/api/social/connect', { method: 'POST' });
      if (access_url) window.open(access_url, '_blank', 'noopener');
    } catch (e) {
      alert('Could not open the calendar. Please try again.');
    }
  };

  const handleProcess = async (data, forceLowQuality = false) => {
    // Hosted: must be signed in AND on an active plan/trial. Self-host: BYOK keys.
    if (billingEnabled) {
      if (!isSignedIn) { setShowLogin(true); return; }
      if (!isManaged) { window.location.hash = '#/pricing'; return; }
    } else if (keysMissing) {
      setShowKeyModal(true);
      return;
    }
    setStatus('processing');
    setLogs(["Starting process..."]);
    setResults(null);
    // Studio handovers have no local media object; the preview switches to the
    // backend-served source once the job id is known.
    setProcessingMedia(data.type === 'thumbnail_session' ? null : data);
    setQualityGate(null);
    setProjectState(null);
    setNoSource(false);

    try {
      let body;
      // BYOK sends the Gemini header; managed users rely on the bearer token
      // that apiFetch attaches automatically.
      const headers = apiKey ? { 'X-Gemini-Key': apiKey } : {};

      // Advanced generation controls: only sent when the user set them, so the
      // default request stays byte-identical to the pre-feature one.
      const advanced = {
        target_clips: data.targetClips || null,
        clip_min_seconds: data.clipMinSeconds || null,
        clip_max_seconds: data.clipMaxSeconds || null,
        // Sent explicitly both ways: absent means off for raw API callers,
        // but the dashboard always states the user's choice.
        auto_hook: data.autoHook ? '1' : '0',
        auto_hook_style: data.autoHook ? (data.autoHookStyle || 'classic') : null,
        // 'auto' is the server default, so only a deliberate choice travels.
        layouts: data.layout && data.layout !== 'auto' ? data.layout : null,
        // Same rule: absent = the deployment's default caption look.
        caption_preset: data.captionPreset || null,
      };

      if (data.type === 'url') {
        headers['Content-Type'] = 'application/json';
        body = JSON.stringify({
          url: data.payload,
          acknowledged: !!data.acknowledged,
          output_format: data.outputFormat || 'auto',
          force_low_quality: forceLowQuality,
          ...Object.fromEntries(Object.entries(advanced).filter(([, v]) => v != null)),
        });
      } else if (data.type === 'thumbnail_session') {
        // Handover from Thumbnail Studio (issue #68): the video and transcript
        // already live server-side, keyed by the Studio session.
        headers['Content-Type'] = 'application/json';
        body = JSON.stringify({
          thumbnail_session_id: data.payload,
          acknowledged: !!data.acknowledged,
          output_format: data.outputFormat || 'auto',
          ...Object.fromEntries(Object.entries(advanced).filter(([, v]) => v != null)),
        });
      } else {
        const formData = new FormData();
        formData.append('file', data.payload);
        formData.append('acknowledged', data.acknowledged ? 'true' : 'false');
        formData.append('output_format', data.outputFormat || 'auto');
        for (const [k, v] of Object.entries(advanced)) {
          if (v != null) formData.append(k, v);
        }
        body = formData;
      }

      const res = await apiFetch('/api/process', { method: 'POST', headers, body });

      if (!res.ok) throw new Error(await res.text());
      const resData = await res.json();

      // Quality gate: the source is below the min resolution — ask before burning
      // 20 min on it. On confirm we resend with force_low_quality.
      if (resData.needs_confirmation) {
        setStatus('idle');
        setQualityGate({ info: resData.quality_check, data });
        return;
      }

      setJobId(resData.job_id);
      if (data.type === 'thumbnail_session') {
        setProcessingMedia({ type: 'server', payload: `/api/source/${resData.job_id}` });
      }
      // Minutes are reserved at job start, not at complete.
      refreshMe();

    } catch (e) {
      if (e instanceof QuotaError) {
        setStatus('idle');
        refreshMe();
        // Trial users hit the trial minute cap → prompt them to activate the plan
        // now (unlocks full minutes). Active users → offer a top-up.
        if (me?.status === 'trialing') {
          setShowTrialUpgrade(true);
        } else {
          setTopUpInfo({ required: e.minutesRequired, remaining: e.minutesRemaining });
          setShowTopUp(true);
        }
        return;
      }
      setStatus('error');
      setLogs(l => [...l, `Error starting job: ${e.message}`]);
    }
  };

  const handleReset = () => {
    // Flush any pending edit-state sync before dropping the project: the clips
    // themselves are already archived to R2 as they were edited.
    flushClipState();
    setStatus('idle');
    setJobId(null);
    setResults(null);
    setLogs([]);
    setProcessingMedia(null);
    setProjectState(null);
    setNoSource(false);
    localStorage.removeItem(SESSION_KEY);
  };

  // --- UI Components ---

  // One nav definition drives all three surfaces: the desktop rail, the mobile
  // drawer, and the bottom tab bar. `short` is the tab-bar label — the full one
  // wraps to two lines in a 5-up bar on a 360px phone.
  const navItems = [
    { id: 'dashboard', ord: '01', icon: LayoutDashboard, label: 'Clip Generator', short: 'clips', primary: true },
    { id: 'saasshorts', ord: '02', icon: Sparkles, label: 'AI Shorts', short: 'ai shorts', byok: true, primary: true },
    { id: 'ai-agent', ord: '03', icon: Bot, label: 'AI Agent', short: 'agent', byok: true },
    { id: 'ugc-gallery', ord: '04', icon: LayoutGrid, label: 'UGC Gallery', short: 'gallery', primary: true },
    { id: 'thumbnails', ord: '05', icon: Image, label: 'YouTube Studio', short: 'studio', primary: true },
    ...(billingEnabled && isSignedIn ? [{ id: 'history', ord: '06', icon: History, label: 'History', short: 'history' }] : []),
    { id: 'settings', ord: '07', icon: Settings, label: 'Settings', short: 'settings' },
  ];
  const activeNav = navItems.find((n) => n.id === activeTab);

  // Escape closes the mobile drawer. The shell itself is overflow-hidden, so
  // there is no body scroll to lock behind it.
  useEffect(() => {
    if (!navOpen) return;
    const onKey = (e) => { if (e.key === 'Escape') setNavOpen(false); };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [navOpen]);

  const goToTab = (id) => {
    if (tutorialLock && id !== 'dashboard') return;
    setActiveTab(id);
    setNavOpen(false);
  };
  const tabLocked = (id) => tutorialLock && id !== 'dashboard';

  // Shared footer links (landing, repo, pricing, contact) — same list in the
  // desktop rail and the mobile drawer, so they can never drift apart.
  const NavFooterLinks = ({ collapsed = false }) => (
    <>
      <a
        href="#landing"
        className="flex items-center gap-2 px-3 py-2 text-xs lowercase text-muted hover:text-ink2 transition-colors"
      >
        <Globe size={14} className="shrink-0" />
        <span className={collapsed ? 'hidden lg:block truncate' : 'truncate'}>landing page</span>
      </a>
      <a
        href="https://github.com/mutonby/openshorts"
        target="_blank"
        rel="noopener noreferrer"
        className="flex items-center gap-2 px-3 py-2 text-xs lowercase text-muted hover:text-ink2 transition-colors"
      >
        <svg height="14" viewBox="0 0 16 16" version="1.1" width="14" aria-hidden="true" fill="currentColor" className="shrink-0"><path fillRule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path></svg>
        <span className={collapsed ? 'hidden lg:block truncate' : 'truncate'}>open source</span>
      </a>
      {billingEnabled && (
        <a
          href="#/pricing"
          className="flex items-center gap-2 px-3 py-2 text-xs lowercase text-muted hover:text-ink2 transition-colors"
        >
          <Sparkles size={14} className="shrink-0" />
          <span className={collapsed ? 'hidden lg:block truncate' : 'truncate'}>plans &amp; pricing</span>
        </a>
      )}
      <a
        href="mailto:info@openshorts.app"
        className="flex items-center gap-2 px-3 py-2 text-xs lowercase text-muted hover:text-ink2 transition-colors"
      >
        <Mail size={14} className="shrink-0" />
        <span className={collapsed ? 'hidden lg:block truncate' : 'truncate'}>info@openshorts.app</span>
      </a>
    </>
  );

  // Desktop rail: icon-only from md, labelled from lg. Below md it is gone
  // entirely — an unlabelled 80px rail ate a fifth of a phone screen.
  const Sidebar = () => (
    <div className="hidden md:flex w-20 lg:w-64 bg-paper2 border-r border-rule flex-col h-full shrink-0 transition-all duration-300">
      <a href="#landing" className="p-6 flex items-center gap-3" title="go to landing page">
        <div className="w-8 h-8 bg-paper3 rounded-input flex items-center justify-center shrink-0 overflow-hidden border border-rule">
          <img src="/logo-openshorts.png" alt="Logo" className="w-full h-full object-cover" />
        </div>
        <span className="font-display lowercase text-lg text-ink hidden lg:block">openshorts</span>
      </a>

      <nav className="flex-1 px-4 py-4 space-y-1">
        {navItems.map((item) => {
          const NavIcon = item.icon;
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              data-tutorial={item.id === 'dashboard' ? 'nav-clips' : undefined}
              onClick={() => goToTab(item.id)}
              title={tabLocked(item.id) ? 'Finish your first clips to unlock' : item.label}
              disabled={tabLocked(item.id)}
              className={`relative w-full flex items-center gap-3 px-3 py-2.5 rounded-input transition-colors ${isActive ? 'bg-paper3 text-ink' : 'text-muted hover:text-ink2 hover:bg-paper3/50'} ${tabLocked(item.id) ? 'opacity-40 cursor-not-allowed hover:bg-transparent hover:text-muted' : ''}`}
            >
              {isActive && (
                <span className="absolute left-0 top-1.5 bottom-1.5 w-0.5 bg-brass rounded-full" aria-hidden="true" />
              )}
              <NavIcon size={18} className={`shrink-0 ${isActive ? 'text-brass' : ''}`} />
              <span className="text-sm lowercase hidden lg:block flex-1 text-left truncate">{item.label}</span>
              {tabLocked(item.id)
                ? <Lock size={12} className="shrink-0 hidden lg:block" />
                : item.byok ? <span className="readout hidden lg:block">BYOK</span> : null}
              <span className="readout hidden lg:block">{item.ord}</span>
            </button>
          );
        })}
      </nav>

      <div className="p-4 border-t border-rule space-y-1">
        <NavFooterLinks collapsed />
      </div>
    </div>
  );

  // Mobile drawer: the complete nav, reachable from the header's menu button.
  const MobileNavDrawer = () => (
    <div
      className="md:hidden fixed inset-0 z-[90] flex"
      role="dialog"
      aria-modal="true"
      aria-label="Navigation"
    >
      <div
        className="absolute inset-0 bg-black/60 animate-fade"
        onClick={() => setNavOpen(false)}
        aria-hidden="true"
      />
      <div className="relative w-[17rem] max-w-[82vw] h-full bg-paper2 border-r border-rule flex flex-col animate-slide-in-left">
        <div className="flex items-center justify-between px-5 h-14 border-b border-rule shrink-0">
          <a href="#landing" className="flex items-center gap-2.5" onClick={() => setNavOpen(false)}>
            <div className="w-7 h-7 bg-paper3 rounded-input overflow-hidden border border-rule shrink-0">
              <img src="/logo-openshorts.png" alt="" className="w-full h-full object-cover" />
            </div>
            <span className="font-display lowercase text-lg text-ink">openshorts</span>
          </a>
          <button
            onClick={() => setNavOpen(false)}
            aria-label="close navigation"
            className="p-2 -mr-2 text-muted hover:text-ink transition-colors"
          >
            <X size={18} />
          </button>
        </div>

        <nav className="flex-1 overflow-y-auto custom-scrollbar px-3 py-3 space-y-1">
          {navItems.map((item) => {
            const NavIcon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => goToTab(item.id)}
                disabled={tabLocked(item.id)}
                aria-current={isActive ? 'page' : undefined}
                title={tabLocked(item.id) ? 'Finish your first clips to unlock' : undefined}
                className={`relative w-full flex items-center gap-3 px-3 py-3 rounded-input transition-colors ${isActive ? 'bg-paper3 text-ink' : 'text-muted active:bg-paper3/60'} ${tabLocked(item.id) ? 'opacity-40 cursor-not-allowed' : ''}`}
              >
                {isActive && (
                  <span className="absolute left-0 top-2 bottom-2 w-0.5 bg-brass rounded-full" aria-hidden="true" />
                )}
                <NavIcon size={18} className={`shrink-0 ${isActive ? 'text-brass' : ''}`} />
                <span className="text-[0.95rem] lowercase flex-1 text-left truncate">{item.label}</span>
                {tabLocked(item.id)
                  ? <Lock size={12} className="shrink-0" />
                  : item.byok ? <span className="readout shrink-0">BYOK</span> : null}
              </button>
            );
          })}
        </nav>

        <div className="px-3 py-3 border-t border-rule space-y-0.5 safe-bottom shrink-0">
          <NavFooterLinks />
        </div>
      </div>
    </div>
  );

  // Bottom tab bar: the four everyday destinations plus "more" for the rest.
  // It is a flex sibling of the scrolling pane rather than `fixed`, so nothing
  // ever hides behind it and no pane needs compensating padding.
  const MobileTabBar = () => {
    const tabs = navItems.filter((n) => n.primary);
    const moreActive = !tabs.some((t) => t.id === activeTab);
    return (
      <nav className="md:hidden shrink-0 border-t border-rule bg-paper2/95 backdrop-blur-sm safe-bottom">
        <div className="flex items-stretch">
          {tabs.map((item) => {
            const NavIcon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                data-tutorial={item.id === 'dashboard' ? 'nav-clips' : undefined}
                onClick={() => goToTab(item.id)}
                disabled={tabLocked(item.id)}
                aria-current={isActive ? 'page' : undefined}
                title={tabLocked(item.id) ? 'Finish your first clips to unlock' : undefined}
                className={`flex-1 min-w-0 flex flex-col items-center justify-center gap-1 py-2 min-h-[56px] transition-colors ${isActive ? 'text-ink' : 'text-muted active:text-ink2'} ${tabLocked(item.id) ? 'opacity-40 cursor-not-allowed' : ''}`}
              >
                <NavIcon size={19} className={isActive ? 'text-brass' : ''} />
                <span className="text-[10.5px] lowercase leading-none truncate max-w-full px-0.5">{item.short}</span>
              </button>
            );
          })}
          <button
            onClick={() => setNavOpen(true)}
            aria-label="more sections"
            aria-expanded={navOpen}
            className={`flex-1 min-w-0 flex flex-col items-center justify-center gap-1 py-2 min-h-[56px] transition-colors ${moreActive ? 'text-ink' : 'text-muted active:text-ink2'}`}
          >
            <Menu size={19} className={moreActive ? 'text-brass' : ''} />
            <span className="text-[10.5px] lowercase leading-none">more</span>
          </button>
        </div>
      </nav>
    );
  };

  return (
    /* h-dvh where supported: on mobile Safari/Chrome `100vh` is the tallest the
       viewport ever gets, so a h-screen shell hides its own bottom bar behind
       the browser chrome until the user scrolls. */
    <div className="flex h-screen supports-[height:100dvh]:h-[100dvh] bg-paper overflow-hidden">
      <Sidebar />
      {navOpen && <MobileNavDrawer />}

      <main className="flex-1 min-w-0 flex flex-col h-full overflow-hidden relative">
        {/* Top Header */}
        <header className="h-14 border-b border-rule bg-paper flex items-center justify-between gap-2 px-3 sm:px-6 shrink-0 z-10">
          <div className="flex items-center gap-2 sm:gap-4 min-w-0">
            {/* Mobile: the drawer handle, and the section name the icon rail
                used to carry. Without it a phone has no "where am I". */}
            <button
              onClick={() => setNavOpen(true)}
              aria-label="open navigation"
              className="md:hidden -ml-1 p-2 rounded-input text-muted active:bg-paper3 transition-colors shrink-0"
            >
              <Menu size={20} />
            </button>
            <span data-tutorial="nav-clips" className="md:hidden font-display lowercase text-base text-ink truncate">
              {activeNav?.label || 'openshorts'}
            </span>
            {status !== 'idle' && (
              <button
                onClick={handleReset}
                className="btn-quiet px-3 py-1.5 text-xs shrink-0"
                aria-label="New Project"
              >
                <Plus size={14} />
                <span className="hidden sm:inline">New Project</span>
              </button>
            )}
          </div>

          <div className="flex items-center gap-2 sm:gap-4 shrink-0">
            {userProfiles.length > 0 && (
              <UserProfileSelector
                profiles={userProfiles}
                selectedUserId={uploadUserId}
                onSelect={setUploadUserId}
                onConnect={isManaged ? handleConnectSocials : undefined}
              />
            )}

            {/* Cloud: minutes meter + account/sign-in. For free users the meter
                opens the upgrade modal — otherwise the only path to a plan is
                failing against the quota wall. */}
            {billingEnabled && isManaged && (
              <UsageMeter onClick={() => {
                if (plan === 'free') { setTopUpInfo({ context: 'upsell' }); setShowTopUp(true); }
                else { window.location.hash = '#/account'; }
              }} />
            )}
            {billingEnabled && isSignedIn && !isManaged && (
              <button onClick={() => setShowPlanChoice(true)}
                className="btn-primary px-4 py-2 text-xs">
                Choose a plan
              </button>
            )}
            {billingEnabled && !isSignedIn && (
              <button onClick={() => setShowLogin(true)}
                className="btn-ghost px-4 py-2 text-xs">
                Sign in
              </button>
            )}
            {billingEnabled && isSignedIn && <ProfileMenu />}

            {/* Hidden below sm: the standing banner underneath already says the
                same thing, and two warnings in a 360px header is just noise. */}
            {keysMissing && (
              <button
                onClick={() => (billingEnabled && !isSignedIn ? setShowLogin(true) : goToTab('settings'))}
                className="badge-warn hover:brightness-125 transition-all hidden sm:inline-flex"
                title="Configure API keys or choose a plan"
              >
                <AlertTriangle size={12} />
                <span className="hidden md:inline">
                  {!geminiOk && !uploadPostKey
                    ? 'Gemini & Upload-Post keys missing'
                    : !geminiOk
                      ? 'Gemini API Key Missing'
                      : 'Upload-Post API Key Missing'}
                </span>
                <span className="md:hidden">keys missing</span>
              </button>
            )}
          </div>
        </header>

        {/* Persistent Missing Keys Banner — visible on every screen */}
        {keysMissing && activeTab !== 'settings' && (
          <div className="mx-3 sm:mx-6 mt-3 px-3.5 sm:px-4 py-3 bg-paper2 border border-rule rounded-card flex flex-wrap items-center justify-between gap-2.5 sm:gap-4 shrink-0 animate-fade">
            <div className="flex items-start sm:items-center gap-2.5 sm:gap-3 text-sm text-ink2 min-w-0 flex-1">
              <KeyRound size={16} className="shrink-0 text-warn mt-0.5 sm:mt-0" />
              <div className="min-w-0">
                <span className="font-medium text-ink">Required API keys missing.</span>{' '}
                <span className="text-muted">
                  {!geminiOk && !uploadPostKey
                    ? 'Set your Gemini and Upload-Post API keys to use OpenShorts.'
                    : !geminiOk
                      ? 'Set your Gemini API key to use OpenShorts.'
                      : 'Set your Upload-Post API key to use OpenShorts.'}
                </span>
              </div>
            </div>
            <button
              onClick={() => goToTab('settings')}
              className="btn-quiet px-3 py-1.5 text-xs shrink-0 w-full sm:w-auto"
            >
              Go to Settings
            </button>
          </div>
        )}

        {/* Session Recovery Banner */}
        {sessionRecovered && (
          <div className="mx-3 sm:mx-6 mt-2 px-3.5 sm:px-4 py-3 bg-paper2 border border-rule rounded-card flex items-start justify-between gap-3 animate-fade shrink-0">
            <div className="flex items-start sm:items-center gap-2 text-sm text-ink2 flex-wrap min-w-0">
              <RotateCcw size={16} className="text-brass shrink-0 mt-0.5 sm:mt-0" />
              <span className="font-medium">Session recovered</span>
              <span className="text-muted text-xs">Your previous work has been restored.</span>
            </div>
            <button
              onClick={() => setSessionRecovered(false)}
              aria-label="dismiss"
              className="text-muted hover:text-ink transition-colors shrink-0 -m-1 p-1"
            >
              <X size={16} />
            </button>
          </div>
        )}

        {/* Included tools (Clip Generator, YouTube Studio): non-blocking trial prompt. */}
        {gateThisTab && <TrialGate toolName={TOOL_NAMES[activeTab] || 'this'} />}

        {/* Advanced tools (AI Shorts, AI Agent): BYOK fal.ai + ElevenLabs notice. */}
        {advancedThisTab && <AdvancedBanner needsPlan={needsPlan} onKeys={() => goToTab('settings')} />}

        {/* Main Workspace */}
        <div className="flex-1 overflow-hidden relative">

          {/* View: Settings */}
          {activeTab === 'settings' && (
            <div className="h-full overflow-y-auto p-4 sm:p-8 max-w-2xl mx-auto animate-fade">
              <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-4 mb-8">
                <div>
                  <p className="eyebrow mb-1.5">07 · SETTINGS</p>
                  <h1 className="font-display lowercase text-2xl text-ink">Settings</h1>
                </div>
                <div className="flex items-center gap-2 text-xs text-muted mt-1">
                  <Shield size={12} className="text-ok shrink-0" /> Privacy: keys only live in your browser (sent to backend just to process)
                </div>
              </div>
              {/* Self-hosted installs have no account page, so the agent
                  how-to lives here; cloud users get it (with OAuth) in Account. */}
              {!billingEnabled && <div className="mb-6"><McpConnectCard cloud={false} /></div>}
              {isManaged ? (
                <div className="card p-6 mb-2">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-3">
                      <div className="w-9 h-9 rounded-input bg-paper3 flex items-center justify-center shrink-0">
                        <Shield size={16} className="text-brass" />
                      </div>
                      <h2 className="text-base font-medium text-ink lowercase">Included in your plan</h2>
                    </div>
                    <span className="badge-ok">Managed</span>
                  </div>
                  <p className="text-xs text-muted mb-5 leading-relaxed">
                    Your plan includes the <strong>Clip Generator</strong> and <strong>YouTube Studio</strong>,
                    fully managed — no API keys required. AI Shorts &amp; dubbing use your own fal.ai / ElevenLabs
                    keys (below). Connect your social accounts to publish directly.
                  </p>
                  <div className="flex flex-wrap gap-2">
                    <button onClick={handleConnectSocials} className="btn-primary py-2 px-4 text-sm">
                      <Share2 size={16} /> Connect social accounts
                    </button>
                    <button onClick={handleOpenCalendar} className="btn-quiet py-2 px-4 text-sm">
                      <Calendar size={16} /> Content calendar
                    </button>
                  </div>
                </div>
              ) : billingEnabled ? (
                <div className="card p-6 mb-2">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-3">
                      <div className="w-9 h-9 rounded-input bg-paper3 flex items-center justify-center shrink-0">
                        <Sparkles size={16} className="text-brass" />
                      </div>
                      <h2 className="text-base font-medium text-ink lowercase">Choose your plan</h2>
                    </div>
                    <span className="badge-ok">Free plan available</span>
                  </div>
                  <p className="text-xs text-muted mb-5 leading-relaxed">
                    Generate shorts with zero setup — no API keys needed. Start free with 20 min/month, or go paid from $12/mo. Cancel anytime.
                  </p>
                  <button onClick={() => setShowPlanChoice(true)} className="btn-primary py-2 px-4 text-sm">
                    <Sparkles size={16} /> Choose a plan
                  </button>
                </div>
              ) : (
                <>
              <KeyInput onKeySet={setApiKey} savedKey={apiKey} />

              <div className="card p-4 sm:p-6 mt-8">
                <div className="flex flex-wrap items-center justify-between gap-2 mb-4">
                  <div className="flex items-center gap-3">
                    <div className="w-9 h-9 rounded-input bg-paper3 flex items-center justify-center shrink-0">
                      <Share2 size={16} className="text-brass" />
                    </div>
                    <h2 className="text-base font-medium text-ink lowercase">Social Integration</h2>
                  </div>
                  <span className="badge-warn">Required</span>
                </div>
                <p className="text-xs text-muted mb-6 leading-relaxed">
                  Required to publish your clips to TikTok, Instagram Reels, and YouTube Shorts via <strong>Upload-Post</strong>.
                  Includes a <strong>free tier</strong> (no credit card required).
                </p>
                <div className="space-y-4">
                  <label className="block text-sm text-muted">Upload-Post API Key</label>
                  <div className="flex flex-col sm:flex-row gap-2">
                    <input
                      type="password"
                      value={uploadPostKey}
                      onChange={(e) => setUploadPostKey(e.target.value)}
                      className="input-field"
                      placeholder="ey..."
                    />
                    <button onClick={fetchUserProfiles} className="btn-quiet py-2 px-4 text-sm">
                      Connect
                    </button>
                  </div>
                  <div className="text-xs text-muted leading-relaxed">
                    Connect your Upload-Post account to enable one-click publishing.
                    <div className="mt-3 grid grid-cols-1 sm:grid-cols-3 gap-2">
                      <a href="https://app.upload-post.com/login" target="_blank" rel="noopener noreferrer" className="p-2 border border-rule rounded-input hover:bg-paper3 transition-colors flex flex-col gap-1">
                        <span className="text-ink2 font-medium">1. Login</span>
                        <span className="text-xs text-muted">Register account</span>
                      </a>
                      <a href="https://app.upload-post.com/manage-users" target="_blank" rel="noopener noreferrer" className="p-2 border border-rule rounded-input hover:bg-paper3 transition-colors flex flex-col gap-1">
                        <span className="text-ink2 font-medium">2. Profiles</span>
                        <span className="text-xs text-muted">Create & Connect</span>
                      </a>
                      <a href="https://app.upload-post.com/api-keys" target="_blank" rel="noopener noreferrer" className="p-2 border border-rule rounded-input hover:bg-paper3 transition-colors flex flex-col gap-1">
                        <span className="text-ink2 font-medium">3. API Key</span>
                        <span className="text-xs text-muted">Generate key</span>
                      </a>
                    </div>
                    <br />
                    <span className="text-muted">
                      Keys are only stored in your browser. They are sent to the backend only to process your request, never stored server-side.
                    </span>
                  </div>
                </div>
              </div>

                </>
              )}

              <div className="card p-4 sm:p-6 mt-8">
                <div className="flex flex-wrap items-center justify-between gap-2 mb-4">
                  <div className="flex items-center gap-3">
                    <div className="w-9 h-9 rounded-input bg-paper3 flex items-center justify-center shrink-0">
                      <Globe size={16} className="text-brass" />
                    </div>
                    <h2 className="text-base font-medium text-ink lowercase">Video Translation</h2>
                  </div>
                  <span className="readout">BYOK</span>
                </div>
                <p className="text-xs text-muted mb-6 leading-relaxed">
                  For <strong>AI Shorts &amp; dubbing</strong> — bring your own key. Translate your clips to different
                  languages using <strong>ElevenLabs</strong> AI dubbing (billed by ElevenLabs). Not covered by your plan.
                </p>
                <div className="space-y-4">
                  <label className="block text-sm text-muted">ElevenLabs API Key</label>
                  <div className="flex flex-col sm:flex-row gap-2">
                    <input
                      type="password"
                      value={elevenLabsKey}
                      onChange={(e) => setElevenLabsKey(e.target.value)}
                      className="input-field"
                      placeholder="sk_..."
                    />
                    <button
                      onClick={() => {
                        if (elevenLabsKey) {
                          localStorage.setItem('elevenLabsKey_v1', encrypt(elevenLabsKey));
                          setElevenLabsSaved(true);
                          setTimeout(() => setElevenLabsSaved(false), 2000);
                        }
                      }}
                      className={elevenLabsSaved ? 'badge-ok px-4' : 'btn-quiet py-2 px-4 text-sm'}
                    >
                      {elevenLabsSaved ? <><Check size={12} /> saved</> : 'Save'}
                    </button>
                  </div>
                  <div className="text-xs text-muted leading-relaxed">
                    Get your API key from ElevenLabs to enable video translation.
                    <div className="mt-3 grid grid-cols-1 sm:grid-cols-2 gap-2">
                      <a href="https://elevenlabs.io/sign-up" target="_blank" rel="noopener noreferrer" className="p-2 border border-rule rounded-input hover:bg-paper3 transition-colors flex flex-col gap-1">
                        <span className="text-ink2 font-medium">1. Sign Up</span>
                        <span className="text-xs text-muted">Create account</span>
                      </a>
                      <a href="https://elevenlabs.io/app/settings/api-keys" target="_blank" rel="noopener noreferrer" className="p-2 border border-rule rounded-input hover:bg-paper3 transition-colors flex flex-col gap-1">
                        <span className="text-ink2 font-medium">2. API Key</span>
                        <span className="text-xs text-muted">Generate key</span>
                      </a>
                    </div>
                    <br />
                    <span className="text-muted">
                      Keys are only stored in your browser. They are sent to the backend only to process your request, never stored server-side.
                    </span>
                  </div>
                </div>
              </div>

              <div className="card p-4 sm:p-6 mt-8">
                <div className="flex flex-wrap items-center justify-between gap-2 mb-4">
                  <div className="flex items-center gap-3">
                    <div className="w-9 h-9 rounded-input bg-paper3 flex items-center justify-center shrink-0">
                      <Sparkles size={16} className="text-brass" />
                    </div>
                    <h2 className="text-base font-medium text-ink lowercase">AI Shorts (UGC Videos)</h2>
                  </div>
                  <span className="readout">BYOK</span>
                </div>
                <p className="text-xs text-muted mb-6 leading-relaxed">
                  Generate UGC-style videos with AI actors for any product or business using <strong>fal.ai</strong>.
                  <strong> Not covered by your plan</strong> — bring your own fal.ai + ElevenLabs keys (billed by those
                  providers, ~$0.65-2 per video). Your plan still covers the AI script &amp; orchestration.
                </p>
                <div className="space-y-4">
                  <label className="block text-sm text-muted">fal.ai API Key</label>
                  <div className="flex flex-col sm:flex-row gap-2">
                    <input
                      type="password"
                      value={falKey}
                      onChange={(e) => setFalKey(e.target.value)}
                      className="input-field"
                      placeholder="fal_..."
                    />
                    <button
                      onClick={() => {
                        if (falKey) {
                          localStorage.setItem('falKey_v1', encrypt(falKey));
                          setFalSaved(true);
                          setTimeout(() => setFalSaved(false), 2000);
                        }
                      }}
                      className={falSaved ? 'badge-ok px-4' : 'btn-quiet py-2 px-4 text-sm'}
                    >
                      {falSaved ? <><Check size={12} /> saved</> : 'Save'}
                    </button>
                  </div>
                  <div className="text-xs text-muted leading-relaxed">
                    Get your API key from fal.ai to enable AI actor video generation.
                    <div className="mt-3 grid grid-cols-1 sm:grid-cols-2 gap-2">
                      <a href="https://fal.ai/dashboard/keys" target="_blank" rel="noopener noreferrer" className="p-2 border border-rule rounded-input hover:bg-paper3 transition-colors flex flex-col gap-1">
                        <span className="text-ink2 font-medium">1. Sign Up</span>
                        <span className="text-xs text-muted">Create fal.ai account</span>
                      </a>
                      <a href="https://fal.ai/dashboard/keys" target="_blank" rel="noopener noreferrer" className="p-2 border border-rule rounded-input hover:bg-paper3 transition-colors flex flex-col gap-1">
                        <span className="text-ink2 font-medium">2. API Key</span>
                        <span className="text-xs text-muted">Generate key</span>
                      </a>
                    </div>
                    <br />
                    <span className="text-muted">
                      Keys are only stored in your browser. Sent to backend only to process requests.
                    </span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* View: SaaS Shorts */}
          {activeTab === 'saasshorts' && (
            <SaaShortsTab geminiApiKey={apiKey} elevenLabsKey={elevenLabsKey} falKey={falKey} uploadPostKey={uploadPostKey} uploadUserId={uploadUserId} managed={isManaged} />
          )}

          {/* View: AI Agent */}
          {activeTab === 'ai-agent' && (
            <div className="h-full overflow-y-auto custom-scrollbar p-4 sm:p-6 md:p-10 animate-fade">
              <div className="max-w-4xl mx-auto space-y-8">

                {/* Header */}
                <div className="space-y-3">
                  <p className="eyebrow flex items-center gap-2">
                    <Bot size={12} /> 03 · AI AGENT · AUTONOMOUS SKILL
                  </p>
                  <h1 className="font-display lowercase text-3xl md:text-4xl text-ink">
                    Your Personal Clipping Team
                  </h1>
                  <p className="text-muted text-base md:text-lg leading-relaxed max-w-2xl">
                    Drop your videos in a folder and a team of AI clippers picks the viral moments, edits them, and queues them for your approval — like having a 24/7 short-form editing crew on autopilot.
                  </p>
                </div>

                {/* Mobile-format warning */}
                <div className="px-4 py-3 rounded-card border border-rule bg-paper2 flex items-start gap-3">
                  <Smartphone size={18} className="text-warn shrink-0 mt-0.5" />
                  <div className="text-sm text-ink2">
                    <p className="font-medium text-ink mb-1">Upload videos already in vertical (9:16) mobile format.</p>
                    <p className="text-muted leading-relaxed">
                      The agent does not reframe horizontal footage. Make sure every source video is shot or pre-cropped to mobile/portrait format before dropping it into the input folder.
                    </p>
                  </div>
                </div>

                {/* Workflow */}
                <div className="grid md:grid-cols-3 gap-4">
                  <div className="card p-5 space-y-2">
                    <div className="w-10 h-10 rounded-input bg-paper3 flex items-center justify-center">
                      <Upload size={18} className="text-brass" />
                    </div>
                    <h3 className="font-medium text-ink lowercase">1. Drop your videos</h3>
                    <p className="text-xs text-muted leading-relaxed">
                      Put your long-form vertical footage in the watched folder. The skill picks one video per run.
                    </p>
                  </div>

                  <div className="card p-5 space-y-2">
                    <div className="w-10 h-10 rounded-input bg-paper3 flex items-center justify-center">
                      <Users size={18} className="text-brass" />
                    </div>
                    <h3 className="font-medium text-ink lowercase">2. AI clippers work</h3>
                    <p className="text-xs text-muted leading-relaxed">
                      Whisper transcribes, Gemini 3 Flash spots viral beats, FFmpeg cuts each clip and adds a hook overlay.
                    </p>
                  </div>

                  <div className="card p-5 space-y-2">
                    <div className="w-10 h-10 rounded-input bg-paper3 flex items-center justify-center">
                      <CheckCircle2 size={18} className="text-brass" />
                    </div>
                    <h3 className="font-medium text-ink lowercase">3. You validate, it ships</h3>
                    <p className="text-xs text-muted leading-relaxed">
                      Approve the candidates you like and the skill auto-publishes them to TikTok, Reels and YouTube Shorts via Upload-Post.
                    </p>
                  </div>
                </div>

                {/* Repo CTA */}
                <div className="card p-6 md:p-8 space-y-5">
                  <div className="flex items-start justify-between gap-4 flex-wrap">
                    <div>
                      <h2 className="font-display lowercase text-xl text-ink mb-1">skill-autoshorts</h2>
                      <p className="text-sm text-muted">
                        The Claude Code skill that powers this workflow. Install it once and trigger it whenever you want a fresh batch of clips.
                      </p>
                    </div>
                    <a
                      href="https://github.com/mutonby/skill-autoshorts"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="btn-primary py-2 px-4 text-sm shrink-0"
                    >
                      View on GitHub <ExternalLink size={14} />
                    </a>
                  </div>

                  <div className="bg-paper border border-rule rounded-card p-4 font-mono text-xs text-ink2 flex items-center justify-between gap-3">
                    <span className="truncate">git clone https://github.com/mutonby/skill-autoshorts</span>
                    <button
                      onClick={() => navigator.clipboard.writeText('git clone https://github.com/mutonby/skill-autoshorts')}
                      className="text-muted hover:text-ink transition-colors shrink-0"
                      title="Copy"
                    >
                      <Copy size={14} />
                    </button>
                  </div>

                  <div className="grid sm:grid-cols-2 gap-3 text-sm">
                    <div className="flex items-start gap-2 text-ink2">
                      <Check size={16} className="text-brass shrink-0 mt-0.5" />
                      <span>Daily batch — picks one long video per run</span>
                    </div>
                    <div className="flex items-start gap-2 text-ink2">
                      <Check size={16} className="text-brass shrink-0 mt-0.5" />
                      <span>Whisper transcription with word-level timing</span>
                    </div>
                    <div className="flex items-start gap-2 text-ink2">
                      <Check size={16} className="text-brass shrink-0 mt-0.5" />
                      <span>Gemini 3 Flash multimodal moment detection</span>
                    </div>
                    <div className="flex items-start gap-2 text-ink2">
                      <Check size={16} className="text-brass shrink-0 mt-0.5" />
                      <span>Auto-publish to TikTok, Reels & YouTube Shorts</span>
                    </div>
                  </div>
                </div>

              </div>
            </div>
          )}

          {/* View: UGC Gallery */}
          {activeTab === 'ugc-gallery' && (
            <div className="h-full overflow-y-auto custom-scrollbar animate-fade">
              <div className="max-w-6xl mx-auto p-4 sm:p-6 md:p-8">
                <UGCGallery />
              </div>
            </div>
          )}

          {/* View: History */}
          {activeTab === 'history' && (
            <div className="h-full overflow-y-auto custom-scrollbar animate-fade">
              <div className="max-w-6xl mx-auto p-4 sm:p-6 md:p-8">
                <HistoryTab onReopenProject={restoreProject} />
              </div>
            </div>
          )}

          {activeTab === 'thumbnails' && (
            <ThumbnailStudio
              geminiApiKey={apiKey}
              uploadPostKey={uploadPostKey}
              uploadUserId={uploadUserId}
              managed={isManaged}
              onCreateClips={(sessionId) => {
                setActiveTab('dashboard');
                // The Studio source is the user's own upload, published to their
                // own channel; the handover carries that same attestation.
                handleProcess({ type: 'thumbnail_session', payload: sessionId, acknowledged: true });
              }}
            />
          )}

          {/* View: Gallery */}
          {/* {activeTab === 'gallery' && (
            <Gallery />
          )} */}

          {/* View: Dashboard (Idle) */}
          {activeTab === 'dashboard' && status === 'idle' && (
            <div className="h-full overflow-y-auto custom-scrollbar animate-fade">
              <div className="min-h-full flex flex-col items-center justify-center px-4 py-5 sm:p-6">
              {/* On a phone the hero used to fill the fold on its own and push
                  the uploader — the whole point of the screen — below it. The
                  eyebrow, the display size and the gaps all shrink first. */}
              <div className="max-w-xl w-full text-center space-y-5 sm:space-y-8">
                <div className="space-y-2.5 sm:space-y-4">
                  <p className="eyebrow hidden sm:block">01 · CLIP GENERATOR</p>
                  <h1 className="font-display lowercase text-3xl sm:text-4xl md:text-5xl text-ink">
                    Create Viral Shorts
                  </h1>
                  <p className="text-muted text-[15px] sm:text-lg leading-snug sm:leading-normal max-w-sm sm:max-w-none mx-auto">
                    Drop your long-form video below to instantly generate viral clips with AI.
                  </p>
                  {/* The same pipeline is an MCP server: point people at the
                      one place that explains how to drive it from an agent. */}
                  {!tutorialLock && (
                  <p className="text-xs text-muted">
                    Or let an agent do it:{' '}
                    <a
                      href={billingEnabled ? '#/account' : '#app'}
                      onClick={(e) => { if (!billingEnabled) { e.preventDefault(); goToTab('settings'); } }}
                      className="text-ink2 underline underline-offset-2 hover:text-brass transition-colors"
                    >
                      connect Claude, ChatGPT or n8n →
                    </a>
                  </p>
                  )}
                </div>

                <MediaInput onProcess={handleProcess} isProcessing={status === 'processing'} />

                <div className="flex flex-wrap items-center justify-center gap-4 sm:gap-8 text-muted text-xs sm:text-sm">
                  <span className="flex items-center gap-2"><Youtube size={16} /> YouTube</span>
                  <span className="flex items-center gap-2"><Instagram size={16} /> Instagram</span>
                  <span className="flex items-center gap-2"><TikTokIcon size={16} /> TikTok</span>
                </div>
              </div>
              </div>
            </div>
          )}

          {/* View: Processing / Results (Split View) */}
          {activeTab === 'dashboard' && (status === 'processing' || status === 'complete' || status === 'error') && (
            <div className="h-full flex flex-col md:flex-row gap-3 md:gap-4 p-3 md:p-4 overflow-y-auto md:overflow-y-hidden custom-scrollbar animate-fade">

              {/* Left Panel: Preview & Status */}
              <div className={`${status === 'complete' ? 'w-full md:w-[30%] lg:w-[25%]' : 'w-full md:w-[55%] lg:w-[60%]'} md:h-full flex flex-col shrink-0 md:shrink card p-3.5 sm:p-6 md:overflow-y-auto custom-scrollbar transition-all duration-700 ease-in-out`}>
                <div className="mb-4 sm:mb-6 flex items-center justify-between gap-2">
                  <h2 className="text-sm font-medium text-ink lowercase flex items-center gap-2">
                    <Activity className={`text-brass ${status === 'processing' ? 'animate-pulse' : ''}`} size={18} />
                    Live Analysis
                  </h2>
                  <span className={status === 'processing' ? 'badge-brass' :
                    status === 'complete' ? 'badge-ok' :
                      'badge-danger'
                    }>
                    {status.toUpperCase()}
                  </span>
                </div>

                {/* Video Preview */}
                {processingMedia && (
                  <ProcessingAnimation
                    media={processingMedia}
                    isComplete={status === 'complete'}
                    syncedTime={syncedTime}
                    isSyncedPlaying={isSyncedPlaying}
                    syncTrigger={syncTrigger}
                  />
                )}

                {/* Phones only. The scan box drops its invented telemetry at
                    this size and the log terminal below starts collapsed, so
                    without this the screen would say nothing about what the job
                    is actually doing. The log tail is the real answer. */}
                {status === 'processing' && (
                  <div className="sm:hidden mb-3 flex items-start gap-2 text-xs text-ink2 min-w-0">
                    <Loader2 size={14} className="animate-spin text-brass shrink-0 mt-px" />
                    <span className="min-w-0 leading-snug break-words">
                      {logs.length ? logs[logs.length - 1] : 'starting up…'}
                    </span>
                  </div>
                )}

                {/* The render is dead time: the user is watching a progress bar
                    with nothing to do, so this is where the one star ask goes. */}
                {status === 'processing' && (
                  <div className="my-3">
                    <StarBanner message="Got a minute while this renders?" />
                  </div>
                )}

                {/* Logs Terminal */}
                <div className={`bg-paper rounded-card border border-rule overflow-hidden flex flex-col transition-all duration-500 ${status === 'complete' ? `min-h-0 opacity-50 hover:opacity-100 ${logsVisible ? 'h-32' : 'h-auto'}` : `flex-1 ${logsVisible ? 'min-h-[160px] sm:min-h-[200px]' : 'min-h-0 flex-none'}`}`}>
                  <button
                    type="button"
                    onClick={() => setLogsVisible(!logsVisible)}
                    aria-expanded={logsVisible}
                    className="w-full px-3.5 sm:px-4 py-2.5 border-b border-rule flex items-center justify-between gap-2 bg-paper2 shrink-0 text-left"
                  >
                    <span className="readout flex items-center gap-2">
                      <Terminal size={12} /> System Logs
                    </span>
                    <span className="flex items-center gap-2 text-muted">
                      {!logsVisible && logs.length > 0 && (
                        <span className="readout normal-case">{logs.length}</span>
                      )}
                      <ChevronDown size={16} className={logsVisible ? '' : 'rotate-180'} />
                    </span>
                  </button>
                  {logsVisible && (
                    <div className="flex-1 p-3.5 sm:p-4 overflow-y-auto font-mono text-[11px] sm:text-xs space-y-1.5 custom-scrollbar text-muted break-words">
                      {logs.map((log, i) => (
                        <div key={i} className={`flex gap-2 ${log.toLowerCase().includes('error') ? 'text-danger' : 'text-muted'}`}>
                          <span className="text-muted opacity-50 shrink-0 hidden sm:inline">{new Date().toLocaleTimeString()}</span>
                          <span className="min-w-0 break-words">{log}</span>
                        </div>
                      ))}
                      {status === 'processing' && (
                        <div className="animate-pulse text-brass">_</div>
                      )}
                    </div>
                  )}
                </div>
              </div>

              {/* Right Panel: Results Grid */}
              <div className={`${status === 'complete' ? 'w-full md:w-[70%] lg:w-[75%]' : 'w-full md:w-[45%] lg:w-[40%]'} md:h-full flex flex-col shrink-0 md:shrink card p-3.5 sm:p-6 transition-all duration-700 ease-in-out`}>
                {/* Title + counters on one row, the two actions on their own row
                    below. Wrapping them all together dropped a lone half-width
                    "schedule week" pill under the title on a phone. */}
                <div className="mb-4 sm:mb-6 shrink-0 space-y-3">
                  <h2 className="font-display lowercase text-lg sm:text-xl text-ink flex flex-wrap items-center gap-2">
                    <span className="mr-auto">Generated Shorts</span>
                    {results?.clips?.length > 0 && (
                      <span className="readout bg-paper3 px-2.5 py-1 rounded-full">
                        {results.clips.length} Clips
                      </span>
                    )}
                    {results?.cost_analysis && !isManaged && (
                      <span className="readout bg-paper3 px-2.5 py-1 rounded-full" title={`Input: ${results.cost_analysis.input_tokens} | Output: ${results.cost_analysis.output_tokens}`}>
                        GEMINI · ${results.cost_analysis.total_cost.toFixed(5)}
                      </span>
                    )}
                  </h2>
                  {results?.clips?.length > 0 && status === 'complete' && (
                    <div className="flex flex-col sm:flex-row sm:justify-end items-stretch sm:items-center gap-2">
                      <button
                        onClick={handleDownloadAll}
                        disabled={downloadingAll}
                        className="btn-ghost px-3 py-2 text-xs"
                        title="Download all clips as a ZIP"
                      >
                        {downloadingAll
                          ? <><Loader2 size={14} className="animate-spin" />zipping…</>
                          : <><Download size={14} />download all</>}
                      </button>
                      {results.clips.length > 1 && (
                        <button
                          onClick={() => setShowScheduleWeek(true)}
                          className="btn-primary px-4 py-2 text-xs"
                        >
                          <Calendar size={14} />
                          schedule week
                        </button>
                      )}
                    </div>
                  )}
                </div>

                {status === 'complete' && results?.clips?.length > 0 && (
                  <div className="mb-2 space-y-2">
                    {/* Peak-moment upsell: they just SAW their clips — sell while
                        they're proud of the result, before asking for stars. */}
                    {plan === 'free' && (
                      <button
                        onClick={() => { setTopUpInfo({ context: 'upsell' }); setShowTopUp(true); }}
                        className="w-full text-left px-3 py-2.5 rounded-input bg-paper3 border border-brass/40 hover:border-brass text-sm transition-colors"
                      >
                        <span className="text-ink">Like these clips?</span>{' '}
                        <span className="text-muted">They carry a watermark and delete in 7 days.</span>{' '}
                        <span className="text-brass font-medium">Keep them forever →</span>
                      </button>
                    )}
                    {/* Distribution nudge at the same peak: clips on screen,
                        publishing them is one connect away. Hidden once any
                        network is linked or the user dismisses it. */}
                    {showSocialNudge && (
                      <div className="w-full flex flex-col sm:flex-row sm:items-center gap-2 sm:gap-3 px-3 py-2.5 rounded-input bg-paper3 border border-rule text-sm">
                        <div className="flex items-start gap-3 flex-1 min-w-0">
                          <div className="min-w-0 leading-relaxed">
                            <span className="text-ink">Publish these clips straight from here.</span>{' '}
                            <span className="text-muted">Connect your YouTube, TikTok or Instagram once — after that every clip is one click from posted.</span>
                          </div>
                          {/* On a phone the dismiss X rides the copy, so the CTA
                              below can run the full width of the card. */}
                          <button
                            onClick={() => {
                              track('SocialNudgeDismissed');
                              setSocialNudgeDismissed(true);
                              try { localStorage.setItem('os_social_nudge_dismissed', '1'); } catch (_) { /* ignore */ }
                            }}
                            aria-label="dismiss"
                            className="sm:hidden shrink-0 -m-1 p-1 text-muted hover:text-ink"
                          >
                            <X size={16} />
                          </button>
                        </div>
                        <button
                          onClick={() => { track('SocialNudgeConnect'); handleConnectSocials(); }}
                          className="btn-quiet shrink-0 text-xs py-1.5 px-3 lowercase w-full sm:w-auto"
                        >
                          connect socials →
                        </button>
                        <button
                          onClick={() => {
                            track('SocialNudgeDismissed');
                            setSocialNudgeDismissed(true);
                            try { localStorage.setItem('os_social_nudge_dismissed', '1'); } catch (_) { /* ignore */ }
                          }}
                          aria-label="dismiss"
                          className="hidden sm:block shrink-0 p-1 text-muted hover:text-ink"
                        >
                          <X size={14} />
                        </button>
                      </div>
                    )}
                    {/* Self-host only: cloud archives clips to the video library,
                        here they really are gone once the retention sweep runs. */}
                    {!billingEnabled && jobRetentionSeconds > 0 && (
                      <div className="px-3 py-2.5 rounded-input bg-paper3 border border-paper3 text-sm">
                        <span className="text-ink">Clips are kept for {formatRetention(jobRetentionSeconds)}, then deleted.</span>{' '}
                        <span className="text-muted">Download what you want to keep, or raise JOB_RETENTION_SECONDS in your env.</span>
                      </div>
                    )}
                  </div>
                )}

                <div className="flex-1 overflow-y-auto custom-scrollbar p-1">
                  {results && results.clips && results.clips.length > 0 ? (
                    <div className={`grid gap-4 pb-10 ${status === 'complete' ? 'grid-cols-1 xl:grid-cols-2' : 'grid-cols-1'}`}>
                      {rankedClips.map(({ clip, index: i }) => (
                        <ResultCard
                          key={`${jobId}-${i}-${clip.video_url || ''}`}
                          clip={clip}
                          index={i}
                          jobId={jobId}
                          onEditClip={(index) => setEditingClip(index)}
                          onReframeClip={(index) => setReframingClip(index)}
                          initialState={projectState?.clips?.find((c) => c.index === i) || null}
                          onStateChange={handleClipStateChange}
                          durable={durableClips[i]}
                          uploadPostKey={uploadPostKey}
                          uploadUserId={uploadUserId}
                          geminiApiKey={apiKey}
                          elevenLabsKey={elevenLabsKey}
                          isManaged={isManaged}
                          connectedPlatforms={(userProfiles.find((p) => p.username === uploadUserId) || userProfiles[0])?.connected ?? null}
                          onConnectSocials={isManaged ? handleConnectSocials : null}
                          onPlay={(time) => handleClipPlay(time)}
                          onPause={handleClipPause}
                          onBulkSubtitle={handleBulkSubtitles}
                          clipCount={results.clips.length}
                          bulkProgress={bulkSub}
                        />
                      ))}
                    </div>
                  ) : (
                    status === 'processing' ? (
                      <div className="h-full min-h-[140px] flex flex-col items-center justify-center text-muted space-y-3 text-center px-4">
                        <Loader2 size={28} className="animate-spin text-brass" />
                        <p className="text-sm lowercase">Waiting for clips...</p>
                        <p className="text-xs text-muted/80 max-w-[26ch] leading-snug">
                          They appear here one by one as each finishes rendering.
                        </p>
                      </div>
                    ) : status === 'error' ? (
                      <div className="h-full min-h-[120px] flex flex-col items-center justify-center text-danger space-y-2">
                        <p>Generation failed.</p>
                      </div>
                    ) : null
                  )}
                </div>
              </div>

            </div>
          )}

        </div>

        {/* Phone navigation. A flex sibling of the scrolling pane, not a fixed
            overlay, so content is never trapped behind it. */}
        <MobileTabBar />

      </main>

      {/* Missing API Key Modal */}
      <Modal
        isOpen={showKeyModal}
        onClose={() => setShowKeyModal(false)}
        eyebrow="SETUP"
        title={!geminiOk && !uploadPostKey
          ? 'Required API Keys Missing'
          : !geminiOk
            ? 'Gemini API Key Required'
            : 'Upload-Post API Key Required'}
        footer={
          <div className="flex gap-3">
            <button
              onClick={() => setShowKeyModal(false)}
              className="btn-ghost flex-1 px-4 py-2 text-sm"
            >
              Cancel
            </button>
            <button
              onClick={() => { setShowKeyModal(false); goToTab('settings'); }}
              className="btn-primary flex-1 px-4 py-2 text-sm"
            >
              Go to Settings
            </button>
          </div>
        }
      >
        <div className="space-y-4">
          <p className="text-sm text-muted">
            OpenShorts needs both a <strong className="text-ink2">Gemini</strong> API key and an <strong className="text-ink2">Upload-Post</strong> API key. Both have free tiers.
          </p>

          {/* Gemini block */}
          <div className={`rounded-input p-4 space-y-2 border ${!apiKey ? 'border-rule2' : 'border-rule opacity-70'}`}>
            <p className="text-xs font-medium text-ink flex items-center gap-2">
              {apiKey ? <Check size={12} className="text-ok" /> : <AlertTriangle size={12} className="text-warn" />}
              Gemini API Key {apiKey && <span className="text-ok">— set</span>}
            </p>
            {!apiKey && (
              <>
                <ol className="text-xs text-muted space-y-1 list-decimal list-inside">
                  <li>Go to <a href="https://aistudio.google.com/app/apikey" target="_blank" rel="noopener noreferrer" className="text-brass underline">aistudio.google.com/app/apikey</a></li>
                  <li>Sign in with your Google account</li>
                  <li>Click "Create API Key"</li>
                  <li>Copy the key and paste it below</li>
                </ol>
                <input
                  type="text"
                  placeholder="Paste your Gemini API key here..."
                  className="input-field"
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && e.target.value.trim()) {
                      setApiKey(e.target.value.trim());
                    }
                  }}
                />
              </>
            )}
          </div>

          {/* Upload-Post block */}
          <div className={`rounded-input p-4 space-y-2 border ${!uploadPostKey ? 'border-rule2' : 'border-rule opacity-70'}`}>
            <p className="text-xs font-medium text-ink flex items-center gap-2">
              {uploadPostKey ? <Check size={12} className="text-ok" /> : <AlertTriangle size={12} className="text-warn" />}
              Upload-Post API Key {uploadPostKey && <span className="text-ok">— set</span>}
            </p>
            {!uploadPostKey && (
              <>
                <p className="text-xs text-muted">
                  Required to publish your clips to TikTok, Instagram Reels, and YouTube Shorts. Free tier available, no credit card needed.
                </p>
                <ol className="text-xs text-muted space-y-1 list-decimal list-inside">
                  <li>Register at <a href="https://app.upload-post.com/login" target="_blank" rel="noopener noreferrer" className="text-brass underline">app.upload-post.com</a></li>
                  <li>Connect your TikTok, Instagram, or YouTube accounts</li>
                  <li>Go to <a href="https://app.upload-post.com/api-keys" target="_blank" rel="noopener noreferrer" className="text-brass underline">API Keys</a> and generate one</li>
                  <li>Paste it below</li>
                </ol>
                <input
                  type="text"
                  placeholder="Paste your Upload-Post API key here..."
                  className="input-field"
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && e.target.value.trim()) {
                      setUploadPostKey(e.target.value.trim());
                    }
                  }}
                />
              </>
            )}
          </div>
        </div>
      </Modal>

      <ScheduleWeekModal
        isOpen={showScheduleWeek}
        onClose={() => setShowScheduleWeek(false)}
        clips={results?.clips || []}
        jobId={jobId}
        uploadPostKey={uploadPostKey}
        uploadUserId={uploadUserId}
        isManaged={isManaged}
      />

      {/* Pre-flight quality gate */}
      {qualityGate && (
        <Modal isOpen={true} onClose={() => setQualityGate(null)} size="md" eyebrow="HEADS UP" title="low source quality">
          <div className="space-y-4">
            <p className="text-sm text-ink2">
              YouTube only offers <span className="text-brass font-semibold">{qualityGate.info.max_height}p</span> for this video
              (below the {qualityGate.info.min_height}p we recommend). Processing anyway will produce lower-quality clips.
            </p>
            {qualityGate.info.cookies_invalid && (
              <p className="text-xs text-muted">
                Your YouTube cookies look expired — refreshing them (export again from an incognito window) often unlocks HD.
              </p>
            )}
            <div className="flex gap-2 justify-end pt-2">
              <button onClick={() => setQualityGate(null)} className="btn-ghost">cancel</button>
              <button
                onClick={() => { const d = qualityGate.data; setQualityGate(null); handleProcess(d, true); }}
                className="btn-primary"
              >
                process anyway
              </button>
            </div>
          </div>
        </Modal>
      )}


      {editingClip !== null && results?.clips?.[editingClip] && (
        <ClipEditor
          jobId={jobId}
          clipIndex={editingClip}
          clipTitle={results.clips[editingClip].video_title_for_youtube_short || ''}
          onClose={() => setEditingClip(null)}
          onRerendered={handleClipRerendered}
        />
      )}
      {reframingClip !== null && results?.clips?.[reframingClip] && (
        <ReframeEditor
          jobId={jobId}
          clipIndex={reframingClip}
          clipTitle={results.clips[reframingClip].video_title_for_youtube_short || ''}
          onClose={() => setReframingClip(null)}
          onReframed={handleClipRerendered}
        />
      )}
      {showLogin && <LoginModal onClose={() => setShowLogin(false)} />}
      {tutorialPhase && (
        <ClipTutorial
          phase={tutorialPhase}
          jobStatus={status}
          onStart={startTutorial}
          onSkip={skipTutorial}
          onDismissCelebrate={finishTutorial}
        />
      )}
      {showPlanChoice && <PlanChoiceModal onClose={() => setShowPlanChoice(false)} />}
      {showTopUp && (
        <TopUpModal
          onClose={() => setShowTopUp(false)}
          required={topUpInfo.required}
          remaining={topUpInfo.remaining}
          context={topUpInfo.context || 'wall'}
        />
      )}
      {showTrialUpgrade && (
        <TrialUpgradeModal
          plan={plan}
          onActivated={refreshMe}
          onClose={() => setShowTrialUpgrade(false)}
        />
      )}
    </div>
  );
}

export default App;
