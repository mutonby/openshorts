import React, { useState, useEffect } from 'react';
import { Upload, Sparkles, Youtube, Instagram, Share2, ChevronDown, Check, Activity, LayoutDashboard, Settings, Plus, History, X, Terminal, Shield, LayoutGrid, Image, Globe, RotateCcw, Calendar, AlertTriangle, KeyRound, Bot, Users, Smartphone, ExternalLink, Copy, CheckCircle2, Mail, Loader2 } from 'lucide-react';
import KeyInput from './components/KeyInput';
import MediaInput from './components/MediaInput';
import ResultCard from './components/ResultCard';
import ProcessingAnimation from './components/ProcessingAnimation';
// import Gallery from './components/Gallery';
import ThumbnailStudio from './components/ThumbnailStudio';
import SaaShortsTab from './components/SaaShortsTab';
import UGCGallery from './components/UGCGallery';
import ScheduleWeekModal from './components/ScheduleWeekModal';
import UsageMeter from './components/UsageMeter';
import TopUpModal from './components/TopUpModal';
import TrialUpgradeModal from './components/TrialUpgradeModal';
import LoginModal from './components/LoginModal';
import TrialGate from './components/TrialGate';
import AdvancedBanner from './components/AdvancedBanner';
import HistoryTab from './components/HistoryTab';
import ProfileMenu from './components/ProfileMenu';
import Modal from './components/ui/Modal';
import { useAuth } from './contexts/AuthContext';
import { apiFetch, apiJson, QuotaError } from './lib/api';

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

const UserProfileSelector = ({ profiles, selectedUserId, onSelect }) => {
  const [isOpen, setIsOpen] = useState(false);

  if (!profiles || profiles.length === 0) return null;

  const selectedProfile = profiles.find(p => p.username === selectedUserId) || profiles[0];

  return (
    <div className="relative z-50">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center justify-between bg-paper2 border border-rule2 rounded-input px-3 py-2 text-sm text-ink2 hover:bg-paper3 transition-colors min-w-[180px]"
      >
        <span className="flex items-center gap-2">
          <div className="w-5 h-5 rounded-full bg-paper3 border border-rule flex items-center justify-center font-mono text-micro text-brass">
            {selectedProfile?.username?.substring(0, 1).toUpperCase() || "U"}
          </div>
          <span className="font-medium text-ink truncate max-w-[100px]">{selectedProfile?.username || "Select User"}</span>
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
                    {profile.username.substring(0, 2).toUpperCase()}
                  </div>
                  <div className="min-w-0">
                    <div className="text-sm font-medium text-ink2 group-hover:text-ink transition-colors truncate">
                      {profile.username}
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
        </div>
      )}
    </div>
  );
};

const SESSION_KEY = 'openshorts_session';
const SESSION_MAX_AGE = 3600000; // 1 hour (matches server job retention)

// Mock polling function
const pollJob = async (jobId) => {
  const res = await apiFetch(`/api/status/${jobId}`);
  if (!res.ok) throw new Error('Status check failed');
  return res.json();
};

function App() {
  // Cloud auth/billing session (inert when billing is disabled).
  const { billingEnabled, isManaged, isSignedIn, me, plan, refreshMe } = useAuth();
  const [showLogin, setShowLogin] = useState(false);
  const [showTopUp, setShowTopUp] = useState(false);
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
  const [showKeyModal, setShowKeyModal] = useState(false);
  const [jobId, setJobId] = useState(null);
  const [status, setStatus] = useState('idle'); // idle, processing, complete, error
  const [results, setResults] = useState(null);
  const [logs, setLogs] = useState([]);
  const [logsVisible, setLogsVisible] = useState(true);
  const [processingMedia, setProcessingMedia] = useState(null);
  const [activeTab, setActiveTab] = useState('dashboard'); // dashboard, settings

  const [sessionRecovered, setSessionRecovered] = useState(false);
  const [showScheduleWeek, setShowScheduleWeek] = useState(false);

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
        if (session.processingMedia) setProcessingMedia(session.processingMedia);
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
      const sessionData = {
        jobId,
        status,
        results,
        processingMedia: processingMedia?.type === 'url' ? processingMedia : null,
        activeTab,
        timestamp: Date.now()
      };
      localStorage.setItem(SESSION_KEY, JSON.stringify(sessionData));
    } catch (e) {
      // localStorage full or serialization error - ignore
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId, status, results, activeTab]);

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

  // For managed users, fetch the durable R2 URLs of the current job's clips so the
  // preview can fall back to them when the local files have been cleaned up.
  useEffect(() => {
    if (!isManaged || !jobId || !(results?.clips?.length)) { setDurableClips({}); return; }
    let cancelled = false;
    apiJson('/api/history')
      .then((d) => {
        if (cancelled) return;
        const map = {};
        for (const v of (d.videos || [])) {
          if (v.job_id === jobId && v.clip_index != null) map[v.clip_index] = v.view_url;
        }
        setDurableClips(map);
      })
      .catch(() => {});
    return () => { cancelled = true; };
  }, [isManaged, jobId, results]);

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
          } else if (data.status === 'failed') {
            setStatus('error');
            const errorMsg = data.error || (data.logs && data.logs.length > 0 ? data.logs[data.logs.length - 1] : "Process failed");
            setLogs(prev => [...prev, "Error: " + errorMsg]);
            clearInterval(interval);
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
  }, [status, jobId]);


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
  const keysMissing = !billingEnabled && (!apiKey || !uploadPostKey);
  const needsPlan = billingEnabled && !isManaged;   // hosted, signed-out or no active plan/trial
  // Included in the plan (fully managed, no keys): Clip Generator + YouTube Studio.
  // Advanced (bring your own fal.ai + ElevenLabs keys): AI Shorts + AI Agent.
  const INCLUDED_TOOL_TABS = ['dashboard', 'thumbnails'];
  const ADVANCED_TOOL_TABS = ['saasshorts', 'ai-agent'];
  const TOOL_NAMES = { dashboard: 'the Clip Generator', thumbnails: 'the YouTube Studio' };
  const gateThisTab = needsPlan && INCLUDED_TOOL_TABS.includes(activeTab);      // included tool, no plan yet
  const advancedThisTab = billingEnabled && ADVANCED_TOOL_TABS.includes(activeTab); // BYOK-notice tools

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

  const handleProcess = async (data) => {
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
    setProcessingMedia(data);

    try {
      let body;
      // BYOK sends the Gemini header; managed users rely on the bearer token
      // that apiFetch attaches automatically.
      const headers = apiKey ? { 'X-Gemini-Key': apiKey } : {};

      if (data.type === 'url') {
        headers['Content-Type'] = 'application/json';
        body = JSON.stringify({ url: data.payload, acknowledged: !!data.acknowledged });
      } else {
        const formData = new FormData();
        formData.append('file', data.payload);
        formData.append('acknowledged', data.acknowledged ? 'true' : 'false');
        body = formData;
      }

      const res = await apiFetch('/api/process', { method: 'POST', headers, body });

      if (!res.ok) throw new Error(await res.text());
      const resData = await res.json();
      setJobId(resData.job_id);

    } catch (e) {
      if (e instanceof QuotaError) {
        setStatus('idle');
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
    setStatus('idle');
    setJobId(null);
    setResults(null);
    setLogs([]);
    setProcessingMedia(null);
    localStorage.removeItem(SESSION_KEY);
  };

  // --- UI Components ---

  const Sidebar = () => {
    const navItems = [
      { id: 'dashboard', ord: '01', icon: LayoutDashboard, label: 'Clip Generator' },
      { id: 'saasshorts', ord: '02', icon: Sparkles, label: 'AI Shorts', byok: true },
      { id: 'ai-agent', ord: '03', icon: Bot, label: 'AI Agent', byok: true },
      { id: 'ugc-gallery', ord: '04', icon: LayoutGrid, label: 'UGC Gallery' },
      { id: 'thumbnails', ord: '05', icon: Image, label: 'YouTube Studio' },
      ...(billingEnabled && isSignedIn ? [{ id: 'history', ord: '06', icon: History, label: 'History' }] : []),
      { id: 'settings', ord: '07', icon: Settings, label: 'Settings' },
    ];

    return (
      <div className="w-20 lg:w-64 bg-paper2 border-r border-rule flex flex-col h-full shrink-0 transition-all duration-300">
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
                onClick={() => setActiveTab(item.id)}
                className={`relative w-full flex items-center gap-3 px-3 py-2.5 rounded-input transition-colors ${isActive ? 'bg-paper3 text-ink' : 'text-muted hover:text-ink2 hover:bg-paper3/50'}`}
              >
                {isActive && (
                  <span className="absolute left-0 top-1.5 bottom-1.5 w-0.5 bg-brass rounded-full" aria-hidden="true" />
                )}
                <NavIcon size={18} className={`shrink-0 ${isActive ? 'text-brass' : ''}`} />
                <span className="text-sm lowercase hidden lg:block flex-1 text-left truncate">{item.label}</span>
                {item.byok && <span className="readout hidden lg:block">BYOK</span>}
                <span className="readout hidden lg:block">{item.ord}</span>
              </button>
            );
          })}
        </nav>

        <div className="p-4 border-t border-rule space-y-1">
          <a
            href="#landing"
            className="flex items-center gap-2 px-3 py-1.5 text-xs lowercase text-muted hover:text-ink2 transition-colors"
          >
            <Globe size={14} className="shrink-0" />
            <span className="hidden lg:block truncate">landing page</span>
          </a>
          <a
            href="https://github.com/mutonby/openshorts"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 px-3 py-1.5 text-xs lowercase text-muted hover:text-ink2 transition-colors"
          >
            <svg height="14" viewBox="0 0 16 16" version="1.1" width="14" aria-hidden="true" fill="currentColor" className="shrink-0"><path fillRule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path></svg>
            <span className="hidden lg:block truncate">open source</span>
          </a>
          {billingEnabled && (
            <a
              href="#/pricing"
              className="flex items-center gap-2 px-3 py-1.5 text-xs lowercase text-muted hover:text-ink2 transition-colors"
            >
              <Sparkles size={14} className="shrink-0" />
              <span className="hidden lg:block truncate">plans &amp; pricing</span>
            </a>
          )}
          <a
            href="mailto:info@openshorts.app"
            className="flex items-center gap-2 px-3 py-1.5 text-xs lowercase text-muted hover:text-ink2 transition-colors"
          >
            <Mail size={14} className="shrink-0" />
            <span className="hidden lg:block truncate">info@openshorts.app</span>
          </a>
        </div>
      </div>
    );
  };

  return (
    <div className="flex h-screen bg-paper overflow-hidden">
      <Sidebar />

      <main className="flex-1 flex flex-col h-full overflow-hidden relative">
        {/* Top Header */}
        <header className="h-14 border-b border-rule bg-paper flex items-center justify-between px-6 shrink-0 z-10">
          <div className="flex items-center gap-4">
            {status !== 'idle' && (
              <button
                onClick={handleReset}
                className="btn-quiet px-3 py-1.5 text-xs"
              >
                <Plus size={14} />
                <span className="hidden sm:inline">New Project</span>
              </button>
            )}
          </div>

          <div className="flex items-center gap-4">
            {userProfiles.length > 0 && (
              <UserProfileSelector
                profiles={userProfiles}
                selectedUserId={uploadUserId}
                onSelect={setUploadUserId}
              />
            )}

            {/* Cloud: minutes meter + account/sign-in */}
            {billingEnabled && isManaged && (
              <UsageMeter onClick={() => { window.location.hash = '#/account'; }} />
            )}
            {billingEnabled && isSignedIn && !isManaged && (
              <button onClick={() => { window.location.hash = '#/pricing'; }}
                className="btn-primary px-4 py-2 text-xs">
                Start free trial
              </button>
            )}
            {billingEnabled && !isSignedIn && (
              <button onClick={() => setShowLogin(true)}
                className="btn-ghost px-4 py-2 text-xs">
                Sign in
              </button>
            )}
            {billingEnabled && isSignedIn && <ProfileMenu />}

            {keysMissing && (
              <button
                onClick={() => (billingEnabled && !isSignedIn ? setShowLogin(true) : setActiveTab('settings'))}
                className="badge-warn hover:brightness-125 transition-all"
                title="Configure API keys or choose a plan"
              >
                <AlertTriangle size={12} />
                <span className="hidden sm:inline">
                  {!apiKey && !uploadPostKey
                    ? 'Gemini & Upload-Post keys missing'
                    : !apiKey
                      ? 'Gemini API Key Missing'
                      : 'Upload-Post API Key Missing'}
                </span>
                <span className="sm:hidden">keys missing</span>
              </button>
            )}
          </div>
        </header>

        {/* Persistent Missing Keys Banner — visible on every screen */}
        {keysMissing && activeTab !== 'settings' && (
          <div className="mx-4 sm:mx-6 mt-3 px-4 py-3 bg-paper2 border border-rule rounded-card flex flex-wrap items-center justify-between gap-3 sm:gap-4 shrink-0 animate-fade">
            <div className="flex items-center gap-3 text-sm text-ink2">
              <KeyRound size={16} className="shrink-0 text-warn" />
              <div>
                <span className="font-medium text-ink">Required API keys missing.</span>{' '}
                <span className="text-muted">
                  {!apiKey && !uploadPostKey
                    ? 'Set your Gemini and Upload-Post API keys to use OpenShorts.'
                    : !apiKey
                      ? 'Set your Gemini API key to use OpenShorts.'
                      : 'Set your Upload-Post API key to use OpenShorts.'}
                </span>
              </div>
            </div>
            <button
              onClick={() => setActiveTab('settings')}
              className="btn-quiet px-3 py-1.5 text-xs shrink-0"
            >
              Go to Settings
            </button>
          </div>
        )}

        {/* Session Recovery Banner */}
        {sessionRecovered && (
          <div className="mx-6 mt-2 px-4 py-3 bg-paper2 border border-rule rounded-card flex items-center justify-between animate-fade shrink-0">
            <div className="flex items-center gap-2 text-sm text-ink2">
              <RotateCcw size={16} className="text-brass" />
              <span className="font-medium">Session recovered</span>
              <span className="text-muted text-xs">Your previous work has been restored.</span>
            </div>
            <button onClick={() => setSessionRecovered(false)} className="text-muted hover:text-ink transition-colors">
              <X size={14} />
            </button>
          </div>
        )}

        {/* Included tools (Clip Generator, YouTube Studio): non-blocking trial prompt. */}
        {gateThisTab && <TrialGate toolName={TOOL_NAMES[activeTab] || 'this'} />}

        {/* Advanced tools (AI Shorts, AI Agent): BYOK fal.ai + ElevenLabs notice. */}
        {advancedThisTab && <AdvancedBanner needsPlan={needsPlan} onKeys={() => setActiveTab('settings')} />}

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
                      <h2 className="text-base font-medium text-ink lowercase">Start your free trial</h2>
                    </div>
                    <span className="badge-brass">3 days free</span>
                  </div>
                  <p className="text-xs text-muted mb-5 leading-relaxed">
                    Generate shorts with zero setup — no API keys needed. 3 days free, then from $12/mo. Cancel anytime.
                  </p>
                  <button onClick={() => { window.location.hash = '#/pricing'; }} className="btn-primary py-2 px-4 text-sm">
                    <Sparkles size={16} /> See plans & start trial
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
                  <p className="text-xs text-muted leading-relaxed">
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
                  </p>
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
                  <p className="text-xs text-muted leading-relaxed">
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
                  </p>
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
                  <p className="text-xs text-muted leading-relaxed">
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
                  </p>
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
              <div className="max-w-6xl mx-auto p-6 md:p-8">
                <UGCGallery />
              </div>
            </div>
          )}

          {/* View: History */}
          {activeTab === 'history' && (
            <div className="h-full overflow-y-auto custom-scrollbar animate-fade">
              <div className="max-w-6xl mx-auto p-6 md:p-8">
                <HistoryTab />
              </div>
            </div>
          )}

          {activeTab === 'thumbnails' && (
            <ThumbnailStudio geminiApiKey={apiKey} uploadPostKey={uploadPostKey} uploadUserId={uploadUserId} managed={isManaged} />
          )}

          {/* View: Gallery */}
          {/* {activeTab === 'gallery' && (
            <Gallery />
          )} */}

          {/* View: Dashboard (Idle) */}
          {activeTab === 'dashboard' && status === 'idle' && (
            <div className="h-full overflow-y-auto custom-scrollbar animate-fade">
              <div className="min-h-full flex flex-col items-center justify-center px-4 py-6 sm:p-6">
              <div className="max-w-xl w-full text-center space-y-8">
                <div className="space-y-4">
                  <p className="eyebrow">01 · CLIP GENERATOR</p>
                  <h1 className="font-display lowercase text-4xl md:text-5xl text-ink">
                    Create Viral Shorts
                  </h1>
                  <p className="text-muted text-lg">
                    Drop your long-form video below to instantly generate viral clips with AI.
                  </p>
                </div>

                <MediaInput onProcess={handleProcess} isProcessing={status === 'processing'} />

                <div className="flex flex-wrap items-center justify-center gap-4 sm:gap-8 text-muted text-sm">
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
            <div className="h-full flex flex-col md:flex-row gap-4 p-4 overflow-y-auto md:overflow-y-hidden custom-scrollbar animate-fade">

              {/* Left Panel: Preview & Status */}
              <div className={`${status === 'complete' ? 'w-full md:w-[30%] lg:w-[25%]' : 'w-full md:w-[55%] lg:w-[60%]'} md:h-full flex flex-col shrink-0 md:shrink card p-4 sm:p-6 overflow-y-auto custom-scrollbar transition-all duration-700 ease-in-out`}>
                <div className="mb-6 flex items-center justify-between">
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

                {/* Logs Terminal */}
                <div className={`bg-paper rounded-card border border-rule overflow-hidden flex flex-col transition-all duration-500 ${status === 'complete' ? 'h-32 min-h-0 opacity-50 hover:opacity-100' : 'flex-1 min-h-[200px]'}`}>
                  <div className="px-4 py-2 border-b border-rule flex items-center justify-between bg-paper2 shrink-0">
                    <span className="readout flex items-center gap-2">
                      <Terminal size={12} /> System Logs
                    </span>
                    <button onClick={() => setLogsVisible(!logsVisible)} className="text-muted hover:text-ink transition-colors">
                      {logsVisible ? <ChevronDown size={14} /> : <ChevronDown size={14} className="rotate-180" />}
                    </button>
                  </div>
                  {logsVisible && (
                    <div className="flex-1 p-4 overflow-y-auto font-mono text-xs space-y-1.5 custom-scrollbar text-muted">
                      {logs.map((log, i) => (
                        <div key={i} className={`flex gap-2 ${log.toLowerCase().includes('error') ? 'text-danger' : 'text-muted'}`}>
                          <span className="text-muted opacity-50 shrink-0">{new Date().toLocaleTimeString()}</span>
                          <span>{log}</span>
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
              <div className={`${status === 'complete' ? 'w-full md:w-[70%] lg:w-[75%]' : 'w-full md:w-[45%] lg:w-[40%]'} md:h-full flex flex-col shrink-0 md:shrink card p-4 sm:p-6 transition-all duration-700 ease-in-out`}>
                <h2 className="font-display lowercase text-xl text-ink mb-6 flex flex-wrap items-center gap-2 shrink-0">
                  Generated Shorts
                  {results?.clips?.length > 0 && (
                    <span className="readout bg-paper3 px-2.5 py-1 rounded-full ml-auto">
                      {results.clips.length} Clips
                    </span>
                  )}
                  {results?.cost_analysis && !isManaged && (
                    <span className="readout bg-paper3 px-2.5 py-1 rounded-full ml-2" title={`Input: ${results.cost_analysis.input_tokens} | Output: ${results.cost_analysis.output_tokens}`}>
                      GEMINI · ${results.cost_analysis.total_cost.toFixed(5)}
                    </span>
                  )}
                  {results?.clips?.length > 1 && status === 'complete' && (
                    <button
                      onClick={() => setShowScheduleWeek(true)}
                      className="btn-primary px-4 py-2 text-xs ml-auto"
                    >
                      <Calendar size={14} />
                      schedule week
                    </button>
                  )}
                </h2>

                <div className="flex-1 overflow-y-auto custom-scrollbar p-1">
                  {results && results.clips && results.clips.length > 0 ? (
                    <div className={`grid gap-4 pb-10 ${status === 'complete' ? 'grid-cols-1 xl:grid-cols-2' : 'grid-cols-1'}`}>
                      {results.clips.map((clip, i) => (
                        <ResultCard
                          key={i}
                          clip={clip}
                          index={i}
                          jobId={jobId}
                          durableUrl={durableClips[i]}
                          uploadPostKey={uploadPostKey}
                          uploadUserId={uploadUserId}
                          geminiApiKey={apiKey}
                          elevenLabsKey={elevenLabsKey}
                          isManaged={isManaged}
                          onPlay={(time) => handleClipPlay(time)}
                          onPause={handleClipPause}
                        />
                      ))}
                    </div>
                  ) : (
                    status === 'processing' ? (
                      <div className="h-full flex flex-col items-center justify-center text-muted space-y-4">
                        <Loader2 size={32} className="animate-spin text-brass" />
                        <p className="text-sm lowercase">Waiting for clips...</p>
                      </div>
                    ) : status === 'error' ? (
                      <div className="h-full flex flex-col items-center justify-center text-danger space-y-2">
                        <p>Generation failed.</p>
                      </div>
                    ) : null
                  )}
                </div>
              </div>

            </div>
          )}

        </div>

      </main>

      {/* Missing API Key Modal */}
      <Modal
        isOpen={showKeyModal}
        onClose={() => setShowKeyModal(false)}
        eyebrow="SETUP"
        title={!apiKey && !uploadPostKey
          ? 'Required API Keys Missing'
          : !apiKey
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
              onClick={() => { setShowKeyModal(false); setActiveTab('settings'); }}
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

      {showLogin && <LoginModal onClose={() => setShowLogin(false)} />}
      {showTopUp && (
        <TopUpModal
          onClose={() => setShowTopUp(false)}
          required={topUpInfo.required}
          remaining={topUpInfo.remaining}
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
