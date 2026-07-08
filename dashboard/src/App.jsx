import React, { useState, useEffect, lazy, Suspense } from 'react';
import { Upload, FileVideo, Sparkles, Youtube, Instagram, Share2, LogOut, ChevronDown, Check, Activity, LayoutDashboard, Settings, PlusCircle, History, Menu, X, Terminal, Shield, LayoutGrid, Image, Globe, RotateCcw, Copy, AlertTriangle, Type, Download, Loader2 } from 'lucide-react';
import KeyInput from './components/KeyInput';
import MediaInput from './components/MediaInput';
import ResultCard from './components/ResultCard';
import SubtitleModal from './components/SubtitleModal';
import ProcessingAnimation from './components/ProcessingAnimation';
// import Gallery from './components/Gallery';
import { getApiUrl } from './config';

// Heavy tab components load on first open instead of bloating the main bundle.
const ThumbnailStudio = lazy(() => import('./components/ThumbnailStudio'));
const SaaShortsTab = lazy(() => import('./components/SaaShortsTab'));
const UGCGallery = lazy(() => import('./components/UGCGallery'));

const TabFallback = () => (
  <div className="h-full flex items-center justify-center">
    <div className="w-8 h-8 rounded-full border-2 border-zinc-800 border-t-primary animate-spin" />
  </div>
);

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
    } catch {
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
        className="flex items-center justify-between bg-surface border border-white/10 rounded-lg px-3 py-2 text-sm text-zinc-300 hover:bg-white/5 transition-colors min-w-[180px]"
      >
        <span className="flex items-center gap-2">
          <div className="w-5 h-5 rounded-full bg-gradient-to-br from-primary to-purple-600 flex items-center justify-center text-[10px] font-bold text-white">
            {selectedProfile?.username?.substring(0, 1).toUpperCase() || "U"}
          </div>
          <span className="font-medium text-white truncate max-w-[100px]">{selectedProfile?.username || "Select User"}</span>
        </span>
        <ChevronDown size={14} className={`text-zinc-500 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
      </button>

      {isOpen && (
        <div className="absolute top-full mt-2 right-0 w-64 bg-[#1a1a1a] border border-white/10 rounded-xl shadow-2xl overflow-hidden">
          <div className="max-h-60 overflow-y-auto custom-scrollbar">
            {profiles.map((profile) => (
              <button
                key={profile.username}
                onClick={() => {
                  onSelect(profile.username);
                  setIsOpen(false);
                }}
                className="w-full flex items-center justify-between px-4 py-3 hover:bg-white/5 transition-colors text-left group border-b border-white/5 last:border-0"
              >
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 rounded-full bg-gradient-to-br from-primary/20 to-purple-500/20 flex items-center justify-center text-xs font-bold text-white border border-white/10 shrink-0">
                    {profile.username.substring(0, 2).toUpperCase()}
                  </div>
                  <div className="min-w-0">
                    <div className="text-sm font-medium text-zinc-200 group-hover:text-white transition-colors truncate">
                      {profile.username}
                    </div>
                    <div className="flex gap-2 mt-0.5">
                      {/* Status indicators */}
                      <div className={`flex items-center gap-1 text-[10px] ${profile.connected.includes('tiktok') ? 'text-zinc-300' : 'text-zinc-600'}`}>
                        <TikTokIcon size={10} />
                      </div>
                      <div className={`flex items-center gap-1 text-[10px] ${profile.connected.includes('instagram') ? 'text-pink-400' : 'text-zinc-600'}`}>
                        <Instagram size={10} />
                      </div>
                      <div className={`flex items-center gap-1 text-[10px] ${profile.connected.includes('youtube') ? 'text-red-400' : 'text-zinc-600'}`}>
                        <Youtube size={10} />
                      </div>
                    </div>
                  </div>
                </div>
                {selectedUserId === profile.username && <Check size={14} className="text-primary shrink-0" />}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

const SESSION_KEY = 'openshorts_session';
const SESSION_MAX_AGE = 24 * 3600000; // 24 hours

// Poll job status; signal aborts the request when the effect is cleaned up
const pollJob = async (jobId, signal) => {
  const res = await fetch(getApiUrl(`/api/status/${jobId}`), { signal });
  const data = await res.json().catch(() => ({}));
  if (res.status === 410) return { ...data, archived: true, httpStatus: 410 };
  if (!res.ok) throw new Error(data?.detail || 'Status check failed');
  return { ...data, httpStatus: res.status };
};

const formatDuration = (seconds) => {
  if (seconds == null || Number.isNaN(Number(seconds))) return 'n/a';
  const total = Math.max(0, Math.round(Number(seconds)));
  const hrs = Math.floor(total / 3600);
  const mins = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  if (hrs > 0) return `${hrs}h ${mins}m`;
  if (mins > 0) return `${mins}m ${secs}s`;
  return `${secs}s`;
};

const formatLastSeen = (seconds) => {
  if (seconds == null) return 'n/a';
  if (seconds < 5) return 'gerade eben';
  return `vor ${formatDuration(seconds)}`;
};

const statusTone = (status) => {
  if (status === 'complete') return 'bg-green-500/10 border-green-500/20 text-green-400';
  if (status === 'error' || status === 'archived') return 'bg-red-500/10 border-red-500/20 text-red-400';
  if (status === 'stalled') return 'bg-amber-500/10 border-amber-500/20 text-amber-300';
  if (status === 'queued') return 'bg-white/10 border-white/10 text-zinc-300';
  return 'bg-primary/10 border-primary/20 text-primary';
};

const Sidebar = ({ activeTab, setActiveTab }) => (
  <div className="w-20 lg:w-64 bg-surface border-r border-white/5 flex flex-col h-full shrink-0 transition-all duration-300">
    <div className="p-6 flex items-center gap-3">
      <div className="w-8 h-8 bg-white/5 rounded-lg flex items-center justify-center shrink-0 overflow-hidden border border-white/5">
        <img src="/logo-openshorts.png" alt="Logo" className="w-full h-full object-cover" />
      </div>
      <span className="font-bold text-lg text-white hidden lg:block tracking-tight">OpenShorts</span>
    </div>

    <nav className="flex-1 px-4 py-4 space-y-2">
      <button
        onClick={() => setActiveTab('dashboard')}
        className={`w-full flex items-center gap-3 px-3 py-3 rounded-xl transition-colors ${activeTab === 'dashboard' ? 'bg-primary/10 text-primary' : 'text-zinc-400 hover:text-white hover:bg-white/5'}`}
      >
        <LayoutDashboard size={20} />
        <span className="font-medium hidden lg:block">Clip Generator</span>
      </button>

      <button
        onClick={() => setActiveTab('saasshorts')}
        className={`w-full flex items-center gap-3 px-3 py-3 rounded-xl transition-colors ${activeTab === 'saasshorts' ? 'bg-violet-500/10 text-violet-400' : 'text-zinc-400 hover:text-white hover:bg-white/5'}`}
      >
        <Sparkles size={20} />
        <span className="font-medium hidden lg:block">AI Shorts</span>
      </button>

      <button
        onClick={() => setActiveTab('ugc-gallery')}
        className={`w-full flex items-center gap-3 px-3 py-3 rounded-xl transition-colors ${activeTab === 'ugc-gallery' ? 'bg-violet-500/10 text-violet-400' : 'text-zinc-400 hover:text-white hover:bg-white/5'}`}
      >
        <LayoutGrid size={20} />
        <span className="font-medium hidden lg:block">UGC Gallery</span>
      </button>

      <button
        onClick={() => setActiveTab('thumbnails')}
        className={`w-full flex items-center gap-3 px-3 py-3 rounded-xl transition-colors ${activeTab === 'thumbnails' ? 'bg-primary/10 text-primary' : 'text-zinc-400 hover:text-white hover:bg-white/5'}`}
      >
        <Image size={20} />
        <span className="font-medium hidden lg:block">YouTube Studio</span>
      </button>

      {/* <button
        onClick={() => setActiveTab('gallery')}
        className={`w-full flex items-center gap-3 px-3 py-3 rounded-xl transition-colors ${activeTab === 'gallery' ? 'bg-primary/10 text-primary' : 'text-zinc-400 hover:text-white hover:bg-white/5'}`}
      >
        <LayoutGrid size={20} />
        <span className="font-medium hidden lg:block">Gallery</span>
      </button> */}

      <button
        onClick={() => setActiveTab('settings')}
        className={`w-full flex items-center gap-3 px-3 py-3 rounded-xl transition-colors ${activeTab === 'settings' ? 'bg-primary/10 text-primary' : 'text-zinc-400 hover:text-white hover:bg-white/5'}`}
      >
        <Settings size={20} />
        <span className="font-medium hidden lg:block">Settings</span>
      </button>
    </nav>

    <div className="p-4 border-t border-white/5 space-y-2">
      <a
        href="#"
        onClick={(e) => { e.preventDefault(); localStorage.removeItem('openshorts_skip_landing'); window.location.hash = ''; window.location.reload(); }}
        className="flex items-center gap-2 p-3 bg-white/5 hover:bg-white/10 rounded-xl transition-colors group"
      >
        <div className="w-8 h-8 rounded-full bg-primary/20 text-primary flex items-center justify-center shrink-0">
          <Globe size={16} />
        </div>
        <div className="hidden lg:block overflow-hidden">
          <p className="text-sm font-bold text-white leading-none mb-0.5">Landing Page</p>
          <p className="text-[10px] text-zinc-400 group-hover:text-zinc-300 transition-colors truncate">View website</p>
        </div>
      </a>
      <a
        href="https://github.com/mutonby/openshorts"
        target="_blank"
        rel="noopener noreferrer"
        className="flex items-center gap-2 p-3 bg-white/5 hover:bg-white/10 rounded-xl transition-colors group"
      >
        <div className="w-8 h-8 rounded-full bg-white text-black flex items-center justify-center shrink-0">
          <svg height="20" viewBox="0 0 16 16" version="1.1" width="20" aria-hidden="true"><path fillRule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path></svg>
        </div>
        <div className="hidden lg:block overflow-hidden">
          <p className="text-sm font-bold text-white leading-none mb-0.5">Open Source</p>
          <p className="text-[10px] text-zinc-400 group-hover:text-zinc-300 transition-colors truncate">Free & Community Driven</p>
        </div>
      </a>
    </div>
  </div>
);

function App() {
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
  const [status, setStatus] = useState('idle'); // idle, queued, processing, stalled, complete, error, archived
  const [results, setResults] = useState(null);
  const [logs, setLogs] = useState([]);
  const [logsVisible, setLogsVisible] = useState(true);
  const [logMode, setLogMode] = useState('important');
  const [jobMeta, setJobMeta] = useState(null);
  const [supportCopied, setSupportCopied] = useState(false);
  const [processingMedia, setProcessingMedia] = useState(null);
  const [activeTab, setActiveTab] = useState('dashboard'); // dashboard, settings

  const [sessionRecovered, setSessionRecovered] = useState(false);

  // Bulk subtitles: apply one style to every clip of the job
  const [showBulkSubtitles, setShowBulkSubtitles] = useState(false);
  const [bulkSubProgress, setBulkSubProgress] = useState({ running: false, current: 0, total: 0, errors: 0 });

  // Quality gate: server refused to start because YouTube only offers low-res
  const [lowQualityPrompt, setLowQualityPrompt] = useState(null); // {data, maxHeight, cookiesInvalid}

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
        setJobMeta(session.jobMeta || null);
        if (session.processingMedia) setProcessingMedia(session.processingMedia);
        if (session.activeTab) setActiveTab(session.activeTab);
        setStatus(session.status);
        setSessionRecovered(true);
        // Active jobs keep the banner open so the user can choose
        // continue vs. cancel; finished sessions auto-hide it.
        if (!['queued', 'processing', 'stalled'].includes(session.status)) {
          setTimeout(() => setSessionRecovered(false), 5000);
        }
      }
    } catch {
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
        jobMeta,
        processingMedia: processingMedia?.type === 'url' ? processingMedia : null,
        activeTab,
        timestamp: Date.now()
      };
      localStorage.setItem(SESSION_KEY, JSON.stringify(sessionData));
    } catch {
      // localStorage full or serialization error - ignore
    }
  }, [jobId, status, results, jobMeta, activeTab, processingMedia]);

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
    if (uploadPostKey && userProfiles.length === 0) {
      fetchUserProfiles({ silent: true });
    }
    // Intentionally only re-fetch when the key changes; adding
    // fetchUserProfiles/userProfiles.length would re-trigger every render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [uploadPostKey]);

  useEffect(() => {
    // Note: no polling in 'complete' — a finished job's 249 KB status payload
    // was previously re-fetched every 2s forever. Bulk actions refresh once
    // explicitly instead (see handleBulkSubtitles).
    if (!(['queued', 'processing', 'stalled'].includes(status) && jobId)) {
      return undefined;
    }
    // Abort in-flight requests and ignore late responses once this effect is
    // cleaned up (job switch, reset, unmount) so stale data can't clobber state.
    const controller = new AbortController();
    let cancelled = false;
    let consecutiveErrors = 0;

    const interval = setInterval(async () => {
      try {
        const data = await pollJob(jobId, controller.signal);
        if (cancelled) return;
        consecutiveErrors = 0;
        console.log("Job status:", data);
        setJobMeta(data);

        // Update results if available (real-time)
        if (data.result) {
          setResults(data.result);
        }

        if (data.archived || data.status === 'archived') {
          setStatus('archived');
          clearInterval(interval);
        } else if (data.status === 'completed') {
          setStatus('complete');
          if (data.logs) setLogs(data.logs);
          clearInterval(interval);
        } else if (data.status === 'failed') {
          setStatus('error');
          const errorMsg = data.error || (data.logs && data.logs.length > 0 ? data.logs[data.logs.length - 1] : "Process failed");
          setLogs(prev => [...prev, "Error: " + errorMsg]);
          clearInterval(interval);
        } else if (data.status === 'stalled') {
          setStatus('stalled');
          if (data.logs) setLogs(data.logs);
        } else {
          setStatus(data.status === 'queued' ? 'queued' : 'processing');
          // Update logs if available
          if (data.logs) setLogs(data.logs);
        }
      } catch (e) {
        if (cancelled || e.name === 'AbortError') return;
        console.error("Polling error", e);
        // A single transient failure shouldn't kill the poll; repeated
        // failures surface as an error instead of an endless spinner.
        consecutiveErrors += 1;
        if (consecutiveErrors >= 5) {
          clearInterval(interval);
          setStatus('error');
          setLogs(prev => [...prev, `Error: Lost connection to server (${consecutiveErrors} failed status checks): ${e.message}`]);
        }
      }
    }, 2000);

    return () => {
      cancelled = true;
      controller.abort();
      clearInterval(interval);
    };
  }, [status, jobId]);


  const fetchUserProfiles = async ({ silent = false } = {}) => {
    if (!uploadPostKey) return;
    try {
      const res = await fetch(getApiUrl('/api/social/user'), {
        headers: { 'X-Upload-Post-Key': uploadPostKey }
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
      } else if (!localStorage.getItem('uploadpost_no_profiles_notice')) {
        // Startup check: tell the user exactly once, then stay quiet forever.
        localStorage.setItem('uploadpost_no_profiles_notice', '1');
        alert(
          "Upload-Post: your API key has no social profiles yet, so direct " +
          "posting stays disabled. Create a profile at upload-post.com to use it. " +
          "(This message won't appear again.)"
        );
      }
    } catch (e) {
      if (!silent) alert("Error fetching User Profiles. Please check key.");
      console.error(e);
    }
  };

  const handleProcess = async (data, forceLowQuality = false) => {
    if (!apiKey) {
      setShowKeyModal(true);
      return;
    }
    setStatus('processing');
    setLogs(["Starting process..."]);
    setResults(null);
    setJobMeta(null);
    setProcessingMedia(data);

    try {
      let body;
      const headers = { 'X-Gemini-Key': apiKey };

      if (data.type === 'url') {
        headers['Content-Type'] = 'application/json';
        body = JSON.stringify({
          url: data.payload,
          force_low_quality: forceLowQuality,
          output_format: data.outputFormat || 'auto',
        });
      } else {
        const formData = new FormData();
        formData.append('file', data.payload);
        formData.append('output_format', data.outputFormat || 'auto');
        body = formData;
      }

      const res = await fetch(getApiUrl('/api/process'), {
        method: 'POST',
        headers: data.type === 'url' ? headers : { 'X-Gemini-Key': apiKey },
        body
      });

      if (!res.ok) throw new Error(await res.text());
      const resData = await res.json();

      // Quality gate: server did NOT start the job — ask the user first.
      if (resData.needs_confirmation) {
        setStatus('idle');
        setLogs([]);
        setProcessingMedia(null);
        setLowQualityPrompt({
          data,
          maxHeight: resData.quality_check?.max_height,
          cookiesInvalid: resData.quality_check?.cookies_invalid,
        });
        return;
      }

      setJobId(resData.job_id);

    } catch (e) {
      setStatus('error');
      setLogs(l => [...l, `Error starting job: ${e.message}`]);
    }
  };

  const cancelJobOnServer = (id) => {
    if (!id) return;
    // Fire-and-forget: frees the server slot and kills running FFmpeg processes.
    fetch(getApiUrl(`/api/jobs/${id}/cancel`), { method: 'POST' }).catch(() => {});
  };

  const handleReset = () => {
    // Don't leave an orphaned job burning CPU on the server.
    if (jobId && ['queued', 'processing', 'stalled'].includes(status)) {
      cancelJobOnServer(jobId);
    }
    setStatus('idle');
    setJobId(null);
    setResults(null);
    setLogs([]);
    setJobMeta(null);
    setProcessingMedia(null);
    localStorage.removeItem(SESSION_KEY);
  };

  const handleBulkSubtitles = async (options) => {
    const clips = results?.clips || [];
    if (!jobId || clips.length === 0) return;
    const total = clips.length;
    let errors = 0;
    setBulkSubProgress({ running: true, current: 0, total, errors: 0 });

    // Sequential on purpose: each burn is an FFmpeg run; parallel requests
    // would hammer the box without finishing sooner.
    for (let i = 0; i < total; i++) {
      setBulkSubProgress({ running: true, current: i + 1, total, errors });
      try {
        const res = await fetch(getApiUrl('/api/subtitle'), {
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
            style: options.style,
            highlight_color: options.highlightColor,
            effect: options.effect,
            base_opacity: options.baseOpacity,
            uppercase: options.uppercase,
            // no input_filename: the server picks the clip's current video and
            // replaces existing subtitles instead of stacking them
          }),
        });
        if (!res.ok) errors += 1;
      } catch {
        errors += 1;
      }
    }

    setBulkSubProgress({ running: false, current: total, total, errors });
    setShowBulkSubtitles(false);
    // One targeted refresh so the cards pick up the new video_urls
    // (replaces the old permanent 2s poll loop in 'complete').
    try {
      const data = await pollJob(jobId);
      setJobMeta(data);
      if (data.result) setResults(data.result);
    } catch {
      // Cards keep their previous URLs; user can refresh manually.
    }
  };

  const handleCopySupportLog = async () => {
    if (!jobId) return;
    try {
      const res = await fetch(getApiUrl(`/api/jobs/${jobId}/support-log`));
      const data = await res.json();
      await navigator.clipboard.writeText(data.support_log || 'No support log available.');
      setSupportCopied(true);
      setTimeout(() => setSupportCopied(false), 2000);
    } catch (e) {
      console.error('Copy support log failed', e);
    }
  };

  const handleResumeJob = async () => {
    if (!jobId || !apiKey) return;
    try {
      const res = await fetch(getApiUrl(`/api/jobs/${jobId}/resume`), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Gemini-Key': apiKey,
        },
        body: JSON.stringify({ phase: 'analyze' }),
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || 'Resume failed');
      }
      setStatus('queued');
      setLogs((prev) => [...prev, 'Job resumed.']);
    } catch (e) {
      console.error('Resume failed', e);
      setLogs((prev) => [...prev, `Resume error: ${e.message}`]);
    }
  };

  // --- UI Components ---

  const displayLogs = logMode === 'important'
    ? (jobMeta?.important_logs || []).map((entry) => ({ ...entry, message: entry.message || '' }))
    : (jobMeta?.raw_logs || []).map((entry) => ({ ...entry, message: entry.message || '' }));

  const fallbackLogs = logs.map((message) => ({
    iso_timestamp: null,
    level: message.toLowerCase().includes('error') ? 'error' : 'info',
    message,
  }));

  const renderedLogs = displayLogs.length > 0 ? displayLogs : fallbackLogs;
  const progressPercent = Number(jobMeta?.progress_percent || 0);
  const phaseLabel = jobMeta?.phase_label || (status === 'queued' ? 'Queued' : status === 'processing' ? 'Processing' : 'Idle');

  return (
    <div className="flex h-screen bg-background overflow-hidden selection:bg-primary/30">
      <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} />

      <main className="flex-1 flex flex-col h-full overflow-hidden relative">
        {/* Background Gradients */}
        <div className="absolute inset-0 overflow-hidden -z-10 pointer-events-none">
          <div className="absolute -top-[10%] -right-[10%] w-[50%] h-[50%] bg-primary/5 rounded-full blur-[120px]" />
        </div>

        {/* Top Header */}
        <header className="h-16 border-b border-white/5 bg-background/50 backdrop-blur-md flex items-center justify-between px-6 shrink-0 z-10">
          <div className="flex items-center gap-4">
            {status !== 'idle' && (
              <button
                onClick={handleReset}
                className="flex items-center gap-2 text-sm text-zinc-400 hover:text-white transition-colors"
              >
                {['queued', 'processing', 'stalled'].includes(status) ? (
                  <>
                    <X size={16} />
                    <span className="hidden sm:inline">Stop & New</span>
                  </>
                ) : (
                  <>
                    <PlusCircle size={16} />
                    <span className="hidden sm:inline">New Project</span>
                  </>
                )}
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

            {!apiKey && (
              <span className="text-xs text-amber-500 bg-amber-500/10 px-3 py-1 rounded-full border border-amber-500/20">
                API Key Missing
              </span>
            )}
          </div>
        </header>

        {/* Session Recovery Banner */}
        {sessionRecovered && (
          <div className="mx-6 mt-2 p-3 bg-primary/10 border border-primary/20 rounded-xl flex items-center justify-between animate-[fadeIn_0.3s_ease-out] shrink-0">
            <div className="flex items-center gap-2 text-sm text-primary">
              <RotateCcw size={16} />
              <span className="font-medium">Session recovered</span>
              <span className="text-zinc-400 text-xs">
                {['queued', 'processing', 'stalled'].includes(status)
                  ? 'Your job is still running on the server.'
                  : 'Your previous work has been restored.'}
              </span>
            </div>
            <div className="flex items-center gap-2">
              {['queued', 'processing', 'stalled'].includes(status) && (
                <>
                  <button
                    onClick={() => setSessionRecovered(false)}
                    className="text-xs bg-primary/20 border border-primary/30 text-primary hover:bg-primary/30 px-3 py-1 rounded-full transition-colors font-medium"
                  >
                    Continue
                  </button>
                  <button
                    onClick={() => { setSessionRecovered(false); handleReset(); }}
                    className="text-xs bg-white/5 border border-white/10 text-zinc-300 hover:bg-red-500/10 hover:text-red-400 hover:border-red-500/20 px-3 py-1 rounded-full transition-colors"
                  >
                    Cancel & start new
                  </button>
                </>
              )}
              <button onClick={() => setSessionRecovered(false)} className="text-zinc-500 hover:text-white transition-colors">
                <X size={14} />
              </button>
            </div>
          </div>
        )}

        {/* Main Workspace */}
        <div className="flex-1 overflow-hidden relative">

          {/* View: Settings */}
          {activeTab === 'settings' && (
            <div className="h-full overflow-y-auto p-8 max-w-2xl mx-auto animate-[fadeIn_0.3s_ease-out]">
              <div className="flex items-center justify-between mb-8">
                <h1 className="text-2xl font-bold">Settings</h1>
                <div className="px-3 py-1 bg-green-500/10 border border-green-500/20 rounded-full text-[10px] text-green-400 font-medium flex items-center gap-2">
                  <Shield size={12} /> Privacy: keys only live in your browser (sent to backend just to process)
                </div>
              </div>
              <KeyInput onKeySet={setApiKey} savedKey={apiKey} />

              <div className="glass-panel p-6 mt-8">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-lg font-semibold">Social Integration</h2>
                  <span className="text-[10px] bg-white/5 border border-white/5 px-2 py-0.5 rounded text-zinc-500 uppercase tracking-wider">Optional</span>
                </div>
                <p className="text-xs text-zinc-500 mb-6 leading-relaxed">
                  Automatically publish your clips to TikTok, Instagram Reels, and YouTube Shorts via <strong>Upload-Post</strong>.
                  Includes a <strong>free tier</strong> (no credit card required).
                  If you prefer, you can skip this and manually download/upload your videos.
                </p>
                <div className="space-y-4">
                  <label className="block text-sm text-zinc-400">Upload-Post API Key</label>
                  <div className="flex gap-2">
                    <input
                      type="password"
                      value={uploadPostKey}
                      onChange={(e) => setUploadPostKey(e.target.value)}
                      className="input-field"
                      placeholder="ey..."
                    />
                    <button onClick={() => fetchUserProfiles()} className="btn-primary py-2 px-4 text-sm">
                      Connect
                    </button>
                  </div>
                  <p className="text-xs text-zinc-500 leading-relaxed">
                    Connect your Upload-Post account to enable one-click publishing.
                    <div className="mt-3 grid grid-cols-1 sm:grid-cols-3 gap-2">
                      <a href="https://app.upload-post.com/login" target="_blank" rel="noopener noreferrer" className="p-2 border border-white/5 rounded-lg hover:bg-white/5 transition-colors flex flex-col gap-1">
                        <span className="text-zinc-400 font-medium">1. Login</span>
                        <span className="text-[10px] text-zinc-600">Register account</span>
                      </a>
                      <a href="https://app.upload-post.com/manage-users" target="_blank" rel="noopener noreferrer" className="p-2 border border-white/5 rounded-lg hover:bg-white/5 transition-colors flex flex-col gap-1">
                        <span className="text-zinc-400 font-medium">2. Profiles</span>
                        <span className="text-[10px] text-zinc-600">Create & Connect</span>
                      </a>
                      <a href="https://app.upload-post.com/api-keys" target="_blank" rel="noopener noreferrer" className="p-2 border border-white/5 rounded-lg hover:bg-white/5 transition-colors flex flex-col gap-1">
                        <span className="text-zinc-400 font-medium">3. API Key</span>
                        <span className="text-[10px] text-zinc-600">Generate key</span>
                      </a>
                    </div>
                    <br />
                    <span className="text-zinc-600 italic">
                      Keys are only stored in your browser. They are sent to the backend only to process your request, never stored server-side.
                    </span>
                  </p>
                </div>
              </div>

              <div className="glass-panel p-6 mt-8">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-lg font-semibold">Video Translation</h2>
                  <span className="text-[10px] bg-white/5 border border-white/5 px-2 py-0.5 rounded text-zinc-500 uppercase tracking-wider">Optional</span>
                </div>
                <p className="text-xs text-zinc-500 mb-6 leading-relaxed">
                  Translate your clips to different languages using <strong>ElevenLabs</strong> AI dubbing.
                  Automatically translates speech while preserving the original voice characteristics.
                </p>
                <div className="space-y-4">
                  <label className="block text-sm text-zinc-400">ElevenLabs API Key</label>
                  <div className="flex gap-2">
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
                          alert('ElevenLabs API Key saved!');
                        }
                      }}
                      className="btn-primary py-2 px-4 text-sm"
                    >
                      Save
                    </button>
                  </div>
                  <p className="text-xs text-zinc-500 leading-relaxed">
                    Get your API key from ElevenLabs to enable video translation.
                    <div className="mt-3 grid grid-cols-1 sm:grid-cols-2 gap-2">
                      <a href="https://elevenlabs.io/sign-up" target="_blank" rel="noopener noreferrer" className="p-2 border border-white/5 rounded-lg hover:bg-white/5 transition-colors flex flex-col gap-1">
                        <span className="text-zinc-400 font-medium">1. Sign Up</span>
                        <span className="text-[10px] text-zinc-600">Create account</span>
                      </a>
                      <a href="https://elevenlabs.io/app/settings/api-keys" target="_blank" rel="noopener noreferrer" className="p-2 border border-white/5 rounded-lg hover:bg-white/5 transition-colors flex flex-col gap-1">
                        <span className="text-zinc-400 font-medium">2. API Key</span>
                        <span className="text-[10px] text-zinc-600">Generate key</span>
                      </a>
                    </div>
                    <br />
                    <span className="text-zinc-600 italic">
                      Keys are only stored in your browser. They are sent to the backend only to process your request, never stored server-side.
                    </span>
                  </p>
                </div>
              </div>

              <div className="glass-panel p-6 mt-8">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-lg font-semibold">AI Shorts (UGC Videos)</h2>
                  <span className="text-[10px] bg-violet-500/10 border border-violet-500/20 px-2 py-0.5 rounded text-violet-400 uppercase tracking-wider">New</span>
                </div>
                <p className="text-xs text-zinc-500 mb-6 leading-relaxed">
                  Generate UGC-style videos with AI actors for any product or business using <strong>fal.ai</strong>.
                  Just describe your product or paste a URL. Requires fal.ai + ElevenLabs API keys.
                </p>
                <div className="space-y-4">
                  <label className="block text-sm text-zinc-400">fal.ai API Key</label>
                  <div className="flex gap-2">
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
                          alert('fal.ai API Key saved!');
                        }
                      }}
                      className="btn-primary py-2 px-4 text-sm"
                    >
                      Save
                    </button>
                  </div>
                  <p className="text-xs text-zinc-500 leading-relaxed">
                    Get your API key from fal.ai to enable AI actor video generation.
                    <div className="mt-3 grid grid-cols-1 sm:grid-cols-2 gap-2">
                      <a href="https://fal.ai/dashboard/keys" target="_blank" rel="noopener noreferrer" className="p-2 border border-white/5 rounded-lg hover:bg-white/5 transition-colors flex flex-col gap-1">
                        <span className="text-zinc-400 font-medium">1. Sign Up</span>
                        <span className="text-[10px] text-zinc-600">Create fal.ai account</span>
                      </a>
                      <a href="https://fal.ai/dashboard/keys" target="_blank" rel="noopener noreferrer" className="p-2 border border-white/5 rounded-lg hover:bg-white/5 transition-colors flex flex-col gap-1">
                        <span className="text-zinc-400 font-medium">2. API Key</span>
                        <span className="text-[10px] text-zinc-600">Generate key</span>
                      </a>
                    </div>
                    <br />
                    <span className="text-zinc-600 italic">
                      Keys are only stored in your browser. Sent to backend only to process requests.
                    </span>
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* View: SaaS Shorts */}
          {activeTab === 'saasshorts' && (
            <Suspense fallback={<TabFallback />}>
              <SaaShortsTab geminiApiKey={apiKey} elevenLabsKey={elevenLabsKey} falKey={falKey} uploadPostKey={uploadPostKey} uploadUserId={uploadUserId} />
            </Suspense>
          )}

          {/* View: UGC Gallery */}
          {activeTab === 'ugc-gallery' && (
            <Suspense fallback={<TabFallback />}>
              <UGCGallery />
            </Suspense>
          )}

          {/* View: Thumbnails */}
          {activeTab === 'thumbnails' && (
            <Suspense fallback={<TabFallback />}>
              <ThumbnailStudio geminiApiKey={apiKey} uploadPostKey={uploadPostKey} uploadUserId={uploadUserId} />
            </Suspense>
          )}

          {/* View: Gallery */}
          {/* {activeTab === 'gallery' && (
            <Gallery />
          )} */}

          {/* View: Dashboard (Idle) */}
          {activeTab === 'dashboard' && status === 'idle' && (
            <div className="h-full flex flex-col items-center justify-center p-6 animate-[fadeIn_0.3s_ease-out]">
              <div className="max-w-xl w-full text-center space-y-8">
                <div className="space-y-4">
                  <h1 className="text-4xl md:text-5xl font-black bg-gradient-to-b from-white to-white/60 bg-clip-text text-transparent">
                    Create Viral Shorts
                  </h1>
                  <p className="text-zinc-400 text-lg">
                    Drop your long-form video URL or file below to instantly generate viral clips with AI.
                  </p>
                </div>

                <MediaInput onProcess={handleProcess} isProcessing={status === 'processing' || status === 'queued'} />

                <div className="flex items-center justify-center gap-8 text-zinc-500 text-sm">
                  <span className="flex items-center gap-2"><Youtube size={16} /> YouTube</span>
                  <span className="flex items-center gap-2"><Instagram size={16} /> Instagram</span>
                  <span className="flex items-center gap-2"><TikTokIcon size={16} /> TikTok</span>
                </div>
              </div>
            </div>
          )}

          {/* View: Processing / Results (Split View) */}
          {activeTab === 'dashboard' && (status === 'queued' || status === 'processing' || status === 'stalled' || status === 'complete' || status === 'error' || status === 'archived') && (
            <div className="h-full flex flex-col md:flex-row animate-[fadeIn_0.3s_ease-out]">

              {/* Left Panel: Preview & Status */}
              <div className={`${status === 'complete' ? 'w-full md:w-[30%] lg:w-[25%]' : 'w-full md:w-[55%] lg:w-[60%]'} h-full flex flex-col border-r border-white/5 bg-black/20 p-6 overflow-y-auto custom-scrollbar transition-all duration-700 ease-in-out`}>
                <div className="mb-6 flex items-center justify-between">
                  <h2 className="text-lg font-semibold flex items-center gap-2">
                    <Activity className={`text-primary ${status === 'processing' || status === 'queued' ? 'animate-pulse' : ''}`} size={20} />
                    Job Status
                  </h2>
                  <span className={`text-xs px-2 py-1 rounded-full border ${statusTone(status)}`}>
                    {status.toUpperCase()}
                  </span>
                </div>

                <div className="mb-6 glass-panel p-4 space-y-4">
                  <div className="space-y-2">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-zinc-400">{phaseLabel}</span>
                      <span className="text-white font-medium">{progressPercent.toFixed(1)}%</span>
                    </div>
                    <div className="h-2 rounded-full bg-white/5 overflow-hidden">
                      <div
                        className={`h-full rounded-full transition-all duration-500 ${status === 'error' || status === 'archived' ? 'bg-red-400' : status === 'stalled' ? 'bg-amber-400' : 'bg-primary'}`}
                        style={{ width: `${Math.max(2, progressPercent)}%` }}
                      />
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-3 text-xs">
                    <div className="p-3 rounded-xl bg-white/5 border border-white/5">
                      <div className="text-zinc-500 mb-1">ETA</div>
                      <div className="text-white font-medium">{formatDuration(jobMeta?.eta_seconds)}</div>
                      {jobMeta?.total_estimate_seconds ? (
                        <div className="text-[10px] text-zinc-500 mt-0.5">
                          Gesamt geschätzt: ~{formatDuration(jobMeta.total_estimate_seconds)}
                        </div>
                      ) : null}
                    </div>
                    <div className="p-3 rounded-xl bg-white/5 border border-white/5">
                      <div className="text-zinc-500 mb-1">Laufzeit</div>
                      <div className="text-white font-medium">{formatDuration(jobMeta?.elapsed_seconds)}</div>
                    </div>
                    <div className="p-3 rounded-xl bg-white/5 border border-white/5">
                      <div className="text-zinc-500 mb-1">Letzte Aktivität</div>
                      <div className="text-white font-medium">{formatLastSeen(jobMeta?.seconds_since_heartbeat)}</div>
                    </div>
                    <div className="p-3 rounded-xl bg-white/5 border border-white/5">
                      <div className="text-zinc-500 mb-1">Gemini / Resume</div>
                      <div className="text-white font-medium">{jobMeta?.attempt || 0} / {jobMeta?.resume_count || 0}</div>
                    </div>
                  </div>

                  <div className="flex flex-wrap gap-2">
                    {jobMeta?.stall_state === 'slow' && (
                      <span className="text-[10px] px-2 py-1 rounded-full border border-amber-500/20 bg-amber-500/10 text-amber-300">
                        Läuft langsamer als erwartet
                      </span>
                    )}
                    {jobMeta?.stall_state === 'stalled' && (
                      <span className="text-[10px] px-2 py-1 rounded-full border border-amber-500/20 bg-amber-500/10 text-amber-300">
                        Hängt wahrscheinlich
                      </span>
                    )}
                    {jobMeta?.processing_mode === 'full_video_fallback' && (
                      <span className="text-[10px] px-2 py-1 rounded-full border border-white/10 bg-white/5 text-zinc-300">
                        Fallback-Video
                      </span>
                    )}
                    {jobMeta?.error_summary && (
                      <span className="text-[10px] px-2 py-1 rounded-full border border-red-500/20 bg-red-500/10 text-red-300">
                        Fehler erkannt
                      </span>
                    )}
                  </div>

                  {jobMeta?.error_summary && (
                    <div className="text-xs text-red-300 bg-red-500/5 border border-red-500/20 rounded-xl p-3 flex items-start gap-2">
                      <AlertTriangle size={14} className="shrink-0 mt-0.5" />
                      <span>{jobMeta.error_summary}</span>
                    </div>
                  )}

                  <div className="flex flex-wrap gap-2">
                    <button
                      onClick={handleCopySupportLog}
                      className="btn-secondary px-3 py-2 text-xs flex items-center gap-2"
                    >
                      <Copy size={14} />
                      {supportCopied ? 'Support-Logs kopiert' : 'Support-Logs kopieren'}
                    </button>
                    {jobMeta?.is_resumable && (
                      <button
                        onClick={handleResumeJob}
                        className="btn-primary px-3 py-2 text-xs flex items-center gap-2"
                      >
                        <RotateCcw size={14} />
                        Fortsetzen
                      </button>
                    )}
                  </div>
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
                <div className={`bg-[#0c0c0e] rounded-xl border border-white/10 overflow-hidden flex flex-col transition-all duration-500 ${status === 'complete' ? 'h-32 min-h-0 opacity-50 hover:opacity-100' : 'flex-1 min-h-[200px]'}`}>
                  <div className="px-4 py-2 border-b border-white/5 flex items-center justify-between bg-white/5 shrink-0">
                    <span className="text-xs font-mono text-zinc-400 flex items-center gap-2">
                      <Terminal size={12} /> {logMode === 'important' ? 'Important Logs' : 'All Logs'}
                    </span>
                    <div className="flex items-center gap-2">
                      <div className="flex rounded-lg overflow-hidden border border-white/10">
                        <button
                          onClick={() => setLogMode('important')}
                          className={`px-2 py-1 text-[10px] ${logMode === 'important' ? 'bg-white/10 text-white' : 'text-zinc-500 hover:text-white'}`}
                        >
                          Wichtig
                        </button>
                        <button
                          onClick={() => setLogMode('all')}
                          className={`px-2 py-1 text-[10px] ${logMode === 'all' ? 'bg-white/10 text-white' : 'text-zinc-500 hover:text-white'}`}
                        >
                          Alle
                        </button>
                      </div>
                      <button onClick={() => setLogsVisible(!logsVisible)} className="text-zinc-500 hover:text-white transition-colors">
                        {logsVisible ? <ChevronDown size={14} /> : <ChevronDown size={14} className="rotate-180" />}
                      </button>
                    </div>
                  </div>
                  {logsVisible && (
                    <div className="flex-1 p-4 overflow-y-auto font-mono text-xs space-y-1.5 custom-scrollbar text-zinc-400">
                      {renderedLogs.map((log, i) => (
                        <div key={i} className={`flex gap-2 ${String(log.message || '').toLowerCase().includes('error') || log.level === 'error' ? 'text-red-400' : log.level === 'warning' ? 'text-amber-300' : 'text-zinc-400'}`}>
                          <span className="text-zinc-700 shrink-0">{log.iso_timestamp ? new Date(log.iso_timestamp).toLocaleTimeString() : '--:--:--'}</span>
                          <span>{log.message}</span>
                        </div>
                      ))}
                      {(status === 'processing' || status === 'queued') && (
                        <div className="animate-pulse text-primary/70">_</div>
                      )}
                    </div>
                  )}
                </div>
              </div>

              {/* Right Panel: Results Grid */}
              <div className={`${status === 'complete' ? 'w-full md:w-[70%] lg:w-[75%]' : 'w-full md:w-[45%] lg:w-[40%]'} h-full flex flex-col bg-background p-6 transition-all duration-700 ease-in-out`}>
                <h2 className="text-lg font-semibold mb-6 flex items-center gap-2 shrink-0">
                  <Sparkles className="text-yellow-400" size={20} />
                  Generated Shorts
                  {results?.clips?.length > 0 && (
                    <span className="text-xs bg-white/10 text-white px-2 py-0.5 rounded-full ml-auto">
                      {results.clips.length} Clips
                    </span>
                  )}
                  {status === 'complete' && results?.clips?.length > 0 && (
                    <>
                      <button
                        onClick={() => setShowBulkSubtitles(true)}
                        disabled={bulkSubProgress.running}
                        className="text-xs flex items-center gap-1.5 bg-yellow-500/10 border border-yellow-500/20 text-yellow-400 hover:bg-yellow-500/20 px-2.5 py-1 rounded-full transition-colors disabled:opacity-60"
                        title="Untertitel-Style einmal wählen und auf alle Clips anwenden"
                      >
                        {bulkSubProgress.running ? (
                          <>
                            <Loader2 size={12} className="animate-spin" />
                            Subtitles {bulkSubProgress.current}/{bulkSubProgress.total}
                          </>
                        ) : (
                          <>
                            <Type size={12} />
                            Subtitles für alle
                          </>
                        )}
                      </button>
                      <a
                        href={getApiUrl(`/api/jobs/${jobId}/download-all`)}
                        className="text-xs flex items-center gap-1.5 bg-white/5 border border-white/10 text-zinc-300 hover:bg-white/10 px-2.5 py-1 rounded-full transition-colors"
                        title="Alle Clips als ZIP herunterladen"
                      >
                        <Download size={12} />
                        Alle laden
                      </a>
                    </>
                  )}
                  {results?.cost_analysis && (
                    <span className="text-xs bg-green-500/10 border border-green-500/20 text-green-400 px-2 py-0.5 rounded-full ml-2" title={`Input: ${results.cost_analysis.input_tokens} | Output: ${results.cost_analysis.output_tokens}`}>
                      ${results.cost_analysis.total_cost.toFixed(5)}
                    </span>
                  )}
                </h2>

                <div className="flex-1 overflow-y-auto custom-scrollbar p-1">
                  {results && results.clips && results.clips.length > 0 ? (
                    <div className={`grid gap-4 pb-10 ${status === 'complete' ? 'grid-cols-1 xl:grid-cols-2' : 'grid-cols-1'}`}>
                      {results.clips.map((clip, i) => (
                        <ResultCard
                          key={`${i}-${clip.video_url || ''}`}
                          clip={clip}
                          index={i}
                          jobId={jobId}
                          uploadPostKey={uploadPostKey}
                          uploadUserId={uploadUserId}
                          geminiApiKey={apiKey}
                          elevenLabsKey={elevenLabsKey}
                          onPlay={(time) => handleClipPlay(time)}
                          onPause={handleClipPause}
                        />
                      ))}
                    </div>
                  ) : (
                    status === 'processing' || status === 'queued' ? (
                      <div className="h-full flex flex-col items-center justify-center text-zinc-500 space-y-4 opacity-50">
                        <div className="w-12 h-12 rounded-full border-2 border-zinc-800 border-t-primary animate-spin" />
                        <p className="text-sm">{status === 'queued' ? 'Warte auf freien Slot...' : 'Warte auf Clips...'}</p>
                      </div>
                    ) : status === 'stalled' ? (
                      <div className="h-full flex flex-col items-center justify-center text-amber-300 space-y-3">
                        <p>Der Job hängt wahrscheinlich oder wartet zu lange.</p>
                        {jobMeta?.is_resumable && (
                          <button onClick={handleResumeJob} className="btn-primary px-4 py-2 text-sm flex items-center gap-2">
                            <RotateCcw size={14} />
                            Fortsetzen
                          </button>
                        )}
                      </div>
                    ) : status === 'archived' ? (
                      <div className="h-full flex flex-col items-center justify-center text-red-400 space-y-2">
                        <p>Dieser Job wurde archiviert oder gepurgt.</p>
                      </div>
                    ) : status === 'error' ? (
                      <div className="h-full flex flex-col items-center justify-center text-red-400 space-y-2">
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

      {/* Bulk Subtitles Modal: one style, applied to every clip */}
      <SubtitleModal
        isOpen={showBulkSubtitles}
        onClose={() => setShowBulkSubtitles(false)}
        onGenerate={handleBulkSubtitles}
        isProcessing={bulkSubProgress.running}
        videoUrl={results?.clips?.[0]?.video_url ? getApiUrl(results.clips[0].video_url) : undefined}
        bulkCount={results?.clips?.length || 0}
      />

      {/* Low Quality Confirmation Modal (quality gate) */}
      {lowQualityPrompt && (
        <div className="fixed inset-0 z-[110] flex items-center justify-center bg-black/70 backdrop-blur-sm">
          <div className="bg-[#18181b] border border-amber-500/30 rounded-2xl p-6 max-w-md w-full mx-4 space-y-4 shadow-2xl">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-full bg-amber-500/15 flex items-center justify-center shrink-0">
                <AlertTriangle size={20} className="text-amber-400" />
              </div>
              <h2 className="text-lg font-bold text-white">
                Only {lowQualityPrompt.maxHeight}p available
              </h2>
            </div>
            <p className="text-sm text-zinc-400">
              YouTube is only offering <span className="text-amber-400 font-semibold">{lowQualityPrompt.maxHeight}p</span> for
              this video right now — your clips would look soft. Processing has <span className="font-semibold text-white">not</span> started yet.
            </p>
            {lowQualityPrompt.cookiesInvalid && (
              <div className="bg-amber-500/10 border border-amber-500/20 rounded-lg p-3 text-xs text-amber-300 space-y-1">
                <p className="font-semibold">Your YouTube cookies were rotated by the browser.</p>
                <p className="text-amber-300/80">
                  Long-lasting export: open an <span className="font-semibold">incognito window</span> → log into YouTube →
                  open <code className="text-amber-200">youtube.com/robots.txt</code> in the same tab → export cookies →
                  <span className="font-semibold"> close the incognito window</span> and never reuse that session.
                  Cookies exported this way last ~2 weeks instead of hours.
                </p>
              </div>
            )}
            <p className="text-xs text-zinc-500">
              Also worth trying: update yt-dlp
              (<code className="text-zinc-400">venv\Scripts\pip install -U yt-dlp</code>), then retry.
            </p>
            <div className="flex gap-3 pt-1">
              <button
                onClick={() => setLowQualityPrompt(null)}
                className="flex-1 py-2.5 rounded-xl bg-white/5 border border-white/10 text-sm text-zinc-200 hover:bg-white/10 transition-colors font-medium"
              >
                Cancel & fix cookies
              </button>
              <button
                onClick={() => { const p = lowQualityPrompt; setLowQualityPrompt(null); handleProcess(p.data, true); }}
                className="flex-1 py-2.5 rounded-xl bg-amber-500/90 hover:bg-amber-400 text-black text-sm font-bold transition-colors"
              >
                Process anyway ({lowQualityPrompt.maxHeight}p)
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Missing API Key Modal */}
      {showKeyModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm" onClick={() => setShowKeyModal(false)}>
          <div className="bg-[#18181b] border border-white/10 rounded-2xl p-6 max-w-md w-full mx-4 space-y-4 shadow-2xl" onClick={(e) => e.stopPropagation()}>
            <h2 className="text-lg font-bold text-white">Gemini API Key Required</h2>
            <p className="text-sm text-zinc-400">
              You need a Google Gemini API key to use the Clip Generator. It's free and takes 30 seconds to get.
            </p>
            <div className="bg-white/5 border border-white/10 rounded-lg p-4 space-y-2">
              <p className="text-xs font-semibold text-zinc-300">How to get your free key:</p>
              <ol className="text-xs text-zinc-400 space-y-1 list-decimal list-inside">
                <li>Go to <a href="https://aistudio.google.com/app/apikey" target="_blank" rel="noopener noreferrer" className="text-blue-400 underline">aistudio.google.com/app/apikey</a></li>
                <li>Sign in with your Google account</li>
                <li>Click "Create API Key"</li>
                <li>Copy the key and paste it below</li>
              </ol>
            </div>
            <input
              type="text"
              placeholder="Paste your Gemini API key here..."
              className="w-full bg-black/50 border border-white/20 rounded-lg px-4 py-2.5 text-sm text-white placeholder-zinc-600 focus:outline-none focus:border-blue-500"
              onKeyDown={(e) => {
                if (e.key === 'Enter' && e.target.value.trim()) {
                  setApiKey(e.target.value.trim());
                  setShowKeyModal(false);
                }
              }}
            />

            {/* Upload-Post info */}
            <div className="bg-violet-500/5 border border-violet-500/20 rounded-lg p-4 space-y-2">
              <p className="text-xs font-semibold text-violet-300">Optional: Auto-publish to social media</p>
              <p className="text-xs text-zinc-400">
                With an <strong className="text-zinc-300">Upload-Post</strong> API key you can publish your clips directly to TikTok, Instagram Reels, and YouTube Shorts — or schedule them for later. Free tier available, no credit card needed.
              </p>
              <ol className="text-xs text-zinc-400 space-y-1 list-decimal list-inside">
                <li>Register at <a href="https://app.upload-post.com/login" target="_blank" rel="noopener noreferrer" className="text-violet-400 underline">app.upload-post.com</a></li>
                <li>Connect your TikTok, Instagram, or YouTube accounts</li>
                <li>Go to API Keys and generate one</li>
                <li>Paste it in Settings — done!</li>
              </ol>
            </div>

            <div className="flex gap-3">
              <button
                onClick={() => setShowKeyModal(false)}
                className="flex-1 text-sm text-zinc-400 py-2 rounded-lg border border-white/10 hover:bg-white/5 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={() => { setShowKeyModal(false); setActiveTab('settings'); }}
                className="flex-1 text-sm text-white py-2 rounded-lg bg-blue-600 hover:bg-blue-500 transition-colors font-medium"
              >
                Go to Settings
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
