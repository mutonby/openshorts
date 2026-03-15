import React, { useState, useEffect } from 'react';
import { Globe, Sparkles, Download, Copy, Check, ChevronRight, ChevronLeft, Loader2, AlertCircle, Volume2, User, Film, Terminal, ChevronDown, RefreshCw, Zap, Target, TrendingUp, MessageSquare, Eye } from 'lucide-react';
import { getApiUrl } from '../config';

const STYLE_OPTIONS = [
  { id: 'ugc', label: 'UGC Natural', desc: 'Authentic, talking to camera' },
  { id: 'educational', label: 'Educational', desc: 'Clear explanations' },
  { id: 'shock', label: 'Shock/Discovery', desc: 'Surprising opener' },
  { id: 'story', label: 'Storytelling', desc: 'Mini narrative arc' },
  { id: 'comparison', label: 'Before/After', desc: 'Comparison style' },
];

const CACHE_KEY = 'saasshorts_cache';
const CACHE_MAX_AGE = 24 * 60 * 60 * 1000; // 24 hours

function loadCache() {
  try {
    const raw = localStorage.getItem(CACHE_KEY);
    if (!raw) return null;
    const cache = JSON.parse(raw);
    if (Date.now() - cache.timestamp > CACHE_MAX_AGE) {
      localStorage.removeItem(CACHE_KEY);
      return null;
    }
    return cache;
  } catch { return null; }
}

function saveCache(url, analysis, webResearch, scripts) {
  try {
    localStorage.setItem(CACHE_KEY, JSON.stringify({
      url, analysis, webResearch, scripts, timestamp: Date.now(),
    }));
  } catch { /* localStorage full */ }
}

export default function SaaShortsTab({ geminiApiKey, elevenLabsKey, falKey }) {
  // Wizard state
  const [step, setStep] = useState(() => {
    const cache = loadCache();
    return cache ? 1 : 0;
  });

  // Step 0: URL input
  const [url, setUrl] = useState(() => loadCache()?.url || '');
  const [style, setStyle] = useState('ugc');
  const [language, setLanguage] = useState('en');
  const [actorGender, setActorGender] = useState('female');
  const [numScripts, setNumScripts] = useState(3);
  const [analyzing, setAnalyzing] = useState(false);
  const [analyzeError, setAnalyzeError] = useState('');
  const [fromCache, setFromCache] = useState(() => !!loadCache());

  // Step 1: Analysis results
  const [analysis, setAnalysis] = useState(() => loadCache()?.analysis || null);
  const [webResearch, setWebResearch] = useState(() => loadCache()?.webResearch || null);
  const [scripts, setScripts] = useState(() => loadCache()?.scripts || []);
  const [selectedScript, setSelectedScript] = useState(0);

  // Step 2: Configure
  const [voices, setVoices] = useState([]);
  const [selectedVoice, setSelectedVoice] = useState('21m00Tcm4TlvDq8ikWAM');
  const [actorDescription, setActorDescription] = useState('');
  const [editedNarration, setEditedNarration] = useState('');
  const [actorOptions, setActorOptions] = useState([]);
  const [selectedActor, setSelectedActor] = useState(null);
  const [generatingActors, setGeneratingActors] = useState(false);

  // Step 3: Generate
  const [generating, setGenerating] = useState(false);
  const [jobId, setJobId] = useState(null);
  const [genLogs, setGenLogs] = useState([]);
  const [genStatus, setGenStatus] = useState('idle');
  const [genResult, setGenResult] = useState(null);

  // UI
  const [copied, setCopied] = useState('');
  const [logsExpanded, setLogsExpanded] = useState(true);

  // Pre-fill from cache on mount
  useEffect(() => {
    if (fromCache && scripts.length > 0 && !actorDescription) {
      setActorDescription(scripts[0].actor_description || '');
      setEditedNarration(scripts[0].full_narration || '');
    }
  }, []);

  // Fetch voices on mount
  useEffect(() => {
    if (elevenLabsKey) {
      fetchVoices();
    }
  }, [elevenLabsKey]);

  // Poll generation status
  useEffect(() => {
    let interval;
    if (jobId && genStatus === 'processing') {
      interval = setInterval(async () => {
        try {
          const res = await fetch(getApiUrl(`/api/saasshorts/status/${jobId}`));
          if (res.status === 404) {
            // Job lost (server restart) — treat as failed so Retry appears
            setGenStatus('failed');
            setGenerating(false);
            setGenLogs((prev) => [...prev, 'Job lost after server restart. Click Retry to resume from cached assets.']);
            clearInterval(interval);
            return;
          }
          if (!res.ok) return;
          const data = await res.json();
          if (data.logs) setGenLogs(data.logs);
          if (data.status === 'completed') {
            setGenStatus('completed');
            setGenResult(data.result);
            setGenerating(false);
            setStep(4);
            clearInterval(interval);
          } else if (data.status === 'failed') {
            setGenStatus('failed');
            setGenerating(false);
            clearInterval(interval);
          }
        } catch (e) {
          console.error('Poll error:', e);
        }
      }, 2000);
    }
    return () => clearInterval(interval);
  }, [jobId, genStatus]);

  const fetchVoices = async () => {
    try {
      const res = await fetch(getApiUrl('/api/saasshorts/voices'), {
        headers: { 'X-ElevenLabs-Key': elevenLabsKey },
      });
      if (res.ok) {
        const data = await res.json();
        setVoices(data.voices || []);
      }
    } catch (e) {
      console.error('Voices fetch error:', e);
    }
  };

  const handleAnalyze = async () => {
    if (!url.trim()) return;
    if (!geminiApiKey) {
      setAnalyzeError('Gemini API key required. Set it in Settings.');
      return;
    }

    setAnalyzing(true);
    setAnalyzeError('');

    try {
      const res = await fetch(getApiUrl('/api/saasshorts/analyze'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Gemini-Key': geminiApiKey,
        },
        body: JSON.stringify({ url: url.trim(), num_scripts: numScripts, style, language, actor_gender: actorGender }),
      });

      if (!res.ok) {
        let msg = 'Analysis failed';
        try { const err = await res.json(); msg = err.detail || msg; } catch { msg = await res.text() || msg; }
        throw new Error(msg);
      }

      const data = await res.json();
      setAnalysis(data.analysis);
      setWebResearch(data.web_research || null);
      setScripts(data.scripts);
      setSelectedScript(0);
      setFromCache(false);

      // Cache results
      saveCache(url.trim(), data.analysis, data.web_research, data.scripts);

      // Pre-fill actor description and narration from first script
      if (data.scripts.length > 0) {
        setActorDescription(data.scripts[0].actor_description || '');
        setEditedNarration(data.scripts[0].full_narration || '');
      }

      setStep(1);
    } catch (e) {
      setAnalyzeError(e.message);
    } finally {
      setAnalyzing(false);
    }
  };

  const handleSelectScript = (idx) => {
    setSelectedScript(idx);
    if (scripts[idx]) {
      setActorDescription(scripts[idx].actor_description || '');
      setEditedNarration(scripts[idx].full_narration || '');
    }
  };

  const handleGenerate = async () => {
    if (!falKey) {
      alert('fal.ai API key required. Set it in Settings.');
      return;
    }
    if (!elevenLabsKey) {
      alert('ElevenLabs API key required. Set it in Settings.');
      return;
    }

    setGenerating(true);
    setGenLogs(['Starting video generation...']);
    setGenStatus('processing');
    setGenResult(null);
    setStep(3);

    try {
      // Update script with edited narration
      const scriptToSend = { ...scripts[selectedScript] };
      if (editedNarration !== scriptToSend.full_narration) {
        scriptToSend.full_narration = editedNarration;
      }

      const res = await fetch(getApiUrl('/api/saasshorts/generate'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Fal-Key': falKey,
          'X-ElevenLabs-Key': elevenLabsKey,
        },
        body: JSON.stringify({
          script: scriptToSend,
          voice_id: selectedVoice,
          actor_description: actorDescription || undefined,
          selected_actor_url: selectedActor || undefined,
        }),
      });

      if (!res.ok) {
        let msg = 'Generation failed';
        try { const err = await res.json(); msg = err.detail || msg; } catch { msg = await res.text() || msg; }
        throw new Error(msg);
      }

      const data = await res.json();
      setJobId(data.job_id);
    } catch (e) {
      setGenStatus('failed');
      setGenLogs((prev) => [...prev, `Error: ${e.message}`]);
      setGenerating(false);
    }
  };

  const handleRetry = async () => {
    if (!jobId) return;
    setGenerating(true);
    setGenLogs(['Retrying from cached assets...']);
    setGenStatus('processing');
    setGenResult(null);

    try {
      const scriptToSend = { ...scripts[selectedScript] };
      if (editedNarration !== scriptToSend.full_narration) {
        scriptToSend.full_narration = editedNarration;
      }

      const res = await fetch(getApiUrl('/api/saasshorts/generate'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Fal-Key': falKey,
          'X-ElevenLabs-Key': elevenLabsKey,
        },
        body: JSON.stringify({
          script: scriptToSend,
          voice_id: selectedVoice,
          actor_description: actorDescription || undefined,
          retry_job_id: jobId,
        }),
      });

      if (!res.ok) {
        let msg = 'Retry failed';
        try { const err = await res.json(); msg = err.detail || msg; } catch { msg = await res.text() || msg; }
        throw new Error(msg);
      }

      const data = await res.json();
      setJobId(data.job_id);
    } catch (e) {
      setGenStatus('failed');
      setGenLogs((prev) => [...prev, `Retry error: ${e.message}`]);
      setGenerating(false);
    }
  };

  const handleCopy = (text, label) => {
    navigator.clipboard.writeText(text);
    setCopied(label);
    setTimeout(() => setCopied(''), 2000);
  };

  const handleReset = () => {
    setStep(0);
    setUrl('');
    setAnalyzeError('');
    setAnalysis(null);
    setWebResearch(null);
    setScripts([]);
    setFromCache(false);
    localStorage.removeItem(CACHE_KEY);
    setSelectedScript(0);
    setJobId(null);
    setGenLogs([]);
    setGenStatus('idle');
    setGenResult(null);
    setGenerating(false);
    setActorDescription('');
    setEditedNarration('');
  };

  // ─── Render Steps ─────────────────────────────────────────────────

  return (
    <div className="h-full overflow-y-auto custom-scrollbar">
      <div className="max-w-5xl mx-auto p-6 lg:p-8">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-2xl font-bold flex items-center gap-3">
              <div className="w-10 h-10 bg-gradient-to-br from-violet-500 to-fuchsia-500 rounded-xl flex items-center justify-center">
                <Zap size={20} className="text-white" />
              </div>
              SaaS Shorts
            </h1>
            <p className="text-sm text-zinc-500 mt-1">
              Generate viral UGC videos for your SaaS with AI actors
            </p>
          </div>
          {step > 0 && (
            <button onClick={handleReset} className="text-sm text-zinc-400 hover:text-white flex items-center gap-1 transition-colors">
              <RefreshCw size={14} /> Start over
            </button>
          )}
        </div>

        {/* Progress Steps */}
        <div className="flex items-center gap-2 mb-8">
          {['URL & Style', 'Analysis', 'Configure', 'Generate', 'Result'].map((label, i) => (
            <React.Fragment key={i}>
              {i > 0 && <div className={`flex-1 h-px ${i <= step ? 'bg-violet-500' : 'bg-white/10'}`} />}
              <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium transition-all ${
                i === step ? 'bg-violet-500/20 text-violet-300 border border-violet-500/30' :
                i < step ? 'bg-violet-500/10 text-violet-400' :
                'bg-white/5 text-zinc-600'
              }`}>
                <span className={`w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold ${
                  i < step ? 'bg-violet-500 text-white' :
                  i === step ? 'bg-violet-500/30 text-violet-300' :
                  'bg-white/10 text-zinc-600'
                }`}>
                  {i < step ? <Check size={10} /> : i + 1}
                </span>
                <span className="hidden sm:inline">{label}</span>
              </div>
            </React.Fragment>
          ))}
        </div>

        {/* ── Step 0: URL Input ────────────────────────────────── */}
        {step === 0 && (
          <div className="animate-[fadeIn_0.3s_ease-out] space-y-6">
            <div className="glass-panel p-8 space-y-6">
              <div>
                <label className="block text-sm font-medium text-zinc-300 mb-2">SaaS Website URL</label>
                <div className="flex gap-3">
                  <div className="relative flex-1">
                    <Globe size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-zinc-500" />
                    <input
                      type="url"
                      value={url}
                      onChange={(e) => setUrl(e.target.value)}
                      placeholder="https://your-saas.com"
                      className="input-field pl-10"
                      onKeyDown={(e) => e.key === 'Enter' && handleAnalyze()}
                    />
                  </div>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-zinc-300 mb-3">Language</label>
                <div className="flex gap-2 mb-6">
                  {[
                    { id: 'en', label: 'English', flag: '🇺🇸' },
                    { id: 'es', label: 'Español', flag: '🇪🇸' },
                  ].map((l) => (
                    <button
                      key={l.id}
                      onClick={() => setLanguage(l.id)}
                      className={`flex-1 p-3 rounded-xl border text-center transition-all ${
                        language === l.id
                          ? 'border-violet-500/50 bg-violet-500/10 text-violet-300'
                          : 'border-white/10 bg-white/5 text-zinc-400 hover:bg-white/10'
                      }`}
                    >
                      <span className="text-lg">{l.flag}</span>
                      <div className="text-xs font-medium mt-1">{l.label}</div>
                    </button>
                  ))}
                </div>

                <label className="block text-sm font-medium text-zinc-300 mb-3">Actor</label>
                <div className="flex gap-2 mb-6">
                  {[
                    { id: 'female', label: 'Woman', icon: '👩' },
                    { id: 'male', label: 'Man', icon: '👨' },
                  ].map((g) => (
                    <button
                      key={g.id}
                      onClick={() => setActorGender(g.id)}
                      className={`flex-1 p-3 rounded-xl border text-center transition-all ${
                        actorGender === g.id
                          ? 'border-violet-500/50 bg-violet-500/10 text-violet-300'
                          : 'border-white/10 bg-white/5 text-zinc-400 hover:bg-white/10'
                      }`}
                    >
                      <span className="text-lg">{g.icon}</span>
                      <div className="text-xs font-medium mt-1">{g.label}</div>
                    </button>
                  ))}
                </div>

                <label className="block text-sm font-medium text-zinc-300 mb-3">Video Style</label>
                <div className="grid grid-cols-2 sm:grid-cols-5 gap-2">
                  {STYLE_OPTIONS.map((s) => (
                    <button
                      key={s.id}
                      onClick={() => setStyle(s.id)}
                      className={`p-3 rounded-xl border text-left transition-all ${
                        style === s.id
                          ? 'border-violet-500/50 bg-violet-500/10 text-violet-300'
                          : 'border-white/10 bg-white/5 text-zinc-400 hover:bg-white/10'
                      }`}
                    >
                      <div className="text-xs font-medium">{s.label}</div>
                      <div className="text-[10px] text-zinc-500 mt-0.5">{s.desc}</div>
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-zinc-300 mb-2">Number of Scripts</label>
                <div className="flex gap-2">
                  {[1, 2, 3, 5].map((n) => (
                    <button
                      key={n}
                      onClick={() => setNumScripts(n)}
                      className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                        numScripts === n
                          ? 'bg-violet-500/20 text-violet-300 border border-violet-500/30'
                          : 'bg-white/5 text-zinc-400 border border-white/10 hover:bg-white/10'
                      }`}
                    >
                      {n}
                    </button>
                  ))}
                </div>
              </div>

              {analyzeError && (
                <div className="flex items-center gap-2 text-sm text-red-400 bg-red-500/10 border border-red-500/20 rounded-lg p-3">
                  <AlertCircle size={14} />
                  {analyzeError}
                </div>
              )}

              <button
                onClick={handleAnalyze}
                disabled={analyzing || !url.trim()}
                className="btn-primary w-full py-3 text-sm font-semibold flex items-center justify-center gap-2 disabled:opacity-50"
              >
                {analyzing ? (
                  <>
                    <Loader2 size={16} className="animate-spin" />
                    Scraping + Researching web + Generating scripts... (45-90s)
                  </>
                ) : (
                  <>
                    <Sparkles size={16} />
                    Research & Generate Scripts
                  </>
                )}
              </button>
            </div>

            {/* Info cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="glass-panel p-4">
                <Target size={16} className="text-violet-400 mb-2" />
                <h3 className="text-sm font-medium text-zinc-300">Deep Research</h3>
                <p className="text-xs text-zinc-500 mt-1">AI scrapes your website + searches Google for reviews, Reddit threads, competitors &amp; real user feedback.</p>
              </div>
              <div className="glass-panel p-4">
                <MessageSquare size={16} className="text-violet-400 mb-2" />
                <h3 className="text-sm font-medium text-zinc-300">Pain Point Scripts</h3>
                <p className="text-xs text-zinc-500 mt-1">Generates hook-problem-solution scripts targeting your audience&apos;s real pain points.</p>
              </div>
              <div className="glass-panel p-4">
                <Film size={16} className="text-violet-400 mb-2" />
                <h3 className="text-sm font-medium text-zinc-300">AI Actor Videos</h3>
                <p className="text-xs text-zinc-500 mt-1">Realistic AI-generated actors with lip-sync, b-roll, and viral subtitles. ~$2.50/video.</p>
              </div>
            </div>
          </div>
        )}

        {/* ── Step 1: Analysis Results ─────────────────────────── */}
        {step === 1 && analysis && (
          <div className="animate-[fadeIn_0.3s_ease-out] space-y-6">
            {/* Analysis Summary */}
            <div className="glass-panel p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold flex items-center gap-2">
                  <Eye size={18} className="text-violet-400" />
                  {analysis.product_name || 'Analysis'}
                </h2>
                <div className="flex items-center gap-2">
                  {fromCache && (
                    <span className="text-[10px] bg-amber-500/10 text-amber-400 px-2 py-0.5 rounded-full border border-amber-500/20 flex items-center gap-1">
                      Cached
                      <button onClick={() => { setStep(0); setFromCache(false); }} className="hover:text-white ml-1" title="Re-analyze">
                        <RefreshCw size={9} />
                      </button>
                    </span>
                  )}
                  <span className="text-xs bg-violet-500/10 text-violet-400 px-2 py-1 rounded-full border border-violet-500/20">
                    {analysis.industry}
                  </span>
                </div>
              </div>
              <p className="text-sm text-zinc-400 mb-4">{analysis.one_liner}</p>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <h3 className="text-xs font-semibold text-zinc-500 uppercase tracking-wider mb-2">Pain Points</h3>
                  <div className="space-y-1.5">
                    {(analysis.pain_points || []).map((pp, i) => (
                      <div key={i} className="flex items-start gap-2 text-sm">
                        <span className={`mt-0.5 w-2 h-2 rounded-full shrink-0 ${
                          pp.intensity === 'high' ? 'bg-red-400' : pp.intensity === 'medium' ? 'bg-yellow-400' : 'bg-green-400'
                        }`} />
                        <div>
                          <span className="text-zinc-300">{pp.pain}</span>
                          {pp.source && pp.source !== 'website' && (
                            <span className="ml-1.5 text-[9px] bg-blue-500/10 text-blue-400 px-1 py-0.5 rounded">{pp.source}</span>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
                <div>
                  <h3 className="text-xs font-semibold text-zinc-500 uppercase tracking-wider mb-2">Emotional Hooks</h3>
                  <div className="space-y-1.5">
                    {(analysis.emotional_hooks || []).map((h, i) => (
                      <div key={i} className="text-sm text-zinc-300 flex items-start gap-2">
                        <TrendingUp size={12} className="text-violet-400 mt-1 shrink-0" />
                        {h}
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* Web Research Results */}
            {webResearch && (
              <div className="glass-panel p-6">
                <h3 className="text-sm font-semibold text-zinc-300 mb-3 flex items-center gap-2">
                  <Globe size={14} className="text-blue-400" />
                  Web Research (Google Search)
                  {webResearch.grounding_sources && (
                    <span className="text-[10px] bg-blue-500/10 text-blue-400 px-2 py-0.5 rounded-full ml-auto">
                      {webResearch.grounding_sources.length} sources
                    </span>
                  )}
                </h3>

                {/* Real user reviews */}
                {webResearch.real_reviews && webResearch.real_reviews.length > 0 && (
                  <div className="mb-4">
                    <h4 className="text-xs font-semibold text-zinc-500 uppercase tracking-wider mb-2">Real User Reviews</h4>
                    <div className="space-y-2">
                      {webResearch.real_reviews.slice(0, 5).map((review, i) => (
                        <div key={i} className="text-xs bg-white/5 rounded-lg p-2.5 border border-white/5">
                          <p className="text-zinc-300 italic">&quot;{review.quote}&quot;</p>
                          <div className="flex items-center gap-2 mt-1">
                            <span className="text-zinc-600">{review.source}</span>
                            <span className={`px-1 py-0.5 rounded text-[9px] ${
                              review.sentiment === 'positive' ? 'bg-green-500/10 text-green-400' :
                              review.sentiment === 'negative' ? 'bg-red-500/10 text-red-400' :
                              'bg-zinc-500/10 text-zinc-400'
                            }`}>{review.sentiment}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Competitors */}
                {webResearch.competitors && webResearch.competitors.length > 0 && (
                  <div className="mb-4">
                    <h4 className="text-xs font-semibold text-zinc-500 uppercase tracking-wider mb-2">Competitors</h4>
                    <div className="flex flex-wrap gap-2">
                      {webResearch.competitors.map((c, i) => (
                        <span key={i} className="text-xs bg-white/5 px-2 py-1 rounded-lg text-zinc-400 border border-white/5" title={c.comparison}>
                          {c.name}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* Sources */}
                {webResearch.grounding_sources && webResearch.grounding_sources.length > 0 && (
                  <div>
                    <h4 className="text-xs font-semibold text-zinc-500 uppercase tracking-wider mb-2">Sources</h4>
                    <div className="flex flex-wrap gap-1.5">
                      {webResearch.grounding_sources.slice(0, 8).map((src, i) => (
                        <a
                          key={i}
                          href={src.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-[10px] text-blue-400 hover:text-blue-300 bg-blue-500/5 px-2 py-0.5 rounded-full border border-blue-500/10 hover:border-blue-500/30 transition-colors truncate max-w-[200px]"
                          title={src.title}
                        >
                          {src.title || (() => { try { return new URL(src.url).hostname; } catch { return src.url; } })()}
                        </a>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Scripts */}
            <div>
              <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <Sparkles size={18} className="text-yellow-400" />
                Generated Scripts
                <span className="text-xs bg-white/10 px-2 py-0.5 rounded-full ml-auto">{scripts.length} scripts</span>
              </h2>

              <div className="grid grid-cols-1 gap-4">
                {scripts.map((script, i) => (
                  <div
                    key={i}
                    onClick={() => handleSelectScript(i)}
                    className={`glass-panel p-5 cursor-pointer transition-all ${
                      selectedScript === i
                        ? 'border-violet-500/50 bg-violet-500/5 ring-1 ring-violet-500/20'
                        : 'hover:bg-white/5'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center gap-3">
                        <span className={`w-7 h-7 rounded-lg flex items-center justify-center text-xs font-bold ${
                          selectedScript === i ? 'bg-violet-500 text-white' : 'bg-white/10 text-zinc-400'
                        }`}>
                          {i + 1}
                        </span>
                        <div>
                          <h3 className="text-sm font-semibold text-zinc-200">{script.title}</h3>
                          <span className="text-[10px] text-zinc-500">{script.duration_seconds}s &middot; {script.style} &middot; {script.target_platform}</span>
                        </div>
                      </div>
                      {selectedScript === i && (
                        <span className="text-[10px] bg-violet-500/20 text-violet-300 px-2 py-0.5 rounded-full">Selected</span>
                      )}
                    </div>

                    {/* Segments preview */}
                    <div className="flex gap-1 mb-3">
                      {(script.segments || []).map((seg, j) => (
                        <div
                          key={j}
                          className={`h-1.5 rounded-full ${
                            seg.type === 'hook' ? 'bg-red-400' :
                            seg.type === 'problem' ? 'bg-yellow-400' :
                            seg.type === 'solution' ? 'bg-green-400' :
                            'bg-blue-400'
                          }`}
                          style={{ flex: (seg.end - seg.start) }}
                          title={`${seg.type}: ${seg.start}s-${seg.end}s`}
                        />
                      ))}
                    </div>

                    <div className="space-y-2">
                      {(script.segments || []).map((seg, j) => (
                        <div key={j} className="flex gap-3 text-xs">
                          <span className={`shrink-0 px-1.5 py-0.5 rounded font-mono uppercase tracking-wider ${
                            seg.type === 'hook' ? 'bg-red-500/20 text-red-300' :
                            seg.type === 'problem' ? 'bg-yellow-500/20 text-yellow-300' :
                            seg.type === 'solution' ? 'bg-green-500/20 text-green-300' :
                            'bg-blue-500/20 text-blue-300'
                          }`}>
                            {seg.type}
                          </span>
                          <span className="text-zinc-400 leading-relaxed">{seg.narration}</span>
                        </div>
                      ))}
                    </div>

                    {/* Hook text & hashtags */}
                    <div className="mt-3 pt-3 border-t border-white/5 flex items-center gap-3 flex-wrap">
                      <span className="text-[10px] bg-red-500/10 text-red-300 px-2 py-0.5 rounded-full">
                        Hook: &quot;{script.hook_text}&quot;
                      </span>
                      {(script.hashtags || []).slice(0, 4).map((tag, j) => (
                        <span key={j} className="text-[10px] text-zinc-500">{tag}</span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="flex justify-between">
              <button onClick={() => setStep(0)} className="btn-secondary px-4 py-2 text-sm flex items-center gap-2">
                <ChevronLeft size={14} /> Back
              </button>
              <button onClick={() => setStep(2)} className="btn-primary px-6 py-2 text-sm flex items-center gap-2">
                Configure Video <ChevronRight size={14} />
              </button>
            </div>
          </div>
        )}

        {/* ── Step 2: Configure ────────────────────────────────── */}
        {step === 2 && scripts[selectedScript] && (
          <div className="animate-[fadeIn_0.3s_ease-out] space-y-6">
            <div className="glass-panel p-6 space-y-5">
              <h2 className="text-lg font-semibold">Configure Video</h2>
              <p className="text-sm text-zinc-500">
                Script: <strong className="text-zinc-300">{scripts[selectedScript].title}</strong>
              </p>

              {/* Voice Selection */}
              <div>
                <label className="block text-sm font-medium text-zinc-300 mb-2 flex items-center gap-2">
                  <Volume2 size={14} /> Voice {language === 'es' ? '(Spanish)' : '(English)'}
                </label>
                {(() => {
                  // Filter voices by language/accent
                  const filtered = voices.length > 0
                    ? voices.filter((v) => {
                        const gender = (v.labels?.gender || '').toLowerCase();
                        // Filter by gender only — for Spanish, multilingual_v2 makes any voice work
                        return !gender || gender === actorGender;
                      })
                      .sort((a, b) => {
                        const aAccent = (a.labels?.accent || '').toLowerCase();
                        const bAccent = (b.labels?.accent || '').toLowerCase();
                        if (language === 'es') {
                          // Spanish/latin accents first, then everything else
                          const aScore = (aAccent.includes('spanish') || aAccent.includes('latin')) ? 0 : 1;
                          const bScore = (bAccent.includes('spanish') || bAccent.includes('latin')) ? 0 : 1;
                          return aScore - bScore;
                        }
                        // English: american/british first
                        const aScore = (aAccent.includes('american') || aAccent.includes('british')) ? 0 : 1;
                        const bScore = (bAccent.includes('american') || bAccent.includes('british')) ? 0 : 1;
                        return aScore - bScore;
                      })
                    : [];

                  if (filtered.length > 0) {
                    return (
                      <div className="space-y-1.5 max-h-48 overflow-y-auto custom-scrollbar">
                        {filtered.map((v) => (
                          <button
                            key={v.voice_id}
                            onClick={() => setSelectedVoice(v.voice_id)}
                            className={`w-full flex items-center gap-3 p-2.5 rounded-lg border text-left transition-all ${
                              selectedVoice === v.voice_id
                                ? 'border-violet-500/50 bg-violet-500/10 text-violet-300'
                                : 'border-white/10 bg-white/5 text-zinc-400 hover:bg-white/10'
                            }`}
                          >
                            <div className="flex-1 min-w-0">
                              <div className="text-sm font-medium truncate">{v.name}</div>
                              <div className="text-[10px] text-zinc-500">
                                {v.labels?.accent || ''} {v.labels?.gender || ''} {v.category ? `· ${v.category}` : ''}
                              </div>
                            </div>
                            {v.preview_url && (
                              <button
                                onClick={(e) => { e.stopPropagation(); new Audio(v.preview_url).play(); }}
                                className="shrink-0 w-7 h-7 rounded-full bg-white/10 hover:bg-violet-500/30 flex items-center justify-center transition-colors"
                                title="Preview voice"
                              >
                                <Volume2 size={12} />
                              </button>
                            )}
                            {selectedVoice === v.voice_id && <Check size={14} className="text-violet-400 shrink-0" />}
                          </button>
                        ))}
                      </div>
                    );
                  }

                  // Fallback defaults by gender + language
                  const defaults = {
                    'en-female': [
                      { id: '21m00Tcm4TlvDq8ikWAM', name: 'Rachel (calm)' },
                      { id: 'EXAVITQu4vr4xnSDxMaL', name: 'Bella (soft)' },
                    ],
                    'en-male': [
                      { id: '29vD33N1CtxCmqQRPOHJ', name: 'Drew (confident)' },
                      { id: 'TxGEqnHWrfWFTfGW9XjX', name: 'Josh (deep)' },
                      { id: 'yoZ06aMxZJJ28mfd3POQ', name: 'Sam (raspy)' },
                    ],
                    'es-female': [
                      { id: 'EXAVITQu4vr4xnSDxMaL', name: 'Bella (suave)' },
                      { id: '21m00Tcm4TlvDq8ikWAM', name: 'Rachel (calmada)' },
                    ],
                    'es-male': [
                      { id: 'ErXwobaYiN019PkySvjV', name: 'Antoni (cálido)' },
                      { id: '29vD33N1CtxCmqQRPOHJ', name: 'Drew (confiado)' },
                    ],
                  };
                  const key = `${language}-${actorGender}`;
                  const opts = defaults[key] || defaults['en-female'];
                  return (
                    <select value={selectedVoice} onChange={(e) => setSelectedVoice(e.target.value)} className="input-field">
                      {opts.map((o) => <option key={o.id} value={o.id}>{o.name}</option>)}
                    </select>
                  );
                })()}
                <p className="text-[10px] text-zinc-600 mt-1">
                  {language === 'es'
                    ? `Voces ${actorGender === 'female' ? 'femeninas' : 'masculinas'} · Todas hablan español con modelo multilingual · Click altavoz para preview`
                    : `${actorGender === 'female' ? 'Female' : 'Male'} voices · Click speaker to preview`}
                </p>
              </div>

              {/* Actor Description + Mandatory Selection */}
              <div>
                <label className="block text-sm font-medium text-zinc-300 mb-2 flex items-center gap-2">
                  <User size={14} /> AI Actor — Choose Your Actor
                </label>
                <textarea
                  value={actorDescription}
                  onChange={(e) => { setActorDescription(e.target.value); setActorOptions([]); setSelectedActor(null); }}
                  rows={2}
                  className="input-field resize-none text-sm"
                  placeholder="e.g. A young woman in her late 20s, dark hair, casual outfit..."
                />
                <button
                  onClick={async () => {
                    if (!falKey || !actorDescription) return;
                    setGeneratingActors(true);
                    setActorOptions([]);
                    setSelectedActor(null);
                    try {
                      const res = await fetch(getApiUrl('/api/saasshorts/actor-options'), {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json', 'X-Fal-Key': falKey },
                        body: JSON.stringify({ actor_description: actorDescription, num_options: 3 }),
                      });
                      if (res.ok) {
                        const data = await res.json();
                        setActorOptions(data.images || []);
                      }
                    } catch (e) { console.error(e); }
                    finally { setGeneratingActors(false); }
                  }}
                  disabled={generatingActors || !falKey || !actorDescription}
                  className="mt-2 w-full text-sm bg-violet-500/20 text-violet-300 px-4 py-2.5 rounded-lg hover:bg-violet-500/30 transition-colors disabled:opacity-40 flex items-center justify-center gap-2 font-medium"
                >
                  {generatingActors ? <><Loader2 size={14} className="animate-spin" /> Generating 3 actors...</> : <><User size={14} /> {actorOptions.length > 0 ? 'Regenerate Actors' : 'Generate 3 Actor Options'} (~$0.06)</>}
                </button>

                {/* Actor Options Grid — Must select one */}
                {actorOptions.length > 0 && (
                  <div className="mt-3">
                    <p className="text-xs text-zinc-400 mb-2">Select an actor to continue:</p>
                    <div className="grid grid-cols-3 gap-3">
                      {actorOptions.map((imgUrl, i) => (
                        <button
                          key={i}
                          onClick={() => setSelectedActor(imgUrl)}
                          className={`relative rounded-xl overflow-hidden border-2 transition-all aspect-[9/16] ${
                            selectedActor === imgUrl ? 'border-violet-500 ring-2 ring-violet-500/30 scale-[1.02]' : 'border-white/10 hover:border-white/30'
                          }`}
                        >
                          <img src={getApiUrl(imgUrl)} alt={`Option ${i+1}`} className="w-full h-full object-cover" />
                          {selectedActor === imgUrl && (
                            <div className="absolute top-2 right-2 w-6 h-6 bg-violet-500 rounded-full flex items-center justify-center shadow-lg">
                              <Check size={12} className="text-white" />
                            </div>
                          )}
                          <div className="absolute bottom-0 inset-x-0 bg-gradient-to-t from-black/60 to-transparent p-2">
                            <span className="text-[10px] text-white/80">Option {i+1}</span>
                          </div>
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                {!selectedActor && actorOptions.length > 0 && (
                  <p className="text-xs text-amber-400 mt-2 flex items-center gap-1"><AlertCircle size={12} /> Select an actor to continue</p>
                )}
              </div>

              {/* Narration Edit */}
              <div>
                <label className="block text-sm font-medium text-zinc-300 mb-2 flex items-center gap-2">
                  <MessageSquare size={14} /> Narration Script
                </label>
                <textarea
                  value={editedNarration}
                  onChange={(e) => setEditedNarration(e.target.value)}
                  rows={5}
                  className="input-field resize-none font-mono text-xs"
                />
                <p className="text-[10px] text-zinc-600 mt-1">{editedNarration.length} chars &middot; ~{Math.round(editedNarration.split(' ').length / 2.5)}s speech</p>
              </div>

              {/* Cost Estimate */}
              <div className="p-3 bg-white/5 rounded-lg border border-white/10">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-zinc-400">Estimated cost</span>
                  <span className="text-green-400 font-semibold">~$2.50</span>
                </div>
                <div className="text-[10px] text-zinc-600 mt-1">
                  Flux image ($0.05) + ElevenLabs voice ($0.10) + Kling avatar ($1.69) + Kling b-roll ($0.70)
                </div>
              </div>

              {/* Missing keys warning */}
              {(!falKey || !elevenLabsKey) && (
                <div className="p-3 bg-amber-500/10 border border-amber-500/20 rounded-lg flex items-center gap-2 text-sm text-amber-400">
                  <AlertCircle size={14} />
                  {!falKey && 'fal.ai API key missing. '}{!elevenLabsKey && 'ElevenLabs API key missing. '}
                  Set them in Settings.
                </div>
              )}
            </div>

            <div className="flex justify-between">
              <button onClick={() => setStep(1)} className="btn-secondary px-4 py-2 text-sm flex items-center gap-2">
                <ChevronLeft size={14} /> Back
              </button>
              <button
                onClick={handleGenerate}
                disabled={!falKey || !elevenLabsKey || !selectedActor || generating}
                className="btn-primary px-6 py-2 text-sm flex items-center gap-2 disabled:opacity-50"
              >
                {generating ? (
                  <><Loader2 size={14} className="animate-spin" /> Generating...</>
                ) : !selectedActor ? (
                  <><User size={14} /> Select an actor first</>
                ) : (
                  <><Film size={14} /> Generate Video (~$2.00)</>
                )}
              </button>
            </div>
          </div>
        )}

        {/* ── Step 3: Generation Progress ──────────────────────── */}
        {step === 3 && (
          <div className="animate-[fadeIn_0.3s_ease-out] space-y-6">
            <div className="glass-panel p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold flex items-center gap-2">
                  <Film size={18} className={genStatus === 'processing' ? 'text-violet-400 animate-pulse' : genStatus === 'completed' ? 'text-green-400' : 'text-red-400'} />
                  Video Generation
                </h2>
                <span className={`text-xs px-2 py-1 rounded-full border ${
                  genStatus === 'processing' ? 'bg-violet-500/10 border-violet-500/20 text-violet-300' :
                  genStatus === 'completed' ? 'bg-green-500/10 border-green-500/20 text-green-400' :
                  'bg-red-500/10 border-red-500/20 text-red-400'
                }`}>
                  {genStatus.toUpperCase()}
                </span>
              </div>

              {/* Progress steps */}
              <div className="space-y-2 mb-4">
                {[
                  'Generating actor image + voiceover',
                  'Creating talking head video (2-5 min)',
                  'Generating b-roll clips',
                  'Compositing final video',
                ].map((label, i) => {
                  const logStr = genLogs.join(' ').toLowerCase();
                  const stepDone =
                    i === 0 ? logStr.includes('[2/6]') || logStr.includes('[3/6]') :
                    i === 1 ? logStr.includes('[3/6]') && (logStr.includes('[4/6]') || logStr.includes('talking head ready')) :
                    i === 2 ? logStr.includes('[5/6]') || logStr.includes('[6/6]') :
                    genStatus === 'completed';
                  const stepActive =
                    i === 0 ? logStr.includes('[1/6]') && !stepDone :
                    i === 1 ? (logStr.includes('[3/6]') && !logStr.includes('[4/6]')) :
                    i === 2 ? (logStr.includes('[4/6]') && !logStr.includes('[5/6]') && !logStr.includes('[6/6]')) :
                    logStr.includes('[6/6]') && genStatus !== 'completed';

                  return (
                    <div key={i} className="flex items-center gap-3 text-sm">
                      {stepDone ? (
                        <Check size={14} className="text-green-400" />
                      ) : stepActive ? (
                        <Loader2 size={14} className="text-violet-400 animate-spin" />
                      ) : (
                        <div className="w-3.5 h-3.5 rounded-full border border-white/20" />
                      )}
                      <span className={stepDone ? 'text-zinc-400' : stepActive ? 'text-white' : 'text-zinc-600'}>
                        {label}
                      </span>
                    </div>
                  );
                })}
              </div>

              {/* Logs Terminal */}
              <div className="bg-[#0c0c0e] rounded-xl border border-white/10 overflow-hidden">
                <div className="px-4 py-2 border-b border-white/5 flex items-center justify-between bg-white/5">
                  <span className="text-xs font-mono text-zinc-400 flex items-center gap-2">
                    <Terminal size={12} /> Generation Logs
                  </span>
                  <button onClick={() => setLogsExpanded(!logsExpanded)} className="text-zinc-500 hover:text-white transition-colors">
                    <ChevronDown size={14} className={logsExpanded ? '' : 'rotate-180'} />
                  </button>
                </div>
                {logsExpanded && (
                  <div className="p-4 max-h-64 overflow-y-auto font-mono text-xs space-y-1 custom-scrollbar">
                    {genLogs.map((log, i) => (
                      <div key={i} className={`${log.toLowerCase().includes('error') ? 'text-red-400' : log.includes('✅') ? 'text-green-400' : 'text-zinc-400'}`}>
                        {log}
                      </div>
                    ))}
                    {genStatus === 'processing' && (
                      <div className="animate-pulse text-violet-400/70">_</div>
                    )}
                  </div>
                )}
              </div>

              {/* Retry button when failed */}
              {genStatus === 'failed' && (
                <div className="mt-4 p-4 bg-red-500/5 border border-red-500/20 rounded-xl space-y-3">
                  <div className="flex items-center gap-2">
                    <AlertCircle size={16} className="text-red-400 shrink-0" />
                    <span className="text-sm text-red-300">Generation failed. You can retry or go back to change settings.</span>
                  </div>
                  <div className="flex gap-3">
                    <button
                      onClick={() => { setStep(2); setGenStatus('idle'); setGenerating(false); }}
                      className="btn-secondary px-4 py-2 text-sm flex items-center gap-2"
                    >
                      <ChevronLeft size={14} /> Change Voice/Settings
                    </button>
                    <button
                      onClick={handleRetry}
                      disabled={generating}
                      className="btn-primary px-4 py-2 text-sm flex items-center gap-2"
                    >
                      <RefreshCw size={14} /> Retry
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* ── Step 4: Results ──────────────────────────────────── */}
        {step === 4 && genResult && (
          <div className="animate-[fadeIn_0.3s_ease-out] space-y-6">
            <div className="glass-panel p-6">
              <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <Sparkles className="text-yellow-400" size={18} />
                Your SaaS Short is Ready!
              </h2>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Video Player */}
                <div className="aspect-[9/16] max-h-[500px] bg-black rounded-xl overflow-hidden relative">
                  <video
                    src={getApiUrl(genResult.video_url)}
                    controls
                    className="w-full h-full object-contain"
                    autoPlay
                  />
                </div>

                {/* Details */}
                <div className="space-y-4">
                  <div>
                    <h3 className="text-sm font-medium text-zinc-300 mb-1">{genResult.script?.title}</h3>
                    <p className="text-xs text-zinc-500">{genResult.duration?.toFixed(1)}s &middot; 9:16 vertical</p>
                  </div>

                  {/* Cost breakdown */}
                  {genResult.cost_estimate && (
                    <div className="p-3 bg-white/5 rounded-lg border border-white/10 space-y-1">
                      <div className="text-xs font-semibold text-zinc-300 mb-2">Cost Breakdown</div>
                      {Object.entries(genResult.cost_estimate).filter(([k]) => k !== 'total').map(([k, v]) => (
                        <div key={k} className="flex justify-between text-xs">
                          <span className="text-zinc-500">{k.replace(/_/g, ' ')}</span>
                          <span className="text-zinc-400">${v}</span>
                        </div>
                      ))}
                      <div className="flex justify-between text-sm font-semibold border-t border-white/10 pt-1 mt-1">
                        <span className="text-zinc-300">Total</span>
                        <span className="text-green-400">${genResult.cost_estimate.total}</span>
                      </div>
                    </div>
                  )}

                  {/* Caption */}
                  {genResult.script?.caption && (
                    <div>
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-xs font-medium text-zinc-400">Caption</span>
                        <button
                          onClick={() => handleCopy(genResult.script.caption, 'caption')}
                          className="text-xs text-zinc-500 hover:text-white flex items-center gap-1"
                        >
                          {copied === 'caption' ? <Check size={10} /> : <Copy size={10} />}
                          {copied === 'caption' ? 'Copied' : 'Copy'}
                        </button>
                      </div>
                      <p className="text-xs text-zinc-400 bg-white/5 p-2 rounded-lg">{genResult.script.caption}</p>
                    </div>
                  )}

                  {/* Hashtags */}
                  {genResult.script?.hashtags && (
                    <div>
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-xs font-medium text-zinc-400">Hashtags</span>
                        <button
                          onClick={() => handleCopy(genResult.script.hashtags.join(' '), 'hashtags')}
                          className="text-xs text-zinc-500 hover:text-white flex items-center gap-1"
                        >
                          {copied === 'hashtags' ? <Check size={10} /> : <Copy size={10} />}
                          {copied === 'hashtags' ? 'Copied' : 'Copy'}
                        </button>
                      </div>
                      <div className="flex flex-wrap gap-1.5">
                        {genResult.script.hashtags.map((tag, i) => (
                          <span key={i} className="text-[10px] bg-violet-500/10 text-violet-300 px-2 py-0.5 rounded-full">{tag}</span>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Actions */}
                  <div className="flex gap-3 pt-2">
                    <a
                      href={getApiUrl(genResult.video_url)}
                      download
                      className="btn-primary px-4 py-2 text-sm flex items-center gap-2"
                    >
                      <Download size={14} /> Download
                    </a>
                    <button
                      onClick={handleReset}
                      className="btn-secondary px-4 py-2 text-sm flex items-center gap-2"
                    >
                      <RefreshCw size={14} /> New Video
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
