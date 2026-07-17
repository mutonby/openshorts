import { useState, useRef, useCallback } from 'react';
import { Upload, Image, Loader2, Send, Check, Download, ArrowRight, ArrowLeft, Sparkles, Video, Type, X, Plus, MessageSquare, FileText, Youtube, AlertCircle, Settings } from 'lucide-react';
import { getApiUrl } from '../config';
import { apiFetch } from '../lib/api';
import StepIndicator from './ui/StepIndicator';
import SegmentedControl from './ui/SegmentedControl';

const STEPS = ['Input', 'Titles', 'Generate', 'Description', 'Publish'];

function DragDropZone({ label, accept, onFile, file, onClear, icon }) {
  const Icon = icon;
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef(null);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) onFile(f);
  }, [onFile]);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  if (file) {
    return (
      <div className="relative border border-rule2 rounded-card p-3 bg-paper3">
        <div className="flex items-center gap-3">
          {file.type?.startsWith('image/') ? (
            <img src={URL.createObjectURL(file)} className="w-12 h-12 rounded-input object-cover" alt="" />
          ) : (
            <div className="w-12 h-12 rounded-input bg-paper2 border border-rule flex items-center justify-center">
              <Icon size={18} className="text-muted" />
            </div>
          )}
          <div className="flex-1 min-w-0">
            <p className="text-sm text-ink truncate">{file.name}</p>
            <p className="readout mt-0.5">{(file.size / 1024 / 1024).toFixed(1)} MB</p>
          </div>
          <button onClick={onClear} className="text-muted hover:text-ink transition-colors">
            <X size={16} />
          </button>
        </div>
      </div>
    );
  }

  return (
    <div
      onClick={() => inputRef.current?.click()}
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={() => setIsDragging(false)}
      className={`border-2 border-dashed rounded-card p-6 text-center cursor-pointer transition-colors duration-200 ${isDragging ? 'border-brass bg-paper3' : 'border-rule2 hover:border-brass'
        }`}
    >
      <Icon size={18} className="mx-auto text-muted mb-2" />
      <p className="text-sm text-ink2 lowercase">{label}</p>
      <p className="text-xs text-muted mt-1 lowercase">Drop or click to upload</p>
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        className="hidden"
        onChange={(e) => e.target.files[0] && onFile(e.target.files[0])}
      />
    </div>
  );
}

export default function ThumbnailStudio({ geminiApiKey, uploadPostKey, uploadUserId, managed = false }) {
  // Managed (hosted plan): Gemini runs server-side via the bearer token, no BYOK key.
  // Only send X-Gemini-Key for self-host BYOK. apiFetch attaches the bearer token.
  const keyHeader = geminiApiKey ? { 'X-Gemini-Key': geminiApiKey } : {};
  const needsKey = !geminiApiKey && !managed;
  // Step management
  const [step, setStep] = useState(0);
  const [mode, setMode] = useState(null); // 'video' or 'manual'

  // Step 1 state
  const [videoFile, setVideoFile] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  // Step 2 state
  const [sessionId, setSessionId] = useState(null);
  const [titles, setTitles] = useState([]);
  const [selectedTitle, setSelectedTitle] = useState('');
  const [manualTitle, setManualTitle] = useState('');
  const [chatInput, setChatInput] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [isRefining, setIsRefining] = useState(false);
  const [recommended, setRecommended] = useState([]); // [{index, reason}]

  // Step 3 state
  const [faceImage, setFaceImage] = useState(null);
  const [bgImage, setBgImage] = useState(null);
  const [extraPrompt, setExtraPrompt] = useState('');
  const [thumbnailCount, setThumbnailCount] = useState(3);
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedThumbnails, setGeneratedThumbnails] = useState([]);

  // Description state
  const [description, setDescription] = useState('');
  const [isDescribing, setIsDescribing] = useState(false);

  // Step 4 (Publish) state
  const [selectedThumbnail, setSelectedThumbnail] = useState(null);
  const [isPublishing, setIsPublishing] = useState(false);
  const [publishResult, setPublishResult] = useState(null);

  // Background preprocessing state
  const [preprocessSessionId, setPreprocessSessionId] = useState(null);
  const [isPreprocessing, setIsPreprocessing] = useState(false);

  const chatEndRef = useRef(null);

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  // --- Background Pre-upload (starts Whisper immediately) ---
  const handlePreUpload = async (file) => {
    setPreprocessSessionId(null);
    setIsPreprocessing(true);
    try {
      const formData = new FormData();
      formData.append('file', file);

      const res = await apiFetch('/api/thumbnail/upload', {
        method: 'POST',
        body: formData
      });

      if (res.ok) {
        const data = await res.json();
        setPreprocessSessionId(data.session_id);
        console.log(`🎙️ Background Whisper started: ${data.session_id}`);
      }
    } catch (e) {
      console.error('Pre-upload failed:', e);
    } finally {
      setIsPreprocessing(false);
    }
  };

  // --- Step 1: Analyze Video ---
  const handleAnalyze = async () => {
    if (needsKey) return alert('Please set your Gemini API key in Settings first.');
    setIsAnalyzing(true);

    try {
      const formData = new FormData();

      if (preprocessSessionId) {
        // Use pre-uploaded session (Whisper already running/done in background)
        formData.append('session_id', preprocessSessionId);
      } else if (videoFile) {
        formData.append('file', videoFile);
      } else {
        return alert('Please upload a video file.');
      }

      const res = await apiFetch('/api/thumbnail/analyze', {
        method: 'POST',
        headers: keyHeader,
        body: formData
      });

      if (!res.ok) {
        const err = await res.text();
        throw new Error(err);
      }

      const data = await res.json();
      setSessionId(data.session_id);
      setTitles(data.titles || []);
      setRecommended(data.recommended || []);
      setChatHistory([{
        role: 'assistant',
        content: `Here are 10 viral title suggestions based on your video. Titles marked TOP PICK are my top picks. Click one to select it, or tell me how to refine them.`
      }]);
      setStep(1);
    } catch (e) {
      alert(`Analysis failed: ${e.message}`);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleManualMode = () => {
    setMode('manual');
    setStep(1);
  };

  // --- Step 2: Title Selection / Refinement ---
  const handleSelectTitle = (title) => {
    setSelectedTitle(title);
  };

  const handleConfirmTitle = () => {
    if (mode === 'manual' && manualTitle) {
      setSelectedTitle(manualTitle);
      // Create session for manual mode
      const newSessionId = sessionId || crypto.randomUUID();
      setSessionId(newSessionId);
      apiFetch('/api/thumbnail/titles', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...keyHeader
        },
        body: JSON.stringify({ title: manualTitle, session_id: newSessionId })
      }).catch(() => { });
    }
    if (selectedTitle || (mode === 'manual' && manualTitle)) {
      setStep(2);
    }
  };

  const handleRefine = async () => {
    if (!chatInput.trim() || !sessionId) return;
    setIsRefining(true);

    const userMsg = chatInput.trim();
    setChatInput('');
    setChatHistory(prev => [...prev, { role: 'user', content: userMsg }]);

    try {
      const res = await apiFetch('/api/thumbnail/titles', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...keyHeader
        },
        body: JSON.stringify({ session_id: sessionId, message: userMsg })
      });

      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setTitles(data.titles || []);
      setChatHistory(prev => [...prev, {
        role: 'assistant',
        content: `Here are refined titles based on your feedback. Click one to select it.`
      }]);
      setTimeout(scrollToBottom, 100);
    } catch (e) {
      setChatHistory(prev => [...prev, {
        role: 'assistant',
        content: `Failed to refine: ${e.message}`
      }]);
    } finally {
      setIsRefining(false);
    }
  };

  // --- Step 3: Generate Thumbnails ---
  const handleGenerate = async () => {
    if (needsKey) return alert('Please set your Gemini API key in Settings first.');
    const finalTitle = selectedTitle || manualTitle;
    if (!finalTitle) return alert('Please select or enter a title first.');

    setIsGenerating(true);
    setGeneratedThumbnails([]);

    try {
      const formData = new FormData();
      formData.append('session_id', sessionId || 'manual');
      formData.append('title', finalTitle);
      formData.append('extra_prompt', extraPrompt);
      formData.append('count', thumbnailCount);
      if (faceImage) formData.append('face', faceImage);
      if (bgImage) formData.append('background', bgImage);

      const res = await apiFetch('/api/thumbnail/generate', {
        method: 'POST',
        headers: keyHeader,
        body: formData
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => null);
        throw new Error(errData?.detail || `Server error ${res.status}`);
      }

      const data = await res.json();
      if (!data.thumbnails || data.thumbnails.length === 0) {
        throw new Error('No thumbnails were generated. Your Gemini API key may not have access to image generation.');
      }
      setGeneratedThumbnails(data.thumbnails);
    } catch (e) {
      alert(`Generation failed: ${e.message}`);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleDownload = async (url) => {
    try {
      const response = await fetch(getApiUrl(url));
      const blob = await response.blob();
      const blobUrl = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = blobUrl;
      a.download = url.split('/').pop() || 'thumbnail.png';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(blobUrl);
    } catch {
      // Fallback: open in new tab if fetch fails
      window.open(getApiUrl(url), '_blank');
    }
  };

  // --- Description Generation ---
  const handleGenerateDescription = async () => {
    if (needsKey) return alert('Please set your Gemini API key in Settings first.');
    const finalTitle = selectedTitle || manualTitle;
    if (!finalTitle) return alert('Please select a title first.');
    if (!sessionId) return alert('No session available.');

    setIsDescribing(true);
    try {
      const res = await apiFetch('/api/thumbnail/describe', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...keyHeader
        },
        body: JSON.stringify({ session_id: sessionId, title: finalTitle })
      });

      if (!res.ok) {
        const err = await res.text();
        throw new Error(err);
      }

      const data = await res.json();
      setDescription(data.description || '');
    } catch (e) {
      alert(`Description generation failed: ${e.message}`);
    } finally {
      setIsDescribing(false);
    }
  };

  // --- Publish to YouTube ---
  const handlePublish = async () => {
    if (!managed && (!uploadPostKey || !uploadUserId)) return alert('Please configure your Upload-Post API key and user in Settings first.');
    const finalTitle = selectedTitle || manualTitle;
    if (!finalTitle) return alert('No title selected.');
    if (!selectedThumbnail) return alert('Please select a thumbnail first.');
    if (!description) return alert('Please generate or write a description first.');

    setIsPublishing(true);
    setPublishResult(null);
    try {
      const formData = new FormData();
      formData.append('session_id', sessionId);
      formData.append('title', finalTitle);
      formData.append('description', description);
      formData.append('thumbnail_url', selectedThumbnail);
      formData.append('api_key', uploadPostKey);
      formData.append('user_id', uploadUserId);

      // Submit the publish job — returns immediately with a publish_id
      const res = await apiFetch('/api/thumbnail/publish', {
        method: 'POST',
        body: formData
      });

      if (!res.ok) {
        const err = await res.text();
        throw new Error(err);
      }

      const { publish_id } = await res.json();

      // Poll for status every 2 seconds (upload can take minutes for large videos)
      await new Promise((resolve, reject) => {
        const interval = setInterval(async () => {
          try {
            const statusRes = await fetch(getApiUrl(`/api/thumbnail/publish/status/${publish_id}`));
            if (!statusRes.ok) { clearInterval(interval); reject(new Error('Status check failed')); return; }
            const statusData = await statusRes.json();

            if (statusData.status === 'done') {
              clearInterval(interval);
              setPublishResult({ success: true, data: statusData.result });
              resolve();
            } else if (statusData.status === 'failed') {
              clearInterval(interval);
              reject(new Error(statusData.error || 'Upload failed'));
            }
            // 'uploading' → keep polling
          } catch (e) {
            clearInterval(interval);
            reject(e);
          }
        }, 2000);
      });

    } catch (e) {
      setPublishResult({ success: false, error: e.message });
    } finally {
      setIsPublishing(false);
    }
  };

  const handleReset = () => {
    setStep(0);
    setMode(null);
    setVideoFile(null);
    setSessionId(null);
    setTitles([]);
    setSelectedTitle('');
    setManualTitle('');
    setChatInput('');
    setChatHistory([]);
    setFaceImage(null);
    setBgImage(null);
    setExtraPrompt('');
    setGeneratedThumbnails([]);
    setDescription('');
    setIsDescribing(false);
    setSelectedThumbnail(null);
    setIsPublishing(false);
    setPublishResult(null);
    setPreprocessSessionId(null);
    setIsPreprocessing(false);
    setRecommended([]);
  };

  return (
    <div className="h-full overflow-y-auto p-6 md:p-8 animate-fade">
      <div className="max-w-5xl mx-auto">
        {/* Header */}
        <div className="flex items-end justify-between mb-2">
          <div>
            <p className="eyebrow mb-2">05 · YOUTUBE STUDIO</p>
            <h1 className="font-display lowercase text-2xl text-ink flex items-center gap-3">
              <span className="w-10 h-10 rounded-card bg-paper3 flex items-center justify-center">
                <Image size={18} className="text-brass" />
              </span>
              YouTube Studio
            </h1>
          </div>
          {step > 0 && (
            <button onClick={handleReset} className="text-xs lowercase text-muted hover:text-ink transition-colors flex items-center gap-1">
              <Plus size={12} /> New Project
            </button>
          )}
        </div>
        <p className="text-sm lowercase text-muted mb-6">Generate viral titles, AI thumbnails, descriptions and publish directly to YouTube</p>

        <div className="mb-8">
          <StepIndicator steps={STEPS} current={step} />
        </div>

        {/* Gemini API Key Warning (self-host BYOK only; managed uses server key) */}
        {needsKey && (
          <div className="mb-6 p-5 bg-warn/10 rounded-card flex items-start gap-3">
            <AlertCircle size={18} className="text-warn shrink-0 mt-0.5" />
            <div>
              <p className="text-sm font-medium text-warn lowercase">Gemini API Key Required</p>
              <p className="text-xs text-muted mt-1">YouTube Studio requires a Google Gemini API key to function. Please configure it in the <strong>Settings</strong> tab before using this feature. Gemini's free tier includes 1,500 requests per day.</p>
            </div>
          </div>
        )}

        {/* ===== STEP 0: Input Mode Selection ===== */}
        {step === 0 && (
          <div className={`grid md:grid-cols-2 gap-6 ${needsKey ? 'opacity-50 pointer-events-none select-none' : ''}`}>
            {/* Mode A: Video Analysis */}
            <div className="card card-hover p-6 space-y-4">
              <div className="flex items-center gap-3 mb-2">
                <div className="w-8 h-8 rounded-input bg-paper3 flex items-center justify-center">
                  <Video size={16} className="text-brass" />
                </div>
                <div>
                  <p className="eyebrow">A · ANALYZE VIDEO</p>
                  <p className="text-xs text-muted mt-0.5">AI suggests viral titles from your content</p>
                </div>
              </div>

              <DragDropZone
                label="Upload video file"
                accept="video/*"
                onFile={(f) => { setVideoFile(f); setMode('video'); handlePreUpload(f); }}
                file={videoFile}
                onClear={() => { setVideoFile(null); setPreprocessSessionId(null); }}
                icon={Video}
              />

              {isPreprocessing && (
                <div className="flex items-center gap-2 text-xs lowercase text-muted bg-paper3 rounded-input px-3 py-2">
                  <Loader2 size={12} className="animate-spin text-brass" />
                  Pre-processing video (Whisper transcription starting)...
                </div>
              )}
              {preprocessSessionId && !isPreprocessing && (
                <div className="flex items-center gap-2 text-xs lowercase text-ok bg-ok/10 rounded-input px-3 py-2">
                  <Check size={12} />
                  Video uploaded — transcription running in background
                </div>
              )}

              <button
                onClick={handleAnalyze}
                disabled={isAnalyzing || !videoFile}
                className="w-full btn-primary"
              >
                {isAnalyzing ? (
                  <>
                    <Loader2 size={16} className="animate-spin" />
                    Analyzing video...
                  </>
                ) : (
                  <>
                    <Sparkles size={16} className="hidden sm:block" />
                    <span className="whitespace-nowrap">Analyze & Get Titles</span>
                  </>
                )}
              </button>
            </div>

            {/* Mode B: Manual Title */}
            <div className="card card-hover p-6 space-y-4 flex flex-col">
              <div className="flex items-center gap-3 mb-2">
                <div className="w-8 h-8 rounded-input bg-paper3 flex items-center justify-center">
                  <Type size={16} className="text-brass" />
                </div>
                <div>
                  <p className="eyebrow">B · WRITE YOUR OWN</p>
                  <p className="text-xs text-muted mt-0.5">Skip analysis, enter your title directly</p>
                </div>
              </div>

              <div className="flex-1 flex flex-col justify-center">
                <input
                  type="text"
                  value={manualTitle}
                  onChange={(e) => setManualTitle(e.target.value)}
                  placeholder="Enter your YouTube title..."
                  className="input-field text-sm mb-4"
                  maxLength={70}
                />
                <p className="readout mb-4">{manualTitle.length} / 70</p>
              </div>

              <button
                onClick={handleManualMode}
                disabled={!manualTitle.trim()}
                className="w-full btn-ghost disabled:opacity-45 disabled:cursor-not-allowed"
              >
                <ArrowRight size={16} />
                Use This Title
              </button>
            </div>
          </div>
        )}

        {/* ===== STEP 1: Title Selection ===== */}
        {step === 1 && (
          <div className="grid md:grid-cols-5 gap-6">
            {/* Left: Chat / Controls */}
            <div className="md:col-span-2 flex flex-col gap-4">
              {mode === 'manual' ? (
                <div className="card p-6 space-y-4">
                  <p className="eyebrow">YOUR TITLE</p>
                  <input
                    type="text"
                    value={manualTitle}
                    onChange={(e) => setManualTitle(e.target.value)}
                    className="input-field text-sm"
                    maxLength={70}
                  />
                  <p className="readout">{manualTitle.length} / 70</p>
                  <button
                    onClick={handleConfirmTitle}
                    disabled={!manualTitle.trim()}
                    className="w-full btn-primary"
                  >
                    <ArrowRight size={16} />
                    Continue to Thumbnails
                  </button>
                </div>
              ) : (
                <div className="card p-4 flex flex-col h-[500px]">
                  <div className="flex items-center gap-2 mb-3 pb-3 border-b border-rule">
                    <MessageSquare size={14} className="text-brass" />
                    <span className="eyebrow">TITLE REFINEMENT CHAT</span>
                  </div>

                  {/* Chat messages */}
                  <div className="flex-1 overflow-y-auto space-y-3 custom-scrollbar mb-3">
                    {chatHistory.map((msg, i) => (
                      <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                        <div className={`max-w-[90%] px-3 py-2 rounded-card text-xs ${msg.role === 'user'
                          ? 'bg-paper3 text-ink2'
                          : 'border border-rule text-ink2'
                          }`}>
                          {msg.content}
                        </div>
                      </div>
                    ))}
                    <div ref={chatEndRef} />
                  </div>

                  {/* Chat input */}
                  <div className="flex gap-2">
                    <input
                      type="text"
                      value={chatInput}
                      onChange={(e) => setChatInput(e.target.value)}
                      onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && handleRefine()}
                      placeholder="Make them more clickbait..."
                      className="input-field text-xs flex-1"
                      disabled={isRefining}
                    />
                    <button
                      onClick={handleRefine}
                      disabled={isRefining || !chatInput.trim()}
                      className="btn-quiet px-3 disabled:opacity-45 disabled:cursor-not-allowed"
                    >
                      {isRefining ? <Loader2 size={14} className="animate-spin text-brass" /> : <Send size={14} />}
                    </button>
                  </div>
                </div>
              )}

              {mode !== 'manual' && selectedTitle && (
                <button
                  onClick={handleConfirmTitle}
                  className="w-full btn-primary"
                >
                  <ArrowRight size={16} />
                  Use Selected Title
                </button>
              )}
            </div>

            {/* Right: Title Cards */}
            <div className="md:col-span-3 space-y-3">
              {selectedTitle && (
                <div className="p-3 bg-ok/10 rounded-card flex items-center gap-2 text-sm">
                  <Check size={14} className="text-ok shrink-0" />
                  <span className="text-ok font-medium truncate">Selected: {selectedTitle}</span>
                </div>
              )}

              {titles.length > 0 && (
                <div className="space-y-2">
                  {titles.map((title, i) => {
                    const rec = recommended.find(r => r.index === i);
                    const recRank = recommended.findIndex(r => r.index === i);
                    return (
                      <button
                        key={i}
                        onClick={() => handleSelectTitle(title)}
                        className={`w-full text-left p-4 rounded-card border transition-colors duration-200 text-sm ${selectedTitle === title
                          ? 'bg-paper3 border-brass text-ink'
                          : 'border-rule text-ink2 hover:bg-paper3 hover:border-rule2'
                          }`}
                      >
                        <div className="flex items-start gap-3">
                          <span className={`w-6 h-6 rounded-full border flex items-center justify-center font-mono text-micro shrink-0 mt-0.5 ${selectedTitle === title ? 'bg-brass border-brass text-brassink' :
                            rec ? 'border-rule2 text-brass' :
                              'border-rule text-muted'
                            }`}>
                            {selectedTitle === title ? <Check size={10} /> : rec ? '★' : i + 1}
                          </span>
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2 flex-wrap">
                              <span className="leading-relaxed">{title}</span>
                              {rec && (
                                <span className="badge-brass shrink-0">
                                  {recRank === 0 ? 'TOP PICK' : '2ND PICK'}
                                </span>
                              )}
                            </div>
                            {rec && (
                              <p className="text-xs text-muted mt-1.5 leading-relaxed">{rec.reason}</p>
                            )}
                          </div>
                        </div>
                      </button>
                    );
                  })}
                </div>
              )}

              {isRefining && (
                <div className="flex items-center justify-center py-8 text-muted">
                  <Loader2 size={18} className="animate-spin mr-2 text-brass" />
                  <span className="text-sm lowercase">Refining titles...</span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* ===== STEP 2: Thumbnail Generation ===== */}
        {step === 2 && (
          <div className="grid md:grid-cols-5 gap-6">
            {/* Left: Controls */}
            <div className="md:col-span-2 space-y-4">
              <div className="card p-6 space-y-4">
                <p className="eyebrow mb-1">TITLE</p>
                <div className="p-3 bg-paper3 border border-rule rounded-input text-sm text-ink">
                  {selectedTitle || manualTitle}
                </div>

                <button
                  onClick={() => setStep(1)}
                  className="text-xs lowercase text-muted hover:text-ink transition-colors flex items-center gap-1"
                >
                  <ArrowLeft size={12} /> Change title
                </button>
              </div>

              <div className="card p-6 space-y-4">
                <p className="eyebrow">FACE IMAGE · OPTIONAL</p>
                <DragDropZone
                  label="Upload face / person photo"
                  accept="image/*"
                  onFile={setFaceImage}
                  file={faceImage}
                  onClear={() => setFaceImage(null)}
                  icon={Upload}
                />
              </div>

              <div className="card p-6 space-y-4">
                <p className="eyebrow">BACKGROUND · OPTIONAL</p>
                <DragDropZone
                  label="Upload background image"
                  accept="image/*"
                  onFile={setBgImage}
                  file={bgImage}
                  onClear={() => setBgImage(null)}
                  icon={Image}
                />
              </div>

              <div className="card p-6 space-y-4">
                <p className="eyebrow">INSTRUCTIONS · OPTIONAL</p>
                <textarea
                  value={extraPrompt}
                  onChange={(e) => setExtraPrompt(e.target.value)}
                  placeholder="e.g. Use red and black colors, dramatic lighting, include money emojis..."
                  className="input-field text-sm resize-none h-20"
                />
              </div>

              <div className="card p-6 space-y-4">
                <p className="eyebrow">COUNT</p>
                <SegmentedControl
                  options={[1, 2, 3, 4].map(n => ({ value: n, label: String(n) }))}
                  value={thumbnailCount}
                  onChange={setThumbnailCount}
                  size="sm"
                />
              </div>

              <button
                onClick={handleGenerate}
                disabled={isGenerating}
                className="w-full btn-primary"
              >
                {isGenerating ? (
                  <>
                    <Loader2 size={16} className="animate-spin" />
                    Generating thumbnails...
                  </>
                ) : (
                  <>
                    <Sparkles size={16} />
                    Generate Thumbnails
                  </>
                )}
              </button>

            </div>

            {/* Right: Generated Thumbnails */}
            <div className="md:col-span-3">
              {generatedThumbnails.length > 0 ? (
                <div className="space-y-4">
                  <p className="text-sm lowercase text-muted">Generated Thumbnails — click to select for publishing</p>
                  <div className="grid gap-4">
                    {generatedThumbnails.map((url, i) => (
                      <div
                        key={i}
                        onClick={() => setSelectedThumbnail(url)}
                        className={`glass-panel overflow-hidden group relative cursor-pointer transition-colors duration-200 ${selectedThumbnail === url ? 'border-2 border-brass' : ''
                          }`}
                      >
                        <img
                          src={getApiUrl(url)}
                          alt={`Thumbnail ${i + 1}`}
                          className="w-full aspect-video object-cover"
                        />
                        <div className="absolute inset-0 bg-black/0 group-hover:bg-black/40 transition-all flex items-center justify-center gap-3 opacity-0 group-hover:opacity-100">
                          <button
                            onClick={(e) => { e.stopPropagation(); handleDownload(url); }}
                            className="btn-quiet"
                          >
                            <Download size={14} />
                            Download
                          </button>
                        </div>
                        <div className="p-3 flex items-center justify-between">
                          <span className="text-xs lowercase text-muted flex items-center gap-2">
                            Thumbnail {i + 1}
                            {selectedThumbnail === url && (
                              <span className="text-brass flex items-center gap-1"><Check size={10} /> Selected</span>
                            )}
                          </span>
                          <button
                            onClick={(e) => { e.stopPropagation(); handleDownload(url); }}
                            className="text-xs lowercase text-muted hover:text-ink transition-colors flex items-center gap-1"
                          >
                            <Download size={12} /> Save
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* Regenerate */}
                  <button
                    onClick={handleGenerate}
                    disabled={isGenerating}
                    className="w-full btn-ghost"
                  >
                    {isGenerating ? (
                      <>
                        <Loader2 size={14} className="animate-spin text-brass" />
                        Regenerating...
                      </>
                    ) : (
                      <>
                        <Sparkles size={14} />
                        Regenerate
                      </>
                    )}
                  </button>

                  {/* Proceed to Description */}
                  {selectedThumbnail && (
                    <button
                      onClick={() => setStep(3)}
                      className="w-full btn-primary"
                    >
                      <ArrowRight size={16} />
                      Next: Description
                    </button>
                  )}
                </div>
              ) : isGenerating ? (
                <div className="h-full flex flex-col items-center justify-center text-muted space-y-4 min-h-[400px]">
                  <div className="w-16 h-16 rounded-full border-2 border-rule2 border-t-brass animate-spin" />
                  <div className="text-center">
                    <p className="text-sm lowercase font-medium text-ink2">Generating thumbnails...</p>
                    <p className="text-xs lowercase text-muted mt-1">This may take a minute per thumbnail</p>
                  </div>
                </div>
              ) : (
                <div className="h-full flex flex-col items-center justify-center text-muted space-y-4 min-h-[400px]">
                  <div className="w-20 h-20 rounded-card bg-paper3 border border-rule flex items-center justify-center">
                    <Image size={28} className="text-muted" />
                  </div>
                  <div className="text-center">
                    <p className="text-sm lowercase text-ink2">Your thumbnails will appear here</p>
                    <p className="text-xs lowercase text-muted mt-1">Configure options and click Generate</p>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* ===== STEP 3: YouTube Description ===== */}
        {step === 3 && (
          <div className="grid md:grid-cols-5 gap-6">
            {/* Left: Context & Controls */}
            <div className="md:col-span-2 space-y-4">
              <button
                onClick={() => setStep(2)}
                className="text-xs lowercase text-muted hover:text-ink transition-colors flex items-center gap-1 mb-2"
              >
                <ArrowLeft size={12} /> Back to Generate
              </button>

              {/* Selected Thumbnail Preview */}
              {selectedThumbnail && (
                <div className="glass-panel overflow-hidden">
                  <img
                    src={getApiUrl(selectedThumbnail)}
                    alt="Selected thumbnail"
                    className="w-full aspect-video object-cover"
                  />
                  <div className="p-3">
                    <span className="text-xs lowercase text-brass flex items-center gap-1"><Check size={10} /> Selected Thumbnail</span>
                  </div>
                </div>
              )}

              {/* Title */}
              <div className="glass-panel p-6 space-y-3">
                <p className="eyebrow">TITLE</p>
                <div className="p-3 bg-paper3 border border-rule rounded-input text-sm text-ink">
                  {selectedTitle || manualTitle}
                </div>
              </div>

              {/* Generate Description Button */}
              {mode === 'video' && (
                <div className="glass-panel p-6 space-y-4">
                  <div className="flex items-center justify-between">
                    <p className="eyebrow flex items-center gap-2">
                      <Sparkles size={14} className="text-brass" />
                      AI DESCRIPTION
                    </p>
                    <span className="readout">WITH CHAPTERS</span>
                  </div>
                  <p className="text-xs text-muted">
                    Generate a YouTube description with chapter timestamps from your video transcript.
                  </p>
                  <button
                    onClick={handleGenerateDescription}
                    disabled={isDescribing}
                    className="w-full btn-ghost"
                  >
                    {isDescribing ? (
                      <>
                        <Loader2 size={14} className="animate-spin text-brass" />
                        Generating description...
                      </>
                    ) : (
                      <>
                        <FileText size={14} />
                        {description ? 'Regenerate Description' : 'Generate Description'}
                      </>
                    )}
                  </button>
                </div>
              )}

              {/* Next: Publish */}
              {description && (
                <button
                  onClick={() => setStep(4)}
                  className="w-full btn-primary"
                >
                  <ArrowRight size={16} />
                  Next: Publish
                </button>
              )}
            </div>

            {/* Right: Editable Description */}
            <div className="md:col-span-3 space-y-4">
              <div className="glass-panel p-6 space-y-4 h-full flex flex-col">
                <div className="flex items-center justify-between">
                  <p className="eyebrow flex items-center gap-2">
                    <FileText size={14} className="text-muted" />
                    YOUTUBE DESCRIPTION
                  </p>
                  <span className="readout">{description.length} / 5000</span>
                </div>

                <textarea
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  placeholder={mode === 'video'
                    ? "Click 'Generate Description' to auto-generate with chapters, or write your own..."
                    : "Write your YouTube video description here..."
                  }
                  className="input-field text-sm resize-none flex-1 min-h-[500px] font-mono custom-scrollbar"
                  maxLength={5000}
                />

                {!description && (
                  <p className="text-xs text-muted">
                    {mode === 'video'
                      ? "AI will generate a compelling description with chapter timestamps from your video's Whisper transcript."
                      : "Write a description for your YouTube video. You can proceed to publish once you have a description."}
                  </p>
                )}
              </div>
            </div>
          </div>
        )}

        {/* ===== STEP 4: Publish to YouTube ===== */}
        {step === 4 && (
          <div className="grid md:grid-cols-5 gap-6">
            {/* Left: Summary & Publish */}
            <div className="md:col-span-2 space-y-4">
              <button
                onClick={() => setStep(3)}
                className="text-xs lowercase text-muted hover:text-ink transition-colors flex items-center gap-1 mb-2"
              >
                <ArrowLeft size={12} /> Back to Description
              </button>

              {/* Selected Thumbnail Preview */}
              {selectedThumbnail && (
                <div className="glass-panel overflow-hidden">
                  <img
                    src={getApiUrl(selectedThumbnail)}
                    alt="Selected thumbnail"
                    className="w-full aspect-video object-cover"
                  />
                  <div className="p-3">
                    <span className="text-xs lowercase text-brass flex items-center gap-1"><Check size={10} /> Selected Thumbnail</span>
                  </div>
                </div>
              )}

              {/* Editable Title */}
              <div className="glass-panel p-6 space-y-3">
                <p className="eyebrow">TITLE</p>
                <input
                  type="text"
                  value={selectedTitle || manualTitle}
                  onChange={(e) => selectedTitle ? setSelectedTitle(e.target.value) : setManualTitle(e.target.value)}
                  className="input-field text-sm"
                  maxLength={100}
                />
              </div>

              {/* Publish Button */}
              {(!managed && (!uploadPostKey || !uploadUserId)) ? (
                <div className="glass-panel p-6 space-y-3">
                  <div className="flex items-center gap-2 text-warn">
                    <AlertCircle size={16} />
                    <span className="text-sm font-medium lowercase">Upload-Post Not Configured</span>
                  </div>
                  <p className="text-xs text-muted">
                    To publish directly to YouTube, configure your Upload-Post API key and connect a profile in Settings.
                  </p>
                  <button
                    onClick={() => { }}
                    className="text-xs lowercase text-brass hover:underline flex items-center gap-1"
                  >
                    <Settings size={12} /> Go to Settings
                  </button>
                </div>
              ) : (
                <button
                  onClick={handlePublish}
                  disabled={isPublishing}
                  className="w-full btn-primary"
                >
                  {isPublishing ? (
                    <>
                      <Loader2 size={16} className="animate-spin" />
                      Publishing to YouTube...
                    </>
                  ) : (
                    <>
                      <Youtube size={16} />
                      Publish to YouTube
                    </>
                  )}
                </button>
              )}

              {/* Polling status */}
              {isPublishing && (
                <div className="readout flex items-center gap-2">
                  <Loader2 size={12} className="animate-spin text-brass" />
                  UPLOADING — POLLING STATUS EVERY 2S
                </div>
              )}

              {/* Publish Result */}
              {publishResult && (
                <div className="glass-panel p-4">
                  {publishResult.success ? (
                    <div className="space-y-2">
                      <span className="badge-ok">PUBLISHED</span>
                      <p className="text-sm lowercase font-medium text-ink">Published successfully!</p>
                      <p className="text-xs text-muted">Your video is being uploaded to YouTube asynchronously.</p>
                    </div>
                  ) : (
                    <div className="space-y-2">
                      <span className="badge-danger">FAILED</span>
                      <p className="text-sm lowercase font-medium text-danger">Publish failed</p>
                      <p className="text-xs text-muted">{publishResult.error}</p>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Right: Description Preview (read-only feel, still editable) */}
            <div className="md:col-span-3 space-y-4">
              <div className="glass-panel p-6 space-y-4 h-full flex flex-col">
                <div className="flex items-center justify-between">
                  <p className="eyebrow flex items-center gap-2">
                    <FileText size={14} className="text-muted" />
                    YOUTUBE DESCRIPTION
                  </p>
                  <button
                    onClick={() => setStep(3)}
                    className="text-xs lowercase text-muted hover:text-ink flex items-center gap-1 transition-colors"
                  >
                    <ArrowLeft size={10} /> Edit
                  </button>
                </div>

                <textarea
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  className="input-field text-sm resize-none flex-1 min-h-[500px] font-mono custom-scrollbar"
                  maxLength={5000}
                />
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
