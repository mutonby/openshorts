import React from 'react';
import { Sparkles, Zap, Globe, FileVideo, Subtitles, Youtube, Instagram, Shield, Github, ArrowRight, Check, ChevronDown, Monitor, Cpu, Languages, Type, Upload, Scissors } from 'lucide-react';
import PricingSection from './components/PricingSection';
import { useAuth } from './contexts/AuthContext';
import './landing.css';

// 64 deterministic tick heights: gaussian envelope × sine wave (no randomness)
const METER_TICKS = Array.from({ length: 64 }, (_, i) => {
  const t = i / 63;
  const envelope = Math.exp(-((t - 0.5) ** 2) / (2 * 0.18 * 0.18));
  const wave = 0.55 + 0.45 * Math.sin(i * 1.7);
  return Math.round((4 + 24 * envelope * wave) * 10) / 10;
});

const APPARATUS_CALLOUTS = ['RATIO · 9:16', 'CLIPS · 3–15', 'DUB · 30+ LANGS', 'SUBS · WORD-LEVEL'];

const SectionHeader = ({ eyebrow, title, children }) => (
  <div className="mb-12">
    <p className="eyebrow mb-3">{eyebrow}</p>
    <h2 className="font-display text-3xl md:text-4xl lowercase text-ink tracking-tight mb-4">{title}</h2>
    {children && <p className="text-muted max-w-2xl leading-relaxed">{children}</p>}
  </div>
);

const FeatureCard = ({ icon, title, description }) => {
  const Icon = icon;
  return (
    <div className="card card-hover p-6">
      <div className="w-10 h-10 rounded-input bg-paper3 flex items-center justify-center mb-4">
        <Icon size={18} className="text-brass" />
      </div>
      <h3 className="font-display text-xl lowercase text-ink mb-2">{title}</h3>
      <p className="text-muted text-sm leading-relaxed">{description}</p>
    </div>
  );
};

const StepCard = ({ number, title, description }) => (
  <div className="flex gap-5">
    <span className="font-mono text-micro text-brass uppercase pt-1.5 flex-shrink-0">
      {String(number).padStart(2, '0')}
    </span>
    <div>
      <h3 className="text-ink font-medium mb-1">{title}</h3>
      <p className="text-muted text-sm leading-relaxed">{description}</p>
    </div>
  </div>
);

const ComparisonRow = ({ feature, openshorts, opusclip, kapwing }) => (
  <tr className="border-b border-rule">
    <td className="py-3 px-4 text-sm text-ink2">{feature}</td>
    <td className="py-3 px-4 text-center">{openshorts}</td>
    <td className="py-3 px-4 text-center">{opusclip}</td>
    <td className="py-3 px-4 text-center">{kapwing}</td>
  </tr>
);

const FAQItem = ({ question, answer, isOpen, onClick }) => (
  <div>
    <button
      onClick={onClick}
      className="w-full flex items-center justify-between px-1 py-5 text-left"
    >
      <span className="text-ink font-medium pr-4">{question}</span>
      <ChevronDown size={18} className={`text-muted flex-shrink-0 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
    </button>
    {isOpen && (
      <div className="px-1 pb-6">
        <p className="faq-answer text-muted text-sm leading-relaxed">{answer}</p>
      </div>
    )}
  </div>
);

export default function Landing({ onLaunchApp }) {
  const { billingEnabled } = useAuth();
  const [openFaq, setOpenFaq] = React.useState(null);

  const features = [
    {
      icon: Sparkles,
      title: "AI Viral Moment Detection",
      description: "Google Gemini 3.0 Flash scores your transcript and scenes to find the 3-15 most engaging moments. Automatic AI clipping, no manual scrubbing."
    },
    {
      icon: Scissors,
      title: "Smart 9:16 Vertical Cropping",
      description: "Dual-mode AI reframing with MediaPipe face tracking and YOLOv8 fallback."
    },
    {
      icon: Subtitles,
      title: "Automatic Subtitle Generation",
      description: "faster-whisper subtitles with word-level timestamps, styled and burned into your clips."
    },
    {
      icon: Languages,
      title: "AI Voice Dubbing in 30+ Languages",
      description: "ElevenLabs AI dubbing translates your audio while preserving the speaker's voice."
    },
    {
      icon: Type,
      title: "Hook Text Overlays",
      description: "AI-generated hook titles that capture viewers in the first 3 seconds."
    },
    {
      icon: Zap,
      title: "AI Video Effects",
      description: "Gemini-generated FFmpeg filters: color grading, transitions, visual enhancements."
    },
    {
      icon: Upload,
      title: "Local Video Upload",
      description: "Upload podcasts, webinars, livestreams, and vlogs at full resolution."
    },
    {
      icon: Shield,
      title: "100% Self-Hosted & Private",
      description: "Run it with Docker on your own machine — videos never leave your infrastructure."
    },
    {
      icon: Monitor,
      title: "Free AI YouTube Studio",
      description: "Free AI thumbnail generator, 10 viral title suggestions, and auto descriptions with chapters."
    },
    {
      icon: Globe,
      title: "Direct Social Publishing",
      description: "Post to TikTok, Instagram Reels, and YouTube Shorts from the dashboard."
    },
    {
      icon: Sparkles,
      title: "AI UGC Video Generator",
      description: "AI writes the script and generates a lip-synced avatar video — from $0.65/video."
    },
    {
      icon: FileVideo,
      title: "AI Actors & Lip-Sync",
      description: "Pick an AI actor or upload a photo for a lip-synced talking head video."
    }
  ];

  const steps = [
    { title: "Upload a Long-Form Video", description: "Drop any video file you own — podcasts, webinars, livestreams, interviews." },
    { title: "AI Detects the Best Viral Moments", description: "Google Gemini 3.0 Flash finds 3-15 high-potential clips of 15-60 seconds." },
    { title: "Smart Cropping to Vertical 9:16", description: "AI reframes to vertical with face tracking — subjects stay centered." },
    { title: "Add Subtitles, Hooks & Effects", description: "Auto subtitles, hook overlays, AI effects — optionally dub into 30+ languages." },
    { title: "Download or Post to Social Media", description: "Export your clips or post directly to TikTok, Instagram Reels, and YouTube Shorts." }
  ];

  const faqs = [
    {
      question: "Is OpenShorts really free? What's the catch?",
      answer: "There are two ways to use OpenShorts. (1) Self-hosted is 100% free and open source: you run it with Docker on your own machine, bring your own API keys, and there are no watermarks, no usage limits, and no subscription. Google Gemini API (required) powers the AI analysis — its free tier includes 1,500 requests/day. ElevenLabs (optional) enables AI dubbing in 30+ languages. Upload-Post (optional) publishes to YouTube, TikTok, and Instagram. (2) Hosted at openshorts.app is the no-setup option: we run all the AI and compute for you and you don't need any API keys — it's a paid service with a 3-day free trial, then plans from $12/mo for 100 minutes of video. So: free if you self-host, paid if you want us to host it for you. Both are far cheaper than Opus Clip ($15-228/month) or Kapwing ($24-79/month)."
    },
    {
      question: "What is OpenShorts and how does it work?",
      answer: "OpenShorts is a free, open source AI clip generator that transforms your long-form videos — podcasts, webinars, livestreams, vlogs, interviews — into viral-ready short clips in 9:16 vertical format. It uses a multi-step AI pipeline: faster-whisper for transcription with word-level timestamps, PySceneDetect for scene boundary detection, and Google Gemini 3.0 Flash AI for identifying the most engaging viral moments. According to HubSpot's 2025 State of Marketing report, short-form video delivers the highest ROI of any content format, and repurposing long-form content into shorts increases total reach by up to 300%."
    },
    {
      question: "How does OpenShorts compare to Opus Clip?",
      answer: "OpenShorts is a free, self-hosted alternative to Opus Clip. Both offer AI viral moment detection and smart vertical cropping. Key differences: OpenShorts is completely free vs Opus Clip's $15-228/month pricing. OpenShorts runs on your infrastructure (full data privacy) vs cloud-only. OpenShorts uses Google Gemini 3.0 Flash for AI analysis vs Opus Clip's proprietary model. OpenShorts adds AI voice dubbing in 30+ languages, AI-generated video effects, and hook text overlays. The trade-off is that OpenShorts requires Docker self-hosting, while Opus Clip is a ready-to-use cloud service."
    },
    {
      question: "How do I turn a long-form video into TikTok or Reels clips?",
      answer: "Upload your long-form video into OpenShorts, enter your free Gemini API key, and click Process. The AI transcribes it with faster-whisper, detects the best viral moments using Google Gemini 3.0 Flash, and crops them to 9:16 vertical format with MediaPipe face tracking. According to Wyzowl's 2025 Video Marketing Statistics report, 91% of businesses use video as a marketing tool, and repurposed short-form clips drive 2.5x more engagement than original content."
    },
    {
      question: "Can OpenShorts generate YouTube thumbnails and titles for free?",
      answer: "Yes. OpenShorts includes a free AI YouTube thumbnail generator, a free AI YouTube title generator, and a free AI YouTube description generator — all powered by Google Gemini 3.0 Flash. Upload your video and the AI suggests 10 viral title options with an interactive refinement chat. Then it generates multiple thumbnail designs using AI image generation — upload a face photo and background image for personalized results. The studio also auto-generates YouTube descriptions with chapter timestamps and lets you publish directly to YouTube. Everything is 100% free with the Gemini free tier."
    },
    {
      question: "What is the AI UGC Video Generator?",
      answer: "OpenShorts includes an AI UGC (User Generated Content) video creator that generates marketing videos with AI actors for any product or business. You describe your product or paste a website URL — the AI writes a viral script, generates a realistic AI actor with lip-synced voiceover, adds b-roll visuals, TikTok-style subtitles, and hook text overlays. The result is a ready-to-post vertical video for TikTok, Instagram Reels, or YouTube Shorts. Two cost modes: Low Cost (~$0.65/video using Hailuo + VEED Lipsync) and Premium (~$2/video using Kling Avatar v2)."
    },
    {
      question: "Can I use the AI UGC Video Generator for any type of business?",
      answer: "Yes. The AI Shorts generator works for any product, service, or business — not just SaaS. You can use it for restaurants, e-commerce stores, coaching services, local businesses, personal brands, apps, and more. Just describe your business in the text field (e.g. 'Artisan pizza restaurant in Madrid, wood-fired oven, home delivery') or paste your website URL, and the AI generates viral marketing scripts tailored to your business."
    },
    {
      question: "How much does it cost to generate an AI UGC video?",
      answer: "OpenShorts itself is free, but the AI Shorts feature uses external APIs (fal.ai for video generation, ElevenLabs for voiceover) that charge per use. Low Cost mode costs approximately $0.65 per video (Flux image $0.05 + ElevenLabs voice $0.10 + Hailuo img2video $0.19 + VEED Lipsync $0.20 + b-roll $0.10). Premium mode costs approximately $2.00 per video using Kling Avatar v2 for higher quality. Both modes are significantly cheaper than hiring UGC creators ($50-500 per video) or using platforms like HeyGen ($24-180/month)."
    },
    {
      question: "What AI does OpenShorts use for viral moment detection?",
      answer: "OpenShorts uses Google Gemini 3.0 Flash, Google's latest multimodal AI model, for viral moment detection and title generation. The AI receives the full video transcript with timestamps, scene boundary data from PySceneDetect, and analyzes engagement patterns to identify the 3-15 most shareable moments. Each clip is scored based on emotional impact, hook strength, and viral potential — similar to how platforms like TikTok and YouTube rank content."
    },
    {
      question: "Can OpenShorts translate and dub videos into other languages?",
      answer: "Yes. OpenShorts integrates with ElevenLabs AI dubbing to translate your video audio into over 30 languages while preserving the original speaker's voice characteristics. After dubbing, the system automatically re-transcribes the new audio and generates subtitles in the target language. This makes it easy to repurpose content for global audiences — studies show that dubbed content receives 2-3x more engagement in non-English markets."
    },
    {
      question: "How does the smart vertical cropping work?",
      answer: "OpenShorts offers two intelligent cropping modes for converting 16:9 horizontal video to 9:16 vertical format. TRACK mode uses MediaPipe face detection with YOLOv8 as fallback to follow a single subject with 'Heavy Tripod' stabilization — the camera moves smoothly like a professional cameraman. GENERAL mode handles group shots and landscapes by creating a blurred background layout. A SpeakerTracker prevents rapid switching between subjects and handles temporary occlusions for smooth results."
    },
    {
      question: "Is there a free open source clip generator?",
      answer: "Yes — OpenShorts is a 100% free, open source clip generator (also known as open source clipping software or an AI video clipper). Unlike paid clip generators like Opus Clip ($15-228/month) or Kapwing ($24-79/month), OpenShorts lets you generate unlimited clips with no watermarks, no usage limits, and no subscription fees. It also includes a free AI YouTube thumbnail generator, free AI YouTube title generator, and free AI YouTube description generator — features that other clip generators charge extra for. You self-host it with Docker on your own machine for full privacy and control."
    },
    {
      question: "What are the system requirements to run OpenShorts?",
      answer: "OpenShorts runs on any system with Docker installed. The recommended setup is 8GB+ RAM and a modern multi-core CPU. GPU acceleration (NVIDIA CUDA) is optional but speeds up video processing significantly. The Docker Compose setup handles all dependencies automatically — Python 3.11, FFmpeg, YOLOv8, MediaPipe, faster-whisper, and the React dashboard. It works on Linux, macOS, and Windows (via WSL2/Docker Desktop)."
    }
  ];

  const checkIcon = <Check size={16} className="text-brass mx-auto" />;
  const checkMuted = <Check size={16} className="text-muted mx-auto" />;
  const xIcon = <span className="text-muted text-sm">Paid</span>;

  return (
    <div className="min-h-screen bg-paper text-ink2 overflow-x-clip">
      {/* Navigation — N9 edge-aligned minimal */}
      <nav className="fixed top-0 w-full z-50 bg-paper border-b border-rule">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <img src="/logo-openshorts.png" alt="OpenShorts logo" className="w-7 h-7" />
            <span className="font-display text-lg lowercase text-ink">OpenShorts</span>
          </div>
          <div className="hidden md:flex items-center gap-7 text-sm lowercase text-muted">
            <a href="#features" className="hover:text-ink transition-colors">Features</a>
            <a href="#how-it-works" className="hover:text-ink transition-colors">How It Works</a>
            {billingEnabled && <a href="#pricing" className="hover:text-ink transition-colors">Pricing</a>}
            <a href="#comparison" className="hover:text-ink transition-colors">Comparison</a>
            <a href="#faq" className="hover:text-ink transition-colors">FAQ</a>
          </div>
          <div className="flex items-center gap-3">
            <a
              href="https://github.com/mutonby/openshorts"
              target="_blank"
              rel="noopener noreferrer"
              className="hidden sm:flex items-center gap-2 text-sm lowercase text-muted hover:text-ink transition-colors"
            >
              <Github size={16} />
              <span>GitHub</span>
            </a>
            <button onClick={onLaunchApp} className="btn-primary px-5 py-2 whitespace-nowrap">
              Launch App
            </button>
          </div>
        </div>
      </nav>

      {/* Hero — Marquee Hero: blueprint grid, content left, apparatus right */}
      <section className="hero-blueprint relative overflow-clip border-b border-rule pt-32 pb-20 px-6">
        <div className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_auto] gap-14 items-center">
          <div className="min-w-0">
            <p className="eyebrow mb-6">00 · AI Clip Generator · Cloud or Self-Hosted</p>

            <h1 className="hero-h1 mb-6">
              the free open source ai <em>clip generator</em>, built to clip what people actually watch.
            </h1>

            <p className="hero-description text-muted max-w-2xl mb-8 leading-relaxed lowercase">
              turn long videos into viral 9:16 shorts, or generate ugc marketing videos with ai actors. online, in the cloud, zero setup.
            </p>

            <div className="flex flex-wrap items-center gap-4 mb-5">
              {billingEnabled ? (
                <a href="#pricing" className="btn-primary whitespace-nowrap">
                  start free trial
                  <ArrowRight size={16} />
                </a>
              ) : (
                <button onClick={onLaunchApp} className="btn-primary whitespace-nowrap">
                  launch openshorts
                  <ArrowRight size={16} />
                </button>
              )}
              {billingEnabled && (
                <button onClick={onLaunchApp} className="btn-ghost whitespace-nowrap">
                  launch the app
                </button>
              )}
            </div>

            <p className="text-sm text-muted lowercase">
              3-day free trial · 20 free minutes · from $12/mo · no watermarks. prefer to run it yourself?{' '}
              <a
                href="https://github.com/mutonby/openshorts"
                target="_blank"
                rel="noopener noreferrer"
                className="text-ink2 underline hover:text-ink transition-colors"
              >
                self-host free on github →
              </a>
            </p>
          </div>

          {/* Apparatus — instrument bezel holding a real 9:16 clip */}
          <figure className="apparatus" aria-label="example vertical clip generated by openshorts">
            <div className="apparatus-shell">
              <span className="apparatus-glow" aria-hidden="true" />
              <div className="apparatus-chamber">
                <video
                  src="/demo/clip-vertical.mp4"
                  autoPlay
                  muted
                  loop
                  playsInline
                  preload="metadata"
                  className="w-full h-full object-cover"
                />
                <span className="apparatus-stencil">OS-9:16</span>
              </div>
            </div>
            <ul className="apparatus-callouts" aria-hidden="true">
              {APPARATUS_CALLOUTS.map((c) => (
                <li key={c}><span className="apparatus-leader" />{c}</li>
              ))}
            </ul>
          </figure>
        </div>
      </section>

      {/* Meter strip */}
      <section className="border-b border-rule" aria-hidden="true">
        <div className="max-w-6xl mx-auto px-6 meter-strip">
          <span className="readout whitespace-nowrap">Signal · 9:16</span>
          <div className="meter-ticks">
            {METER_TICKS.map((h, i) => (
              <span key={i} className="meter-tick" style={{ height: `${h}px` }} />
            ))}
          </div>
          <span className="readout whitespace-nowrap hidden sm:inline">Clips · 3–15 / video</span>
        </div>
      </section>

      {/* Stats — three-stat row */}
      <section className="border-b border-rule">
        <div className="max-w-6xl mx-auto px-6 py-12 grid grid-cols-3 divide-x divide-rule text-center">
          <div className="px-4">
            <div className="font-display text-4xl md:text-5xl text-ink tabular-nums">3–15</div>
            <div className="eyebrow mt-2">Clips per Video</div>
          </div>
          <div className="px-4">
            <div className="font-display text-4xl md:text-5xl text-ink tabular-nums">30+</div>
            <div className="eyebrow mt-2">Dubbing Languages</div>
          </div>
          <div className="px-4">
            <div className="font-display text-4xl md:text-5xl text-ink tabular-nums">100%</div>
            <div className="eyebrow mt-2">Open Source</div>
          </div>
        </div>
      </section>

      {/* Smart crop — real product output: 16:9 source to 9:16 result */}
      <section className="py-20 px-6">
        <div className="max-w-5xl mx-auto">
          <SectionHeader eyebrow="01 · Smart Crop" title="one video in. the moment, reframed.">
            Real output: AI face tracking reframes 16:9 to vertical 9:16 — no manual positioning.
          </SectionHeader>
          <div className="flex flex-col md:flex-row items-center gap-8 md:gap-10">
            <figure className="crop-frame w-full max-w-xl min-w-0" aria-label="original 16:9 source video">
              <video
                src="/demo/clip-source.mp4"
                autoPlay
                muted
                loop
                playsInline
                preload="metadata"
              />
            </figure>
            <div className="crop-leader" aria-hidden="true">
              <span className="readout whitespace-nowrap">AI Tracking → 9:16</span>
              <span className="crop-leader-line" />
            </div>
            <figure className="crop-frame w-[180px] md:w-[210px] flex-none" aria-label="vertical 9:16 clip generated by openshorts">
              <video
                src="/demo/clip-vertical.mp4"
                autoPlay
                muted
                loop
                playsInline
                preload="metadata"
              />
            </figure>
          </div>
        </div>
      </section>

      {/* Two ways to use it: free self-host vs paid hosted */}
      <section className="py-20 px-6 border-t border-rule">
        <div className="max-w-4xl mx-auto">
          <SectionHeader eyebrow="02 · Deploy" title="Two ways to use OpenShorts">
            Let us run everything — or run it yourself for free.
          </SectionHeader>
          <div className="grid md:grid-cols-2 gap-6">
            <div className="card p-8 flex flex-col border-brass">
              <div className="flex flex-wrap items-center gap-3 mb-3">
                <Sparkles size={18} className="text-brass" />
                <h3 className="font-display text-2xl lowercase text-ink">cloud — openshorts.app</h3>
                <span className="badge-brass">Recommended · 3-Day Trial</span>
              </div>
              <ul className="space-y-1.5 mb-6 flex-1">
                {['Zero setup', 'No API keys — Gemini included', 'Social publishing built in', 'From $12/mo'].map((f, i) => (
                  <li key={i} className="flex items-center gap-2 text-sm text-muted"><Check size={14} className="text-ok shrink-0" />{f}</li>
                ))}
              </ul>
              {billingEnabled ? (
                <a href="#pricing" className="btn-primary whitespace-nowrap">
                  start free trial <ArrowRight size={16} />
                </a>
              ) : (
                <button onClick={onLaunchApp} className="btn-primary whitespace-nowrap">
                  launch openshorts <ArrowRight size={16} />
                </button>
              )}
            </div>
            <div className="card p-8 flex flex-col">
              <div className="flex flex-wrap items-center gap-3 mb-3">
                <Github size={18} className="text-muted" />
                <h3 className="font-display text-2xl lowercase text-ink">self-hosted — for developers</h3>
                <span className="readout border border-rule rounded-full px-2.5 py-1">Free · Docker</span>
              </div>
              <ul className="space-y-1.5 mb-6 flex-1">
                {['Your infrastructure', 'Bring your own API keys', 'You maintain it', '100% free forever'].map((f, i) => (
                  <li key={i} className="flex items-center gap-2 text-sm text-muted"><Check size={14} className="text-ok shrink-0" />{f}</li>
                ))}
              </ul>
              <a href="https://github.com/mutonby/openshorts" target="_blank" rel="noopener noreferrer"
                className="btn-ghost whitespace-nowrap">
                <Github size={16} /> view on github
              </a>
            </div>
          </div>
        </div>
      </section>

      {/* Pricing Section — hosted plans (self-host stays free) */}
      {billingEnabled && (
        <section id="pricing" className="py-20 px-6 border-t border-rule">
          <div className="max-w-6xl mx-auto">
            <SectionHeader eyebrow="03 · Pricing" title="Simple, transparent pricing">
              3-day free trial with 20 free minutes. Cancel anytime.
            </SectionHeader>
            <PricingSection onRequireLogin={() => { window.location.hash = '#/pricing'; }} />
          </div>
        </section>
      )}

      {/* 3 Tools in 1 Section */}
      <section className="py-20 px-6 border-t border-rule">
        <div className="max-w-6xl mx-auto">
          <SectionHeader eyebrow="04 · Tools" title="3 Free Tools in 1 Platform">
            Everything below is included in cloud plans — no keys, no setup.
          </SectionHeader>
          <div className="grid md:grid-cols-3 gap-5">
            <div className="card p-8">
              <p className="eyebrow mb-4">01 · Clips</p>
              <Scissors size={20} className="text-brass mb-4" />
              <h3 className="font-display text-2xl lowercase text-ink mb-2">Clip Generator</h3>
              <p className="text-muted text-sm leading-relaxed mb-4">Open source AI clipping tool: turn long-form videos into viral-ready 9:16 shorts.</p>
              <ul className="space-y-1.5">
                {['AI viral moment detection', 'Smart face-tracking crop', 'Auto subtitles + AI dubbing in 30+ languages'].map((f, i) => (
                  <li key={i} className="flex items-center gap-2 text-xs text-muted"><Check size={12} className="text-ok shrink-0" />{f}</li>
                ))}
              </ul>
            </div>
            <div className="card p-8">
              <p className="eyebrow mb-4">02 · AI Shorts</p>
              <Sparkles size={20} className="text-brass mb-4" />
              <h3 className="font-display text-2xl lowercase text-ink mb-2">AI Shorts</h3>
              <p className="text-muted text-sm leading-relaxed mb-4">UGC marketing videos with AI actors for any business.</p>
              <ul className="space-y-1.5">
                {['AI actor generation + lip-sync', 'B-roll + TikTok-style subtitles', 'From $0.65 per video'].map((f, i) => (
                  <li key={i} className="flex items-center gap-2 text-xs text-muted"><Check size={12} className="text-ok shrink-0" />{f}</li>
                ))}
              </ul>
            </div>
            <div className="card p-8">
              <p className="eyebrow mb-4">03 · Studio</p>
              <Monitor size={20} className="text-brass mb-4" />
              <h3 className="font-display text-2xl lowercase text-ink mb-2">YouTube Studio</h3>
              <p className="text-muted text-sm leading-relaxed mb-4">Free AI YouTube toolkit: thumbnails, titles, descriptions.</p>
              <ul className="space-y-1.5">
                {['AI thumbnail generator (with face upload)', '10 viral title suggestions + chat', 'Direct publish to YouTube'].map((f, i) => (
                  <li key={i} className="flex items-center gap-2 text-xs text-muted"><Check size={12} className="text-ok shrink-0" />{f}</li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-20 px-6 border-t border-rule">
        <div className="max-w-6xl mx-auto">
          <SectionHeader eyebrow="05 · Features" title="Free AI Clip Generator + UGC Video Creator">
            The free open source clip generator & AI UGC video creator. A smart AI video clipper for TikTok, Reels & Shorts.
          </SectionHeader>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-5">
            {features.map((feature, i) => (
              <FeatureCard key={i} {...feature} />
            ))}
          </div>
        </div>
      </section>

      {/* API Keys Section */}
      <section className="py-20 px-6 border-t border-rule">
        <div className="max-w-5xl mx-auto">
          <SectionHeader eyebrow="06 · API Keys" title="Self-hosting? Every API has a free tier">
            Cloud plans include managed Gemini and social publishing — you never touch an API key. Self-hosters bring their own keys (all with generous free tiers):
          </SectionHeader>
          <div className="grid md:grid-cols-3 gap-5">
            <div className="card p-6 relative">
              <span className="badge-brass absolute top-4 right-4">Required</span>
              <div className="w-10 h-10 rounded-input bg-paper3 flex items-center justify-center mb-4">
                <Cpu size={18} className="text-brass" />
              </div>
              <h3 className="font-display text-xl lowercase text-ink mb-1">Google Gemini API</h3>
              <div className="mb-3"><span className="badge-ok">Free tier: 1,500 req/day</span></div>
              <p className="text-muted text-sm leading-relaxed">Powers all AI features: viral moment detection, title generation, video effects, YouTube thumbnail creation, and description writing. The core engine of OpenShorts.</p>
            </div>
            <div className="card p-6 relative">
              <span className="readout absolute top-4 right-4 border border-rule rounded-full px-2.5 py-1">Optional</span>
              <div className="w-10 h-10 rounded-input bg-paper3 flex items-center justify-center mb-4">
                <Languages size={18} className="text-brass" />
              </div>
              <h3 className="font-display text-xl lowercase text-ink mb-1">ElevenLabs API</h3>
              <div className="mb-3"><span className="badge-ok">Free tier included</span></div>
              <p className="text-muted text-sm leading-relaxed">Enables AI voice dubbing and translation in 30+ languages. Preserves the original speaker's voice while translating audio. Dubbed clips are auto-subtitled.</p>
            </div>
            <div className="card p-6 relative">
              <span className="readout absolute top-4 right-4 border border-rule rounded-full px-2.5 py-1">Optional</span>
              <div className="w-10 h-10 rounded-input bg-paper3 flex items-center justify-center mb-4">
                <Globe size={18} className="text-brass" />
              </div>
              <h3 className="font-display text-xl lowercase text-ink mb-1">Upload-Post API</h3>
              <div className="mb-3"><span className="badge-ok">Free tier included</span></div>
              <p className="text-muted text-sm leading-relaxed">Enables direct publishing to YouTube, TikTok, and Instagram Reels from the dashboard. <a href="https://www.upload-post.com" target="_blank" rel="noopener noreferrer" className="text-brass underline hover:brightness-110">Social media API</a> that lets you post your clips and thumbnails without leaving OpenShorts.</p>
            </div>
          </div>
          <div className="grid md:grid-cols-2 gap-5 mt-5">
            <div className="card p-6 relative">
              <span className="readout absolute top-4 right-4 border border-rule rounded-full px-2.5 py-1">AI Shorts</span>
              <div className="w-10 h-10 rounded-input bg-paper3 flex items-center justify-center mb-4">
                <Zap size={18} className="text-brass" />
              </div>
              <h3 className="font-display text-xl lowercase text-ink mb-1">fal.ai API</h3>
              <div className="mb-3"><span className="badge-ok">Pay-per-use from $0.04</span></div>
              <p className="text-muted text-sm leading-relaxed">Powers AI Shorts: generates AI actor images (Flux), talking head videos (Hailuo/Kling), and lip-sync (VEED). Required only for the AI UGC video generator.</p>
            </div>
            <div className="card p-6 relative">
              <span className="readout absolute top-4 right-4 border border-rule rounded-full px-2.5 py-1">AI Shorts</span>
              <div className="w-10 h-10 rounded-input bg-paper3 flex items-center justify-center mb-4">
                <Languages size={18} className="text-brass" />
              </div>
              <h3 className="font-display text-xl lowercase text-ink mb-1">ElevenLabs TTS</h3>
              <div className="mb-3"><span className="badge-ok">Free tier included</span></div>
              <p className="text-muted text-sm leading-relaxed">Generates natural voiceovers for AI Shorts from the script. Multiple voice options for male and female actors in English and Spanish.</p>
            </div>
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section id="how-it-works" className="py-20 px-6 border-t border-rule">
        <div className="max-w-4xl mx-auto">
          <SectionHeader eyebrow="07 · Pipeline" title="How It Works">
            From long-form video to viral-ready clips in 5 automated steps.
          </SectionHeader>
          <div className="space-y-8">
            {steps.map((step, i) => (
              <StepCard key={i} number={i + 1} {...step} />
            ))}
          </div>
        </div>
      </section>

      {/* Tech Stack */}
      <section className="py-20 px-6 border-t border-rule">
        <div className="max-w-5xl mx-auto">
          <SectionHeader eyebrow="08 · Stack" title="Built with Proven Technology">
            Industry-leading AI models and open source tools in one pipeline.
          </SectionHeader>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              { name: "Google Gemini 3.0", desc: "AI Analysis" },
              { name: "faster-whisper", desc: "Transcription" },
              { name: "YOLOv8", desc: "Object Detection" },
              { name: "MediaPipe", desc: "Face Tracking" },
              { name: "FFmpeg", desc: "Video Processing" },
              { name: "ElevenLabs", desc: "Voice & TTS" },
              { name: "fal.ai", desc: "AI Video Gen" },
              { name: "React + Vite", desc: "Dashboard" },
              { name: "Docker", desc: "Deployment" }
            ].map((tech, i) => (
              <div key={i} className="border border-rule rounded-input bg-paper2 px-4 py-3 text-center">
                <div className="readout text-ink2">{tech.name}</div>
                <div className="text-xs text-muted mt-1">{tech.desc}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Comparison Table */}
      <section id="comparison" className="py-20 px-6 border-t border-rule">
        <div className="max-w-4xl mx-auto">
          <SectionHeader eyebrow="09 · Comparison" title="Free Clip Generator vs Paid Alternatives">
            Hosted OpenShorts starts at $12/mo — or self-host free. Opus Clip runs $15-228/month, Kapwing $24-79/month.
          </SectionHeader>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-rule2">
                  <th className="py-3 px-4 text-left text-sm text-muted font-medium">Feature</th>
                  <th className="py-3 px-4 text-center text-sm font-medium">
                    <span className="text-brass">OpenShorts</span>
                  </th>
                  <th className="py-3 px-4 text-center text-sm text-muted font-medium">Opus Clip</th>
                  <th className="py-3 px-4 text-center text-sm text-muted font-medium">Kapwing</th>
                </tr>
              </thead>
              <tbody>
                <ComparisonRow feature="Price" openshorts={<span className="text-ok font-medium">$0 Free</span>} opusclip={xIcon} kapwing={xIcon} />
                <ComparisonRow feature="AI Viral Moment Detection" openshorts={checkIcon} opusclip={checkMuted} kapwing={checkMuted} />
                <ComparisonRow feature="Smart Vertical Cropping" openshorts={checkIcon} opusclip={checkMuted} kapwing={checkMuted} />
                <ComparisonRow feature="Auto Subtitles" openshorts={checkIcon} opusclip={checkMuted} kapwing={checkMuted} />
                <ComparisonRow feature="AI Voice Dubbing (30+ langs)" openshorts={checkIcon} opusclip={<span className="text-muted text-sm">Limited</span>} kapwing={<span className="text-muted text-sm">No</span>} />
                <ComparisonRow feature="AI Video Effects" openshorts={checkIcon} opusclip={<span className="text-muted text-sm">No</span>} kapwing={checkMuted} />
                <ComparisonRow feature="Hook Text Overlays" openshorts={checkIcon} opusclip={checkMuted} kapwing={checkMuted} />
                <ComparisonRow feature="Self-Hosted / Privacy" openshorts={checkIcon} opusclip={<span className="text-muted text-sm">Cloud only</span>} kapwing={<span className="text-muted text-sm">Cloud only</span>} />
                <ComparisonRow feature="No Watermark" openshorts={checkIcon} opusclip={<span className="text-muted text-sm">Free tier only</span>} kapwing={<span className="text-muted text-sm">Paid</span>} />
                <ComparisonRow feature="Open Source" openshorts={checkIcon} opusclip={<span className="text-muted text-sm">No</span>} kapwing={<span className="text-muted text-sm">No</span>} />
                <ComparisonRow feature="AI YouTube Thumbnail Generator" openshorts={checkIcon} opusclip={<span className="text-muted text-sm">No</span>} kapwing={<span className="text-muted text-sm">Paid</span>} />
                <ComparisonRow feature="AI Title & Description Generator" openshorts={checkIcon} opusclip={<span className="text-muted text-sm">Limited</span>} kapwing={<span className="text-muted text-sm">Paid</span>} />
                <ComparisonRow feature="AI UGC Video Generator" openshorts={checkIcon} opusclip={<span className="text-muted text-sm">No</span>} kapwing={<span className="text-muted text-sm">No</span>} />
                <ComparisonRow feature="AI Actors with Lip-Sync" openshorts={checkIcon} opusclip={<span className="text-muted text-sm">No</span>} kapwing={<span className="text-muted text-sm">No</span>} />
                <ComparisonRow feature="Usage Limits" openshorts={<span className="text-ok text-sm">Unlimited</span>} opusclip={<span className="text-muted text-sm">Per plan</span>} kapwing={<span className="text-muted text-sm">Per plan</span>} />
              </tbody>
            </table>
          </div>
        </div>
      </section>

      {/* Use Cases */}
      <section className="py-20 px-6 border-t border-rule">
        <div className="max-w-5xl mx-auto">
          <SectionHeader eyebrow="10 · Use Cases" title="Who Uses OpenShorts?">
            Creators, marketers, and agencies scaling short-form video production.
          </SectionHeader>
          <div className="grid md:grid-cols-3 gap-5">
            {[
              {
                title: "Content Creators",
                description: "Repurpose long-form videos into TikTok and Reels clips automatically.",
                icon: Youtube
              },
              {
                title: "Social Media Managers",
                description: "Batch-process videos and publish for multiple clients from one dashboard.",
                icon: Instagram
              },
              {
                title: "Podcasters & Educators",
                description: "Extract the most engaging moments from episodes and lessons.",
                icon: FileVideo
              },
              {
                title: "Businesses & Brands",
                description: "UGC-style marketing videos with AI actors — from $0.65 per video.",
                icon: Sparkles
              }
            ].map((useCase, i) => (
              <div key={i} className="card p-6">
                <useCase.icon size={18} className="text-brass mb-4" />
                <h3 className="font-display text-xl lowercase text-ink mb-2">{useCase.title}</h3>
                <p className="text-muted text-sm leading-relaxed">{useCase.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* FAQ Section */}
      <section id="faq" className="py-20 px-6 border-t border-rule">
        <div className="max-w-3xl mx-auto">
          <SectionHeader eyebrow="11 · FAQ" title="Frequently Asked Questions">
            Everything you need to know about OpenShorts, from setup to features.
          </SectionHeader>
          <div className="divide-y divide-rule border-y border-rule">
            {faqs.map((faq, i) => (
              <FAQItem
                key={i}
                question={faq.question}
                answer={faq.answer}
                isOpen={openFaq === i}
                onClick={() => setOpenFaq(openFaq === i ? null : i)}
              />
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section — final statement */}
      <section className="py-24 px-6 border-t border-rule">
        <div className="max-w-3xl mx-auto text-center">
          <h2 className="font-display text-4xl md:text-5xl lowercase text-ink tracking-tight mb-5">start creating viral videos today.</h2>
          <p className="text-muted mb-10 max-w-xl mx-auto leading-relaxed lowercase">3-day free trial · 20 free minutes · no watermarks — or self-host free with docker.</p>
          <div className="flex flex-wrap items-center justify-center gap-4">
            {billingEnabled ? (
              <a href="#pricing" className="btn-primary whitespace-nowrap">
                start free trial
                <ArrowRight size={16} />
              </a>
            ) : (
              <button onClick={onLaunchApp} className="btn-primary whitespace-nowrap">
                launch openshorts
                <ArrowRight size={16} />
              </button>
            )}
            <a
              href="https://github.com/mutonby/openshorts"
              target="_blank"
              rel="noopener noreferrer"
              className="btn-ghost whitespace-nowrap"
            >
              <Github size={16} />
              Star on GitHub
            </a>
          </div>
        </div>
      </section>

      {/* Footer — Ft5 Statement */}
      <footer className="border-t border-rule py-16 px-6">
        <div className="max-w-6xl mx-auto">
          <p className="font-display text-3xl md:text-5xl lowercase text-ink tracking-tight mb-10">clip it before it scrolls past.</p>
          <div className="border-t border-rule pt-6 flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <img src="/logo-openshorts.png" alt="OpenShorts" className="w-6 h-6" />
              <span className="text-sm text-muted">OpenShorts — Free Open Source Clip Generator & AI UGC Video Creator</span>
            </div>
            <div className="flex flex-wrap items-center gap-x-6 gap-y-2 text-sm lowercase text-muted">
              <a href="https://github.com/mutonby/openshorts" target="_blank" rel="noopener noreferrer" className="hover:text-ink transition-colors">GitHub</a>
              <a href="#features" className="hover:text-ink transition-colors">Features</a>
              <a href="#faq" className="hover:text-ink transition-colors">FAQ</a>
              <a href="#legal" className="hover:text-ink transition-colors whitespace-nowrap">Terms & Privacy</a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
