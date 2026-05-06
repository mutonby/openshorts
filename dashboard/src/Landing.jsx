import React from 'react';
import { Sparkles, Zap, Globe, FileVideo, Subtitles, Youtube, Instagram, Shield, Github, ArrowRight, Play, Check, ChevronDown, Monitor, Cpu, Languages, Type, Upload, Scissors } from 'lucide-react';

const TikTokIcon = ({ size = 16, className = "" }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor" className={className}>
    <path d="M19.589 6.686a4.793 4.793 0 0 1-3.77-4.245V2h-3.445v13.672a2.896 2.896 0 0 1-5.201 1.743l-.002-.001.002.001a2.895 2.895 0 0 1 3.183-4.51v-3.5a6.329 6.329 0 0 0-5.394 10.692 6.33 6.33 0 0 0 10.857-4.424V8.687a8.182 8.182 0 0 0 4.773 1.526V6.79a4.831 4.831 0 0 1-1.003-.104z" />
  </svg>
);

const FeatureCard = ({ icon: Icon, title, description }) => (
  <div className="group bg-surface/50 backdrop-blur-xl border border-white/10 rounded-2xl p-6 hover:border-primary/30 transition-all duration-300 hover:shadow-lg hover:shadow-primary/5">
    <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center mb-4 group-hover:bg-primary/20 transition-colors">
      <Icon size={24} className="text-primary" />
    </div>
    <h3 className="text-lg font-semibold text-white mb-2">{title}</h3>
    <p className="text-zinc-400 text-sm leading-relaxed">{description}</p>
  </div>
);

const StepCard = ({ number, title, description }) => (
  <div className="flex gap-4">
    <div className="flex-shrink-0 w-10 h-10 rounded-full bg-primary/20 border border-primary/30 flex items-center justify-center text-primary font-bold text-sm">
      {number}
    </div>
    <div>
      <h3 className="text-white font-semibold mb-1">{title}</h3>
      <p className="text-zinc-400 text-sm leading-relaxed">{description}</p>
    </div>
  </div>
);

const ComparisonRow = ({ feature, openshorts, opusclip, kapwing }) => (
  <tr className="border-b border-white/5">
    <td className="py-3 px-4 text-sm text-zinc-300">{feature}</td>
    <td className="py-3 px-4 text-center">{openshorts}</td>
    <td className="py-3 px-4 text-center">{opusclip}</td>
    <td className="py-3 px-4 text-center">{kapwing}</td>
  </tr>
);

const FAQItem = ({ question, answer, isOpen, onClick }) => (
  <div className="border border-white/10 rounded-xl overflow-hidden">
    <button
      onClick={onClick}
      className="w-full flex items-center justify-between px-6 py-4 text-left hover:bg-white/5 transition-colors"
    >
      <span className="text-white font-medium pr-4">{question}</span>
      <ChevronDown size={18} className={`text-zinc-400 flex-shrink-0 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
    </button>
    {isOpen && (
      <div className="px-6 pb-5">
        <p className="faq-answer text-zinc-400 text-sm leading-relaxed">{answer}</p>
      </div>
    )}
  </div>
);

export default function Landing({ onLaunchApp }) {
  const [openFaq, setOpenFaq] = React.useState(null);

  const features = [
    {
      icon: Sparkles,
      title: "AI Viral Moment Detection",
      description: "Google Gemini 3.0 Flash analyzes your video transcript and scene boundaries to detect the 3-15 most engaging moments. Each clip is scored for viral potential based on emotional impact, hook strength, and shareability — similar to how TikTok's algorithm ranks content for the For You page."
    },
    {
      icon: Scissors,
      title: "Smart 9:16 Vertical Cropping",
      description: "Dual-mode AI reframing: TRACK mode follows subjects with MediaPipe face detection + YOLOv8 fallback. GENERAL mode creates blurred backgrounds for group shots and landscapes."
    },
    {
      icon: Subtitles,
      title: "Automatic Subtitle Generation",
      description: "Powered by faster-whisper with word-level timestamps. According to Verizon Media research, 80% of viewers are more likely to watch a video to completion when captions are available. Subtitles are auto-generated, styled, and burned into your clips."
    },
    {
      icon: Languages,
      title: "AI Voice Dubbing in 30+ Languages",
      description: "ElevenLabs AI integration translates and dubs your video audio while preserving the original speaker's voice characteristics. According to CSA Research, 76% of consumers prefer content in their native language — dubbing unlocks global audiences."
    },
    {
      icon: Type,
      title: "Hook Text Overlays",
      description: "Add attention-grabbing text overlays with styled fonts. AI-generated hook titles capture viewers in the first 3 seconds — critical for TikTok and Reels engagement."
    },
    {
      icon: Zap,
      title: "AI Video Effects",
      description: "Google Gemini generates dynamic FFmpeg filters for professional video effects — color grading, transitions, and visual enhancements applied automatically."
    },
    {
      icon: Upload,
      title: "Local Video Upload",
      description: "Upload your long-form videos — podcasts, webinars, livestreams, vlogs — at full original resolution and audio quality. Process content you own or have rights to."
    },
    {
      icon: Shield,
      title: "100% Self-Hosted & Private",
      description: "Deploy with Docker on your own machine. Your videos never leave your infrastructure. API keys are encrypted client-side and never stored on the server."
    },
    {
      icon: Monitor,
      title: "Free AI YouTube Studio",
      description: "Free AI YouTube thumbnail generator, AI title suggestions (10 viral options with refinement chat), and auto-generated descriptions with chapter timestamps — all free. Upload a face photo for personalized thumbnails. Publish directly to YouTube from one workflow."
    },
    {
      icon: Globe,
      title: "Direct Social Publishing",
      description: "Post directly to TikTok, Instagram Reels, and YouTube Shorts from the dashboard. Async uploads with progress tracking and S3 cloud backup."
    },
    {
      icon: Sparkles,
      title: "AI UGC Video Generator",
      description: "Generate marketing videos with AI actors for any product or business. Paste a URL or describe your product — AI writes the script, generates a realistic avatar with lip-sync, adds b-roll, subtitles, and hook overlays. From $0.65/video."
    },
    {
      icon: FileVideo,
      title: "AI Actors & Lip-Sync",
      description: "Choose from a gallery of AI-generated actors or upload your own photo. The pipeline generates a talking head video with natural movement and lip-synced voiceover in English or Spanish. Two modes: Low Cost ($0.65) and Premium ($2.00)."
    }
  ];

  const steps = [
    { title: "Upload a Long-Form Video", description: "Drop any video file you own — podcasts, webinars, livestreams, interviews. OpenShorts supports all common formats and resolutions." },
    { title: "AI Detects the Best Viral Moments", description: "Google Gemini 3.0 Flash transcribes, analyzes scene boundaries, and identifies 3-15 high-potential clips of 15-60 seconds each." },
    { title: "Smart Cropping to Vertical 9:16", description: "AI reframes each clip to vertical format with face tracking. Subjects stay centered with stabilized camera movement — no manual positioning." },
    { title: "Add Subtitles, Hooks & Effects", description: "Auto-generate styled subtitles, add hook text overlays, and apply AI video effects. Optionally dub into 30+ languages." },
    { title: "Download or Post to Social Media", description: "Export your viral-ready clips or post directly to TikTok, Instagram Reels, and YouTube Shorts from the dashboard." }
  ];

  const faqs = [
    {
      question: "What is OpenShorts and how does it work?",
      answer: "OpenShorts is a free, open source AI clip generator that transforms your long-form videos — podcasts, webinars, livestreams, vlogs, interviews — into viral-ready short clips in 9:16 vertical format. It uses a multi-step AI pipeline: faster-whisper for transcription with word-level timestamps, PySceneDetect for scene boundary detection, and Google Gemini 3.0 Flash AI for identifying the most engaging viral moments. According to HubSpot's 2025 State of Marketing report, short-form video delivers the highest ROI of any content format, and repurposing long-form content into shorts increases total reach by up to 300%."
    },
    {
      question: "Is OpenShorts really free? What's the catch?",
      answer: "OpenShorts is 100% free and open source. You self-host it using Docker on your own machine or server. It uses three external APIs — all with free tiers. Google Gemini API (required) powers the AI analysis, viral moment detection, and thumbnail generation — its free tier includes 1,500 requests per day. ElevenLabs API (optional) enables AI voice dubbing in 30+ languages — free tier included. Upload-Post API (optional) is a social media API that allows direct publishing to YouTube, TikTok, and Instagram — 10 free uploads/month, no credit card required. There are no watermarks, no usage limits, no monthly subscriptions, and no per-video fees — unlike Opus Clip ($15-228/month) or Kapwing ($24-79/month)."
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
      question: "Can OpenShorts generate YouTube thumbnails and titles for free?",
      answer: "Yes. OpenShorts includes a free AI YouTube thumbnail generator, a free AI YouTube title generator, and a free AI YouTube description generator — all powered by Google Gemini 3.0 Flash. Upload your video and the AI suggests 10 viral title options with an interactive refinement chat. Then it generates multiple thumbnail designs using AI image generation — upload a face photo and background image for personalized results. The studio also auto-generates YouTube descriptions with chapter timestamps and lets you publish directly to YouTube. Everything is 100% free with the Gemini free tier."
    },
    {
      question: "What are the system requirements to run OpenShorts?",
      answer: "OpenShorts runs on any system with Docker installed. The recommended setup is 8GB+ RAM and a modern multi-core CPU. GPU acceleration (NVIDIA CUDA) is optional but speeds up video processing significantly. The Docker Compose setup handles all dependencies automatically — Python 3.11, FFmpeg, YOLOv8, MediaPipe, faster-whisper, and the React dashboard. It works on Linux, macOS, and Windows (via WSL2/Docker Desktop)."
    },
    {
      question: "Is there a free open source clip generator?",
      answer: "Yes — OpenShorts is a 100% free, open source clip generator. Unlike paid clip generators like Opus Clip ($15-228/month) or Kapwing ($24-79/month), OpenShorts lets you generate unlimited clips with no watermarks, no usage limits, and no subscription fees. It also includes a free AI YouTube thumbnail generator, free AI YouTube title generator, and free AI YouTube description generator — features that other clip generators charge extra for. You self-host it with Docker on your own machine for full privacy and control."
    },
    {
      question: "What is the AI UGC Video Generator?",
      answer: "OpenShorts includes an AI UGC (User Generated Content) video creator that generates marketing videos with AI actors for any product or business. You describe your product or paste a website URL — the AI writes a viral script, generates a realistic AI actor with lip-synced voiceover, adds b-roll visuals, TikTok-style subtitles, and hook text overlays. The result is a ready-to-post vertical video for TikTok, Instagram Reels, or YouTube Shorts. Two cost modes: Low Cost (~$0.65/video using Hailuo + VEED Lipsync) and Premium (~$2/video using Kling Avatar v2)."
    },
    {
      question: "How much does it cost to generate an AI UGC video?",
      answer: "OpenShorts itself is free, but the AI Shorts feature uses external APIs (fal.ai for video generation, ElevenLabs for voiceover) that charge per use. Low Cost mode costs approximately $0.65 per video (Flux image $0.05 + ElevenLabs voice $0.10 + Hailuo img2video $0.19 + VEED Lipsync $0.20 + b-roll $0.10). Premium mode costs approximately $2.00 per video using Kling Avatar v2 for higher quality. Both modes are significantly cheaper than hiring UGC creators ($50-500 per video) or using platforms like HeyGen ($24-180/month)."
    },
    {
      question: "Can I use the AI UGC Video Generator for any type of business?",
      answer: "Yes. The AI Shorts generator works for any product, service, or business — not just SaaS. You can use it for restaurants, e-commerce stores, coaching services, local businesses, personal brands, apps, and more. Just describe your business in the text field (e.g. 'Artisan pizza restaurant in Madrid, wood-fired oven, home delivery') or paste your website URL, and the AI generates viral marketing scripts tailored to your business."
    }
  ];

  const checkIcon = <Check size={16} className="text-green-400 mx-auto" />;
  const xIcon = <span className="text-zinc-500 text-sm">Paid</span>;

  return (
    <div className="min-h-screen bg-background text-white">
      {/* Navigation */}
      <nav className="fixed top-0 w-full z-50 bg-background/80 backdrop-blur-xl border-b border-white/5">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <img src="/logo-openshorts.png" alt="OpenShorts logo" className="w-8 h-8" />
            <span className="text-lg font-bold">OpenShorts</span>
          </div>
          <div className="hidden md:flex items-center gap-8 text-sm text-zinc-400">
            <a href="#features" className="hover:text-white transition-colors">Features</a>
            <a href="#how-it-works" className="hover:text-white transition-colors">How It Works</a>
            <a href="#comparison" className="hover:text-white transition-colors">Comparison</a>
            <a href="#faq" className="hover:text-white transition-colors">FAQ</a>
          </div>
          <div className="flex items-center gap-3">
            <a
              href="https://github.com/mutonby/openshorts"
              target="_blank"
              rel="noopener noreferrer"
              className="hidden sm:flex items-center gap-2 text-sm text-zinc-400 hover:text-white transition-colors"
            >
              <Github size={18} />
              <span>GitHub</span>
            </a>
            <button
              onClick={onLaunchApp}
              className="bg-primary hover:bg-blue-600 text-white px-5 py-2 rounded-xl text-sm font-medium transition-all active:scale-[0.98] shadow-lg shadow-primary/20"
            >
              Launch App
            </button>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="pt-32 pb-20 px-6">
        <div className="max-w-5xl mx-auto text-center">
          <div className="inline-flex items-center gap-2 bg-primary/10 border border-primary/20 rounded-full px-4 py-1.5 text-sm text-primary mb-8">
            <Sparkles size={14} />
            <span>Free & Open Source AI Clip Generator + UGC Video Creator</span>
          </div>

          <h1 className="text-4xl md:text-6xl lg:text-7xl font-bold leading-tight mb-6 tracking-tight">
            Free Open Source
            <span className="bg-gradient-to-r from-primary via-purple-400 to-pink-500 bg-clip-text text-transparent"> Clip Generator </span>
            & AI UGC Video Creator
          </h1>

          <p className="hero-description text-lg md:text-xl text-zinc-400 max-w-3xl mx-auto mb-10 leading-relaxed">
            Three tools in one. <strong className="text-white">Clip Generator:</strong> turn your long-form videos into viral shorts with AI moment detection, smart 9:16 crop, and auto subtitles. <strong className="text-white">AI Shorts:</strong> generate UGC marketing videos with AI actors and lip-sync for any business. <strong className="text-white">YouTube Studio:</strong> free AI thumbnail generator, 10 viral title suggestions with refinement chat, and auto descriptions with chapters. Self-hosted, open source, no limits.
          </p>

          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-12">
            <button
              onClick={onLaunchApp}
              className="flex items-center gap-2 bg-primary hover:bg-blue-600 text-white px-8 py-3.5 rounded-xl font-medium transition-all active:scale-[0.98] shadow-lg shadow-primary/20 text-lg"
            >
              Get Started Free
              <ArrowRight size={20} />
            </button>
            <a
              href="https://github.com/mutonby/openshorts"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 bg-white/5 border border-white/10 text-white px-8 py-3.5 rounded-xl font-medium transition-all hover:bg-white/10 text-lg"
            >
              <Github size={20} />
              View on GitHub
            </a>
          </div>

          {/* Platform Icons */}
          <div className="flex items-center justify-center gap-6 text-zinc-500">
            <span className="text-sm">Export to:</span>
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-1.5 text-zinc-400">
                <TikTokIcon size={18} />
                <span className="text-sm">TikTok</span>
              </div>
              <div className="flex items-center gap-1.5 text-zinc-400">
                <Instagram size={18} />
                <span className="text-sm">Reels</span>
              </div>
              <div className="flex items-center gap-1.5 text-zinc-400">
                <Youtube size={18} />
                <span className="text-sm">Shorts</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Stats Bar */}
      <section className="border-y border-white/5 bg-surface/30">
        <div className="max-w-5xl mx-auto px-6 py-10 grid grid-cols-2 md:grid-cols-4 gap-8 text-center">
          <div>
            <div className="text-3xl font-bold text-white">100%</div>
            <div className="text-sm text-zinc-400 mt-1">Free & Open Source</div>
          </div>
          <div>
            <div className="text-3xl font-bold text-white">3</div>
            <div className="text-sm text-zinc-400 mt-1">Tools in One</div>
          </div>
          <div>
            <div className="text-3xl font-bold text-white">30+</div>
            <div className="text-sm text-zinc-400 mt-1">Dubbing Languages</div>
          </div>
          <div>
            <div className="text-3xl font-bold text-white">$0</div>
            <div className="text-sm text-zinc-400 mt-1">No Watermarks</div>
          </div>
        </div>
      </section>

      {/* 3 Tools in 1 Section */}
      <section className="py-20 px-6">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-14">
            <h2 className="text-3xl md:text-4xl font-bold mb-4">3 Free Tools in 1 Platform</h2>
            <p className="text-zinc-400 max-w-2xl mx-auto">Everything you need to create, optimize, and publish short-form video content — all free and open source.</p>
          </div>
          <div className="grid md:grid-cols-3 gap-6">
            <div className="bg-surface/50 border border-primary/20 rounded-2xl p-8 relative overflow-hidden">
              <div className="absolute top-0 right-0 w-32 h-32 bg-primary/5 rounded-full -translate-y-1/2 translate-x-1/2" />
              <Scissors size={28} className="text-primary mb-4" />
              <h3 className="text-xl font-bold text-white mb-2">Clip Generator</h3>
              <p className="text-zinc-400 text-sm leading-relaxed mb-4">Turn your long-form videos into viral-ready 9:16 shorts. AI detects the best moments, crops to vertical with face tracking, and adds subtitles automatically.</p>
              <ul className="space-y-1.5">
                {['AI viral moment detection', 'Smart face-tracking crop', 'Auto subtitles + hook overlays', 'AI dubbing in 30+ languages'].map((f, i) => (
                  <li key={i} className="flex items-center gap-2 text-xs text-zinc-400"><Check size={12} className="text-green-400 shrink-0" />{f}</li>
                ))}
              </ul>
            </div>
            <div className="bg-surface/50 border border-violet-500/20 rounded-2xl p-8 relative overflow-hidden">
              <div className="absolute top-0 right-0 w-32 h-32 bg-violet-500/5 rounded-full -translate-y-1/2 translate-x-1/2" />
              <Sparkles size={28} className="text-violet-400 mb-4" />
              <h3 className="text-xl font-bold text-white mb-2">AI Shorts</h3>
              <p className="text-zinc-400 text-sm leading-relaxed mb-4">Generate UGC marketing videos with AI actors for any product or business. No camera, no studio. Just describe your product and get a viral-ready video.</p>
              <ul className="space-y-1.5">
                {['AI actor generation + lip-sync', 'Script writing from URL or description', 'B-roll + TikTok-style subtitles', 'From $0.65 per video'].map((f, i) => (
                  <li key={i} className="flex items-center gap-2 text-xs text-zinc-400"><Check size={12} className="text-green-400 shrink-0" />{f}</li>
                ))}
              </ul>
            </div>
            <div className="bg-surface/50 border border-pink-500/20 rounded-2xl p-8 relative overflow-hidden">
              <div className="absolute top-0 right-0 w-32 h-32 bg-pink-500/5 rounded-full -translate-y-1/2 translate-x-1/2" />
              <Monitor size={28} className="text-pink-400 mb-4" />
              <h3 className="text-xl font-bold text-white mb-2">YouTube Studio</h3>
              <p className="text-zinc-400 text-sm leading-relaxed mb-4">Complete free AI YouTube toolkit. Generate thumbnails with your face, get 10 viral title suggestions with refinement chat, and auto-generate descriptions with timestamps.</p>
              <ul className="space-y-1.5">
                {['AI thumbnail generator (with face upload)', '10 viral title suggestions + chat', 'Auto descriptions with chapters', 'Direct publish to YouTube'].map((f, i) => (
                  <li key={i} className="flex items-center gap-2 text-xs text-zinc-400"><Check size={12} className="text-green-400 shrink-0" />{f}</li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-20 px-6">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-14">
            <h2 className="text-3xl md:text-4xl font-bold mb-4">Free AI Clip Generator + UGC Video Creator</h2>
            <p className="text-zinc-400 max-w-2xl mx-auto">Three tools in one: clip long videos into viral shorts, generate UGC marketing videos with AI actors, and a complete YouTube Studio for thumbnails, titles, and descriptions.</p>
          </div>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-5">
            {features.map((feature, i) => (
              <FeatureCard key={i} {...feature} />
            ))}
          </div>
        </div>
      </section>

      {/* API Keys Section */}
      <section className="py-20 px-6 bg-surface/20">
        <div className="max-w-5xl mx-auto">
          <div className="text-center mb-14">
            <h2 className="text-3xl md:text-4xl font-bold mb-4">All APIs Have Free Tiers</h2>
            <p className="text-zinc-400 max-w-2xl mx-auto">OpenShorts uses three external APIs — all with generous free tiers. Only Gemini is required. Your API keys are encrypted client-side and never stored on the server.</p>
          </div>
          <div className="grid md:grid-cols-3 gap-5">
            <div className="bg-surface/50 border border-white/10 rounded-2xl p-6 relative">
              <div className="absolute top-4 right-4 bg-primary/20 text-primary text-[10px] font-bold px-2 py-0.5 rounded-full border border-primary/30">REQUIRED</div>
              <div className="w-12 h-12 rounded-xl bg-blue-500/10 flex items-center justify-center mb-4">
                <Cpu size={24} className="text-blue-400" />
              </div>
              <h3 className="text-lg font-semibold text-white mb-1">Google Gemini API</h3>
              <span className="inline-block text-xs text-green-400 bg-green-500/10 border border-green-500/20 px-2 py-0.5 rounded-full mb-3">Free tier: 1,500 req/day</span>
              <p className="text-zinc-400 text-sm leading-relaxed">Powers all AI features: viral moment detection, title generation, video effects, YouTube thumbnail creation, and description writing. The core engine of OpenShorts.</p>
            </div>
            <div className="bg-surface/50 border border-white/10 rounded-2xl p-6 relative">
              <div className="absolute top-4 right-4 bg-zinc-700/50 text-zinc-400 text-[10px] font-bold px-2 py-0.5 rounded-full border border-zinc-600/30">OPTIONAL</div>
              <div className="w-12 h-12 rounded-xl bg-purple-500/10 flex items-center justify-center mb-4">
                <Languages size={24} className="text-purple-400" />
              </div>
              <h3 className="text-lg font-semibold text-white mb-1">ElevenLabs API</h3>
              <span className="inline-block text-xs text-green-400 bg-green-500/10 border border-green-500/20 px-2 py-0.5 rounded-full mb-3">Free tier included</span>
              <p className="text-zinc-400 text-sm leading-relaxed">Enables AI voice dubbing and translation in 30+ languages. Preserves the original speaker's voice while translating audio. Dubbed clips are auto-subtitled.</p>
            </div>
            <div className="bg-surface/50 border border-white/10 rounded-2xl p-6 relative">
              <div className="absolute top-4 right-4 bg-zinc-700/50 text-zinc-400 text-[10px] font-bold px-2 py-0.5 rounded-full border border-zinc-600/30">OPTIONAL</div>
              <div className="w-12 h-12 rounded-xl bg-pink-500/10 flex items-center justify-center mb-4">
                <Globe size={24} className="text-pink-400" />
              </div>
              <h3 className="text-lg font-semibold text-white mb-1">Upload-Post API</h3>
              <span className="inline-block text-xs text-green-400 bg-green-500/10 border border-green-500/20 px-2 py-0.5 rounded-full mb-3">Free tier included</span>
              <p className="text-zinc-400 text-sm leading-relaxed">Enables direct publishing to YouTube, TikTok, and Instagram Reels from the dashboard. <a href="https://www.upload-post.com" target="_blank" rel="noopener noreferrer" className="text-pink-400 hover:text-pink-300 underline">Social media API</a> that lets you post your clips and thumbnails without leaving OpenShorts.</p>
            </div>
          </div>
          <div className="grid md:grid-cols-2 gap-5 mt-5">
            <div className="bg-surface/50 border border-white/10 rounded-2xl p-6 relative">
              <div className="absolute top-4 right-4 bg-violet-700/50 text-violet-300 text-[10px] font-bold px-2 py-0.5 rounded-full border border-violet-500/30">AI SHORTS</div>
              <div className="w-12 h-12 rounded-xl bg-violet-500/10 flex items-center justify-center mb-4">
                <Zap size={24} className="text-violet-400" />
              </div>
              <h3 className="text-lg font-semibold text-white mb-1">fal.ai API</h3>
              <span className="inline-block text-xs text-green-400 bg-green-500/10 border border-green-500/20 px-2 py-0.5 rounded-full mb-3">Pay-per-use from $0.04</span>
              <p className="text-zinc-400 text-sm leading-relaxed">Powers AI Shorts: generates AI actor images (Flux), talking head videos (Hailuo/Kling), and lip-sync (VEED). Required only for the AI UGC video generator.</p>
            </div>
            <div className="bg-surface/50 border border-white/10 rounded-2xl p-6 relative">
              <div className="absolute top-4 right-4 bg-violet-700/50 text-violet-300 text-[10px] font-bold px-2 py-0.5 rounded-full border border-violet-500/30">AI SHORTS</div>
              <div className="w-12 h-12 rounded-xl bg-violet-500/10 flex items-center justify-center mb-4">
                <Languages size={24} className="text-violet-400" />
              </div>
              <h3 className="text-lg font-semibold text-white mb-1">ElevenLabs TTS</h3>
              <span className="inline-block text-xs text-green-400 bg-green-500/10 border border-green-500/20 px-2 py-0.5 rounded-full mb-3">Free tier included</span>
              <p className="text-zinc-400 text-sm leading-relaxed">Generates natural voiceovers for AI Shorts from the script. Multiple voice options for male and female actors in English and Spanish.</p>
            </div>
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section id="how-it-works" className="py-20 px-6">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-14">
            <h2 className="text-3xl md:text-4xl font-bold mb-4">How It Works</h2>
            <p className="text-zinc-400 max-w-2xl mx-auto">From a long-form video to viral-ready clips in 5 automated steps. The entire pipeline runs on your machine with AI doing the heavy lifting.</p>
          </div>
          <div className="space-y-8">
            {steps.map((step, i) => (
              <StepCard key={i} number={i + 1} {...step} />
            ))}
          </div>
        </div>
      </section>

      {/* Tech Stack */}
      <section className="py-20 px-6">
        <div className="max-w-5xl mx-auto">
          <div className="text-center mb-14">
            <h2 className="text-3xl md:text-4xl font-bold mb-4">Built with Proven Technology</h2>
            <p className="text-zinc-400 max-w-2xl mx-auto">OpenShorts combines industry-leading AI models and open source tools into a production-ready video processing pipeline.</p>
          </div>
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
              <div key={i} className="bg-surface/50 border border-white/10 rounded-xl p-4 text-center">
                <div className="text-white font-medium text-sm">{tech.name}</div>
                <div className="text-zinc-500 text-xs mt-1">{tech.desc}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Comparison Table */}
      <section id="comparison" className="py-20 px-6 bg-surface/20">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-14">
            <h2 className="text-3xl md:text-4xl font-bold mb-4">Free Clip Generator vs Paid Alternatives</h2>
            <p className="text-zinc-400 max-w-2xl mx-auto">Why pay $15-228/month for an AI clip generator when you can self-host the same capabilities for free? OpenShorts includes a free YouTube thumbnail generator, AI title suggestions, and auto descriptions — features that paid tools charge extra for.</p>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-white/10">
                  <th className="py-3 px-4 text-left text-sm text-zinc-400 font-medium">Feature</th>
                  <th className="py-3 px-4 text-center text-sm font-medium">
                    <span className="text-primary">OpenShorts</span>
                  </th>
                  <th className="py-3 px-4 text-center text-sm text-zinc-400 font-medium">Opus Clip</th>
                  <th className="py-3 px-4 text-center text-sm text-zinc-400 font-medium">Kapwing</th>
                </tr>
              </thead>
              <tbody>
                <ComparisonRow feature="Price" openshorts={<span className="text-green-400 font-semibold">$0 Free</span>} opusclip={xIcon} kapwing={xIcon} />
                <ComparisonRow feature="AI Viral Moment Detection" openshorts={checkIcon} opusclip={checkIcon} kapwing={checkIcon} />
                <ComparisonRow feature="Smart Vertical Cropping" openshorts={checkIcon} opusclip={checkIcon} kapwing={checkIcon} />
                <ComparisonRow feature="Auto Subtitles" openshorts={checkIcon} opusclip={checkIcon} kapwing={checkIcon} />
                <ComparisonRow feature="AI Voice Dubbing (30+ langs)" openshorts={checkIcon} opusclip={<span className="text-zinc-500 text-sm">Limited</span>} kapwing={<span className="text-zinc-500 text-sm">No</span>} />
                <ComparisonRow feature="AI Video Effects" openshorts={checkIcon} opusclip={<span className="text-zinc-500 text-sm">No</span>} kapwing={checkIcon} />
                <ComparisonRow feature="Hook Text Overlays" openshorts={checkIcon} opusclip={checkIcon} kapwing={checkIcon} />
                <ComparisonRow feature="Self-Hosted / Privacy" openshorts={checkIcon} opusclip={<span className="text-zinc-500 text-sm">Cloud only</span>} kapwing={<span className="text-zinc-500 text-sm">Cloud only</span>} />
                <ComparisonRow feature="No Watermark" openshorts={checkIcon} opusclip={<span className="text-zinc-500 text-sm">Free tier only</span>} kapwing={<span className="text-zinc-500 text-sm">Paid</span>} />
                <ComparisonRow feature="Open Source" openshorts={checkIcon} opusclip={<span className="text-zinc-500 text-sm">No</span>} kapwing={<span className="text-zinc-500 text-sm">No</span>} />
                <ComparisonRow feature="AI YouTube Thumbnail Generator" openshorts={checkIcon} opusclip={<span className="text-zinc-500 text-sm">No</span>} kapwing={<span className="text-zinc-500 text-sm">Paid</span>} />
                <ComparisonRow feature="AI Title & Description Generator" openshorts={checkIcon} opusclip={<span className="text-zinc-500 text-sm">Limited</span>} kapwing={<span className="text-zinc-500 text-sm">Paid</span>} />
                <ComparisonRow feature="AI UGC Video Generator" openshorts={checkIcon} opusclip={<span className="text-zinc-500 text-sm">No</span>} kapwing={<span className="text-zinc-500 text-sm">No</span>} />
                <ComparisonRow feature="AI Actors with Lip-Sync" openshorts={checkIcon} opusclip={<span className="text-zinc-500 text-sm">No</span>} kapwing={<span className="text-zinc-500 text-sm">No</span>} />
                <ComparisonRow feature="Usage Limits" openshorts={<span className="text-green-400 text-sm">Unlimited</span>} opusclip={<span className="text-zinc-500 text-sm">Per plan</span>} kapwing={<span className="text-zinc-500 text-sm">Per plan</span>} />
              </tbody>
            </table>
          </div>
        </div>
      </section>

      {/* Use Cases */}
      <section className="py-20 px-6">
        <div className="max-w-5xl mx-auto">
          <div className="text-center mb-14">
            <h2 className="text-3xl md:text-4xl font-bold mb-4">Who Uses OpenShorts?</h2>
            <p className="text-zinc-400 max-w-2xl mx-auto">Content creators, marketers, and agencies use OpenShorts to scale their short-form video production. According to HubSpot's 2025 report, short-form video is the #1 content format with the highest ROI.</p>
          </div>
          <div className="grid md:grid-cols-3 gap-5">
            {[
              {
                title: "Content Creators",
                description: "Repurpose your long-form videos into TikTok and Reels clips automatically. According to YouTube's Creator Insider data, channels that post Shorts alongside long-form videos see 20-30% more subscriber growth.",
                icon: Youtube
              },
              {
                title: "Social Media Managers",
                description: "Scale short-form content production for multiple clients. According to Sprout Social's 2025 Index, 66% of consumers find short-form video the most engaging content type. Process videos in batch and publish directly from one dashboard.",
                icon: Instagram
              },
              {
                title: "Podcasters & Educators",
                description: "Extract the most engaging moments from podcast episodes and educational content. Research by Headliner shows that podcast clips on social media increase episode downloads by 72% on average.",
                icon: FileVideo
              },
              {
                title: "Businesses & Brands",
                description: "Generate UGC-style marketing videos for any product or business with AI actors. No camera, no studio, no influencer budget. Just describe your product and get a viral-ready video with lip-synced AI avatar, voiceover, b-roll, and subtitles — from $0.65 per video.",
                icon: Sparkles
              }
            ].map((useCase, i) => (
              <div key={i} className="bg-surface/50 border border-white/10 rounded-2xl p-6">
                <useCase.icon size={24} className="text-primary mb-4" />
                <h3 className="text-lg font-semibold text-white mb-2">{useCase.title}</h3>
                <p className="text-zinc-400 text-sm leading-relaxed">{useCase.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* FAQ Section */}
      <section id="faq" className="py-20 px-6 bg-surface/20">
        <div className="max-w-3xl mx-auto">
          <div className="text-center mb-14">
            <h2 className="text-3xl md:text-4xl font-bold mb-4">Frequently Asked Questions</h2>
            <p className="text-zinc-400">Everything you need to know about OpenShorts, from setup to features.</p>
          </div>
          <div className="space-y-3">
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

      {/* CTA Section */}
      <section className="py-20 px-6">
        <div className="max-w-3xl mx-auto text-center">
          <h2 className="text-3xl md:text-4xl font-bold mb-4">Start Creating Viral Videos for Free</h2>
          <p className="text-zinc-400 mb-8 max-w-xl mx-auto">No sign-up, no credit card, no watermarks. Generate viral clips from long videos or create AI UGC marketing videos with AI actors for any business. Self-host with Docker.</p>
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <button
              onClick={onLaunchApp}
              className="flex items-center gap-2 bg-primary hover:bg-blue-600 text-white px-8 py-3.5 rounded-xl font-medium transition-all active:scale-[0.98] shadow-lg shadow-primary/20 text-lg"
            >
              Launch OpenShorts
              <ArrowRight size={20} />
            </button>
            <a
              href="https://github.com/mutonby/openshorts"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 text-zinc-400 hover:text-white transition-colors text-sm"
            >
              <Github size={18} />
              Star on GitHub
            </a>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-white/5 py-10 px-6">
        <div className="max-w-5xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <img src="/logo-openshorts.png" alt="OpenShorts" className="w-6 h-6" />
            <span className="text-sm text-zinc-400">OpenShorts — Free Open Source Clip Generator & AI UGC Video Creator</span>
          </div>
          <div className="flex items-center gap-6 text-sm text-zinc-500">
            <a href="https://github.com/mutonby/openshorts" target="_blank" rel="noopener noreferrer" className="hover:text-white transition-colors">GitHub</a>
            <a href="#features" className="hover:text-white transition-colors">Features</a>
            <a href="#faq" className="hover:text-white transition-colors">FAQ</a>
          </div>
        </div>
        <div className="max-w-5xl mx-auto mt-6 pt-4 border-t border-white/5 text-center">
          <span className="text-xs text-zinc-600">Made with ❤️ by <a href="https://www.upload-post.com" target="_blank" rel="noopener noreferrer" className="text-zinc-500 hover:text-white transition-colors">Upload-Post</a></span>
        </div>
      </footer>
    </div>
  );
}
