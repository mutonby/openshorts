import React from 'react';
import { Sparkles, Zap, Globe, FileVideo, Subtitles, Youtube, Instagram, Shield, Github, ArrowRight, Check, ChevronDown, Monitor, Cpu, Languages, Type, Upload, Scissors, Link2, Bot } from 'lucide-react';
import PricingSection from './components/PricingSection';
import SiteNav from './components/SiteNav';
import { useAuth } from './contexts/AuthContext';
import './landing.css';

const APPARATUS_CALLOUTS = ['RATIO · 9:16', 'CLIPS · 3–15', 'HOOK · AUTO', 'SUBS · WORD-LEVEL'];

// Real clips OpenShorts made from Creative Commons (CC BY) sources. `video` is
// the original and the clip side by side, in sync; `vertical` is the clip on
// its own for the hero. The credit is what the licence asks for, so it stays
// next to the video that uses it.
// The six clips OpenShorts cut from that episode with no clip-count setting,
// in source order, with the score the moment picker gave each.
const EPISODE_CLIPS = [
  { n: 1, score: 88, title: 'I knew I was marrying her on our third date' },
  { n: 2, score: 82, title: 'Why we do absolutely everything together' },
  { n: 3, score: 75, title: 'She did a deep dive on my old social media' },
  { n: 4, score: 85, title: 'How we won a bidding war with a photo' },
  { n: 5, score: 72, title: 'Using our past to help others find freedom' },
  { n: 6, score: 78, title: 'The real definition of a winner' },
];

// Mounts the video only once it scrolls near the viewport, so five clip
// previews cost nothing to a visitor who never gets that far.
function LazyLoopVideo({ src, poster }) {
  const ref = React.useRef(null);
  const [show, setShow] = React.useState(false);
  React.useEffect(() => {
    const el = ref.current;
    if (!el || typeof IntersectionObserver === 'undefined') { setShow(true); return undefined; }
    const io = new IntersectionObserver(([e]) => { if (e.isIntersecting) { setShow(true); io.disconnect(); } }, { rootMargin: '200px' });
    io.observe(el);
    return () => io.disconnect();
  }, []);
  return (
    <div ref={ref} className="aspect-[9/16] w-full">
      {show ? (
        <video src={src} poster={poster} autoPlay muted loop playsInline preload="metadata" className="w-full h-full object-cover block" />
      ) : (
        <img src={poster} alt="" loading="lazy" className="w-full h-full object-cover block" />
      )}
    </div>
  );
}

const DEMOS = [
  {
    id: 'split',
    tab: 'two speakers',
    video: '/demo/case-split.mp4',
    poster: '/screens/case-split.webp',
    vertical: '/demo/split-vertical.mp4',
    posterV: '/screens/split-vertical.webp',
    credit: 'Turn the Tables with Dan and Shoshana Jordan',
    creditBy: 'Heritage of Faith',
    creditUrl: 'https://www.youtube.com/watch?v=-KbQj_vboOU',
  },
  {
    id: 'talk',
    tab: 'face tracking',
    video: '/demo/case-talk.mp4',
    poster: '/screens/case-talk.webp',
    vertical: '/demo/talk-vertical.mp4',
    posterV: '/screens/talk-vertical.webp',
    credit: 'Mark Bush on How to Overcome Adversity',
    creditBy: 'Foundation for Economic Education',
    creditUrl: 'https://www.youtube.com/watch?v=aqFhllJJzoQ',
  },
];

const SectionHeader = ({ eyebrow, title, children }) => (
  <div className="mb-12">
    <p className="eyebrow mb-3">{eyebrow}</p>
    <h2 className="font-display text-3xl md:text-4xl lowercase text-ink tracking-tight mb-4">{title}</h2>
    {children && <p className="text-muted max-w-2xl leading-relaxed">{children}</p>}
  </div>
);

// Every feature with its real visual: screenshots of the app and clips it
// made. Rotates on its own until the visitor picks a tab.
const SHOWCASE = [
  {
    id: 'moments', tab: 'finds the moments',
    title: 'every clip scored before you see it',
    body: 'Gemini reads the word-level transcript and the scene cuts, scores each moment out of 100, and writes the title and the description for every platform.',
    media: { type: 'img', src: '/screens/app-clipcard.webp', w: 1153, h: 441, alt: 'A clip in OpenShorts with its viral score, titles and edit actions' },
  },
  {
    id: 'reframe', tab: 'reframes any layout',
    title: 'one face, two speakers, a whole screen',
    body: 'Face tracking for one person, both speakers stacked for a two-shot, the screen kept whole for a screencast. Picked per video, nothing to configure.',
    media: { type: 'video', src: '/demo/case-split.mp4', poster: '/screens/case-split.webp', w: 960, h: 540 },
  },
  {
    id: 'captions', tab: '17 caption styles',
    title: 'captions people actually read',
    body: 'Word-level captions burned in, in 17 styles: Hormozi, pill, lime box, one word at a time, karaoke and more. Change the style of any clip in one click.',
    media: { type: 'captions' },
  },
  {
    id: 'hook', tab: 'hooks that stop the scroll',
    title: 'a hook on the first second',
    body: 'Every clip opens with a hook line written from what is said in it. Rewrite it, restyle it or move it before you post.',
    media: { type: 'img', src: '/screens/app-hook.webp', w: 548, h: 587, alt: 'The hook editor with text, style, font and position' },
  },
  {
    id: 'editor', tab: 'a real editor',
    title: 'fix anything, down to the word',
    body: 'Source and program monitors side by side with the full transcript: click a word to set a cut, add segments, change the framing, re-render in seconds.',
    media: { type: 'img', src: '/screens/app-editor.webp', w: 1568, h: 652, alt: 'The OpenShorts clip editor with source monitor, program monitor, transcript and segments' },
  },
  {
    id: 'autopilot', tab: 'autopilot',
    title: 'your channel, clipped on its own',
    body: 'Connect YouTube and every new upload becomes clips, optionally scheduled to your socials one a day. On paid plans.',
    media: { type: 'img', src: '/screens/autopilot-settings.webp', w: 920, h: 672, alt: 'Autopilot settings' },
  },
  {
    id: 'agents', tab: 'inside claude & chatgpt',
    title: 'clip from a chat',
    body: 'Add OpenShorts to Claude or ChatGPT with one URL and ask for clips. Same pipeline, same minutes, no API key to manage.',
    media: { type: 'img', src: '/screens/connect-an-agent.webp', w: 684, h: 366, alt: 'Connecting OpenShorts to an AI agent' },
  },
  {
    id: 'ugc', tab: 'ai ugc videos',
    title: 'product videos with AI actors',
    body: 'Describe a product or paste its URL: script, lip-synced AI actor, B-roll and captions, for well under a dollar a video on the low-cost mode.',
    media: { type: 'img', src: '/screens/app-ugc.webp', w: 1114, h: 612, alt: 'The UGC gallery with AI actor videos' },
  },
];

const CAPTION_STYLES = [
  ['default', 'default'], ['hormozi', 'hormozi'], ['pill', 'pill'], ['lime', 'lime box'], ['oneword', 'one word'],
];

function ShowcaseMedia({ media }) {
  if (media.type === 'video') {
    return (
      <video src={media.src} poster={media.poster} autoPlay muted loop playsInline preload="metadata"
        width={media.w} height={media.h} className="w-full h-auto block rounded-card" />
    );
  }
  if (media.type === 'captions') {
    return (
      <div className="grid grid-cols-5 gap-2 sm:gap-3">
        {CAPTION_STYLES.map(([id, label]) => (
          <figure key={id} className="m-0 min-w-0">
            <video src={`/demo/caption-${id}.mp4`} poster={`/screens/caption-${id}.webp`} autoPlay muted loop playsInline
              preload="metadata" className="w-full aspect-[9/16] object-cover block rounded-card border border-rule" />
            <figcaption className="readout text-[10px] text-muted mt-2 text-center">{label}</figcaption>
          </figure>
        ))}
      </div>
    );
  }
  return (
    <img src={media.src} alt={media.alt} width={media.w} height={media.h} loading="lazy"
      className="w-full h-auto block rounded-card border border-rule" />
  );
}

function FeatureShowcase() {
  const [active, setActive] = React.useState(0);
  const [auto, setAuto] = React.useState(true);
  React.useEffect(() => {
    if (!auto) return undefined;
    const t = setTimeout(() => setActive((i) => (i + 1) % SHOWCASE.length), 7000);
    return () => clearTimeout(t);
  }, [active, auto]);
  const item = SHOWCASE[active];
  return (
    <div className="grid lg:grid-cols-[17rem_minmax(0,1fr)] gap-6 lg:gap-10 items-start">
      <div className="flex lg:flex-col gap-2 overflow-x-auto pb-1 -mx-1 px-1" role="tablist" aria-label="Features">
        {SHOWCASE.map((f, i) => (
          <button
            key={f.id}
            type="button"
            role="tab"
            aria-selected={i === active}
            onClick={() => { setActive(i); setAuto(false); }}
            className={`relative shrink-0 text-left rounded-card border px-4 py-3 text-sm lowercase transition-colors overflow-hidden ${i === active ? 'border-brass text-ink bg-paper2' : 'border-rule text-muted hover:text-ink'}`}
          >
            {f.tab}
            {i === active && auto && (
              <span key={`p-${active}`} className="showcase-progress absolute left-0 bottom-0 h-0.5 bg-brass" aria-hidden="true" />
            )}
          </button>
        ))}
      </div>
      <div key={item.id} className="showcase-in min-w-0" role="tabpanel">
        <h3 className="font-display text-3xl md:text-4xl lowercase text-ink mb-3">{item.title}</h3>
        <p className="text-muted max-w-2xl mb-6 leading-relaxed">{item.body}</p>
        <ShowcaseMedia media={item.media} />
      </div>
    </div>
  );
}

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

// Internal links to the per-platform pages. The same links, in the same order,
// are in seo/landing-fallback.js so they exist in the static HTML too.
const LINK = 'text-ink2 underline underline-offset-2 hover:text-ink transition-colors';

const WORKS_WITH = [
  ['/platforms/youtube', 'youtube'],
  ['/platforms/twitch', 'twitch'],
  ['/platforms/kick', 'kick'],
  ['/platforms/tiktok', 'tiktok'],
  ['/podcast-to-shorts', 'podcasts'],
];

const PLATFORMS = [
  {
    icon: Youtube,
    title: 'YouTube clips',
    text: 'Paste a YouTube link and get vertical clips back, Shorts for your own channel, or the whole video turned into 9:16 pieces.',
    links: [['/platforms/youtube', 'YouTube clip maker'], ['/platforms/youtube-shorts', 'YouTube Shorts maker'], ['/youtube-to-shorts-converter', 'YouTube to Shorts converter']],
  },
  {
    icon: Instagram,
    title: 'TikTok & Reels',
    text: 'The same clips, captioned and with a hook, sent to TikTok and Instagram Reels from the dashboard.',
    links: [['/platforms/tiktok', 'TikTok clip maker'], ['/platforms/instagram-reels', 'AI reel generator']],
  },
  {
    icon: Monitor,
    title: 'Streams',
    text: 'Hours-long Twitch and Kick VODs clipped in parts, with the facecam kept and quiet gameplay still handled.',
    links: [['/platforms/twitch', 'Twitch clip maker'], ['/platforms/kick', 'Kick clip maker'], ['/gta-5-clips', 'GTA 5 clips'], ['/gta-6-clips', 'GTA 6 clips'], ['/wow-forever-clips', 'WoW Forever clips']],
  },
  {
    icon: FileVideo,
    title: 'Podcasts & talks',
    text: 'Both podcast hosts kept in frame, only the sermon clipped out of a service, and a speaker who walks the stage tracked.',
    links: [['/podcast-to-shorts', 'Podcast clip maker'], ['/use-cases/sermons', 'Sermon clip maker'], ['/auto-reframe', 'Auto reframe']],
  },
];

const SWITCHING = [
  ['/alternatives/opus-clip', 'vs opus clip'],
  ['/alternatives/klap', 'vs klap'],
  ['/alternatives/vizard', 'vs vizard'],
  ['/alternatives/submagic', 'vs submagic'],
  ['/free-ai-clipping-tools-compared', 'free plans compared'],
  ['/alternatives', 'all alternatives'],
];

// A landing section anchor (#pricing, #features…), not an app route (#app,
// #/pricing, #legal…).
const NOT_SECTIONS = new Set(['app', 'landing', 'legal']);
const sectionId = (hash) => {
  const m = /^#([a-z][\w-]*)$/i.exec(hash || '');
  return m && !NOT_SECTIONS.has(m[1]) ? m[1] : null;
};

// Arriving on /#pricing (the nav's "pricing" from every static page): the
// browser jumps to the anchor in the static fallback, then React replaces the
// fallback and the jump is lost. The pricing section also renders only once
// /api/config says billing is on. So scroll once the section exists, keep it
// aligned for a moment while late content above settles, and give up after a
// few seconds or as soon as the visitor scrolls on their own.
function useScrollToHashSection() {
  React.useEffect(() => {
    const id = sectionId(window.location.hash);
    if (!id) return undefined;
    let stopped = false;
    let foundAt = 0;
    let timer;
    const start = Date.now();
    const stop = () => { stopped = true; };
    const events = ['wheel', 'touchstart', 'keydown', 'mousedown'];
    events.forEach((e) => window.addEventListener(e, stop, { passive: true }));
    const tick = () => {
      if (stopped) return;
      const el = document.getElementById(id);
      if (el) {
        const margin = parseFloat(getComputedStyle(el).scrollMarginTop) || 0;
        if (Math.abs(el.getBoundingClientRect().top - margin) > 4) el.scrollIntoView({ block: 'start' });
        if (!foundAt) foundAt = Date.now();
        if (Date.now() - foundAt > 1500) return;
      } else if (Date.now() - start > 6000) {
        return;
      }
      timer = setTimeout(tick, 100);
    };
    tick();
    return () => {
      stopped = true;
      clearTimeout(timer);
      events.forEach((e) => window.removeEventListener(e, stop));
    };
  }, []);
}

// A click on a link to a section of this page (the nav's /#pricing, the
// in-page #pricing buttons) scrolls smoothly instead of jumping, and closes
// the mobile menu or the open dropdown it came from.
function useSmoothSectionLinks() {
  React.useEffect(() => {
    const onClick = (e) => {
      if (e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
      const a = e.target.closest ? e.target.closest('a[href]') : null;
      if (!a || (a.target && a.target !== '_self')) return;
      let url;
      try { url = new URL(a.getAttribute('href'), window.location.href); } catch { return; }
      if (url.origin !== window.location.origin || url.pathname !== window.location.pathname) return;
      const id = sectionId(url.hash);
      const el = id && document.getElementById(id);
      if (!el) return;
      e.preventDefault();
      const reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      el.scrollIntoView({ behavior: reduce ? 'auto' : 'smooth', block: 'start' });
      if (window.location.hash !== url.hash) window.history.pushState(null, '', url.hash);
      document.querySelectorAll('.osn-toggle').forEach((t) => { t.checked = false; });
      document.querySelectorAll('[data-osn-dd][open]').forEach((d) => { d.open = false; });
    };
    document.addEventListener('click', onClick);
    return () => document.removeEventListener('click', onClick);
  }, []);
}

export default function Landing({ onLaunchApp }) {
  useScrollToHashSection();
  useSmoothSectionLinks();
  const { billingEnabled } = useAuth();
  const [openFaq, setOpenFaq] = React.useState(null);
  // Start from whatever was typed into the static paste box of the crawler
  // fallback (seo/landing-fallback.js) before this component mounted: the
  // fallback DOM is still in #root while the first render runs.
  const [heroUrl, setHeroUrl] = React.useState(() => {
    try { return document.getElementById('seo-hero-url')?.value || ''; } catch { return ''; }
  });
  const [heroDemo, setHeroDemo] = React.useState(0);
  const [cropDemo, setCropDemo] = React.useState(0);
  // Floating CTA: shown once the hero's own input has scrolled away, hidden
  // again near the end where the closing CTA takes over.
  const [stickyCta, setStickyCta] = React.useState(false);
  React.useEffect(() => {
    const onScroll = () => {
      const form = document.querySelector('.hero-input-row');
      const pastHero = form ? form.getBoundingClientRect().bottom < 0 : window.scrollY > 700;
      const nearEnd = window.innerHeight + window.scrollY > document.documentElement.scrollHeight - 1100;
      setStickyCta(pastHero && !nearEnd);
    };
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  // Hand the pasted link to the app: MediaInput picks it up on mount, so the
  // user lands with their own video ready instead of on a pricing page.
  const handleHeroSubmit = (e) => {
    e.preventDefault();
    const url = heroUrl.trim();
    if (url) {
      try { localStorage.setItem('os_pending_url', url); } catch { /* ignore */ }
    }
    onLaunchApp();
  };

  const features = [
    {
      icon: Sparkles,
      title: "AI Viral Moment Detection",
      description: "Google Gemini 3.1 Flash-Lite scores your transcript and scenes to find the 3-15 most engaging moments. Automatic AI clipping, no manual scrubbing."
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
      description: "Bring your own ElevenLabs key and dub a clip into 30+ languages, keeping the speaker's voice. Works in the cloud and self-hosted."
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
      icon: Bot,
      title: "MCP Server, API & CLI for AI Agents",
      description: "Connect Claude, ChatGPT or n8n to an always-on endpoint, or run pip install openshorts and clip from the terminal. Same pipeline, no dashboard needed."
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
    { title: "AI Detects the Best Viral Moments", description: "Google Gemini 3.1 Flash-Lite finds 3-15 high-potential clips of 15-60 seconds." },
    { title: "Smart Cropping to Vertical 9:16", description: "AI reframes to vertical with face tracking — subjects stay centered." },
    { title: "Add Subtitles, Hooks & Effects", description: "Auto subtitles, hook overlays and AI effects. Optional dubbing with your own ElevenLabs key." },
    { title: "Download or Post to Social Media", description: "Export your clips or post directly to TikTok, Instagram Reels, and YouTube Shorts." }
  ];

  const faqs = [
    {
      question: "Is OpenShorts really free?",
      answer: "Self-hosting is free under the MIT licence, with no watermark and no cap; OpenShorts Cloud has a free plan (your first video up to 60 minutes, then 20 minutes a month, watermarked) and paid plans from $12/month. There is no catch, but there are two different things on offer. (1) Self-hosted is 100% free and open source: you run it with Docker on your own machine, bring your own API keys, and there are no watermarks, no usage limits and no subscription. What it costs you is hardware and time. On a typical CPU an 8-minute video takes 5 to 8 minutes to process, and you need your own Google Gemini key (required, free tier is 1,500 requests/day), plus ElevenLabs for dubbing and fal.ai for AI Shorts if you want those. (2) Hosted at openshorts.app is the same software with the running costs covered: our NVIDIA GPU clips that same 8-minute video in about 50 seconds, the Gemini key is included so there is nothing to create or paste, auto-posting to TikTok, Instagram and YouTube is already wired up, and your clips are stored and re-openable from any browser. Its free plan clips your first video whole up to 60 minutes, then 20 minutes a month, with a watermark and no credit card; paid plans start at $12/mo for 100 minutes without watermark. So: free if you are happy to run it yourself, paid if you would rather it just ran fast. For reference, Opus Clip starts at $15/month and bills a credit per minute of source video."
    },
    {
      question: "Do the clips have a watermark?",
      answer: "Only on the free Cloud plan. Clips made on the free plan of OpenShorts Cloud carry a small watermark; every paid plan, from $12/month, has none, and upgrading removes the mark from the clips you already made. The self-hosted edition, free under MIT, never adds one."
    },
    {
      question: "Do I need a credit card to start?",
      answer: "No. The free plan of OpenShorts Cloud needs no credit card: your first video is clipped free up to 60 minutes, then you get 20 free minutes every month, with a watermark. You only add a card if you choose a paid plan, from $12/month."
    },
    {
      question: "How many clips do I get from one video?",
      answer: "Between 3 and 15, each 15 to 60 seconds long. The AI scores every moment of the video and keeps the ones that hold up as standalone clips, so the number depends on the source, not on a fixed quota."
    },
    {
      question: "Can I clip someone else's YouTube video?",
      answer: "Only if you own it or have the rights or permission to use it. Before a video is processed, OpenShorts asks you to confirm you own the content or have the rights to it, and you are responsible for what you submit. Fair use depends on the country and on how the clip is used, so it is never guaranteed; reuploading someone else's video without permission is copyright infringement, and platforms remove it and strike the account. Clip your own videos, channels you manage, or footage you have licensed."
    },
    {
      question: "What is OpenShorts and how does it work?",
      answer: "OpenShorts is an open source AI clip generator (free to self-host under MIT, Cloud plans from $12/month) that transforms your long-form videos — podcasts, webinars, livestreams, vlogs, interviews — into viral-ready short clips in 9:16 vertical format. It uses a multi-step AI pipeline: faster-whisper for transcription with word-level timestamps, PySceneDetect for scene boundary detection, and Google Gemini 3.1 Flash-Lite AI for identifying the most engaging viral moments. According to HubSpot's 2025 State of Marketing report, short-form video delivers the highest ROI of any content format, and repurposing long-form content into shorts increases total reach by up to 300%."
    },
    {
      question: "How does OpenShorts compare to Opus Clip?",
      answer: "OpenShorts is an open source alternative to Opus Clip. Both offer AI viral moment detection and smart vertical cropping. Key differences: OpenShorts is free when self-hosted and $12/month hosted, against Opus Clip from $15/month billed per source minute. OpenShorts can run on your own infrastructure (full data privacy); Opus Clip is cloud-only. OpenShorts uses Google Gemini 3.1 Flash-Lite for AI analysis vs Opus Clip's proprietary model. OpenShorts adds two-speaker and screencast layouts, AI UGC videos with lip-synced actors, and an MCP server for Claude and ChatGPT. The honest trade-off: Opus Clip has the larger caption-style library."
    },
    {
      question: "How do I turn a long-form video into TikTok or Reels clips?",
      answer: "Upload your long-form video into OpenShorts, enter your free Gemini API key, and click Process. The AI transcribes it with faster-whisper, detects the best viral moments using Google Gemini 3.1 Flash-Lite, and crops them to 9:16 vertical format with MediaPipe face tracking. According to Wyzowl's 2025 Video Marketing Statistics report, 91% of businesses use video as a marketing tool, and repurposed short-form clips drive 2.5x more engagement than original content."
    },
    {
      question: "Can OpenShorts generate YouTube thumbnails and titles for free?",
      answer: "Yes. OpenShorts includes a free AI YouTube thumbnail generator, a free AI YouTube title generator, and a free AI YouTube description generator — all powered by Google Gemini 3.1 Flash-Lite. Upload your video and the AI suggests 10 viral title options with an interactive refinement chat. Then it generates multiple thumbnail designs using AI image generation — upload a face photo and background image for personalized results. The studio also auto-generates YouTube descriptions with chapter timestamps and lets you publish directly to YouTube. Everything is 100% free with the Gemini free tier."
    },
    {
      question: "What is the AI UGC Video Generator?",
      answer: "OpenShorts includes an AI UGC (User Generated Content) video creator that generates marketing videos with AI actors for any product or business. You describe your product or paste a website URL — the AI writes a viral script, generates a realistic AI actor with lip-synced voiceover, adds b-roll visuals and TikTok-style subtitles, with the hook spoken in the first seconds. The result is a ready-to-post vertical video for TikTok, Instagram Reels, or YouTube Shorts. Two cost modes: Low Cost (~$0.65/video using Hailuo + VEED Lipsync) and Premium (~$2/video using Kling Avatar v2)."
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
      answer: "OpenShorts uses Google Gemini 3.1 Flash-Lite, Google's latest multimodal AI model, for viral moment detection and title generation. The AI receives the full video transcript with timestamps, scene boundary data from PySceneDetect, and analyzes engagement patterns to identify the 3-15 most shareable moments. Each clip is scored based on emotional impact, hook strength, and viral potential — similar to how platforms like TikTok and YouTube rank content."
    },
    {
      question: "Can OpenShorts translate and dub videos into other languages?",
      answer: "Yes, with your own ElevenLabs API key, both in the cloud and self-hosted: add the key in Settings and a clip is dubbed into any of 30+ languages while keeping the original speaker's voice. After dubbing, OpenShorts re-transcribes the new audio and burns subtitles in the target language. ElevenLabs bills the dubbing to your key; it is not included in the OpenShorts plans."
    },
    {
      question: "How does the smart vertical cropping work?",
      answer: "OpenShorts offers two intelligent cropping modes for converting 16:9 horizontal video to 9:16 vertical format. TRACK mode uses MediaPipe face detection with YOLOv8 as fallback to follow a single subject with 'Heavy Tripod' stabilization — the camera moves smoothly like a professional cameraman. GENERAL mode handles group shots and landscapes by creating a blurred background layout. A SpeakerTracker prevents rapid switching between subjects and handles temporary occlusions for smooth results."
    },
    {
      question: "Is there a free open source clip generator?",
      answer: "Yes. OpenShorts is an open source clip generator (also known as open source clipping software or an AI video clipper) under the MIT licence. Self-hosted, it generates unlimited clips with no watermarks, no usage limits and no subscription fees. OpenShorts Cloud, the hosted version, clips your first video free up to 60 minutes, then 20 minutes a month with a watermark, and paid plans start at $12/month without one. It also includes a free AI YouTube thumbnail generator, free AI YouTube title generator, and free AI YouTube description generator — features that other clip generators charge extra for. You self-host it with Docker on your own machine for full privacy and control."
    },
    {
      question: "Can I automate OpenShorts from Claude, ChatGPT or n8n?",
      answer: "Yes. OpenShorts has a native MCP server at mcp.openshorts.app/mcp plus a REST API with per-user keys and completion webhooks, so an AI agent can run the whole flow: submit a video URL, wait for processing, list the clips and publish them to TikTok, Instagram or YouTube. This is where the hosted service shines: an agent needs an endpoint that is always on, and the hosted one is, with the API key created in your account page in one click. The self-hosted edition serves the same /mcp endpoint, but only while your own machine is running. There is also a zero-dependency CLI (pip install openshorts) and an importable n8n workflow. Full guide at openshorts.app/mcp."
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
      {/* Navigation: the shared site nav (seo/nav.js), same items as every page */}
      <SiteNav className="fixed top-0 w-full z-50" />

      {/* Hero — Marquee Hero: blueprint grid, content left, apparatus right */}
      <section className="hero-blueprint relative overflow-clip border-b border-rule pt-32 pb-20 px-6">
        <div className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_auto] gap-14 items-center">
          <div className="min-w-0">
            <p className="eyebrow mb-6">00 · Free AI Clip Generator · Open Source</p>

            <h1 className="hero-h1 mb-6">
              <span className="block">free ai clip maker:</span>{' '}
              <span className="block">long video in,</span>{' '}
              <span className="block"><em>viral shorts</em> out.</span>
            </h1>

            <p className="hero-description text-muted max-w-2xl mb-8 leading-relaxed lowercase">
              paste a youtube link or upload a podcast, stream or interview. the ai picks the 3 to 15 moments worth posting,
              reframes them to 9:16 around the people talking, and burns in the captions. ready for tiktok, reels and shorts in minutes.
            </p>

            {/* The hero CTA is the product itself: paste a link and land in the
                app with it loaded. Sign-in is asked for at generate time, not
                before the user has seen anything. */}
            <form onSubmit={handleHeroSubmit} className="mb-5">
              <div className="hero-input-row flex flex-col sm:flex-row items-stretch gap-3">
                <div className="relative flex-1 min-w-0">
                  <Link2 size={16} className="absolute left-4 top-1/2 -translate-y-1/2 text-muted pointer-events-none" />
                  <input
                    type="url"
                    value={heroUrl}
                    onChange={(e) => setHeroUrl(e.target.value)}
                    placeholder="paste a video link"
                    className="input-field pl-11"
                    aria-label="Video link"
                  />
                </div>
                <button type="submit" className="btn-primary whitespace-nowrap">
                  get free clips
                  <ArrowRight size={16} />
                </button>
                <button type="button" onClick={onLaunchApp} className="btn-ghost whitespace-nowrap">
                  <Upload size={16} />
                  upload a file
                </button>
              </div>
            </form>

            {/* Trust line right under the CTA: the objection this removes
                (a card) is the whole point of the free plan. */}
            <div className="flex flex-wrap items-center gap-x-5 gap-y-2 mb-5 text-sm">
              <span className="badge-ok whitespace-nowrap">
                <Check size={12} /> no credit card required
              </span>
              <span className="text-muted lowercase">first video free (up to 60 min) · then 20 min every month</span>
            </div>

            <p className="text-sm text-muted lowercase">
              <a href="#pricing" className="underline underline-offset-2 hover:text-ink">paid plans from $12/mo</a> without watermark. prefer to run it yourself?{' '}
              <a
                href="https://github.com/mutonby/openshorts"
                target="_blank"
                rel="noopener noreferrer"
                className="text-ink2 underline hover:text-ink transition-colors"
              >
                self-host free on github →
              </a>
            </p>

            <p className="mt-4 text-sm text-muted lowercase">
              works with links from{' '}
              {WORKS_WITH.map(([href, label], i) => (
                <React.Fragment key={href}>
                  {i > 0 && ' · '}
                  <a href={href} className={LINK}>{label}</a>
                </React.Fragment>
              ))}
            </p>
          </div>

          {/* Apparatus — instrument bezel holding a real 9:16 clip */}
          <figure className="apparatus" aria-label="example vertical clip generated by openshorts">
            <div className="apparatus-shell">
              <span className="apparatus-glow" aria-hidden="true" />
              <div className="apparatus-chamber">
                <video
                  key={DEMOS[heroDemo].id}
                  src={DEMOS[heroDemo].vertical}
                  poster={DEMOS[heroDemo].posterV}
                  autoPlay
                  muted
                  loop={DEMOS.length === 1}
                  onEnded={() => setHeroDemo((i) => (i + 1) % DEMOS.length)}
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

      {/* Proof: real usage numbers. Update by hand; never round them up. */}
      <section className="border-b border-rule">
        <div className="max-w-6xl mx-auto px-6 py-12 grid grid-cols-2 md:grid-cols-4 gap-y-8 md:divide-x divide-rule text-center">
          {[
            ['45,000+', 'accounts created'],
            ['32,900+', 'videos clipped'],
            ['8,300+', 'hours of video processed'],
            ['7.5k', 'github stars'],
          ].map(([n, label]) => (
            <div key={label} className="px-4">
              <div className="font-display text-4xl md:text-5xl text-ink tabular-nums">{n}</div>
              <div className="eyebrow mt-2">{label}</div>
            </div>
          ))}
        </div>
      </section>

      {/* By platform: one described card per source/destination, linking the
          per-platform pages. */}
      <section id="platforms" className="py-20 px-6">
        <div className="max-w-6xl mx-auto">
          <SectionHeader eyebrow="Platforms" title="Made for every platform: YouTube, TikTok, streams and podcasts">
            One clip generator, tuned to where the video comes from and where the clips are going.
          </SectionHeader>
          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-5">
            {PLATFORMS.map((p) => (
              <div key={p.title} className="card p-6 flex flex-col">
                <p.icon size={18} className="text-brass mb-4" />
                <h3 className="font-display text-xl lowercase text-ink mb-2">{p.title}</h3>
                <p className="text-muted text-sm leading-relaxed mb-4">{p.text}</p>
                <ul className="mt-auto space-y-1.5 text-sm lowercase">
                  {p.links.map(([href, label]) => (
                    <li key={href}><a href={href} className={LINK}>{label} →</a></li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
          <p className="mt-8 text-sm lowercase">
            <a href="/platforms" className={LINK}>every platform and its specs →</a>
          </p>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-20 px-6 border-t border-rule">
        <div className="max-w-6xl mx-auto">
          <SectionHeader eyebrow="01 · Features" title="AI clipping that finds the moments worth posting">
            Screens from the app and clips it made. Pick a feature, or let it play.
          </SectionHeader>
          <FeatureShowcase />
          <details className="mt-12 group">
            <summary className="list-none cursor-pointer text-sm text-muted hover:text-ink lowercase [&::-webkit-details-marker]:hidden">
              the full feature list <ChevronDown size={14} className="inline transition-transform group-open:rotate-180" />
            </summary>
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-5 mt-6">
              {features.map((feature, i) => (
                <FeatureCard key={i} {...feature} />
              ))}
            </div>
          </details>
        </div>
      </section>

      {/* One video in, many clips out: the real clips OpenShorts cut from one
          CC BY episode with default settings, with the score the AI gave each.
          The pattern the market leader opens with, shown with our own output. */}
      <section className="py-20 px-6 border-b border-rule">
        <div className="max-w-6xl mx-auto">
          <SectionHeader eyebrow="02 · One Video, Six Clips" title="one 27-minute episode in. six ready-to-post clips out.">
            No prompts, no settings and no scrubbing: the AI scored every moment, kept the six that stand on their own and cut them with captions and a hook.
          </SectionHeader>
          <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,15rem)_auto_minmax(0,1fr)] gap-6 items-center">
            <figure className="m-0 max-w-[16rem] lg:max-w-none">
              <div className="relative rounded-card overflow-hidden border border-rule">
                <img src="/screens/ep-source.webp" alt="The original 16:9 podcast episode" width="640" height="360" loading="lazy" className="block w-full h-auto" />
                <span className="absolute bottom-2 right-2 readout bg-paper/80 px-2 py-0.5 rounded">26:54</span>
              </div>
              <figcaption className="text-xs text-muted mt-2">the full episode, 16:9</figcaption>
            </figure>
            <ArrowRight size={28} className="text-brass mx-auto rotate-90 lg:rotate-0" aria-hidden="true" />
            <div className="grid grid-cols-3 sm:grid-cols-6 gap-3">
              {EPISODE_CLIPS.map((c) => (
                <figure key={c.n} className="m-0 min-w-0">
                  <div className="relative rounded-card overflow-hidden border border-rule bg-paper2">
                    <LazyLoopVideo src={`/demo/ep-clip-${c.n}.mp4`} poster={`/screens/ep-clip-${c.n}.webp`} />
                    <span className="absolute top-1.5 left-1.5 readout text-[10px] bg-paper/85 text-ok px-1.5 py-0.5 rounded">score {c.score}</span>
                  </div>
                  <figcaption className="text-xs text-ink2 mt-2 leading-snug">{c.title}</figcaption>
                </figure>
              ))}
            </div>
          </div>
          <p className="mt-6 text-xs text-muted">
            Source: <a href="https://www.youtube.com/watch?v=-KbQj_vboOU" target="_blank" rel="noopener noreferrer" className="underline underline-offset-2 hover:text-ink">Turn the Tables with Dan and Shoshana Jordan</a>, Heritage of Faith, licensed CC BY. Clipped by OpenShorts, scores as the AI gave them.
          </p>
        </div>
      </section>

      {/* Smart crop — real product output: 16:9 source to 9:16 result */}
      <section className="py-20 px-6">
        <div className="max-w-5xl mx-auto">
          <SectionHeader eyebrow="03 · Smart Crop" title="one video in. the moment, reframed.">
            Real output, not a mock-up: the AI picks the moment, reframes 16:9 to vertical 9:16 and burns in the captions.
          </SectionHeader>
          {DEMOS.length > 1 && (
            <div className="flex flex-wrap gap-2 mb-8" role="tablist" aria-label="Example clips">
              {DEMOS.map((d, i) => (
                <button
                  key={d.id}
                  type="button"
                  role="tab"
                  aria-selected={cropDemo === i}
                  onClick={() => setCropDemo(i)}
                  className={`px-4 py-1.5 rounded-full border text-sm lowercase transition-colors ${cropDemo === i ? 'bg-brass text-brassink border-brass' : 'border-rule text-muted hover:text-ink'}`}
                >
                  {d.tab}
                </button>
              ))}
            </div>
          )}
          <figure className="crop-frame crop-frame-16-9 w-full" aria-label="original video next to the vertical clip OpenShorts made from it">
            <video
              key={DEMOS[cropDemo].id}
              src={DEMOS[cropDemo].video}
              poster={DEMOS[cropDemo].poster}
              autoPlay
              muted
              loop
              playsInline
              preload="metadata"
              width="960"
              height="540"
              className="w-full h-auto block"
            />
          </figure>
          <p className="mt-6 text-xs text-muted">
            Source: <a href={DEMOS[cropDemo].creditUrl} target="_blank" rel="noopener noreferrer" className="underline underline-offset-2 hover:text-ink">{DEMOS[cropDemo].credit}</a>, {DEMOS[cropDemo].creditBy}, licensed CC BY. Clipped by OpenShorts.
          </p>
          <div className="mt-10 flex flex-wrap items-center gap-4">
            <button
              type="button"
              onClick={() => {
                window.scrollTo({ top: 0, behavior: 'smooth' });
                document.querySelector('.hero-input-row input')?.focus({ preventScroll: true });
              }}
              className="btn-primary whitespace-nowrap"
            >
              try it on your own video <ArrowRight size={16} />
            </button>
            <span className="text-sm text-muted">first video free up to 60 min, no credit card</span>
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section id="how-it-works" className="py-20 px-6 border-t border-rule">
        <div className="max-w-4xl mx-auto">
          <SectionHeader eyebrow="04 · Pipeline" title="How to turn a long video into Shorts with AI (5 steps)">
            From long-form video to viral-ready clips in 5 automated steps.
          </SectionHeader>
          <div className="space-y-8">
            {steps.map((step, i) => (
              <StepCard key={i} number={i + 1} {...step} />
            ))}
          </div>
        </div>
      </section>

      {/* Two ways to use it: free self-host vs paid hosted */}
      <section className="py-20 px-6 border-t border-rule">
        <div className="max-w-4xl mx-auto">
          <SectionHeader eyebrow="05 · Deploy" title="Two ways to use OpenShorts">
            The same open source software, running either on our GPU or on your machine.
          </SectionHeader>
          <div className="grid md:grid-cols-2 gap-6">
            <div className="card p-8 flex flex-col border-brass">
              <div className="flex flex-wrap items-center gap-3 mb-3">
                <Sparkles size={18} className="text-brass" />
                <h3 className="font-display text-2xl lowercase text-ink">cloud · openshorts.app</h3>
                <span className="badge-brass">Recommended · Free Plan</span>
              </div>
              <ul className="space-y-1.5 mb-6 flex-1">
                {['Our NVIDIA GPU: an 8-min video in about 50s', 'Gemini key included, nothing to set up', 'Social publishing built in', 'Connect Claude or ChatGPT directly: paste one URL, no API key', 'First video free up to 60 min, then 20 min/month, no card'].map((f, i) => (
                  <li key={i} className="flex items-center gap-2 text-sm text-muted"><Check size={14} className="text-ok shrink-0" />{f}</li>
                ))}
              </ul>
              {billingEnabled ? (
                <a href="#pricing" className="btn-primary whitespace-nowrap">
                  start free <ArrowRight size={16} />
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
                <h3 className="font-display text-2xl lowercase text-ink">self-hosted · for developers</h3>
                <span className="readout border border-rule rounded-full px-2.5 py-1">Free · Docker</span>
              </div>
              <ul className="space-y-1.5 mb-6 flex-1">
                {['Your machine: 5 to 8 min on a typical CPU', 'Bring your own Gemini, ElevenLabs and fal.ai keys', 'You install it, you maintain it, you back it up', 'Free forever, and always will be'].map((f, i) => (
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
            <SectionHeader eyebrow="06 · Pricing" title="Free plan vs paid: what you actually get">
              Your first video is free, up to 60 minutes. Then 20 free minutes a month — no credit card. Cancel anytime.
            </SectionHeader>
            <PricingSection onRequireLogin={() => { window.location.hash = '#/pricing'; }} />
          </div>
        </section>
      )}

      {/* Comparison Table */}
      <section id="comparison" className="py-20 px-6 border-t border-rule">
        <div className="max-w-4xl mx-auto">
          <SectionHeader eyebrow="07 · Comparison" title="OpenShorts vs Opus Clip, Klap and Vizard">
            Hosted OpenShorts starts at $12/mo billed monthly, or self-host it free. Opus Clip starts at $15/month, Klap at $14 and Vizard at $14.50 a month on yearly billing, Kapwing at $24/month.
          </SectionHeader>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-rule2">
                  <th className="py-3 px-4 text-left text-sm text-muted font-medium">Feature</th>
                  <th className="py-3 px-4 text-center text-sm font-medium">
                    <span className="text-brass">OpenShorts</span>
                    <span className="block text-xs text-muted font-normal">self-hosted / Cloud</span>
                  </th>
                  <th className="py-3 px-4 text-center text-sm text-muted font-medium">Opus Clip</th>
                  <th className="py-3 px-4 text-center text-sm text-muted font-medium">Kapwing</th>
                </tr>
              </thead>
              <tbody>
                <ComparisonRow feature="Price" openshorts={<span className="text-ok font-medium">$0 / from $12/mo</span>} opusclip={xIcon} kapwing={xIcon} />
                <ComparisonRow feature="AI Viral Moment Detection" openshorts={checkIcon} opusclip={checkMuted} kapwing={checkMuted} />
                <ComparisonRow feature="Smart Vertical Cropping" openshorts={checkIcon} opusclip={checkMuted} kapwing={checkMuted} />
                <ComparisonRow feature="Auto Subtitles" openshorts={checkIcon} opusclip={checkMuted} kapwing={checkMuted} />
                <ComparisonRow feature="AI Voice Dubbing (30+ langs, your ElevenLabs key)" openshorts={checkIcon} opusclip={<span className="text-muted text-sm">Limited</span>} kapwing={<span className="text-muted text-sm">No</span>} />
                <ComparisonRow feature="AI Video Effects" openshorts={checkIcon} opusclip={<span className="text-muted text-sm">No</span>} kapwing={checkMuted} />
                <ComparisonRow feature="Hook Text Overlays" openshorts={checkIcon} opusclip={checkMuted} kapwing={checkMuted} />
                <ComparisonRow feature="Self-Hosted / Privacy" openshorts={checkIcon} opusclip={<span className="text-muted text-sm">Cloud only</span>} kapwing={<span className="text-muted text-sm">Cloud only</span>} />
                <ComparisonRow feature="No Watermark" openshorts={<span className="text-ok text-sm">Self-hosted &amp; paid</span>} opusclip={<span className="text-muted text-sm">Free tier only</span>} kapwing={<span className="text-muted text-sm">Paid</span>} />
                <ComparisonRow feature="Open Source" openshorts={checkIcon} opusclip={<span className="text-muted text-sm">No</span>} kapwing={<span className="text-muted text-sm">No</span>} />
                <ComparisonRow feature="AI YouTube Thumbnail Generator" openshorts={checkIcon} opusclip={<span className="text-muted text-sm">No</span>} kapwing={<span className="text-muted text-sm">Paid</span>} />
                <ComparisonRow feature="AI Title & Description Generator" openshorts={checkIcon} opusclip={<span className="text-muted text-sm">Limited</span>} kapwing={<span className="text-muted text-sm">Paid</span>} />
                <ComparisonRow feature="AI UGC Video Generator" openshorts={checkIcon} opusclip={<span className="text-muted text-sm">No</span>} kapwing={<span className="text-muted text-sm">No</span>} />
                <ComparisonRow feature="AI Actors with Lip-Sync" openshorts={checkIcon} opusclip={<span className="text-muted text-sm">No</span>} kapwing={<span className="text-muted text-sm">No</span>} />
                <ComparisonRow feature="Usage Limits" openshorts={<span className="text-ok text-sm">Unlimited self-hosted</span>} opusclip={<span className="text-muted text-sm">Per plan</span>} kapwing={<span className="text-muted text-sm">Per plan</span>} />
              </tbody>
            </table>
          </div>
          <div className="mt-10">
            <h3 className="font-display text-xl lowercase text-ink mb-4">switching from another tool?</h3>
            <div className="flex flex-wrap gap-2">
              {SWITCHING.map(([href, label]) => (
                <a key={href} href={href} className="px-4 py-1.5 rounded-full border border-rule text-sm lowercase text-muted hover:text-ink hover:border-brass transition-colors">
                  {label}
                </a>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Use Cases */}
      <section className="py-20 px-6 border-t border-rule">
        <div className="max-w-5xl mx-auto">
          <SectionHeader eyebrow="08 · Use Cases" title="Who Uses OpenShorts?">
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

      {/* Also included: the two tools that are not the clipper */}
      <section className="py-20 px-6 border-t border-rule">
        <div className="max-w-6xl mx-auto">
          <SectionHeader eyebrow="09 · Also Included" title="two more tools, same account">
            In the same openshorts.app account, with no keys and no setup.
          </SectionHeader>
          <div className="grid md:grid-cols-2 gap-5">
            <div className="card p-8">
              <p className="eyebrow mb-4">01 · AI Shorts</p>
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
              <p className="eyebrow mb-4">02 · Studio</p>
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

      {/* FAQ Section */}
      <section id="faq" className="py-20 px-6 border-t border-rule">
        <div className="max-w-3xl mx-auto">
          <SectionHeader eyebrow="10 · FAQ" title="FAQ: free AI clipping, watermarks and limits">
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

      {/* Self-hosting details, folded: they matter to people who run it
          themselves and are noise to someone deciding whether to try the cloud.
          The content stays in the DOM, so crawlers still read it. */}
      <section className="py-16 px-6 border-t border-rule">
        <div className="max-w-5xl mx-auto">
          <details className="group">
            <summary className="list-none cursor-pointer flex items-center justify-between gap-6 [&::-webkit-details-marker]:hidden">
              <div>
                <span className="eyebrow block mb-3">11 · Self-Hosting</span>
                <h2 className="font-display text-2xl md:text-3xl lowercase text-ink">Open source: run the same pipeline on your own hardware</h2>
              </div>
              <ChevronDown size={22} className="text-muted shrink-0 transition-transform group-open:rotate-180" />
            </summary>
            {/* API Keys Section */}
            <section className="pt-12">
              <div className="max-w-5xl mx-auto">
                <SectionHeader eyebrow="API Keys" title="Self-hosting? Every API has a free tier">
                  Cloud plans include managed Gemini and social publishing: you never touch an API key. Self-hosters bring their own keys (all with generous free tiers):
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

            {/* Tech Stack */}
            <section className="pt-12">
              <div className="max-w-5xl mx-auto">
                <SectionHeader eyebrow="Stack" title="Built with Proven Technology">
                  Industry-leading AI models and open source tools in one pipeline.
                </SectionHeader>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  {[
                    { name: "Google Gemini 3.1", desc: "AI Analysis" },
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

          </details>
        </div>
      </section>

      {/* CTA Section — final statement */}
      <section className="py-24 px-6 border-t border-rule">
        <div className="max-w-3xl mx-auto text-center">
          <h2 className="font-display text-4xl md:text-5xl lowercase text-ink tracking-tight mb-5">start creating viral videos today.</h2>
          <p className="text-muted mb-10 max-w-xl mx-auto leading-relaxed lowercase">first video free (up to 60 min) · then 20 min/month · no credit card, or self-host it free with docker.</p>
          <div className="flex flex-wrap items-center justify-center gap-4">
            {billingEnabled ? (
              <a href="#pricing" className="btn-primary whitespace-nowrap">
                start free
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

      {/* Floating CTA, the same form as the hero */}
      <form
        onSubmit={handleHeroSubmit}
        aria-hidden={!stickyCta}
        className={`fixed z-40 bottom-4 left-1/2 -translate-x-1/2 w-[calc(100%-2rem)] max-w-xl flex items-center gap-2 p-2 rounded-full border border-rule bg-paper2/95 backdrop-blur shadow-2xl transition-all duration-300 ${stickyCta ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-6 pointer-events-none'}`}
      >
        <Link2 size={16} className="ml-3 text-muted shrink-0 hidden sm:block" />
        <input
          type="url"
          value={heroUrl}
          onChange={(e) => setHeroUrl(e.target.value)}
          placeholder="paste a video link"
          tabIndex={stickyCta ? 0 : -1}
          className="flex-1 min-w-0 bg-transparent outline-none text-sm text-ink placeholder:text-muted px-2"
          aria-label="Video link"
        />
        <button type="submit" tabIndex={stickyCta ? 0 : -1} className="btn-primary whitespace-nowrap rounded-full text-sm">
          get free clips <ArrowRight size={14} />
        </button>
      </form>

      {/* Footer — Ft5 Statement */}
      <footer className="border-t border-rule py-16 px-6">
        <div className="max-w-6xl mx-auto">
          <p className="font-display text-3xl md:text-5xl lowercase text-ink tracking-tight mb-10">clip it before it scrolls past.</p>
          <nav aria-label="Guides and comparisons" className="border-t border-rule pt-6 mb-6 flex flex-wrap items-center gap-x-6 gap-y-2 text-sm lowercase text-muted">
            <a href="/free-ai-clip-generator" className="hover:text-ink transition-colors">free ai clip generator</a>
            <a href="/tools" className="hover:text-ink transition-colors">free tools</a>
            <a href="/youtube-transcript-generator" className="hover:text-ink transition-colors">youtube transcript generator</a>
            <a href="/youtube-tag-generator" className="hover:text-ink transition-colors">youtube tag generator</a>
            <a href="/video-aspect-ratio-converter" className="hover:text-ink transition-colors">video to 9:16 converter</a>
            <a href="/auto-clip" className="hover:text-ink transition-colors">auto clip</a>
            <a href="/youtube-automation" className="hover:text-ink transition-colors">youtube automation</a>
            <a href="/platforms" className="hover:text-ink transition-colors">all platforms</a>
            <a href="/use-cases" className="hover:text-ink transition-colors">use cases</a>
            <a href="/how-to" className="hover:text-ink transition-colors">guides</a>
            <a href="/platforms/youtube"className="hover:text-ink transition-colors">youtube clip maker</a>
            <a href="/platforms/youtube-shorts" className="hover:text-ink transition-colors">youtube shorts maker</a>
            <a href="/platforms/tiktok" className="hover:text-ink transition-colors">tiktok clip maker</a>
            <a href="/platforms/instagram-reels" className="hover:text-ink transition-colors">ai reel generator</a>
            <a href="/platforms/twitch" className="hover:text-ink transition-colors">twitch clip maker</a>
            <a href="/platforms/kick" className="hover:text-ink transition-colors">kick clip maker</a>
            <a href="/auto-reframe" className="hover:text-ink transition-colors">auto reframe</a>
            <a href="/use-cases/sermons" className="hover:text-ink transition-colors">sermon clip maker</a>
            <a href="/how-to/clip-a-youtube-video" className="hover:text-ink transition-colors">how to clip a youtube video</a>
            <a href="/pricing" className="hover:text-ink transition-colors">pricing</a>
            <a href="/open-source-video-clipper" className="hover:text-ink transition-colors">open source video clipper</a>
            <a href="/self-hosted-ai-clip-maker" className="hover:text-ink transition-colors">self-hosted ai clip maker</a>
            <a href="/podcast-to-shorts" className="hover:text-ink transition-colors">podcast clips</a>
            <a href="/youtube-to-shorts-converter" className="hover:text-ink transition-colors">youtube to shorts</a>
            <a href="/how-openshorts-works" className="hover:text-ink transition-colors">how it works</a>
            <a href="/alternatives" className="hover:text-ink transition-colors">alternatives</a>
            <a href="/free-ai-clipping-tools-compared" className="hover:text-ink transition-colors">free ai clipping tools compared</a>
            <a href="/alternativas" className="hover:text-ink transition-colors">alternativas</a>
            <a href="/alternatives/opus-clip" className="hover:text-ink transition-colors">vs opus clip</a>
            <a href="/opus-clip-pricing" className="hover:text-ink transition-colors">opus clip pricing</a>
            <a href="/opus-ai" className="hover:text-ink transition-colors">opus ai</a>
            <a href="/vizard-ai" className="hover:text-ink transition-colors">vizard ai</a>
            <a href="/vizard-ai-video-to-text" className="hover:text-ink transition-colors">vizard ai video to text</a>
            <a href="/alternatives/vidyo-ai" className="hover:text-ink transition-colors">vidyo ai (quso)</a>
            <a href="/alternatives/2short" className="hover:text-ink transition-colors">2short ai</a>
            <a href="/alternatives/sendshort" className="hover:text-ink transition-colors">sendshort</a>
            <a href="/submagic-reviews" className="hover:text-ink transition-colors">submagic review</a>
            <a href="/mcp" className="hover:text-ink transition-colors">mcp server & api</a>
            <a href="/automate-shorts-api" className="hover:text-ink transition-colors">automate shorts</a>
            <a href="/n8n-youtube-shorts-automation" className="hover:text-ink transition-colors">n8n workflow</a>
          </nav>
          <div className="border-t border-rule pt-6 flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <img src="/logo-openshorts.png" alt="OpenShorts" className="w-6 h-6" />
              <span className="text-sm text-muted">OpenShorts — Free Open Source Clip Generator & AI UGC Video Creator</span>
            </div>
            <div className="flex flex-wrap items-center gap-x-6 gap-y-2 text-sm lowercase text-muted">
              <a href="https://github.com/mutonby/openshorts" target="_blank" rel="noopener noreferrer" className="hover:text-ink transition-colors">GitHub</a>
              <a href="#features" className="hover:text-ink transition-colors">Features</a>
              <a href="#faq" className="hover:text-ink transition-colors">FAQ</a>
              <a href="/terms" className="hover:text-ink transition-colors whitespace-nowrap">Terms</a>
              <a href="/privacy" className="hover:text-ink transition-colors whitespace-nowrap">Privacy</a>
              <a href="/legal-notice" className="hover:text-ink transition-colors whitespace-nowrap">Legal Notice</a>
              <a href="/refunds" className="hover:text-ink transition-colors whitespace-nowrap">Refunds</a>
              <a href="/report-content" className="hover:text-ink transition-colors whitespace-nowrap">Report Content</a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
