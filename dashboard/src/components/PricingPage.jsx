import React, { useState } from 'react';
import { ArrowDown, ArrowRight, Check, ChevronDown, Github, MoveRight, Cpu, KeyRound, Send, Server, HardDrive, Repeat } from 'lucide-react';
import PricingSection from './PricingSection';

// The honest hosted-vs-self-hosted trade-off. The software is identical and open
// source either way, so the plans have to earn their price on hardware, keys and
// setup rather than on features. Timings are measured on our own pipeline: an
// 8-minute input takes ~50s on the server GPU and 5-8 min on CPU.
// The landing page carries its own shorter version of this (Landing.jsx "Two ways
// to use OpenShorts"); this detailed table is only for the standalone #/pricing.
const HOSTED_VS_SELF = [
  {
    icon: Cpu,
    label: 'Speed',
    hosted: 'An 8-minute video is clipped in about 50 seconds on our NVIDIA GPU.',
    self: '5 to 8 minutes for the same video on a typical CPU, unless you own a CUDA GPU.',
  },
  {
    icon: KeyRound,
    label: 'API keys',
    hosted: 'The Gemini key is included. Nothing to create, paste or top up.',
    self: 'You create and pay for your own Gemini key, plus ElevenLabs and fal.ai for dubbing and AI Shorts.',
  },
  {
    icon: Send,
    label: 'Auto-posting',
    hosted: 'Paid plans: connect TikTok, Instagram, YouTube, Facebook, LinkedIn, Threads and Bluesky once and every clip posts or schedules straight from the app.',
    self: 'You register your own Upload-Post account and wire up the credentials.',
  },
  {
    icon: Repeat,
    label: 'Autopilot',
    hosted: 'Paid plans: each new video on your YouTube channel is clipped on its own and the best clips are scheduled to your socials, one a day.',
    self: 'Not included: it needs an always-on server watching your channel.',
  },
  {
    icon: Server,
    label: 'Setup',
    hosted: 'Sign in and paste a link. Nothing to install.',
    self: 'Docker, 8GB+ RAM and a few GB of model downloads on the first run.',
  },
  {
    icon: HardDrive,
    label: 'Storage',
    hosted: 'Clips are hosted for you and re-open in any browser.',
    self: 'Your own disk and your own backups.',
  },
];

// Billing / value questions reused verbatim from Landing.jsx faqs, minus the
// two that called the product free with no limits: this page is where every
// cancelled checkout lands, right under the paid plans.
const FAQS = [
  {
    question: "Is OpenShorts really free? What's the catch?",
    answer: "There is no catch, but there are two different things on offer. (1) Self-hosted is 100% free and open source: you run it with Docker on your own machine, bring your own API keys, and there are no watermarks, no usage limits and no subscription. What it costs you is hardware and time. On a typical CPU an 8-minute video takes 5 to 8 minutes to process, and you need your own Google Gemini key (required, free tier is 1,500 requests/day), plus ElevenLabs for dubbing and fal.ai for AI Shorts if you want those. (2) Hosted at openshorts.app is the same software with the running costs covered: our NVIDIA GPU clips that same 8-minute video in about 50 seconds, the Gemini key is included so there is nothing to create or paste, paid plans post every clip to TikTok, Instagram, YouTube, Facebook, LinkedIn, Threads and Bluesky for you and Autopilot clips each new video on your channel on its own, and your clips are stored and re-openable from any browser. It has a free plan (20 minutes a month, watermark, no credit card) and paid plans from $12/mo for 100 minutes without watermark. So: free if you are happy to run it yourself, paid if you would rather it just ran fast. Both are far cheaper than Opus Clip ($15-228/month) or Kapwing ($24-79/month)."
  },
  {
    question: "Can OpenShorts generate YouTube thumbnails and titles for free?",
    answer: "Yes. OpenShorts includes a free AI YouTube thumbnail generator, a free AI YouTube title generator, and a free AI YouTube description generator — all powered by Google Gemini 3.1 Flash-Lite. Upload your video and the AI suggests 10 viral title options with an interactive refinement chat. Then it generates multiple thumbnail designs using AI image generation — upload a face photo and background image for personalized results. The studio also auto-generates YouTube descriptions with chapter timestamps and lets you publish directly to YouTube. Self-hosted, it all runs on your own Gemini key (the free tier works). On OpenShorts Cloud, titles and descriptions work on every plan and thumbnail images need a paid plan, from $12/month."
  },
  {
    question: "How much does it cost to generate an AI UGC video?",
    answer: "OpenShorts itself is free, but the AI Shorts feature uses external APIs (fal.ai for video generation, ElevenLabs for voiceover) that charge per use. Low Cost mode costs approximately $0.65 per video (Flux image $0.05 + ElevenLabs voice $0.10 + Hailuo img2video $0.19 + VEED Lipsync $0.20 + b-roll $0.10). Premium mode costs approximately $2.00 per video using Kling Avatar v2 for higher quality. Both modes are significantly cheaper than hiring UGC creators ($50-500 per video) or using platforms like HeyGen ($24-180/month)."
  },
  {
    question: "Can OpenShorts post my clips to my social networks automatically?",
    answer: "Yes, on every paid plan of OpenShorts Cloud (from $12/month). Connect your TikTok, Instagram, YouTube, Facebook, LinkedIn, Threads and Bluesky accounts once from the app and each clip can be posted to all of them in one go, or scheduled, without downloading anything; posting runs on Upload-Post and is already set up for you. Autopilot goes one step further: connect your YouTube channel and every new video you upload is clipped on its own, and the best clips are scheduled to your socials, one a day at the hour you choose. The free plan is download-only. Self-hosted, you can post with your own Upload-Post account, but Autopilot is Cloud only."
  },
  {
    question: "I only want to post videos I already have. Do I need OpenShorts?",
    answer: "No. OpenShorts is for cutting long videos into clips. If you only want to post videos or photos you already have to TikTok, Instagram, YouTube, LinkedIn, Facebook, X, Threads, Pinterest, Bluesky or Reddit, use Upload-Post (upload-post.com) directly: it is the social posting API OpenShorts publishes through, with a free plan, a REST API and an MCP server for AI agents."
  },
  {
    question: "Does API and MCP access cost extra?",
    answer: "No. Every plan, including the free one, can create API keys in the account page and use the REST API and the MCP server at mcp.openshorts.app/mcp. Calls draw from the same monthly minutes as the dashboard, so automating with Claude, ChatGPT or n8n does not change the price of anything. The value of the hosted endpoint is that it is always on: an agent or a scheduled pipeline can clip and publish while your own machine is off, which is the one thing the self-hosted edition cannot do for you. Full guide at openshorts.app/mcp."
  },
  {
    question: "What are the system requirements to run OpenShorts?",
    answer: "OpenShorts runs on any system with Docker installed. The recommended setup is 8GB+ RAM and a modern multi-core CPU. GPU acceleration (NVIDIA CUDA) is optional but speeds up video processing significantly. The Docker Compose setup handles all dependencies automatically — Python 3.11, FFmpeg, YOLOv8, MediaPipe, faster-whisper, and the React dashboard. It works on Linux, macOS, and Windows (via WSL2/Docker Desktop)."
  }
];

const TRUST_CARDS = [
  {
    eyebrow: 'open source',
    body: (
      <>
        The full code is public on{' '}
        <a
          href="https://github.com/mutonby/openshorts"
          target="_blank"
          rel="noopener noreferrer"
          className="text-ink underline decoration-1 underline-offset-2 hover:text-brass transition-colors"
        >
          GitHub
        </a>
        , and self-hosting it stays free forever. What a plan here buys is the GPU, the API keys
        and the setup, not the software.
      </>
    ),
  },
  {
    eyebrow: 'no watermark on paid plans · no per-clip credits',
    body: (
      <>
        Paid plans export clean clips, with no watermark. Plans meter minutes of input video per billing
        period, never per-clip credits.
      </>
    ),
  },
  {
    eyebrow: 'cancel anytime',
    body: (
      <>
        Start on the free plan — your first video free up to 60 minutes, then 20 minutes a month, no credit card. Billing for paid
        plans runs on Stripe — cancel anytime from your account.
      </>
    ),
  },
];

const FAQItem = ({ question, answer, isOpen, onClick }) => (
  <div>
    <button
      onClick={onClick}
      className="w-full flex items-center justify-between px-1 py-5 text-left"
    >
      <span className="text-ink font-medium pr-4">{question}</span>
      <ChevronDown
        size={18}
        className={`text-muted flex-shrink-0 transition-transform ${isOpen ? 'rotate-180' : ''}`}
      />
    </button>
    {isOpen && (
      <div className="px-1 pb-6">
        <p className="text-muted text-sm leading-relaxed">{answer}</p>
      </div>
    )}
  </div>
);

const DemoFigure = ({ src, label, className = '', children }) => (
  <figure className={`border border-rule rounded-card overflow-hidden bg-paper2 ${className}`}>
    <div className="relative">
      <video
        src={src}
        autoPlay
        muted
        loop
        playsInline
        preload="metadata"
        className="block w-full h-auto"
      />
      {children}
    </div>
    <figcaption className="readout px-3 py-2 border-t border-rule">{label}</figcaption>
  </figure>
);

// Conversion-focused paywall / pricing page (#/pricing). Wraps <PricingSection/>.
export default function PricingPage({ onRequireLogin }) {
  const [openFaq, setOpenFaq] = useState(null);

  const scrollToPlans = () => {
    document.getElementById('pricing-plans')?.scrollIntoView({ behavior: 'smooth' });
  };

  return (
    <div className="min-h-screen bg-paper text-ink2 animate-fade">
      {/* Header */}
      <header className="px-6 pt-20 pb-12">
        <div className="max-w-6xl mx-auto text-center">
          <p className="eyebrow mb-5">Pricing</p>
          <h1 className="font-display lowercase text-ink tracking-tight text-4xl md:text-6xl leading-[1.02] mb-5">
            start clipping in minutes.
          </h1>
          <p className="readout">first video free (up to 60 min) · then 20 min/month · no credit card</p>
        </div>
      </header>

      {/* Demo proof strip */}
      <section className="px-6 pb-16">
        <div className="max-w-4xl mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-[minmax(0,1fr)_auto_auto] items-center gap-5 md:gap-6">
            <DemoFigure src="/demo/clip-source.mp4" label="input · 16:9" />

            <MoveRight size={20} className="text-brass hidden md:block" aria-hidden="true" />
            <ArrowDown size={20} className="text-brass md:hidden justify-self-center" aria-hidden="true" />

            <DemoFigure
              src="/demo/clip-vertical.mp4"
              label="output · 9:16 · tracked"
              className="w-44 sm:w-48 justify-self-center md:justify-self-auto"
            />
          </div>
          <p className="text-center text-xs text-muted lowercase mt-4">
            real output from the clip generator — same footage, reframed by AI face tracking.
          </p>
        </div>
      </section>

      {/* Pricing grid */}
      <section id="pricing-plans" className="px-6 py-16 border-t border-rule scroll-mt-8">
        <div className="max-w-6xl mx-auto">
          <PricingSection onRequireLogin={onRequireLogin} />
        </div>
      </section>

      {/* Hosted vs self-hosted: same software, different running costs. */}
      <section className="px-6 py-16 border-t border-rule">
        <div className="max-w-5xl mx-auto">
          <div className="mb-8">
            <p className="eyebrow mb-3">Hosted or self-hosted</p>
            <h2 className="font-display text-3xl md:text-4xl lowercase text-ink tracking-tight">
              what you are actually paying for
            </h2>
            <p className="text-muted text-sm mt-3 max-w-2xl leading-relaxed">
              The software is the same and it is open source either way. What a plan buys you is the
              hardware, the API keys and the setup, so here is exactly what that means.
            </p>
          </div>

          <div className="hidden md:grid grid-cols-[9rem_1fr_1fr] gap-x-6 pb-2 mb-2 border-b border-rule">
            <span />
            <span className="eyebrow">Hosted on this site</span>
            <span className="eyebrow">Self-hosted</span>
          </div>

          <div className="divide-y divide-rule border-b border-rule">
            {HOSTED_VS_SELF.map(({ icon: Icon, label, hosted, self }) => (
              <div key={label} className="py-4 grid gap-2 md:grid-cols-[9rem_1fr_1fr] md:gap-x-6 md:items-start">
                <div className="flex items-center gap-2 text-ink2">
                  <Icon size={15} className="text-muted shrink-0" />
                  <span className="text-sm font-medium">{label}</span>
                </div>
                <div>
                  <span className="eyebrow block mb-1 md:hidden">Hosted on this site</span>
                  <p className="text-sm text-ink2 leading-relaxed">
                    <Check size={14} className="text-ok inline-block mr-1.5 -mt-0.5" />
                    {hosted}
                  </p>
                </div>
                <div>
                  <span className="eyebrow block mb-1 mt-2 md:hidden">Self-hosted</span>
                  <p className="text-sm text-muted leading-relaxed">{self}</p>
                </div>
              </div>
            ))}
          </div>

          <div className="mt-6 flex flex-col sm:flex-row sm:items-center gap-3 justify-between">
            <p className="text-xs text-muted leading-relaxed max-w-xl">
              Self-hosting is genuinely free and always will be. It costs you a machine, your own API
              keys and the time to keep it running.
            </p>
            <a
              href="https://github.com/mutonby/openshorts"
              target="_blank" rel="noopener noreferrer"
              className="shrink-0 text-sm text-muted hover:text-ink transition-colors inline-flex items-center gap-1.5"
            >
              <Github size={15} /> View the source
            </a>
          </div>
        </div>
      </section>

      {/* Trust strip */}
      <section className="px-6 py-16 border-t border-rule">
        <div className="max-w-5xl mx-auto grid grid-cols-1 md:grid-cols-3 gap-4">
          {TRUST_CARDS.map((card) => (
            <div key={card.eyebrow} className="card p-6">
              <div className="flex items-center gap-2 mb-3">
                <Check size={16} className="text-ok shrink-0" />
                <span className="eyebrow">{card.eyebrow}</span>
              </div>
              <p className="text-sm text-muted leading-relaxed">{card.body}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Mini FAQ */}
      <section className="px-6 py-16 border-t border-rule">
        <div className="max-w-3xl mx-auto">
          <div className="mb-10">
            <p className="eyebrow mb-3">Billing · FAQ</p>
            <h2 className="font-display text-3xl md:text-4xl lowercase text-ink tracking-tight">
              common questions
            </h2>
          </div>
          <div className="divide-y divide-rule border-y border-rule">
            {FAQS.map((faq, i) => (
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

      {/* Closing statement */}
      <section className="px-6 py-24 border-t border-rule">
        <div className="max-w-3xl mx-auto text-center">
          <h2 className="font-display text-4xl md:text-5xl lowercase text-ink tracking-tight mb-10">
            twenty free minutes. every month. no card until you decide.
          </h2>
          <div className="flex flex-wrap items-center justify-center gap-4">
            <button onClick={scrollToPlans} className="btn-primary whitespace-nowrap">
              start free
              <ArrowRight size={16} />
            </button>
            <a
              href="https://github.com/mutonby/openshorts"
              target="_blank"
              rel="noopener noreferrer"
              className="btn-ghost whitespace-nowrap"
            >
              <Github size={16} />
              view source
            </a>
          </div>
        </div>
      </section>
    </div>
  );
}
