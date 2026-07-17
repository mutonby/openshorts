import React, { useState } from 'react';
import { ArrowDown, ArrowRight, Check, ChevronDown, Github, MoveRight } from 'lucide-react';
import PricingSection from './PricingSection';

// Billing / value questions reused verbatim from Landing.jsx faqs
const FAQS = [
  {
    question: "Is OpenShorts really free? What's the catch?",
    answer: "There are two ways to use OpenShorts. (1) Self-hosted is 100% free and open source: you run it with Docker on your own machine, bring your own API keys, and there are no watermarks, no usage limits, and no subscription. Google Gemini API (required) powers the AI analysis — its free tier includes 1,500 requests/day. ElevenLabs (optional) enables AI dubbing in 30+ languages. Upload-Post (optional) publishes to YouTube, TikTok, and Instagram. (2) Hosted at openshorts.app is the no-setup option: we run all the AI and compute for you and you don't need any API keys — it's a paid service with a 3-day free trial, then plans from $12/mo for 100 minutes of video. So: free if you self-host, paid if you want us to host it for you. Both are far cheaper than Opus Clip ($15-228/month) or Kapwing ($24-79/month)."
  },
  {
    question: "How does OpenShorts compare to Opus Clip?",
    answer: "OpenShorts is a free, self-hosted alternative to Opus Clip. Both offer AI viral moment detection and smart vertical cropping. Key differences: OpenShorts is completely free vs Opus Clip's $15-228/month pricing. OpenShorts runs on your infrastructure (full data privacy) vs cloud-only. OpenShorts uses Google Gemini 3.0 Flash for AI analysis vs Opus Clip's proprietary model. OpenShorts adds AI voice dubbing in 30+ languages, AI-generated video effects, and hook text overlays. The trade-off is that OpenShorts requires Docker self-hosting, while Opus Clip is a ready-to-use cloud service."
  },
  {
    question: "Can OpenShorts generate YouTube thumbnails and titles for free?",
    answer: "Yes. OpenShorts includes a free AI YouTube thumbnail generator, a free AI YouTube title generator, and a free AI YouTube description generator — all powered by Google Gemini 3.0 Flash. Upload your video and the AI suggests 10 viral title options with an interactive refinement chat. Then it generates multiple thumbnail designs using AI image generation — upload a face photo and background image for personalized results. The studio also auto-generates YouTube descriptions with chapter timestamps and lets you publish directly to YouTube. Everything is 100% free with the Gemini free tier."
  },
  {
    question: "Is there a free open source clip generator?",
    answer: "Yes — OpenShorts is a 100% free, open source clip generator. Unlike paid clip generators like Opus Clip ($15-228/month) or Kapwing ($24-79/month), OpenShorts lets you generate unlimited clips with no watermarks, no usage limits, and no subscription fees. It also includes a free AI YouTube thumbnail generator, free AI YouTube title generator, and free AI YouTube description generator — features that other clip generators charge extra for. You self-host it with Docker on your own machine for full privacy and control."
  },
  {
    question: "How much does it cost to generate an AI UGC video?",
    answer: "OpenShorts itself is free, but the AI Shorts feature uses external APIs (fal.ai for video generation, ElevenLabs for voiceover) that charge per use. Low Cost mode costs approximately $0.65 per video (Flux image $0.05 + ElevenLabs voice $0.10 + Hailuo img2video $0.19 + VEED Lipsync $0.20 + b-roll $0.10). Premium mode costs approximately $2.00 per video using Kling Avatar v2 for higher quality. Both modes are significantly cheaper than hiring UGC creators ($50-500 per video) or using platforms like HeyGen ($24-180/month)."
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
        . Run it with Docker on your own machine and use it completely free — forever.
      </>
    ),
  },
  {
    eyebrow: 'no watermarks · no per-clip credits',
    body: (
      <>
        Clips export clean — no watermarks. Plans meter minutes of input video per billing
        period, never per-clip credits.
      </>
    ),
  },
  {
    eyebrow: 'cancel anytime',
    body: (
      <>
        3-day free trial with up to 20 minutes included; you are not charged during the trial.
        Billing runs on Stripe — cancel anytime from your account.
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
          <p className="readout">3-day free trial · 20 free minutes · cancel anytime</p>
        </div>
      </header>

      {/* Demo proof strip */}
      <section className="px-6 pb-16">
        <div className="max-w-4xl mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-[minmax(0,1fr)_auto_auto] items-center gap-5 md:gap-6">
            <DemoFigure src="/demo/clip-source.mp4" label="input · 16:9">
              {/* 9:16 crop window over the 16:9 source: window width = height x 9/16 = 31.64% of frame width */}
              <div className="absolute inset-y-0 left-0 bg-black/50" style={{ width: '34.18%' }} />
              <div className="absolute inset-y-0 right-0 bg-black/50" style={{ width: '34.18%' }} />
              <div
                className="absolute inset-y-0 left-1/2 -translate-x-1/2 border border-brass"
                style={{ width: '31.64%' }}
              >
                <span className="absolute top-1.5 left-1/2 -translate-x-1/2 readout text-brass bg-black/70 rounded-full px-2 py-0.5 whitespace-nowrap">
                  9:16
                </span>
              </div>
            </DemoFigure>

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
            three free days. twenty free minutes. no charge until you decide.
          </h2>
          <div className="flex flex-wrap items-center justify-center gap-4">
            <button onClick={scrollToPlans} className="btn-primary whitespace-nowrap">
              start free trial
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
