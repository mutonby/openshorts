// Legacy AI Agent informational page — pulled verbatim from the old
// App.jsx 'ai-agent' tab body. Describes the autonomous clipping skill.

import {
  Bot, Check, CheckCircle2, Copy, ExternalLink, Smartphone, Upload, Users,
} from 'lucide-react';

export default function LegacyAIAgent() {
  return (
    <div className="p-6 md:p-10 max-w-4xl mx-auto space-y-8">
      <div className="space-y-3">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-success/10 border border-success/30 text-[11px] uppercase tracking-wider text-success font-semibold">
          <Bot size={12} /> Autonomous Skill
        </div>
        <h1 className="text-3xl md:text-4xl font-bold text-white">
          Your Personal Clipping Team
        </h1>
        <p className="text-zinc-400 text-base md:text-lg leading-relaxed max-w-2xl">
          Drop your videos in a folder and a team of AI clippers picks the viral moments, edits them, and queues them for your approval — like having a 24/7 short-form editing crew on autopilot.
        </p>
      </div>

      <div className="p-4 rounded-lg border border-amber-500/30 bg-amber-500/10 flex items-start gap-3">
        <Smartphone size={20} className="text-amber-400 shrink-0 mt-0.5" />
        <div className="text-sm text-amber-100">
          <p className="font-semibold text-amber-300 mb-1">Upload videos already in vertical (9:16) mobile format.</p>
          <p className="text-amber-100/80 leading-relaxed">
            The agent does not reframe horizontal footage. Make sure every source video is shot or pre-cropped to mobile/portrait format before dropping it into the input folder.
          </p>
        </div>
      </div>

      <div className="grid md:grid-cols-3 gap-4">
        {[
          { icon: Upload, title: '1. Drop your videos', text: 'Put your long-form vertical footage in the watched folder. The skill picks one video per run.' },
          { icon: Users, title: '2. AI clippers work', text: 'Whisper transcribes, Gemini spots viral beats, FFmpeg cuts each clip and adds a hook overlay.' },
          { icon: CheckCircle2, title: '3. You validate, it ships', text: 'Approve candidates and the skill auto-publishes to TikTok, Reels and YouTube Shorts via Upload-Post.' },
        ].map(({ icon: Icon, title, text }) => (
          <div key={title} className="rounded-xl border border-border bg-surface p-5 space-y-2">
            <div className="w-10 h-10 rounded-lg bg-success/10 text-success flex items-center justify-center">
              <Icon size={18} />
            </div>
            <h3 className="font-semibold text-white">{title}</h3>
            <p className="text-xs text-zinc-400 leading-relaxed">{text}</p>
          </div>
        ))}
      </div>

      <div className="rounded-xl border border-border bg-surface p-6 md:p-8 space-y-5">
        <div className="flex items-start justify-between gap-4 flex-wrap">
          <div>
            <h2 className="text-xl font-bold text-white mb-1">skill-autoshorts</h2>
            <p className="text-sm text-zinc-400">
              The Claude Code skill that powers this workflow. Install it once and trigger it whenever you want a fresh batch of clips.
            </p>
          </div>
          <a
            href="https://github.com/mutonby/skill-autoshorts"
            target="_blank"
            rel="noopener noreferrer"
            className="btn-primary py-2 px-4 text-sm flex items-center gap-2 shrink-0"
          >
            View on GitHub <ExternalLink size={14} />
          </a>
        </div>

        <div className="bg-[#0c0c0e] border border-border rounded-lg p-4 font-mono text-xs text-zinc-300 flex items-center justify-between gap-3">
          <span className="truncate">git clone https://github.com/mutonby/skill-autoshorts</span>
          <button
            onClick={() => navigator.clipboard.writeText('git clone https://github.com/mutonby/skill-autoshorts')}
            className="text-zinc-500 hover:text-white transition-colors shrink-0"
            title="Copy"
          >
            <Copy size={14} />
          </button>
        </div>

        <div className="grid sm:grid-cols-2 gap-3 text-sm">
          {[
            'Daily batch — picks one long video per run',
            'Whisper transcription with word-level timing',
            'Gemini multimodal moment detection',
            'Auto-publish to TikTok, Reels & YouTube Shorts',
          ].map((t) => (
            <div key={t} className="flex items-start gap-2 text-zinc-300">
              <Check size={16} className="text-success shrink-0 mt-0.5" />
              <span>{t}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
