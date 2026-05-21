// Step 4: Long-form editor. 16:9 preview + chapter timeline scrubber +
// right panel with Chapters / Subtitles / Export tabs.
//
// All real backend wiring is deferred (plan TODOs #6 chapter detection,
// #7 segment export). The chapter list seeded by Step 3 is a placeholder
// and "Export segment as short" surfaces a "coming soon" modal.

import { useEffect, useMemo, useRef, useState } from 'react';
import { Download, FileText, Layers, Scissors, X } from 'lucide-react';

const TABS = [
  { id: 'chapters',  label: 'Chapters',  icon: Layers },
  { id: 'subtitles', label: 'Subtitles', icon: FileText },
  { id: 'export',    label: 'Export',    icon: Download },
];

function fmtTime(sec) {
  if (!Number.isFinite(sec)) return '--:--';
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${String(s).padStart(2, '0')}`;
}

export default function Editor({ wizard }) {
  const file = wizard.data.file;
  const chapters = wizard.data.chapters || [];
  const [tab, setTab] = useState('chapters');
  const [activeChapter, setActiveChapter] = useState(chapters[0]?.id || null);
  const [showExportModal, setShowExportModal] = useState(false);
  const [sourceUrl, setSourceUrl] = useState(null);
  const [durationSec, setDurationSec] = useState(null);
  const videoRef = useRef(null);

  useEffect(() => {
    if (!(file?.file instanceof File)) { setSourceUrl(null); return; }
    const url = URL.createObjectURL(file.file);
    setSourceUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file?.file]);

  // Derive the timeline duration: real video metadata wins, else last
  // chapter, else a 10-minute fallback so the bars render.
  const totalDuration = durationSec || chapters[chapters.length - 1]?.endSec || 600;

  function seekTo(sec) {
    if (videoRef.current) {
      videoRef.current.currentTime = sec;
      videoRef.current.play().catch(() => {});
    }
  }

  function selectChapter(c) {
    setActiveChapter(c.id);
    seekTo(c.startSec);
  }

  function renameChapter(id, label) {
    wizard.setData({
      chapters: chapters.map((c) => c.id === id ? { ...c, label } : c),
    });
  }

  if (!sourceUrl) {
    return (
      <div className="h-full flex flex-col items-center justify-center p-12 text-center text-zinc-500">
        <p className="text-[14px] text-white font-medium">No source file available.</p>
        <p className="text-[12px] mt-1">Re-upload to re-run the wizard.</p>
        <button onClick={wizard.reset} className="mt-4 btn-primary px-4 py-2 text-[13px]">
          Start over
        </button>
      </div>
    );
  }

  return (
    <div className="h-full flex">
      <div className="flex-1 flex flex-col bg-background overflow-hidden">
        <div className="flex-1 flex items-center justify-center p-6 overflow-hidden">
          <div className="w-full max-w-4xl">
            <div className="bg-black border border-border rounded-lg overflow-hidden aspect-video">
              <video
                ref={videoRef}
                src={sourceUrl}
                controls
                onLoadedMetadata={(e) => setDurationSec(e.currentTarget.duration)}
                className="w-full h-full object-contain"
              />
            </div>
          </div>
        </div>

        <ChapterTimeline
          chapters={chapters}
          totalDuration={totalDuration}
          active={activeChapter}
          onSelect={selectChapter}
        />
      </div>

      <aside className="w-[320px] shrink-0 border-l border-border bg-surface flex flex-col">
        <div className="border-b border-border flex items-center">
          {TABS.map((t) => {
            const Icon = t.icon;
            const active = tab === t.id;
            return (
              <button
                key={t.id}
                onClick={() => setTab(t.id)}
                className={`flex-1 flex items-center justify-center gap-2 py-3 text-[12px] transition-colors ${
                  active ? 'text-white bg-white/5 border-b-2 border-primary' : 'text-zinc-400 hover:text-white'
                }`}
              >
                <Icon size={12} />
                {t.label}
              </button>
            );
          })}
        </div>

        <div className="flex-1 overflow-y-auto custom-scrollbar p-4">
          {tab === 'chapters' && (
            <ChaptersPanel
              chapters={chapters}
              activeId={activeChapter}
              onSelect={selectChapter}
              onRename={renameChapter}
              onExportSegment={() => setShowExportModal(true)}
            />
          )}
          {tab === 'subtitles' && <SubtitlesPanel />}
          {tab === 'export' && <ExportPanel onSegmentClick={() => setShowExportModal(true)} />}
        </div>
      </aside>

      {showExportModal && (
        <SegmentExportModal onClose={() => setShowExportModal(false)} />
      )}
    </div>
  );
}

function ChapterTimeline({ chapters, totalDuration, active, onSelect }) {
  return (
    <div className="border-t border-border bg-surface px-4 py-3 shrink-0">
      <div className="flex items-center justify-between mb-2">
        <span className="text-[11px] uppercase tracking-wider text-zinc-500">Chapters</span>
        <span className="text-[11px] text-zinc-500 font-mono">{fmtTime(totalDuration)}</span>
      </div>
      <div className="relative h-7 bg-white/5 rounded-md overflow-hidden">
        {chapters.map((c) => {
          const left = (c.startSec / totalDuration) * 100;
          const width = ((c.endSec - c.startSec) / totalDuration) * 100;
          const isActive = c.id === active;
          return (
            <button
              key={c.id}
              onClick={() => onSelect(c)}
              className={`absolute top-0 bottom-0 border-r border-background transition-colors ${
                isActive ? 'bg-primary/50' : 'bg-primary/15 hover:bg-primary/30'
              }`}
              style={{ left: `${left}%`, width: `${width}%` }}
              title={`${c.label} (${fmtTime(c.startSec)} – ${fmtTime(c.endSec)})`}
            >
              <span className="absolute inset-0 flex items-center justify-center text-[10px] text-white truncate px-2">
                {c.label}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function ChaptersPanel({ chapters, activeId, onSelect, onRename, onExportSegment }) {
  return (
    <div className="space-y-2">
      <p className="text-[11px] text-zinc-500 mb-3">
        Click a chapter to seek the preview. Rename inline. Export any chapter as a vertical short.
      </p>
      {chapters.map((c) => {
        const active = c.id === activeId;
        return (
          <div
            key={c.id}
            className={`rounded-lg border p-3 transition-colors ${
              active ? 'border-primary bg-primary/10' : 'border-border hover:bg-white/5'
            }`}
          >
            <div className="flex items-center justify-between gap-2 mb-1">
              <input
                value={c.label}
                onChange={(e) => onRename(c.id, e.target.value)}
                onFocus={() => onSelect(c)}
                className="bg-transparent text-[13px] text-white font-medium flex-1 min-w-0 focus:outline-none"
              />
              <button
                onClick={onExportSegment}
                className="shrink-0 text-[10px] px-2 py-0.5 rounded border border-primary/40 text-primary hover:bg-primary/10"
              >
                Export
              </button>
            </div>
            <div className="text-[10px] font-mono text-zinc-500">
              {fmtTime(c.startSec)} – {fmtTime(c.endSec)}
            </div>
          </div>
        );
      })}
      {chapters.length === 0 && (
        <p className="text-[11px] text-zinc-500 italic">No chapters detected. Backend TODO #6.</p>
      )}
    </div>
  );
}

function SubtitlesPanel() {
  return (
    <div className="space-y-3 text-[12px]">
      <p className="text-[11px] text-zinc-500">
        Edit transcribed lines, retime, restyle, and re-export with the brand-kit subtitle style.
      </p>
      <div className="rounded-lg border border-border bg-background/40 p-4 text-center text-zinc-500 text-[11px]">
        Subtitle editor lands with backend transcript endpoint hookup.
        <br />
        See plan TODO #6 + the existing /api/subtitle route.
      </div>
    </div>
  );
}

function ExportPanel({ onSegmentClick }) {
  return (
    <div className="space-y-3 text-[12px]">
      <p className="text-[11px] text-zinc-500">
        Export the full long-form edit or any single chapter as a vertical short.
      </p>
      <button
        disabled
        title="Long-form export — coming soon"
        className="w-full px-3 py-2 text-[12px] rounded-md border border-border text-zinc-500 cursor-not-allowed flex items-center gap-2"
      >
        <Download size={12} /> Download long-form
      </button>
      <button
        onClick={onSegmentClick}
        className="w-full btn-primary px-3 py-2 text-[12px] flex items-center gap-2"
      >
        <Scissors size={12} /> Export segment as short
      </button>
    </div>
  );
}

function SegmentExportModal({ onClose }) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="bg-surface border border-border rounded-xl p-6 max-w-md w-full mx-4 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-[15px] font-semibold text-white flex items-center gap-2">
            <Scissors size={16} className="text-primary" />
            Export segment as short
          </h2>
          <button onClick={onClose} className="text-zinc-500 hover:text-white">
            <X size={14} />
          </button>
        </div>
        <p className="text-[13px] text-zinc-400 leading-relaxed">
          This will rerun the vertical-reframing pipeline on the selected chapter range and surface the result in the short-form Review step.
        </p>
        <div className="mt-4 rounded-lg border border-amber-500/30 bg-amber-500/10 p-3 text-[12px] text-amber-200">
          Backend route not implemented yet — see plan TODO #7
          (<code className="text-amber-300">POST /api/long-form/export-segment</code>).
          The UI is wired so the button works the moment the route ships.
        </div>
        <div className="flex justify-end gap-2 mt-5">
          <button
            onClick={onClose}
            className="text-[13px] text-zinc-400 hover:text-white px-3 py-2 rounded-md"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
