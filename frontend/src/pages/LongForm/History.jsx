// Past long-form edits. Reads from localStorage 'openshorts.longForm.history'.
// TODO(backend): plan TODO #10 — replace with GET /api/clips/recent once the
// backend grows a unified job index.

import { useEffect, useState } from 'react';
import { Archive } from 'lucide-react';

const HISTORY_KEY = 'openshorts.longForm.history';

function loadHistory() {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export default function History() {
  const [items, setItems] = useState([]);

  useEffect(() => {
    setItems(loadHistory());
  }, []);

  if (items.length === 0) {
    return (
      <div className="h-full flex flex-col items-center justify-center text-zinc-500 p-12">
        <Archive size={36} className="text-zinc-700 mb-3" />
        <div className="text-[14px] text-white font-medium">No long-form edits yet</div>
        <p className="text-[12px] text-zinc-500 mt-1 text-center max-w-md">
          Each completed long-form edit will be saved here so you can re-open the editor or re-export segments without re-uploading the source.
        </p>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-3 overflow-y-auto custom-scrollbar h-full">
      {items.map((item) => (
        <div key={item.id} className="rounded-xl border border-border bg-surface p-4">
          <div className="flex items-center justify-between">
            <div className="min-w-0">
              <div className="text-[13px] font-medium text-white truncate">{item.title || 'Untitled edit'}</div>
              <div className="text-[11px] text-zinc-500 mt-0.5">
                {new Date(item.ts).toLocaleString()} · {item.chapters || 0} chapter{item.chapters === 1 ? '' : 's'}
              </div>
            </div>
            <span className="text-[10px] uppercase tracking-wider px-2 py-0.5 rounded border border-border bg-white/5 text-zinc-500 shrink-0 ml-3">
              Saved
            </span>
          </div>
        </div>
      ))}
    </div>
  );
}
