// Placeholder body used by every page that hasn't been built yet.
// Phase 1 ships these so the sidebar renders and navigation works;
// later phases swap each stub for the real implementation.

export default function PageStub({ title, description, todo }) {
  return (
    <div className="p-8 max-w-3xl">
      <div className="text-[10px] uppercase tracking-[0.12em] text-zinc-500 mb-2">Phase 1 placeholder</div>
      <h2 className="text-[20px] font-semibold text-white mb-3">{title}</h2>
      {description && <p className="text-[13px] text-zinc-400 leading-relaxed mb-6">{description}</p>}
      {todo && todo.length > 0 && (
        <div className="rounded-lg border border-border bg-surface p-5">
          <div className="text-[11px] uppercase tracking-wider text-zinc-500 mb-3">Up next</div>
          <ul className="space-y-2 text-[12px] text-zinc-300">
            {todo.map((item, i) => <li key={i} className="flex gap-2"><span className="text-zinc-600">·</span><span>{item}</span></li>)}
          </ul>
        </div>
      )}
    </div>
  );
}
