// Shared header for every settings section: title + optional description.

export default function SectionHeader({ title, description, badge }) {
  return (
    <div className="mb-6">
      <div className="flex items-center gap-3 mb-1">
        <h1 className="text-[18px] font-semibold text-white">{title}</h1>
        {badge && (
          <span className="text-[10px] px-2 py-0.5 rounded uppercase tracking-wider border bg-white/5 border-border text-zinc-400">
            {badge}
          </span>
        )}
      </div>
      {description && (
        <p className="text-[13px] text-zinc-500 leading-relaxed max-w-2xl">{description}</p>
      )}
    </div>
  );
}
