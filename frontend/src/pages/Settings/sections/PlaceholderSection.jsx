import SectionHeader from './SectionHeader.jsx';

// Reusable placeholder for settings sections that haven't been built yet
// (subtitle style, color presets, export defaults, processing history,
// per-platform overrides). Phase 1/2 ships these as named drop targets;
// later phases fill them in.

export default function PlaceholderSection({ title, description, todo, badge = 'Coming soon' }) {
  return (
    <div>
      <SectionHeader title={title} description={description} badge={badge} />
      {todo && todo.length > 0 && (
        <div className="rounded-lg border border-border bg-surface p-5 space-y-3">
          <div className="text-[10px] uppercase tracking-wider text-zinc-500">Planned controls</div>
          <ul className="space-y-2 text-[13px] text-zinc-300">
            {todo.map((t, i) => (
              <li key={i} className="flex gap-2">
                <span className="text-zinc-600">·</span>
                <span>{t}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
