// CSS-only tooltip. Wraps an arbitrary child element and shows a label
// on hover/focus. Uses Tailwind's group-hover pattern — no portal,
// no positioning library, no animation deps.

export default function Tooltip({ label, side = 'bottom', children, className = '' }) {
  if (!label) return children;

  const sideClass = {
    top:    'bottom-full left-1/2 -translate-x-1/2 mb-1.5',
    bottom: 'top-full  left-1/2 -translate-x-1/2 mt-1.5',
    left:   'right-full top-1/2 -translate-y-1/2 mr-1.5',
    right:  'left-full  top-1/2 -translate-y-1/2 ml-1.5',
  }[side] || '';

  return (
    <span className={`relative inline-flex group ${className}`}>
      {children}
      <span
        role="tooltip"
        className={`pointer-events-none absolute z-50 whitespace-nowrap rounded-md bg-[#0a0a0a] border border-border px-2 py-1 text-[11px] text-zinc-200 shadow-lg opacity-0 group-hover:opacity-100 group-focus-within:opacity-100 transition-opacity ${sideClass}`}
      >
        {label}
      </span>
    </span>
  );
}
