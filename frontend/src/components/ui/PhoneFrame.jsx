// PhoneFrame — pure visual wrapper that gives any 9:16 content a phone-shaped
// bezel + notch. Wraps a <video>, <img>, or any preview component.

export default function PhoneFrame({ children, className = '', size = 'md' }) {
  const widths = { sm: 200, md: 260, lg: 320 };
  const w = widths[size] || widths.md;
  const h = Math.round((w * 16) / 9);
  return (
    <div
      className={`relative bg-zinc-950 border border-zinc-800 rounded-[28px] p-2 shadow-2xl ${className}`}
      style={{ width: w, height: h }}
    >
      <div className="absolute top-2.5 left-1/2 -translate-x-1/2 w-16 h-4 bg-black rounded-full z-10" />
      <div className="w-full h-full bg-black rounded-[22px] overflow-hidden flex items-center justify-center">
        {children}
      </div>
    </div>
  );
}
