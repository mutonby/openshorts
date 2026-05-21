// Tiny "?" / info icon next to a primary action. Hover reveals the
// explanation via Tooltip. Used pervasively per the spec.

import { Info } from 'lucide-react';
import Tooltip from './Tooltip.jsx';

export default function InfoIcon({ label, side = 'bottom', size = 12, className = '' }) {
  return (
    <Tooltip label={label} side={side} className={className}>
      <span
        tabIndex={0}
        className="inline-flex items-center justify-center w-4 h-4 rounded-full text-zinc-500 hover:text-zinc-300 focus:text-zinc-300 transition-colors outline-none"
        aria-label={label}
      >
        <Info size={size} />
      </span>
    </Tooltip>
  );
}
