import { X } from 'lucide-react';

/**
 * The single modal shell for the app (design.md).
 * Plain dark overlay (no backdrop-blur), hairline paper2 panel.
 *
 * Props:
 *  - isOpen / onClose
 *  - title (string, rendered lowercase serif) — optional
 *  - eyebrow (string, mono UPPERCASE micro label above title) — optional
 *  - size: 'sm' | 'md' | 'lg' | 'xl' (max width; default 'md')
 *  - children: body content
 *  - footer: optional node pinned under the body
 *  - hideClose: hide the X button
 */
const SIZES = {
  sm: 'max-w-sm',
  md: 'max-w-md',
  lg: 'max-w-2xl',
  xl: 'max-w-5xl',
};

export default function Modal({ isOpen, onClose, title, eyebrow, size = 'md', children, footer, hideClose = false }) {
  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center bg-black/70 p-4 animate-fade"
      onMouseDown={(e) => { if (e.target === e.currentTarget && onClose) onClose(); }}
      role="dialog"
      aria-modal="true"
    >
      <div className={`card relative w-full ${SIZES[size] || SIZES.md} max-h-[90vh] flex flex-col`}>
        {!hideClose && onClose && (
          <button
            onClick={onClose}
            aria-label="close"
            className="absolute top-4 right-4 z-10 p-1.5 rounded-full text-muted hover:text-ink hover:bg-paper3 transition-colors"
          >
            <X size={16} />
          </button>
        )}
        {(title || eyebrow) && (
          <div className="px-4 sm:px-6 pt-6 pb-4 border-b border-rule shrink-0">
            {eyebrow && <p className="eyebrow mb-1.5">{eyebrow}</p>}
            {title && <h2 className="font-display lowercase text-2xl text-ink leading-tight break-words pr-8">{title}</h2>}
          </div>
        )}
        <div className="px-4 sm:px-6 py-5 overflow-y-auto custom-scrollbar grow">
          {children}
        </div>
        {footer && (
          <div className="px-4 sm:px-6 py-4 border-t border-rule shrink-0">
            {footer}
          </div>
        )}
      </div>
    </div>
  );
}
