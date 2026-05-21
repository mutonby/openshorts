// PlatformBadge — small color-coded chip for a social platform. The static
// class names are spelled out per Tailwind's safelist scan.

import { Facebook, Instagram, Youtube } from 'lucide-react';

const TikTokGlyph = ({ size = 12 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor">
    <path d="M19.589 6.686a4.793 4.793 0 0 1-3.77-4.245V2h-3.445v13.672a2.896 2.896 0 0 1-5.201 1.743 2.895 2.895 0 0 1 3.183-4.51v-3.5a6.329 6.329 0 0 0-5.394 10.692 6.33 6.33 0 0 0 10.857-4.424V8.687a8.182 8.182 0 0 0 4.773 1.526V6.79a4.831 4.831 0 0 1-1.003-.104z" />
  </svg>
);
const SnapGlyph = ({ size = 12 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor">
    <path d="M12 2c5 0 7 4 7 7 0 1 0 3-.4 4.6.4.2 1 .3 1.6.3.5 0 1.4-.1 1.4.7 0 .8-1.4 1.1-2.3 1.4-.4.1-.6.2-.6.5 0 .8 2.8 2.6 4.7 2.6.5 0 .9.4.9.9 0 1.4-2.9 1.7-3.3 2.1-.1.1-.1.3 0 .5.1.4-.1.7-.5.7-.8 0-2-.5-3.4 0-1.3.5-2.2 2.2-5 2.2s-3.7-1.7-5-2.2c-1.4-.5-2.6 0-3.4 0-.4 0-.6-.3-.5-.7.1-.2.1-.4 0-.5-.4-.4-3.3-.7-3.3-2.1 0-.5.4-.9.9-.9 1.9 0 4.7-1.8 4.7-2.6 0-.3-.2-.4-.6-.5-.9-.3-2.3-.6-2.3-1.4 0-.8.9-.7 1.4-.7.6 0 1.2-.1 1.6-.3-.4-1.6-.4-3.6-.4-4.6 0-3 2-7 7-7z" />
  </svg>
);

const PLATFORMS = {
  youtube:   { label: 'YouTube',   icon: Youtube,     class: 'text-platform-youtube border-platform-youtube/30 bg-platform-youtube/10' },
  tiktok:    { label: 'TikTok',    icon: TikTokGlyph, class: 'text-platform-tiktok border-platform-tiktok/30 bg-platform-tiktok/10' },
  instagram: { label: 'Instagram', icon: Instagram,   class: 'text-platform-instagram border-platform-instagram/30 bg-platform-instagram/10' },
  snapchat:  { label: 'Snapchat',  icon: SnapGlyph,   class: 'text-platform-snapchat border-platform-snapchat/30 bg-platform-snapchat/10' },
  facebook:  { label: 'Facebook',  icon: Facebook,    class: 'text-platform-facebook border-platform-facebook/30 bg-platform-facebook/10' },
};

export default function PlatformBadge({ platform, withLabel = true, size = 'sm' }) {
  const meta = PLATFORMS[platform];
  if (!meta) return null;
  const Icon = meta.icon;
  const padding = size === 'sm' ? 'px-1.5 py-0.5 text-[10px]' : 'px-2 py-1 text-[11px]';
  return (
    <span className={`inline-flex items-center gap-1 rounded-md border ${padding} ${meta.class}`}>
      <Icon size={size === 'sm' ? 12 : 14} />
      {withLabel && meta.label}
    </span>
  );
}

export { PLATFORMS };
