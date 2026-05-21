// One panel per social platform. Phase 2 stubs the override forms; the
// real wiring lands once the bell + scheduling backend gaps close.

import { useParams } from 'react-router-dom';
import PlaceholderSection from './PlaceholderSection.jsx';

const PLATFORMS = {
  youtube:   { label: 'YouTube',   dotClass: 'bg-platform-youtube' },
  tiktok:    { label: 'TikTok',    dotClass: 'bg-platform-tiktok' },
  instagram: { label: 'Instagram', dotClass: 'bg-platform-instagram' },
  snapchat:  { label: 'Snapchat',  dotClass: 'bg-platform-snapchat' },
  facebook:  { label: 'Facebook',  dotClass: 'bg-platform-facebook' },
};

export default function PlatformSection() {
  const { platform = 'youtube' } = useParams();
  const meta = PLATFORMS[platform] || PLATFORMS.youtube;

  return (
    <div>
      <div className="flex items-center gap-3 mb-6">
        <span className={`w-2.5 h-2.5 rounded-full ${meta.dotClass}`} />
        <h1 className="text-[18px] font-semibold text-white">{meta.label}</h1>
        <span className="text-[10px] px-2 py-0.5 rounded uppercase tracking-wider border bg-white/5 border-border text-zinc-400">
          Per-platform overrides
        </span>
      </div>
      <p className="text-[13px] text-zinc-500 leading-relaxed max-w-2xl mb-6">
        Override global subtitle style, color grade, export format, and scheduling defaults for {meta.label}. These take precedence over the General settings whenever a clip is published to this platform.
      </p>
      <PlaceholderSection
        title="Overrides"
        description="Each platform will expose: caption position, subtitle font, color grade, export codec, scheduling cadence."
        badge="Coming soon"
        todo={[
          'Subtitle style override (per-platform safe-zone)',
          'Color grade override',
          'Export codec / container override',
          'Default scheduling cadence',
        ]}
      />
    </div>
  );
}
