import PlaceholderSection from './PlaceholderSection.jsx';

export default function SubtitleStyleSection() {
  return (
    <PlaceholderSection
      title="Subtitle style"
      description="Defaults applied to auto-burn-in subtitles. Most controls live in the Brand Kit today; this panel will host extras (animation, dropshadow, casing rules, line-length thresholds)."
      todo={[
        'Word-by-word reveal speed',
        'Per-language font fallback chain',
        'Default burn-in vs. soft-sub on download',
      ]}
    />
  );
}
