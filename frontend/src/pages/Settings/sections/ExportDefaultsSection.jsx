import PlaceholderSection from './PlaceholderSection.jsx';

export default function ExportDefaultsSection() {
  return (
    <PlaceholderSection
      title="Export defaults"
      description="Output settings used when you download or publish a clip — container, codec, bitrate, max duration. Per-platform overrides live under Platforms."
      todo={[
        'Default container (MP4 / MOV)',
        'Codec + bitrate target',
        'Auto-resize policy (crop vs. letterbox)',
        'Filename template',
      ]}
    />
  );
}
