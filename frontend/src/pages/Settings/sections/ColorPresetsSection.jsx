import PlaceholderSection from './PlaceholderSection.jsx';

export default function ColorPresetsSection() {
  return (
    <PlaceholderSection
      title="Color presets"
      description="Cinematic LUTs and color-grade defaults applied during the short-form and long-form auto-edit. Needs the backend LUT integration before it can ship (plan TODO #5)."
      todo={[
        'Upload .cube / .3dl LUT files',
        'Choose a default LUT per workflow (short / long)',
        'Per-platform overrides',
      ]}
    />
  );
}
