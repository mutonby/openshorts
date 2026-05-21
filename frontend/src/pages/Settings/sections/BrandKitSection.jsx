import BrandKit from '../../../components/BrandKit';
import SectionHeader from './SectionHeader.jsx';

export default function BrandKitSection() {
  return (
    <div>
      <SectionHeader
        title="Brand Kit"
        description="Colors, font, and per-aspect-ratio text positioning that every subtitle, hook, and overlay inherits."
      />
      <BrandKit />
    </div>
  );
}
