import { Navigate, Route, Routes } from 'react-router-dom';
import AppShell from './layouts/AppShell.jsx';
import Dashboard from './pages/Dashboard.jsx';
import ShortForm from './pages/ShortForm.jsx';
import LongForm from './pages/LongForm.jsx';
import ClipGenerator from './pages/ClipGenerator.jsx';
import SettingsLayout from './pages/Settings/index.jsx';
import BrandKitSection from './pages/Settings/sections/BrandKitSection.jsx';
import SubtitleStyleSection from './pages/Settings/sections/SubtitleStyleSection.jsx';
import ColorPresetsSection from './pages/Settings/sections/ColorPresetsSection.jsx';
import ExportDefaultsSection from './pages/Settings/sections/ExportDefaultsSection.jsx';
import PlatformSection from './pages/Settings/sections/PlatformSection.jsx';
import ApiKeysSection from './pages/Settings/sections/ApiKeysSection.jsx';
import HistorySection from './pages/Settings/sections/HistorySection.jsx';
import LegacySaaSShorts from './pages/Legacy/SaaSShorts.jsx';
import LegacyThumbnails from './pages/Legacy/Thumbnails.jsx';
import LegacyUGCGallery from './pages/Legacy/UGCGalleryPage.jsx';
import LegacyAIAgent from './pages/Legacy/AIAgent.jsx';

export default function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route index element={<Navigate to="/dashboard" replace />} />
        <Route path="dashboard" element={<Dashboard />} />
        <Route path="short-form/*" element={<ShortForm />} />
        <Route path="long-form/*" element={<LongForm />} />
        <Route path="clip-generator" element={<ClipGenerator />} />

        <Route path="settings" element={<SettingsLayout />}>
          <Route index element={<Navigate to="general/brand-kit" replace />} />
          <Route path="general/brand-kit"        element={<BrandKitSection />} />
          <Route path="general/subtitle-style"   element={<SubtitleStyleSection />} />
          <Route path="general/color-presets"    element={<ColorPresetsSection />} />
          <Route path="general/export-defaults"  element={<ExportDefaultsSection />} />
          <Route path="platforms/:platform"      element={<PlatformSection />} />
          <Route path="system/api-keys"          element={<ApiKeysSection />} />
          <Route path="system/history"           element={<HistorySection />} />
        </Route>

        <Route path="legacy/saasshorts" element={<LegacySaaSShorts />} />
        <Route path="legacy/thumbnails" element={<LegacyThumbnails />} />
        <Route path="legacy/ugc" element={<LegacyUGCGallery />} />
        <Route path="legacy/ai-agent" element={<LegacyAIAgent />} />
        <Route path="*" element={<Navigate to="/dashboard" replace />} />
      </Route>
    </Routes>
  );
}
