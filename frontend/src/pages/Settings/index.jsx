// Settings layout — VS Code style. 150px left nav grouped into
// General / Platforms / System; clicking a nav item loads the
// corresponding panel into <Outlet />.

import { NavLink, Outlet } from 'react-router-dom';
import { Shield } from 'lucide-react';

const SECTIONS = [
  {
    label: 'General',
    items: [
      { to: 'general/brand-kit',       label: 'Brand Kit' },
      { to: 'general/subtitle-style',  label: 'Subtitle style' },
      { to: 'general/color-presets',   label: 'Color presets' },
      { to: 'general/export-defaults', label: 'Export defaults' },
    ],
  },
  {
    label: 'Platforms',
    items: [
      { to: 'platforms/youtube',   label: 'YouTube' },
      { to: 'platforms/tiktok',    label: 'TikTok' },
      { to: 'platforms/instagram', label: 'Instagram' },
      { to: 'platforms/snapchat',  label: 'Snapchat' },
      { to: 'platforms/facebook',  label: 'Facebook' },
    ],
  },
  {
    label: 'System',
    items: [
      { to: 'system/api-keys', label: 'API Keys' },
      { to: 'system/history',  label: 'Processing history' },
    ],
  },
];

export default function SettingsLayout() {
  return (
    <div className="flex h-full">
      <aside className="w-[180px] shrink-0 border-r border-border bg-surface/40 overflow-y-auto custom-scrollbar">
        <div className="px-4 py-4 border-b border-border">
          <div className="flex items-center gap-2 text-[10px] uppercase tracking-wider text-success">
            <Shield size={11} /> Keys live in browser
          </div>
        </div>
        <nav className="py-2">
          {SECTIONS.map((section) => (
            <div key={section.label} className="py-2">
              <div className="px-4 py-1 text-[10px] uppercase tracking-[0.12em] text-zinc-500">
                {section.label}
              </div>
              {section.items.map((item) => (
                <NavLink
                  key={item.to}
                  to={item.to}
                  className={({ isActive }) =>
                    `block px-4 py-1.5 text-[12px] transition-colors ${
                      isActive
                        ? 'bg-primary/10 text-primary border-l-2 border-primary'
                        : 'text-zinc-400 hover:text-white hover:bg-white/[0.03] border-l-2 border-transparent'
                    }`
                  }
                >
                  {item.label}
                </NavLink>
              ))}
            </div>
          ))}
        </nav>
      </aside>
      <div className="flex-1 overflow-y-auto custom-scrollbar">
        <div className="max-w-3xl p-8">
          <Outlet />
        </div>
      </div>
    </div>
  );
}
