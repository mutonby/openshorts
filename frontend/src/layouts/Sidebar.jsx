import { NavLink } from 'react-router-dom';
import { LayoutDashboard, Smartphone, Video, Scissors, Settings as SettingsIcon, Wand2 } from 'lucide-react';

const NAV = [
  { to: '/dashboard',      label: 'Dashboard',      icon: LayoutDashboard },
  { to: '/long-form',      label: 'Long-form',      icon: Video },
  { to: '/ai-restyle',     label: 'AI Restyle',     icon: Wand2 },
  { to: '/short-form',     label: 'Short-form',     icon: Smartphone },
  { to: '/clip-generator', label: 'Clip Generator', icon: Scissors },
  { to: '/settings',       label: 'Settings',       icon: SettingsIcon },
];

export default function Sidebar() {
  return (
    <aside className="w-[210px] shrink-0 bg-sidebar border-r border-border flex flex-col h-full">
      <div className="h-[50px] flex items-center gap-3 px-5 border-b border-border">
        <div className="w-7 h-7 rounded-md overflow-hidden bg-white/5 border border-border shrink-0">
          <img src="/logo-openshorts.png" alt="OpenShorts" className="w-full h-full object-cover" />
        </div>
        <span className="text-[15px] font-semibold tracking-tight text-white">OpenShorts</span>
      </div>

      <nav className="flex-1 px-3 py-4 space-y-1">
        {NAV.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2 rounded-lg text-[13px] transition-colors ${
                isActive
                  ? 'bg-primary/15 text-primary'
                  : 'text-zinc-400 hover:text-white hover:bg-white/5'
              }`
            }
          >
            <Icon size={16} />
            <span className="font-medium">{label}</span>
          </NavLink>
        ))}
      </nav>
    </aside>
  );
}
