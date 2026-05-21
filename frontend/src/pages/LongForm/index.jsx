// Long-form workflow: single-file 4-step wizard plus a sibling History
// tab. App.jsx mounts this under /long-form/*; routing inside is
// router-local.

import { NavLink, Outlet, Route, Routes } from 'react-router-dom';
import Wizard from './Wizard.jsx';
import History from './History.jsx';

function Shell() {
  return (
    <div className="h-full flex flex-col">
      <div className="px-6 pt-5 pb-3 border-b border-border bg-background flex items-center gap-1 shrink-0">
        <NavLink
          to="/long-form"
          end
          className={({ isActive }) =>
            `text-[13px] px-3 py-1.5 rounded-md transition-colors ${
              isActive ? 'bg-white/10 text-white' : 'text-zinc-400 hover:text-white'
            }`
          }
        >
          Wizard
        </NavLink>
        <NavLink
          to="/long-form/history"
          className={({ isActive }) =>
            `text-[13px] px-3 py-1.5 rounded-md transition-colors ${
              isActive ? 'bg-white/10 text-white' : 'text-zinc-400 hover:text-white'
            }`
          }
        >
          History
        </NavLink>
      </div>
      <div className="flex-1 overflow-hidden">
        <Outlet />
      </div>
    </div>
  );
}

export default function LongForm() {
  return (
    <Routes>
      <Route element={<Shell />}>
        <Route index element={<Wizard />} />
        <Route path="history" element={<History />} />
      </Route>
    </Routes>
  );
}
