// Short-form workflow: 4-step wizard for vertical clip generation from up
// to 5 source videos, with a sibling History tab. The parent App.jsx
// route mounts this under /short-form/*; everything inside here is
// router-local.

import { NavLink, Outlet, Route, Routes } from 'react-router-dom';
import Wizard from './Wizard.jsx';
import History from './History.jsx';

function Shell() {
  return (
    <div className="h-full flex flex-col">
      <div className="px-6 pt-5 pb-3 border-b border-border bg-background flex items-center gap-1 shrink-0">
        <NavLink
          to="/short-form"
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
          to="/short-form/history"
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

export default function ShortForm() {
  return (
    <Routes>
      <Route element={<Shell />}>
        <Route index element={<Wizard />} />
        <Route path="history" element={<History />} />
      </Route>
    </Routes>
  );
}
