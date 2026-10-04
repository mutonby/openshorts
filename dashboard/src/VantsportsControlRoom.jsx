import React, { useMemo, useState } from 'react';
import {
  Activity, ArrowUpRight, Bell, CalendarDays, ChevronRight, PlayCircle,
  Clock3, Crown, LayoutDashboard, Medal, MoreHorizontal, Plus,
  Search, Settings2, ShieldCheck, Sparkles, Swords, Trophy, Users, Video,
} from 'lucide-react';
import './vantsports-control-room.css';

const tournaments = [
  { title: 'VANT Open // Community Cup', game: 'VALORANT', status: 'Inscripciones abiertas', meta: '32 plazas · empieza en 4 días', prize: '500 €', tone: 'coral' },
  { title: 'Night Shift // Squad Series', game: 'CS2', status: 'Bracket en preparación', meta: '16 equipos · viernes 20:00', prize: 'Entrada gratis', tone: 'violet' },
  { title: 'VANT Weekend // 1v1', game: 'FC 26', status: 'Próximamente', meta: '64 jugadores · 18 octubre', prize: '250 €', tone: 'gold' },
];

const clips = [
  { title: 'ace retake // VANT Open', game: 'VALORANT', time: '00:37', score: '94', state: 'Listo para revisar', accent: 'coral' },
  { title: 'overtime clutch // Night Shift', game: 'CS2', time: '00:42', score: '88', state: 'Renderizando subtítulos', accent: 'violet' },
  { title: 'último minuto // VANT Weekend', game: 'FC 26', time: '00:29', score: '81', state: 'Programado para publicar', accent: 'gold' },
];

function Logo() {
  return <a className="cr-logo" href="#landing"><span className="cr-mark">V</span><span>VANTSSPORTS<small>CONTROL ROOM</small></span></a>;
}

function NavItem({ icon, label, active, onClick, count }) {
  return <button className={`cr-nav-item ${active ? 'is-active' : ''}`} onClick={onClick}>{React.createElement(icon, { size: 17 })}<span>{label}</span>{count && <b>{count}</b>}</button>;
}

function Metric({ icon, label, value, delta, tone }) {
  return <div className={`cr-metric ${tone}`}><div className="cr-metric-icon">{React.createElement(icon, { size: 17 })}</div><div><span>{label}</span><strong>{value}</strong><em>{delta}</em></div></div>;
}

function TournamentRow({ item, onOpen }) {
  return <button className="cr-tournament-row" onClick={onOpen}><span className={`cr-game-mark ${item.tone}`}>{item.game.slice(0, 2)}</span><span className="cr-row-main"><strong>{item.title}</strong><small>{item.game} · {item.meta}</small></span><span className={`cr-status ${item.tone}`}>{item.status}</span><span className="cr-prize">{item.prize}</span><ChevronRight size={17} className="cr-row-arrow" /></button>;
}

function ClipCard({ item, onOpen }) {
  return <button className="cr-clip-card" onClick={onOpen}><div className={`cr-clip-thumb ${item.accent}`}><PlayCircle size={22} /><span>{item.time}</span><i>AI MOMENT {item.score}</i></div><div className="cr-clip-copy"><small>{item.game}</small><strong>{item.title}</strong><span>{item.state}</span></div><MoreHorizontal size={17} /></button>;
}

export default function VantsportsControlRoom({ onOpenStudio }) {
  const [section, setSection] = useState('overview');
  const [query, setQuery] = useState('');
  const [toast, setToast] = useState('');
  const visibleTournaments = useMemo(() => tournaments.filter((item) => `${item.title} ${item.game}`.toLowerCase().includes(query.toLowerCase())), [query]);
  const notify = (message) => { setToast(message); window.setTimeout(() => setToast(''), 2600); };
  const openStudio = () => { if (onOpenStudio) onOpenStudio(); else window.location.hash = '#app'; };

  return <div className="cr-shell">
    <aside className="cr-sidebar"><Logo /><div className="cr-workspace"><span className="cr-avatar">F</span><span><small>WORKSPACE</small><strong>Feispla Club</strong></span><ChevronRight size={15} /></div><nav className="cr-nav"><p>OPERACIONES</p><NavItem icon={LayoutDashboard} label="Overview" active={section === 'overview'} onClick={() => setSection('overview')} /><NavItem icon={Trophy} label="Torneos" count="3" active={section === 'tournaments'} onClick={() => setSection('tournaments')} /><NavItem icon={Video} label="Clip Studio" count="8" active={section === 'clips'} onClick={() => { setSection('clips'); }} /><NavItem icon={CalendarDays} label="Calendario" active={section === 'calendar'} onClick={() => setSection('calendar')} /><p className="cr-nav-spacer">COMUNIDAD</p><NavItem icon={Users} label="Jugadores" active={section === 'players'} onClick={() => notify('Directorio de jugadores preparado para Supabase')} /><NavItem icon={Medal} label="Clasificaciones" active={section === 'rankings'} onClick={() => notify('Clasificaciones por torneo listas para conectar')} /></nav><div className="cr-sidebar-bottom"><button className="cr-nav-item" onClick={() => notify('Ajustes del workspace')}><Settings2 size={17} /><span>Ajustes</span></button><div className="cr-plan"><Crown size={16} /><span><b>PLAN ELITE</b><small>28 días restantes</small></span><ArrowUpRight size={14} /></div></div></aside>
    <main className="cr-main"><header className="cr-topbar"><div className="cr-breadcrumb"><span>WORKSPACE</span><ChevronRight size={14} /><strong>CONTROL ROOM</strong></div><div className="cr-top-actions"><label className="cr-search"><Search size={15} /><input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Buscar torneo, clip..." /></label><button className="cr-icon-btn" onClick={() => notify('No hay alertas nuevas')}><Bell size={17} /><i /></button><button className="cr-user" onClick={() => notify('Perfil de organizador')}><span>FP</span><b>Feispla</b><ChevronRight size={14} /></button></div></header>
      <div className="cr-content"><div className="cr-heading"><div><span className="cr-eyebrow"><Activity size={13} /> LIVE CIRCUIT / OCT 2026</span><h1>{section === 'overview' ? 'Tu circuito, bajo control.' : section === 'tournaments' ? 'Torneos que mueven la comunidad.' : section === 'clips' ? 'Del match al feed.' : 'La operación del circuito.'}</h1><p>Todo lo que ocurre entre la competición y el contenido, en una sola vista.</p></div><div className="cr-heading-actions"><button className="cr-button secondary" onClick={() => notify('El creador de torneos estará disponible al conectar Supabase')}><Plus size={16} /> Nuevo torneo</button><button className="cr-button primary" onClick={openStudio}><Sparkles size={16} /> Abrir Clip Studio</button></div></div>
        {section === 'overview' && <><div className="cr-metrics"><Metric icon={Trophy} label="TORNEOS ACTIVOS" value="03" delta="+1 esta semana" tone="coral" /><Metric icon={Users} label="PARTICIPANTES" value="1.248" delta="+18,4% vs mes anterior" tone="violet" /><Metric icon={Video} label="CLIPS PUBLICADOS" value="286" delta="+42 esta semana" tone="gold" /><Metric icon={Activity} label="RETENCIÓN MEDIA" value="72,8%" delta="+6,2% vs último ciclo" tone="green" /></div><div className="cr-grid-main"><section className="cr-panel cr-panel-wide"><div className="cr-panel-head"><div><span className="cr-section-kicker">01 / CIRCUITO</span><h2>Actividad del circuito</h2></div><button className="cr-text-button" onClick={() => setSection('tournaments')}>Ver todos <ArrowUpRight size={14} /></button></div><div className="cr-chart"><div className="cr-chart-labels"><span>1.4k</span><span>1.0k</span><span>600</span><span>200</span></div><div className="cr-chart-area"><div className="cr-grid-lines"><i /><i /><i /><i /></div><svg viewBox="0 0 620 170" preserveAspectRatio="none"><defs><linearGradient id="crFill" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stopColor="#ff5361" stopOpacity=".35" /><stop offset="1" stopColor="#ff5361" stopOpacity="0" /></linearGradient></defs><path d="M0,140 C38,128 48,132 78,112 S115,120 145,106 S184,72 214,92 S250,92 278,54 S323,83 350,64 S394,42 425,59 S464,28 495,48 S543,20 570,35 S602,14 620,6 L620,170 L0,170 Z" fill="url(#crFill)" /><path d="M0,140 C38,128 48,132 78,112 S115,120 145,106 S184,72 214,92 S250,92 278,54 S323,83 350,64 S394,42 425,59 S464,28 495,48 S543,20 570,35 S602,14 620,6" fill="none" stroke="#ff5361" strokeWidth="3" /></svg><div className="cr-chart-days"><span>LUN</span><span>MAR</span><span>MIÉ</span><span>JUE</span><span>VIE</span><span>SÁB</span><span>DOM</span></div></div></div><div className="cr-chart-legend"><span><i className="coral" /> Participantes activos</span><span><i className="violet" /> Clips procesados</span><b>+24,8% <small>últimos 7 días</small></b></div></section><section className="cr-panel cr-live-panel"><div className="cr-panel-head"><div><span className="cr-section-kicker">02 / LIVE</span><h2>Ahora mismo</h2></div><span className="cr-live-dot">LIVE</span></div><div className="cr-match"><div className="cr-match-top"><span>VANT OPEN</span><small>ROUND 3 · BO3</small></div><div className="cr-match-score"><strong>2</strong><em>:</em><strong>1</strong></div><div className="cr-match-teams"><span>RED SHIFT</span><span>NEON KINGS</span></div></div><div className="cr-live-stat"><span><Clock3 size={14} /> Tiempo de partida</span><b>34:18</b></div><div className="cr-live-stat"><span><ShieldCheck size={14} /> Estado de producción</span><b className="ok">Conectado</b></div><button className="cr-full-button" onClick={() => notify('Vista de match en desarrollo')}>Abrir match center <ArrowUpRight size={15} /></button></section></div><div className="cr-grid-bottom"><section className="cr-panel"><div className="cr-panel-head"><div><span className="cr-section-kicker">03 / TORNEOS</span><h2>Próximas competiciones</h2></div><button className="cr-text-button" onClick={() => setSection('tournaments')}>Gestionar <ArrowUpRight size={14} /></button></div><div className="cr-list">{visibleTournaments.map((item) => <TournamentRow key={item.title} item={item} onOpen={() => notify(`${item.title}: detalle de torneo`)} />)}</div></section><section className="cr-panel"><div className="cr-panel-head"><div><span className="cr-section-kicker">04 / CLIP PIPELINE</span><h2>Últimos momentos</h2></div><button className="cr-text-button" onClick={() => setSection('clips')}>Abrir Studio <ArrowUpRight size={14} /></button></div><div className="cr-clip-list">{clips.map((item) => <ClipCard key={item.title} item={item} onOpen={openStudio} />)}</div></section></div></>}
        {section !== 'overview' && <section className="cr-panel cr-section-page"><div className="cr-panel-head"><div><span className="cr-section-kicker">VANTSSPORTS / {section.toUpperCase()}</span><h2>{section === 'tournaments' ? 'Gestión de torneos' : section === 'clips' ? 'Biblioteca de clips' : 'Módulo del circuito'}</h2></div><button className="cr-button primary" onClick={section === 'clips' ? openStudio : () => notify('Módulo conectado al backend premium')}><Sparkles size={16} /> {section === 'clips' ? 'Crear clips' : 'Nueva acción'}</button></div><div className="cr-section-placeholder"><Swords size={42} /><h3>Este módulo ya tiene su lugar en el producto.</h3><p>La interfaz premium está lista para conectar datos vivos de Supabase, pagos PayPal y el motor VANTCLIP sin cambiar la experiencia.</p>{section === 'tournaments' && <div className="cr-list">{visibleTournaments.map((item) => <TournamentRow key={item.title} item={item} onOpen={() => notify(`${item.title}: detalle de torneo`)} />)}</div>}</div></section>}
      </div></main>{toast && <div className="cr-toast"><ShieldCheck size={16} />{toast}</div>}
  </div>;
}
