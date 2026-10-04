import React, { useState } from 'react';
import { Activity, ArrowRight, Bell, ChevronDown, ChevronRight, Crown, Gamepad2, Globe2, Menu, Search, Shield, Sparkles, Trophy, UserRound, Users, X, Zap } from 'lucide-react';
import './vantsports-network.css';

const games = [
  { name: 'VALORANT', code: 'VAL', className: 'valorant', players: '842K' },
  { name: 'Counter-Strike 2', code: 'CS2', className: 'cs2', players: '516K' },
  { name: 'League of Legends', code: 'LOL', className: 'lol', players: '391K' },
  { name: 'Fortnite', code: 'FN', className: 'fortnite', players: '284K' },
  { name: 'Rocket League', code: 'RL', className: 'rocket', players: '126K' },
  { name: 'EA SPORTS FC 26', code: 'FC', className: 'fc', players: '98K' },
];

const ranks = [
  ['01', 'NoxVanta', 'VALORANT', '2,418', 'SCARLET'],
  ['02', 'Mauri.exe', 'CS2', '2,204', 'TITÁN'],
  ['03', 'KiroVANTS', 'LOL', '2,091', 'DIAMANTE'],
  ['04', 'LunaShift', 'VALORANT', '1,987', 'DIAMANTE'],
];

const tournaments = [
  { title: 'VANT Open // Community Cup', game: 'VALORANT', date: '12 OCT', teams: '32 / 32', prize: '500 €', state: 'EN DIRECTO', tone: 'red' },
  { title: 'Night Shift // Squad Series', game: 'CS2', date: '18 OCT', teams: '12 / 16', prize: '1.000 €', state: 'INSCRIPCIONES', tone: 'violet' },
  { title: 'VANT Weekend // 1v1', game: 'FC 26', date: '25 OCT', teams: '24 / 64', prize: '250 €', state: 'ABIERTO', tone: 'gold' },
];

function Brand() { return <a className="vn-brand" href="#landing"><img src="/vantsports/favicon-32.png" alt="" /><span>VANTS<small>NETWORK</small></span></a>; }
function GameCard({ game, compact = false }) { return <button className={`vn-game-card ${game.className} ${compact ? 'compact' : ''}`}><div className="vn-game-bg"><span>{game.code}</span></div><div className="vn-game-copy"><strong>{game.name}</strong><small><Activity size={11} /> {game.players} jugadores</small></div><ChevronRight size={15} /></button>; }
function SectionTitle({ eyebrow, title, action }) { return <div className="vn-section-title"><div><span>{eyebrow}</span><h2>{title}</h2></div>{action}</div>; }

export default function VantsportsNetwork({ onOpenCircuit, onOpenStudio }) {
  const [menu, setMenu] = useState(false);
  const [gameFilter, setGameFilter] = useState('POPULAR');
  return <div className="vn-app">
    <header className="vn-topbar"><Brand /><nav className="vn-main-nav"><a className="active" href="#circuit">Inicio</a><a href="#circuit">Juegos <ChevronDown size={12} /></a><a href="#circuit">Ranked</a><a href="#circuit">Torneos</a><a href="#circuit">Noticias</a></nav><div className="vn-top-actions"><button className="vn-search-btn"><Search size={16} /><span>Buscar jugador</span></button><button className="vn-bell"><Bell size={16} /><i /></button><button className="vn-premium"><Crown size={14} /> PREMIUM</button><button className="vn-login" onClick={() => setMenu(!menu)}><UserRound size={15} /> Entrar</button><button className="vn-menu" onClick={() => setMenu(!menu)}>{menu ? <X size={18} /> : <Menu size={18} />}</button></div></header>
    {menu && <div className="vn-mobile-menu"><a href="#circuit">Inicio</a><a href="#circuit">Juegos</a><a href="#circuit">Ranked</a><a href="#circuit">Torneos</a><a href="#circuit">Noticias</a></div>}
    <main className="vn-page"><section className="vn-hero"><div className="vn-hero-copy"><span className="vn-kicker"><i /> PLATAFORMA COMPETITIVA · TEMPORADA 01</span><h1>Juega.<br /><em>Sube.</em><br />Deja marca.</h1><p>Tu identidad competitiva para Valorant, CS2, League of Legends y más. Ranked con MMR, torneos con premios y tus mejores momentos convertidos en clips.</p><div className="vn-hero-actions"><button className="vn-cta" onClick={onOpenCircuit}>Entrar al circuito <ArrowRight size={16} /></button><button className="vn-ghost" onClick={onOpenStudio}><Sparkles size={15} /> Crear un clip</button></div><div className="vn-proof"><span><Shield size={13} /> MMR verificado</span><span><Trophy size={13} /> Premios reales</span><span><Users size={13} /> Comunidad VANTS</span></div></div><div className="vn-hero-visual"><div className="vn-hero-image" /><div className="vn-hero-card"><span>LIVE CIRCUIT</span><strong>VANT OPEN</strong><div><b>2</b><i>:</i><b>1</b></div><small>RED SHIFT vs NEON KINGS</small></div><div className="vn-hero-rank"><span>TOP RANKED</span><strong>NoxVanta</strong><small>SCARLET · 2,418 MMR</small><b>01</b></div></div></section>
      <section className="vn-stats"><div><strong>2.4M+</strong><span>PARTIDAS REGISTRADAS</span></div><div><strong>48K</strong><span>JUGADORES ACTIVOS</span></div><div><strong>126</strong><span>TORNEOS ESTA TEMPORADA</span></div><div><strong>9.8K €</strong><span>PREMIOS EN JUEGO</span></div></section>
      <section className="vn-section vn-games"><SectionTitle eyebrow="01 / GAME HUB" title="Tus juegos. Tu historial." action={<button className="vn-link">Ver todos los juegos <ArrowRight size={14} /></button>} /><div className="vn-game-featured"><GameCard game={games[0]} /><GameCard game={games[1]} /><GameCard game={games[2]} /></div><div className="vn-games-toolbar"><div><button className={gameFilter === 'POPULAR' ? 'active' : ''} onClick={() => setGameFilter('POPULAR')}>Más populares</button><button className={gameFilter === 'ALL' ? 'active' : ''} onClick={() => setGameFilter('ALL')}>Todos los juegos</button></div><span><Globe2 size={13} /> Datos globales en vivo</span></div><div className="vn-game-grid">{games.slice(3).map((game) => <GameCard key={game.name} game={game} compact />)}</div></section>
      <section className="vn-section vn-competitive"><div className="vn-competitive-copy"><span className="vn-kicker">02 / RANKED VANTS</span><h2>Que tu nivel<br /><em>hable por ti.</em></h2><p>Ocho rangos. Un recorrido por juego. Cada victoria, MVP y clutch suma VP a tu perfil. La temporada cambia; tu historial queda.</p><button className="vn-cta" onClick={onOpenCircuit}>Ver mi ranking <ArrowRight size={15} /></button></div><div className="vn-rank-ladder"><div className="vn-ladder-head"><span>LEADERBOARD GLOBAL</span><button>Esta semana <ChevronDown size={13} /></button></div>{ranks.map(([pos, name, game, mmr, rank]) => <div className="vn-rank-row" key={name}><b>{pos}</b><span className="vn-avatar">{name.slice(0, 1)}</span><div><strong>{name}</strong><small>{game}</small></div><span className="vn-rank-name">{rank}</span><strong className="vn-mmr">{mmr} <small>MMR</small></strong><ChevronRight size={15} /></div>)}<button className="vn-board-button" onClick={onOpenCircuit}>Abrir leaderboard completo <ArrowRight size={14} /></button></div></section>
      <section className="vn-section vn-tournaments"><SectionTitle eyebrow="03 / TOURNAMENTS" title="La próxima partida importa." action={<button className="vn-link" onClick={onOpenCircuit}>Ver calendario <ArrowRight size={14} /></button>} /><div className="vn-tournament-list">{tournaments.map((item) => <button className="vn-tournament" key={item.title} onClick={onOpenCircuit}><span className={`vn-date ${item.tone}`}><b>{item.date.split(' ')[0]}</b><small>{item.date.split(' ')[1]}</small></span><div><span className={`vn-state ${item.tone}`}>{item.state}</span><strong>{item.title}</strong><small>{item.game} · {item.teams} participantes</small></div><span className="vn-tournament-prize"><small>PREMIO</small><b>{item.prize}</b></span><ChevronRight size={17} /></button>)}</div></section>
      <section className="vn-section vn-bottom-grid"><div className="vn-news"><SectionTitle eyebrow="04 / VANTS NEWS" title="Lo que pasa en la escena." action={<button className="vn-link">Todas las noticias <ArrowRight size={14} /></button>} /><article><div className="vn-news-image" /><div className="vn-news-copy"><span>COMUNIDAD · HACE 2 DÍAS</span><h3>El circuito VANTS abre su primera temporada competitiva.</h3><p>Ranked, torneos, scrims y un lugar para que cada partida cuente.</p><button className="vn-link">Leer historia <ArrowRight size={13} /></button></div></article></div><div className="vn-membership"><span className="vn-kicker"><Crown size={13} /> VANTS PREMIUM</span><h2>Tu temporada.<br /><em>Sin límites.</em></h2><p>Desbloquea ranked completo, salas privadas, torneos premium y tu zona exclusiva.</p><div><strong>19 €</strong><span>pago único<br />toda la temporada</span></div><button className="vn-cta">Ver planes <ArrowRight size={15} /></button></div></section>
    </main><footer className="vn-footer"><Brand /><span>Vantsports Network · Competición, comunidad y clips.</span><div><a href="#circuit">Discord</a><a href="#circuit">API</a><a href="#circuit">Soporte</a></div></footer>
  </div>;
}
