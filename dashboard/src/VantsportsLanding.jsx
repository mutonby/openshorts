import React from 'react';
import {
  ArrowRight,
  Check,
  Clapperboard,
  Clock3,
  Coins,
  Gamepad2,
  MessageCircle,
  ShieldCheck,
  Sparkles,
  Trophy,
  Video,
  Zap,
} from 'lucide-react';
import { useAuth } from './contexts/AuthContext';
import './vantsports-landing.css';

const tournaments = [
  { game: 'VALORANT', title: 'VANT Open // Community Cup', meta: 'Inscripción abierta · 32 plazas', prize: 'Premio: 500 €', tone: 'coral' },
  { game: 'CS2', title: 'Night Shift // Squad Series', meta: 'Empieza el viernes · 16 equipos', prize: 'Entrada gratuita', tone: 'amber' },
  { game: 'FC 26', title: 'VANT Weekend // 1v1', meta: 'Clasificatorio · 64 jugadores', prize: 'Premio: 250 €', tone: 'blue' },
];

const plans = [
  { name: 'BASIC', price: '9', label: 'Para entrar en escena', features: ['Torneos abiertos', 'Perfil de participante', 'Rol BASIC en Discord', 'Biblioteca de clips propia'] },
  { name: 'PRO', price: '19', label: 'Para equipos que compiten', featured: true, features: ['Todo lo de BASIC', 'Pro Series y scrims', 'Prioridad en tryouts', 'Clips oficiales de torneo'] },
  { name: 'ELITE', price: '39', label: 'Acceso total al circuito', elite: true, features: ['Todo lo de PRO', 'Elite Invitational', 'Canal privado con staff', 'Branding premium para clips'] },
];

const features = [
  { icon: Trophy, title: 'Torneos con propósito', text: 'Crea eventos gratuitos o de pago con grupos, brackets, jornadas y resultados claros.' },
  { icon: Clapperboard, title: 'Clips que nacen del torneo', text: 'VANTCLIP detecta momentos, añade subtítulos y exporta formatos 9:16 listos para publicar.' },
  { icon: MessageCircle, title: 'Discord conectado', text: 'Roles, avisos, canales y estados de inscripción sincronizados con tu comunidad.' },
  { icon: ShieldCheck, title: 'Pagos y permisos seguros', text: 'Stripe y PayPal se confirman por webhook; cada rol solo ve lo que necesita.' },
];

function VantsLogo() {
  return <img className="vs-logo-image" src="/vantsports/favicon-32.png" alt="" width="32" height="32" />;
}

function SectionHeading({ eyebrow, title, text }) {
  return (
    <div className="vs-section-heading">
      <span className="vs-eyebrow">{eyebrow}</span>
      <h2>{title}</h2>
      {text && <p>{text}</p>}
    </div>
  );
}

export default function VantsportsLanding({ onLaunchApp }) {
  const { billingEnabled } = useAuth();

  return (
    <div className="vs-page">
      <header className="vs-header">
        <a className="vs-brand" href="#landing" aria-label="Vantsports inicio"><VantsLogo /><span>VANTS<span className="vs-brand-sub">SPORTS</span></span></a>
        <nav className="vs-nav" aria-label="Navegación principal">
          <a href="#vs-platform">Plataforma</a><a href="#vs-tournaments">Torneos</a><a href="#vs-clips">VANTCLIP</a><a href="#vs-pricing">Planes</a>
        </nav>
        <div className="vs-header-actions"><a className="vs-text-link" href="#circuit">Entrar al circuito</a><button className="vs-button vs-button-small" onClick={onLaunchApp}>Crear clips <ArrowRight size={14} /></button></div>
      </header>

      <main>
        <section className="vs-hero" id="landing">
          <div className="vs-hero-grid" aria-hidden="true" />
          <div className="vs-hero-glow" aria-hidden="true" />
          <div className="vs-hero-copy">
            <div className="vs-kicker"><span className="vs-live-dot" /> CIRCUITO COMPETITIVO · VANTCLIP INSIDE</div>
            <h1>Torneos que generan historia.<br /><em>Clips que se vuelven señal.</em></h1>
            <p>Vantsports une la competición, la comunidad y la creación de contenido en una sola plataforma premium. Organiza tu torneo y convierte cada jugada en un momento compartible.</p>
            <div className="vs-hero-actions"><button className="vs-button" onClick={onLaunchApp}>Crear mi primer clip <ArrowRight size={16} /></button><a className="vs-button vs-button-ghost" href="#vs-tournaments">Explorar torneos <Trophy size={16} /></a></div>
            <div className="vs-hero-meta"><span><Gamepad2 size={14} /> 12+ juegos</span><span><Video size={14} /> 9:16 nativo</span><span><ShieldCheck size={14} /> Pagos verificados</span></div>
          </div>
          <div className="vs-hero-console" aria-label="Vista previa del panel Vantsports">
            <div className="vs-console-top"><span className="vs-console-dot" /><span className="vs-console-label">VANT // CONTROL ROOM</span><span className="vs-console-status">LIVE</span></div>
            <div className="vs-console-main"><div className="vs-console-title">NEXT MATCH</div><div className="vs-match"><span>VANT OPEN</span><strong>2 : 1</strong><span>RED SHIFT</span></div><div className="vs-console-line"><span>CLIP PIPELINE</span><span>03 / 08 READY</span></div><div className="vs-progress"><i /></div></div>
            <div className="vs-console-foot"><span>AI MOMENT DETECTION</span><span className="vs-console-accent">CONNECTED</span></div>
          </div>
        </section>

        <section className="vs-stats" aria-label="Datos de la plataforma"><div><strong>9:16</strong><span>clips verticales</span></div><div><strong>24/7</strong><span>circuito conectado</span></div><div><strong>0</strong><span>rangos globales</span></div><div><strong>100%</strong><span>tu comunidad</span></div></section>

        <section className="vs-section vs-platform" id="vs-platform"><SectionHeading eyebrow="01 · SISTEMA" title="Una capa competitiva sobre un motor de clips." text="Vantsports gestiona la experiencia de competición. VANTCLIP hace el trabajo pesado de vídeo. Dos sistemas, una señal de marca." /><div className="vs-feature-grid">{features.map(({ icon, title, text }) => <article className="vs-feature" key={title}><div className="vs-feature-icon">{React.createElement(icon, { size: 19 })}</div><h3>{title}</h3><p>{text}</p><span className="vs-feature-arrow"><ArrowRight size={14} /></span></article>)}</div></section>

        <section className="vs-section vs-tournament-section" id="vs-tournaments"><div className="vs-section-split"><SectionHeading eyebrow="02 · CALENDARIO" title="Compite por el momento que todos van a recordar." text="Clasificaciones temporales por torneo, resultados verificables y una biblioteca de clips que crece con cada jornada." /><a className="vs-inline-link" href="#vs-pricing">Ver todos los torneos <ArrowRight size={15} /></a></div><div className="vs-tournament-grid">{tournaments.map((t) => <article className={`vs-tournament vs-tone-${t.tone}`} key={t.title}><div className="vs-tournament-top"><span>{t.game}</span><span className="vs-open-pill">OPEN</span></div><h3>{t.title}</h3><p>{t.meta}</p><div className="vs-tournament-bottom"><strong>{t.prize}</strong><span><Clock3 size={13} /> Próximamente</span></div></article>)}</div></section>

        <section className="vs-section vs-clips-section" id="vs-clips"><div className="vs-clips-card"><div><span className="vs-eyebrow">03 · VANTCLIP STUDIO</span><h2>De la partida al feed en minutos.</h2><p>Sube un VOD, elige un torneo y deja que el pipeline encuentre los momentos. Revisa subtítulos, branding y formato antes de exportar.</p><button className="vs-button" onClick={onLaunchApp}>Abrir Clip Studio <Sparkles size={16} /></button></div><div className="vs-pipeline"><div className="vs-pipeline-item"><span>01</span><strong>Detectar</strong><small>momentos IA</small></div><div className="vs-pipeline-line" /><div className="vs-pipeline-item"><span>02</span><strong>Editar</strong><small>subtítulos + marca</small></div><div className="vs-pipeline-line" /><div className="vs-pipeline-item"><span>03</span><strong>Publicar</strong><small>TikTok · Reels · Shorts</small></div></div></div></section>

        <section className="vs-section vs-pricing-section" id="vs-pricing"><SectionHeading eyebrow="04 · MEMBRESÍAS" title="Elige tu lugar en el circuito." text="Planes de temporada para jugadores, equipos y organizadores. Sin rangos globales: tu progreso pertenece a cada torneo." /><div className="vs-plan-grid">{plans.map((plan) => <article className={`vs-plan ${plan.featured ? 'vs-plan-featured' : ''} ${plan.elite ? 'vs-plan-elite' : ''}`} key={plan.name}>{plan.featured && <span className="vs-plan-badge">MÁS POPULAR</span>}<span className="vs-plan-tier">{plan.name}</span><h3>{plan.label}</h3><div className="vs-plan-price"><b>{plan.price}</b><span>€<br />/ temporada</span></div><ul>{plan.features.map((feature) => <li key={feature}><Check size={15} />{feature}</li>)}</ul><button className="vs-button vs-plan-button" onClick={onLaunchApp}>Explorar plan <ArrowRight size={14} /></button></article>)}</div>{billingEnabled && <p className="vs-billing-note"><Coins size={14} /> Pagos seguros y gestión de cuenta disponibles en el área premium.</p>}</section>

        <section className="vs-final-cta"><div><span className="vs-eyebrow">05 · TU PRÓXIMA JUGADA</span><h2>La comunidad ya está lista.<br />Solo falta tu torneo.</h2></div><button className="vs-button" onClick={onLaunchApp}>Entrar a VANTCLIP <Zap size={16} /></button></section>
      </main>
      <footer className="vs-footer"><div className="vs-brand"><VantsLogo /><span>VANTS<span className="vs-brand-sub">SPORTS</span></span></div><span>Vantsports · Competición, comunidad y clips.</span><span>VANTCLIP engine · 9:16 native</span></footer>
    </div>
  );
}
