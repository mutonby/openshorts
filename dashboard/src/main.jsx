import { StrictMode, useState, useEffect, lazy, Suspense } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import VantsportsLanding from './VantsportsLanding.jsx'
import VantsportsControlRoom from './VantsportsControlRoom.jsx'
import VantsportsNetwork from './VantsportsNetwork.jsx'
import { AuthProvider, useAuth } from './contexts/AuthContext'
import { capture as captureAttribution } from './lib/attribution'
import PricingPage from './components/PricingPage'
import AccountPage from './components/AccountPage'
import LoginModal from './components/LoginModal'
import { applyConsent } from './lib/consent'
import CookieBanner from './components/CookieBanner'

const App = lazy(() => import('./App.jsx'))
const Legal = lazy(() => import('./Legal.jsx'))
const OAuthConsent = lazy(() => import('./components/OAuthConsent'))

function PageShell({ title, children }) {
  return (
    <div className="min-h-screen bg-paper text-ink2">
      <header className="h-14 sm:h-16 border-b border-rule bg-paper flex items-center justify-between gap-3 px-4 sm:px-6 sticky top-0 z-20">
        <a href="#app" className="font-display lowercase text-lg text-ink truncate">OpenShorts</a>
        <a href="#app" className="text-sm lowercase text-muted hover:text-ink transition-colors shrink-0">← <span className="hidden sm:inline">Back to app</span><span className="sm:hidden">back</span></a>
      </header>
      <main className="p-4 sm:p-6 md:p-8 pb-[max(2rem,env(safe-area-inset-bottom))]">
        {title && <h1 className="font-display lowercase text-2xl sm:text-3xl text-ink text-center mb-6 sm:mb-10">{title}</h1>}
        {children}
      </main>
    </div>
  );
}

function PricingView() {
  const [showLogin, setShowLogin] = useState(false);
  return (
    <PageShell>
      <PricingPage onRequireLogin={() => setShowLogin(true)} />
      {showLogin && <LoginModal onClose={() => setShowLogin(false)} />}
    </PageShell>
  );
}

function AccountView() {
  const { isSignedIn, loading } = useAuth();
  useEffect(() => {
    if (!loading && !isSignedIn) window.location.hash = '#/pricing';
  }, [loading, isSignedIn]);
  return <PageShell><AccountPage /></PageShell>;
}

// Landing spot after an account is erased. Its own view because the session is
// gone: sending the user to #/account would bounce them to pricing with no
// explanation, and "openshorts_skip_landing" would send them into the app.
function DeletedView() {
  return (
    <div className="min-h-screen bg-paper text-ink2 flex items-center justify-center p-6">
      <div className="max-w-md text-center space-y-4">
        <h1 className="font-display lowercase text-2xl sm:text-3xl text-ink">Your account is deleted</h1>
        <p className="text-sm">
          Your projects, clips and transcripts are gone, any subscription is
          cancelled, and your API keys no longer work. We've emailed you a
          confirmation with the details.
        </p>
        <p className="text-sm text-muted">
          You're welcome back any time — signing up again with the same address
          starts a brand-new, empty account.
        </p>
        <a href="#landing" className="btn-ghost px-4 py-2 inline-flex">Back to openshorts.app</a>
      </div>
    </div>
  );
}

function Root() {
  const resolveView = () => {
    const hash = window.location.hash || '';
    if (hash.startsWith('#/auth/')) return 'auth';       // AuthContext consumes then redirects
    if (hash.startsWith('#/oauth/authorize')) return 'oauth';
    if (hash.startsWith('#/account')) return 'account';
    if (hash.startsWith('#/deleted')) return 'deleted';
    if (hash.startsWith('#/pricing')) return 'pricing';
    if (hash === '#circuit' || hash === '#/circuit') return 'network';
    if (hash === '#/control-room') return 'circuit';
    if (hash === '#/clip-studio') return 'app';
    if (hash === '#legal') return 'legal';
    // #landing = explicit landing view (app logo); section anchors keep the landing mounted
    if (['#landing', '#features', '#how-it-works', '#pricing', '#comparison', '#faq'].includes(hash)) return 'landing';
    if (hash === '#app' || hash.startsWith('#app?') || localStorage.getItem('openshorts_skip_landing') === '1') return 'app';
    return 'landing';
  };

  const [view, setView] = useState(resolveView);

  useEffect(() => {
    const handleHashChange = () => setView(resolveView());
    window.addEventListener('hashchange', handleHashChange);
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, []);

  const handleLaunchApp = () => {
    localStorage.setItem('openshorts_skip_landing', '1');
    window.location.hash = '#app';
    setView('app');
  };

  if (view === 'legal') return <Legal />;
  if (view === 'pricing') return <PricingView />;
  if (view === 'account') return <AccountView />;
  if (view === 'oauth') return <OAuthConsent />;
  if (view === 'deleted') return <DeletedView />;
  if (view === 'auth') {
    return <div className="min-h-screen flex items-center justify-center bg-background text-zinc-400">Signing you in…</div>;
  }
  if (view === 'app') return <App />;
  if (view === 'network') return <VantsportsNetwork onOpenCircuit={() => { window.location.hash = '#/control-room'; setView('circuit'); }} onOpenStudio={handleLaunchApp} />;
  if (view === 'circuit') return <VantsportsControlRoom onOpenStudio={handleLaunchApp} />;
  return <VantsportsLanding onLaunchApp={handleLaunchApp} />;
}

// Before React mounts: AuthContext rewrites the URL on auth redirects, which
// would destroy the referrer and any UTM params we still need to read.
captureAttribution();

// Start whatever the visitor previously agreed to. Nothing at all on a first
// visit: index.html only publishes the analytics initialiser, it never runs it.
applyConsent();

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <AuthProvider>
      <Suspense fallback={<div className="min-h-screen bg-paper flex items-center justify-center text-muted text-sm lowercase">loading…</div>}>
        <Root />
      </Suspense>
      <CookieBanner />
    </AuthProvider>
  </StrictMode>,
)
