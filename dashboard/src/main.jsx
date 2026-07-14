import { StrictMode, useState, useEffect } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'
import Landing from './Landing.jsx'
import Legal from './Legal.jsx'
import { AuthProvider, useAuth } from './contexts/AuthContext'
import PricingSection from './components/PricingSection'
import AccountPage from './components/AccountPage'
import LoginModal from './components/LoginModal'

function PageShell({ title, children }) {
  return (
    <div className="min-h-screen bg-background text-white">
      <header className="h-16 border-b border-white/5 flex items-center justify-between px-6">
        <a href="#app" className="font-bold text-lg">OpenShorts</a>
        <a href="#app" className="text-sm text-zinc-400 hover:text-white">← Back to app</a>
      </header>
      <main className="p-8">
        {title && <h1 className="text-3xl font-bold text-center mb-10">{title}</h1>}
        {children}
      </main>
    </div>
  );
}

function PricingView() {
  const [showLogin, setShowLogin] = useState(false);
  return (
    <PageShell title="Choose your plan">
      <PricingSection onRequireLogin={() => setShowLogin(true)} />
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

function Root() {
  const resolveView = () => {
    const hash = window.location.hash || '';
    if (hash.startsWith('#/auth/')) return 'auth';       // AuthContext consumes then redirects
    if (hash.startsWith('#/account')) return 'account';
    if (hash.startsWith('#/pricing')) return 'pricing';
    if (hash === '#legal') return 'legal';
    if (hash === '#app' || localStorage.getItem('openshorts_skip_landing') === '1') return 'app';
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
  if (view === 'auth') {
    return <div className="min-h-screen flex items-center justify-center bg-background text-zinc-400">Signing you in…</div>;
  }
  if (view === 'app') return <App />;
  return <Landing onLaunchApp={handleLaunchApp} />;
}

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <AuthProvider>
      <Root />
    </AuthProvider>
  </StrictMode>,
)
