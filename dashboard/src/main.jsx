import { StrictMode, useState, useEffect, lazy, Suspense } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'

// Only one of the two is ever shown; lazy-load both so returning app users
// don't download the landing page and first-time visitors don't download the app.
const App = lazy(() => import('./App.jsx'))
const Landing = lazy(() => import('./Landing.jsx'))

function Root() {
  const [showApp, setShowApp] = useState(() => {
    return window.location.hash === '#app' || localStorage.getItem('openshorts_skip_landing') === '1';
  });

  useEffect(() => {
    const handleHashChange = () => {
      setShowApp(window.location.hash === '#app');
    };
    window.addEventListener('hashchange', handleHashChange);
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, []);

  const handleLaunchApp = () => {
    localStorage.setItem('openshorts_skip_landing', '1');
    window.location.hash = '#app';
    setShowApp(true);
  };

  if (showApp) {
    return (
      <Suspense fallback={null}>
        <App />
      </Suspense>
    );
  }

  return (
    <Suspense fallback={null}>
      <Landing onLaunchApp={handleLaunchApp} />
    </Suspense>
  );
}

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <Root />
  </StrictMode>,
)
