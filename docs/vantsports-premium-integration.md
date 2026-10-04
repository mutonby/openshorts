# Integración premium Vantsports × VANTCLIP

## Objetivo

La landing de VANTCLIP pasa a presentar **Vantsports** como la capa de producto premium para competición, comunidad y creación de contenido. El motor de procesamiento de vídeo de VANTCLIP permanece separado y se sigue lanzando desde la aplicación existente.

## Qué se integró

- Nueva landing React en `dashboard/src/VantsportsLanding.jsx`.
- Sistema visual premium en `dashboard/src/vantsports-landing.css`, con estética VANTS coral/ámbar, panel de control, tarjetas de torneos y planes.
- Navegación de producto para plataforma, torneos, VANTCLIP Studio y membresías.
- CTA funcional hacia el flujo actual de creación de clips mediante `onLaunchApp`.
- Previsualización de torneos gratuitos y de pago, Discord conectado, pagos verificados y pipeline de clips 9:16.
- Asset visual de Vantsports en `dashboard/public/vantsports/hero-arena.jpg`.
- Metadatos HTML principales en español para Vantsports × VANTCLIP.
- Host público del sandbox permitido en `dashboard/vite.config.js` para revisión de previews.

## Stack técnico de la nueva web

La integración queda preparada con **React** para la interfaz, **JavaScript** para la lógica existente del dashboard, **CSS** para el sistema visual premium, **TypeScript** para los contratos compartidos en `dashboard/src/vantsports/types.ts`, **HTML** como documento de entrada y SEO, **Python** para el puente firmado entre Vantsports y VANTCLIP en `vantsports/api/bridge.py`, y **PL/pgSQL** para la migración Supabase en `supabase/migrations/20261004190000_vantsports_core.sql`.

## Decisiones de arquitectura

- **React/Vite** se mantiene como base de la web para aprovechar autenticación, billing, app y componentes existentes de VANTCLIP.
- El procesamiento de vídeo, workers, API y render siguen sin cambios.
- Los torneos, pagos, Discord, perfiles y administración se representan en la capa de producto; la conexión real con Supabase, Stripe, PayPal y Discord debe implementarse en módulos backend separados.
- No se presentan rangos globales. La clasificación se describe como temporal y asociada a cada torneo.
- Los planes de la landing son una interfaz de producto; los cobros solo deben activarse cuando la configuración de billing esté conectada a webhooks firmados.

## Validación

- `npm run build` pasa correctamente.
- `npx eslint src/main.jsx src/VantsportsLanding.jsx --report-unused-disable-directives --max-warnings 0` pasa correctamente.
- El lint completo del repositorio todavía contiene errores preexistentes en `public/op1.js`, `Legal.jsx`, `PricingPage.jsx` y `lib/consent.js`; no fueron modificados por esta integración.
- La landing se revisó en la URL de preview del sandbox y carga con la navegación, tarjetas, panel y CTA visibles.

## Siguiente fase recomendada

1. Conectar las tarjetas de torneos a tablas Supabase con `tournament_id`.
2. Añadir rutas de organizador, inscripción y bracket.
3. Implementar el contrato firmado entre Vantsports API y VANTCLIP API.
4. Conectar los planes a Stripe/PayPal mediante webhooks idempotentes.
5. Sustituir los datos de ejemplo de la landing por datos publicados y cacheados.
