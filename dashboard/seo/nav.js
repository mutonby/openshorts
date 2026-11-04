/* The one site navigation, shared by every surface that shows a header:
 *
 *   - the static SEO pages (seo/render.js),
 *   - the crawler-visible homepage fallback (seo/landing-fallback.js),
 *   - the React landing (src/components/SiteNav.jsx, which renders this same
 *     markup so the header does not change when React replaces the fallback).
 *
 * Every link is a plain <a href> in the HTML: crawlers that never run
 * JavaScript must see the whole menu. Dropdowns are <details>/<summary> (open
 * on click, tap or Enter/Space with no script at all); the mobile menu is a
 * checkbox toggle for the same reason. NAV_SCRIPT only adds the niceties
 * (one dropdown open at a time, close on outside click and Escape).
 *
 * Link targets are the same on every page and in every language; only the
 * labels change. Labels of the pages themselves stay in English because the
 * pages are English (same rule as the footer).
 *
 * This file is imported by the browser bundle too, so keep it free of Node
 * APIs and of imports that would pull the SEO page data into the app. */

export const REPO_URL = 'https://github.com/mutonby/openshorts'

/* `/#app` opens the app (main.jsx): the paste box and the clip generator.
 * `/#app?login=1` opens it with the sign-in dialog already up (App.jsx). */
export const APP_HREF = '/#app'
export const LOGIN_HREF = '/#app?login=1'

export const NAV_GROUPS = [
  {
    id: 'platforms',
    label: { en: 'Platforms', es: 'Plataformas' },
    items: [
      { href: '/platforms', label: { en: 'All platforms', es: 'Todas las plataformas' } },
      { href: '/platforms/youtube', label: 'YouTube clip maker' },
      { href: '/platforms/youtube-shorts', label: 'YouTube Shorts maker' },
      { href: '/platforms/tiktok', label: 'TikTok clip maker' },
      { href: '/platforms/instagram-reels', label: 'AI reels maker', sub: 'Instagram & Facebook' },
      { href: '/platforms/twitch', label: 'Twitch clip maker' },
      { href: '/platforms/kick', label: 'Kick clip maker' },
    ],
  },
  {
    id: 'use-cases',
    label: { en: 'Use cases', es: 'Casos de uso' },
    items: [
      { href: '/use-cases', label: { en: 'All use cases', es: 'Todos los casos de uso' } },
      { href: '/podcast-to-shorts', label: 'Podcast clip maker' },
      { href: '/use-cases/sermons', label: 'Sermon clip maker' },
      { href: '/gta-5-clips', label: 'GTA 5 clips', sub: { en: 'Gaming', es: 'Videojuegos' } },
      { href: '/gta-6-clips', label: 'GTA 6 clips', sub: { en: 'Console captures', es: 'Capturas de consola' } },
      { href: '/wow-forever-clips', label: 'WoW Forever clips', sub: { en: 'Raids and streams', es: 'Bandas y directos' } },
      { href: '/auto-reframe', label: 'Auto reframe 9:16' },
      { href: '/auto-clip', label: 'Auto clip', sub: 'Autopilot' },
      { href: '/youtube-automation', label: 'YouTube automation' },
      { href: '/ai-ugc-video-generator', label: 'AI UGC video ads', sub: { en: 'AI actors', es: 'Actores IA' } },
      { href: '/how-to', label: { en: 'How-to guides', es: 'Guías' }, sub: { en: 'Clip, self-host, automate', es: 'Recortar, autoalojar, automatizar' } },
    ],
  },
  {
    id: 'compare',
    label: { en: 'Compare', es: 'Comparativas' },
    items: [
      { href: { en: '/alternatives', es: '/alternativas' }, label: { en: 'All alternatives', es: 'Todas las alternativas' } },
      { href: '/alternatives/opus-clip', label: 'Opus Clip alternative' },
      { href: '/alternatives/klap', label: 'Klap alternative' },
      { href: '/alternatives/vizard', label: 'Vizard alternative' },
      { href: '/alternatives/submagic', label: 'Submagic alternative' },
      { href: '/opus-clip-pricing', label: 'Opus Clip pricing' },
      { href: '/free-ai-clipping-tools-compared', label: 'Free AI clipping tools compared' },
      { href: '/free-ai-clip-generator', label: 'Free AI clip generator' },
    ],
  },
  {
    id: 'tools',
    label: { en: 'Free tools', es: 'Herramientas gratis' },
    items: [
      { href: '/tools', label: { en: 'All free tools', es: 'Todas las herramientas' } },
      { href: '/youtube-transcript-generator', label: 'YouTube transcript generator' },
      { href: '/tools/tiktok-transcript-generator', label: 'TikTok transcript generator' },
      { href: '/tools/ai-thumbnail-maker', label: 'AI thumbnail maker' },
      { href: '/youtube-tag-generator', label: 'Tag & title generator' },
      { href: '/video-aspect-ratio-converter', label: 'Video to 9:16 converter' },
      { href: '/mcp', label: 'MCP & API' },
      { href: '/automate-shorts-api', label: 'Automate Shorts API' },
    ],
  },
]

/* The nav's Pricing scrolls to the plan table on the homepage (#pricing in
 * Landing.jsx). The static /pricing page (seo/pricing-page.js) is linked from
 * the footers only. The plan picker and checkout stay at the hash route
 * #/pricing?plan=..., a different thing: nginx sends /pricing?plan=... there. */
export const NAV_LINKS = [
  { href: '/#pricing', label: { en: 'Pricing', es: 'Precios' } },
  { href: '/self-hosted-ai-clip-maker', label: { en: 'Open source', es: 'Código abierto' } },
]

export const NAV_STRINGS = {
  en: { cta: 'Get free clips', ctaHint: 'First video free up to 60 min, no card. No watermark from $12/month, or self-host free (MIT).', openApp: 'Open app', login: 'Log in', menu: 'Menu', main: 'Main', github: 'OpenShorts on GitHub' },
  es: { cta: 'Clips gratis', ctaHint: 'Primer vídeo gratis hasta 60 min, sin tarjeta. Sin marca de agua desde 12 $/mes, o autoalojado gratis (MIT).', openApp: 'Abrir la app', login: 'Iniciar sesión', menu: 'Menú', main: 'Principal', github: 'OpenShorts en GitHub' },
}

const pick = (v, lang) => (v && typeof v === 'object' ? v[lang] || v.en : v)

const esc = (s) =>
  String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')

const CHEVRON = '<svg class="osn-chev" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m6 9 6 6 6-6"/></svg>'
const GITHUB = '<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M12 .5a11.5 11.5 0 0 0-3.64 22.41c.58.1.79-.25.79-.56v-2c-3.2.7-3.88-1.37-3.88-1.37-.52-1.33-1.28-1.69-1.28-1.69-1.05-.71.08-.7.08-.7 1.16.08 1.77 1.19 1.77 1.19 1.03 1.77 2.7 1.26 3.36.96.1-.75.4-1.26.73-1.55-2.55-.29-5.24-1.28-5.24-5.68 0-1.26.45-2.28 1.19-3.09-.12-.29-.52-1.46.11-3.05 0 0 .97-.31 3.17 1.18a11 11 0 0 1 5.77 0c2.2-1.49 3.17-1.18 3.17-1.18.63 1.59.23 2.76.11 3.05.74.81 1.19 1.83 1.19 3.09 0 4.41-2.69 5.38-5.25 5.67.41.36.78 1.06.78 2.14v3.17c0 .31.21.67.8.56A11.5 11.5 0 0 0 12 .5Z"/></svg>'
const BURGER = '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><path d="M4 7h16M4 12h16M4 17h16"/></svg>'

/* The header markup. `signedIn` is only ever true in the React landing, which
 * knows the session; static HTML always renders the signed-out variant. */
export function renderNav({ lang = 'en', signedIn = false, showLogin = true } = {}) {
  const t = NAV_STRINGS[lang] || NAV_STRINGS.en
  const groups = NAV_GROUPS.map((g) => {
    const items = g.items
      .map((it) => {
        const sub = pick(it.sub, lang)
        return `<li><a href="${esc(pick(it.href, lang))}">${esc(pick(it.label, lang))}${sub ? `<small>${esc(sub)}</small>` : ''}</a></li>`
      })
      .join('')
    return `<li class="osn-item"><details class="osn-dd" data-osn-dd><summary>${esc(pick(g.label, lang))}${CHEVRON}</summary><ul class="osn-panel">${items}</ul></details></li>`
  }).join('')
  const links = NAV_LINKS.map((l) => `<li class="osn-item"><a class="osn-top" href="${esc(l.href)}">${esc(pick(l.label, lang))}</a></li>`).join('')
  const login = signedIn || !showLogin ? '' : `<a class="osn-login" href="${LOGIN_HREF}">${esc(t.login)}</a>`
  return `<header class="osn" data-osn>
<div class="osn-bar">
  <a class="osn-brand" href="/"><img src="/logo-openshorts.png" alt="OpenShorts logo" width="28" height="28"><span>openshorts</span></a>
  <input type="checkbox" id="osn-toggle" class="osn-toggle" aria-label="${esc(t.menu)}">
  <label for="osn-toggle" class="osn-burger" title="${esc(t.menu)}">${BURGER}</label>
  <nav class="osn-menu" aria-label="${esc(t.main)}">
    <ul class="osn-list">${groups}${links}<li class="osn-item"><a class="osn-top osn-gh" href="${REPO_URL}" rel="noopener" aria-label="${esc(t.github)}" title="GitHub">${GITHUB}<span class="osn-gh-label">GitHub</span></a></li></ul>
    ${login}
  </nav>
  <a class="osn-cta" href="${APP_HREF}"${signedIn ? '' : ` title="${esc(t.ctaHint)}"`}>${esc(signedIn ? t.openApp : t.cta)}</a>
</div>
</header>`
}

/* Literal colour values (the brand tokens of tailwind.config.js and render.js)
 * so the same block works inside the SPA and on the static pages, which load
 * neither Tailwind nor each other's variables. */
export const NAV_CSS = `
.osn{position:sticky;top:0;z-index:50;background:oklch(13% 0.014 265);border-bottom:1px solid oklch(96% 0.006 262 / .08);
  font-family:"Geist",ui-sans-serif,system-ui,-apple-system,sans-serif;font-size:.875rem;line-height:1.4}
.osn *{box-sizing:border-box}
.osn a{text-decoration:none}
.osn-bar{max-width:72rem;margin:0 auto;padding:0 1.5rem;height:4rem;display:flex;align-items:center;gap:1.5rem;position:relative}
.osn-brand{display:flex;align-items:center;gap:.6rem;font-family:"Instrument Serif",ui-serif,Georgia,serif;font-size:1.25rem;
  letter-spacing:-.01em;color:oklch(96% 0.006 262);text-transform:lowercase;white-space:nowrap;flex-shrink:0}
.osn-brand img{width:28px;height:28px;display:block}
.osn-menu{flex:1;display:flex;align-items:center;justify-content:space-between;gap:1.5rem;min-width:0}
.osn-list{list-style:none;margin:0;padding:0;display:flex;align-items:center;gap:1.4rem}
.osn-item{margin:0;position:relative}
.osn-top,.osn-dd>summary,.osn-login{color:oklch(64% 0.012 262);text-transform:lowercase;white-space:nowrap;cursor:pointer;transition:color .15s}
.osn-top:hover,.osn-dd>summary:hover,.osn-dd[open]>summary,.osn-login:hover{color:oklch(96% 0.006 262)}
.osn-top:focus-visible,.osn-dd>summary:focus-visible,.osn-login:focus-visible,.osn-cta:focus-visible,.osn-brand:focus-visible,
.osn-panel a:focus-visible{outline:2px solid oklch(76% 0.17 50);outline-offset:3px;border-radius:4px}
.osn-dd>summary{list-style:none;display:flex;align-items:center;gap:.3rem;user-select:none}
.osn-dd>summary::-webkit-details-marker{display:none}
.osn-chev{transition:transform .15s;opacity:.8}
.osn-dd[open] .osn-chev{transform:rotate(180deg)}
.osn-gh{display:flex;align-items:center}
.osn-gh-label{display:none}
.osn-panel{list-style:none;margin:0;padding:.4rem;position:absolute;top:calc(100% + 1.1rem);left:-.9rem;min-width:16rem;
  background:oklch(16.5% 0.015 265);border:1px solid oklch(96% 0.006 262 / .14);border-radius:10px;
  box-shadow:0 18px 40px -12px rgb(0 0 0 / .6)}
.osn-panel li{margin:0}
.osn-panel a{display:block;padding:.5rem .7rem;border-radius:7px;color:oklch(86% 0.01 262);text-transform:lowercase;white-space:nowrap}
.osn-panel a:hover{background:oklch(20% 0.016 265);color:oklch(96% 0.006 262)}
.osn-panel small{display:block;font-size:.75rem;color:oklch(64% 0.012 262)}
.osn-cta{background:oklch(76% 0.17 50);color:oklch(17% 0.03 50);padding:.5rem 1.1rem;border-radius:8px;font-weight:500;
  white-space:nowrap;text-transform:lowercase;flex-shrink:0;transition:filter .15s}
.osn-cta:hover{filter:brightness(1.08);color:oklch(17% 0.03 50)}
.osn-toggle{position:absolute;width:1px;height:1px;opacity:0;pointer-events:none;margin:0}
.osn-burger{display:none;color:oklch(86% 0.01 262);cursor:pointer;padding:.4rem;margin-right:-.4rem;border-radius:6px}
.osn-toggle:focus-visible+.osn-burger{outline:2px solid oklch(76% 0.17 50);outline-offset:2px}
@media (min-width:64rem) and (max-width:74.99rem){.osn-bar{gap:1rem}.osn-menu{gap:1rem}.osn-list{gap:1rem}}
@media (max-width:63.99rem){
  .osn-bar{padding:0 1rem;gap:.75rem}
  .osn-brand{margin-right:auto;font-size:1.1rem}
  .osn-cta{order:2;padding:.45rem .85rem}
  .osn-burger{display:flex;order:3}
  .osn-menu{display:none;position:absolute;top:100%;left:0;right:0;flex-direction:column;align-items:stretch;gap:0;
    background:oklch(13% 0.014 265);border-bottom:1px solid oklch(96% 0.006 262 / .14);padding:.5rem 1rem 1rem;
    max-height:calc(100dvh - 4rem);overflow-y:auto}
  .osn-toggle:checked~.osn-menu{display:flex}
  .osn-list{flex-direction:column;align-items:stretch;gap:0}
  .osn-item{border-bottom:1px solid oklch(96% 0.006 262 / .08)}
  .osn-top,.osn-dd>summary{display:flex;justify-content:space-between;padding:.85rem .25rem;font-size:1rem;color:oklch(86% 0.01 262)}
  .osn-gh{justify-content:flex-start;gap:.5rem}
  .osn-gh-label{display:inline}
  .osn-panel{position:static;min-width:0;border:0;box-shadow:none;background:transparent;padding:0 0 .6rem}
  .osn-panel a{white-space:normal;padding:.55rem .5rem}
  .osn-login{display:block;text-align:center;margin-top:1rem;padding:.7rem;border:1px solid oklch(96% 0.006 262 / .14);border-radius:8px;
    color:oklch(86% 0.01 262);font-size:1rem}
}
@media (prefers-reduced-motion:reduce){.osn-chev,.osn-cta,.osn-top{transition:none}}
`

/* Optional behaviour, run by the static pages right after the header. The
 * React landing does the same in SiteNav.jsx (wireNav below). */
export function wireNav(h) {
  if (!h) return () => {}
  const dds = () => Array.prototype.slice.call(h.querySelectorAll('[data-osn-dd]'))
  const onToggle = (e) => {
    if (!e.target.open) return
    dds().forEach((d) => { if (d !== e.target) d.open = false })
  }
  const onClick = (e) => {
    dds().forEach((d) => { if (d.open && !d.contains(e.target)) d.open = false })
  }
  const onKey = (e) => {
    if (e.key !== 'Escape') return
    dds().forEach((d) => {
      if (!d.open) return
      d.open = false
      const s = d.querySelector('summary')
      if (s && d.contains(document.activeElement)) s.focus()
    })
    const tg = h.querySelector('.osn-toggle')
    if (tg && tg.checked) tg.checked = false
  }
  h.addEventListener('toggle', onToggle, true)
  document.addEventListener('click', onClick)
  document.addEventListener('keydown', onKey)
  return () => {
    h.removeEventListener('toggle', onToggle, true)
    document.removeEventListener('click', onClick)
    document.removeEventListener('keydown', onKey)
  }
}

/* The static pages' copy of wireNav, as source text. Written out rather than
 * taken from wireNav.toString(): the build may transform that function, and the
 * inline script has to be plain ES5 that runs anywhere. No CSP is sent. */
export const NAV_SCRIPT = `<script>(function(){var h=document.querySelector('[data-osn]');if(!h)return;
function dds(){return Array.prototype.slice.call(h.querySelectorAll('[data-osn-dd]'))}
h.addEventListener('toggle',function(e){if(!e.target.open)return;dds().forEach(function(d){if(d!==e.target)d.open=false})},true);
document.addEventListener('click',function(e){dds().forEach(function(d){if(d.open&&!d.contains(e.target))d.open=false})});
document.addEventListener('keydown',function(e){if(e.key!=='Escape')return;dds().forEach(function(d){if(!d.open)return;d.open=false;var s=d.querySelector('summary');if(s&&d.contains(document.activeElement))s.focus()});var t=h.querySelector('.osn-toggle');if(t&&t.checked)t.checked=false});
})()</script>`
