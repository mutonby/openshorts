/* Renders a page definition into a complete, standalone HTML document.
 *
 * These pages are deliberately not React. GPTBot, ClaudeBot and PerplexityBot
 * are plain HTTP clients: they fetch a URL, read the raw HTML and move on. They
 * do not run JavaScript, do not wait for a render and do not come back. Anything
 * that only exists after hydration does not exist to them at all.
 *
 * So the whole page ships as server-ready markup with its CSS inlined, and the
 * structure below follows what actually survives into a generated answer:
 * a TL;DR at the top, one question per H2, self-contained 200-400 token blocks,
 * attributed statistics, a visible byline and a visible date.
 */

import { SITE, SAME_AS, EDITIONS } from './data.js'
import { COMPONENTS_CSS, factTiles, pasteLinkWidget } from './components.js'
import { renderNav, NAV_CSS, NAV_SCRIPT } from './nav.js'

const esc = (s) =>
  String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')

/* Brand tokens, inlined. These pages are served straight from nginx and never
 * go through Vite or Tailwind, so they cannot rely on the app's stylesheet. */
const CSS = `
:root{
  --paper:oklch(13% 0.014 265);--paper2:oklch(16.5% 0.015 265);--paper3:oklch(20% 0.016 265);
  --ink:oklch(96% 0.006 262);--ink2:oklch(86% 0.01 262);--muted:oklch(64% 0.012 262);
  --rule:oklch(96% 0.006 262 / .08);--rule2:oklch(96% 0.006 262 / .14);
  --brass:oklch(76% 0.17 50);--ok:oklch(75% 0.11 150);
  --display:"Instrument Serif",ui-serif,Georgia,serif;
  --body:"Geist",ui-sans-serif,system-ui,-apple-system,sans-serif;
  --mono:"JetBrains Mono",ui-monospace,SFMono-Regular,monospace;
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--paper);color:var(--ink2);font-family:var(--body);
  font-size:16px;line-height:1.65;-webkit-font-smoothing:antialiased}
a{color:var(--ink2);text-decoration:underline;text-underline-offset:3px;
  text-decoration-color:var(--rule2)}
a:hover{color:var(--ink);text-decoration-color:var(--brass)}
.wrap{max-width:56rem;margin:0 auto;padding:0 1.5rem}
main{padding:3rem 0 4rem}
.crumbs{font-family:var(--mono);font-size:.7rem;letter-spacing:.1em;
  text-transform:uppercase;color:var(--muted);margin-bottom:1.5rem}
.crumbs a{text-decoration:none;color:var(--muted)}
.crumbs a:hover{color:var(--brass)}
h1{font-family:var(--display);font-weight:400;font-size:clamp(2.1rem,4.5vw,3.2rem);
  line-height:1.08;letter-spacing:-.03em;color:var(--ink);margin:0 0 1rem}
h2{font-family:var(--display);font-weight:400;font-size:clamp(1.5rem,2.6vw,2rem);
  line-height:1.15;letter-spacing:-.02em;color:var(--ink);margin:3rem 0 .85rem}
h3{font-size:1.05rem;font-weight:600;color:var(--ink);margin:2rem 0 .5rem}
p{margin:0 0 1.1rem}
.byline{font-size:.8rem;color:var(--muted);border-bottom:1px solid var(--rule);
  padding-bottom:1.5rem;margin-bottom:2rem}
.byline .sep{opacity:.4;margin:0 .5rem}
.shot{margin:1.6rem 0}
.shot img{width:100%;height:auto;display:block;border:1px solid var(--rule2);border-radius:8px}
.shot figcaption{font-size:.85rem;color:var(--muted);margin-top:.5rem}
.video,.diagram{margin:1.6rem 0}
.video iframe{width:100%;aspect-ratio:16/9;height:auto;border:0;border-radius:8px;display:block;background:var(--paper2)}
.diagram svg{width:100%;max-width:460px;height:auto;display:block;margin:0 auto}
.video figcaption,.diagram figcaption{font-size:.85rem;color:var(--muted);margin-top:.5rem}
.tldr{background:var(--paper2);border:1px solid var(--rule2);border-left:3px solid var(--brass);
  border-radius:10px;padding:1.4rem 1.5rem;margin:0 0 2.5rem}
.tldr .label{font-family:var(--mono);font-size:.65rem;letter-spacing:.12em;
  text-transform:uppercase;color:var(--brass);display:block;margin-bottom:.6rem}
.tldr p{margin:0 0 .7rem;color:var(--ink2)}
.tldr p:last-child{margin-bottom:0}
ul,ol{margin:0 0 1.1rem;padding-left:1.25rem}
li{margin-bottom:.45rem}
table{width:100%;border-collapse:collapse;margin:1.25rem 0 1.5rem;font-size:.9rem;
  display:block;overflow-x:auto;white-space:nowrap}
@media(min-width:52rem){table{display:table;white-space:normal}}
th,td{text-align:left;padding:.7rem .9rem;border-bottom:1px solid var(--rule);vertical-align:top}
th{font-size:.7rem;font-family:var(--mono);letter-spacing:.08em;text-transform:uppercase;
  color:var(--muted);font-weight:500}
td.os{color:var(--ink)}
.yes{color:var(--ok)}
.note{border:1px solid var(--rule2);border-radius:10px;padding:1.1rem 1.25rem;
  margin:1.5rem 0;background:var(--paper2)}
.note .label{font-family:var(--mono);font-size:.65rem;letter-spacing:.12em;
  text-transform:uppercase;color:var(--muted);display:block;margin-bottom:.5rem}
.note p:last-child{margin-bottom:0}
.checked{font-family:var(--mono);font-size:.7rem;color:var(--muted);
  letter-spacing:.05em;margin-bottom:1.5rem}
.faq dt{font-weight:600;color:var(--ink);margin-top:1.75rem}
.faq dd{margin:.5rem 0 0}
.sources{font-size:.85rem;color:var(--muted)}
.sources li{margin-bottom:.35rem}
.tm-note{font-size:.8rem;color:var(--muted);margin-top:1.2rem}
.cluster{display:grid;gap:.75rem;grid-template-columns:1fr;margin:1.25rem 0}
@media(min-width:40rem){.cluster{grid-template-columns:1fr 1fr}}
.cluster a{display:block;border:1px solid var(--rule);border-radius:10px;padding:.9rem 1.1rem;
  text-decoration:none;background:var(--paper2)}
.cluster a:hover{border-color:var(--rule2)}
.cluster strong{display:block;color:var(--ink);font-weight:500;margin-bottom:.15rem}
.cluster span{font-size:.85rem;color:var(--muted)}
.cta-box{border:1px solid var(--rule2);border-left:3px solid var(--brass);border-radius:10px;
  background:var(--paper2);padding:1.25rem 1.4rem;margin:2rem 0;
  display:flex;flex-wrap:wrap;gap:1rem 1.5rem;align-items:center;justify-content:space-between}
.cta-box .copy{flex:1 1 18rem}
.cta-box .label{font-family:var(--mono);font-size:.65rem;letter-spacing:.12em;
  text-transform:uppercase;color:var(--brass);display:block;margin-bottom:.4rem}
.cta-box strong{display:block;font-family:var(--display);font-weight:400;font-size:1.2rem;
  line-height:1.25;color:var(--ink);margin-bottom:.35rem}
.cta-box p{margin:0;font-size:.9rem;color:var(--muted)}
.cta-box .btn{background:var(--brass);color:oklch(17% 0.03 50);padding:.65rem 1.2rem;
  border-radius:8px;font-size:.9rem;font-weight:500;text-decoration:none;white-space:nowrap}
.cta-act{display:flex;flex-direction:column;align-items:flex-start;gap:.35rem}
.cta-micro{font-size:.72rem;color:var(--muted);max-width:18rem;line-height:1.4}
.best-alts ol{margin:0 0 .4rem}
.plans .btn-primary{display:inline-block;text-decoration:none;white-space:nowrap}
.cta-box .btn:hover{color:oklch(17% 0.03 50);filter:brightness(1.06)}
.lede{font-size:1.05rem;color:var(--ink2);max-width:44rem;margin:0 0 1.5rem}
.sr{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap}
.tool{background:var(--paper2);border:1px solid var(--rule2);border-radius:14px;padding:1.25rem;margin:0 0 2.5rem}
.tool-row{display:flex;flex-wrap:wrap;gap:.6rem}
.tool input[type=url],.tool input[type=text],.tool textarea,.tool select{background:var(--paper);color:var(--ink);
  border:1px solid var(--rule2);border-radius:9px;padding:.75rem .9rem;font:inherit;font-size:1rem;min-width:0}
.tool input[type=url]{flex:1 1 20rem}
.tool textarea{width:100%;resize:vertical;line-height:1.5}
.tool input:focus-visible,.tool textarea:focus-visible,.tool select:focus-visible,.tool button:focus-visible{
  outline:2px solid var(--brass);outline-offset:2px}
.btn-primary{background:var(--brass);color:oklch(17% 0.03 50);border:0;border-radius:9px;padding:.75rem 1.2rem;
  font:inherit;font-weight:600;font-size:.95rem;cursor:pointer;white-space:nowrap;text-decoration:none;display:inline-block}
.btn-primary:disabled{opacity:.55;cursor:wait}
.btn-ghost,.tool-actions button{background:transparent;color:var(--ink2);border:1px solid var(--rule2);border-radius:8px;
  padding:.45rem .8rem;font:inherit;font-size:.85rem;cursor:pointer}
.btn-ghost:hover,.tool-actions button:hover{border-color:var(--brass);color:var(--ink)}
.tool-hint{font-size:.85rem;color:var(--muted);margin:.7rem 0 0}
.tool-status{margin-top:1rem;font-size:.95rem}
.tool-status[data-kind=error]{color:oklch(75% 0.15 25)}
.tool-status[data-kind=busy]{color:var(--muted)}
.tool-status[data-kind=warn]{border:1px solid var(--rule2);border-left:3px solid var(--brass);border-radius:10px;padding:1rem 1.1rem;background:var(--paper)}
.tool-status[data-kind=warn] p{margin:0 0 .7rem}
.tool-meta{margin:1.2rem 0 .6rem}
.tool-meta strong{display:block;color:var(--ink);font-weight:500}
.tool-meta span{font-size:.82rem;color:var(--muted)}
.tool-actions{display:flex;flex-wrap:wrap;gap:.5rem;align-items:center;margin-bottom:.8rem}
.tool-actions label{font-size:.85rem;color:var(--muted);display:flex;gap:.35rem;align-items:center}
.tool-actions select{padding:.45rem .6rem;font-size:.85rem;max-width:14rem}
.transcript{max-height:26rem;overflow:auto;background:var(--paper);border:1px solid var(--rule);border-radius:10px;
  padding:.9rem 1rem;font-size:.92rem;line-height:1.55}
.transcript p{margin:0 0 .35rem}
.transcript .ts{font-family:var(--mono);font-size:.75rem;color:var(--brass);text-decoration:none;margin-right:.35rem}
.cta-inline{display:flex;flex-wrap:wrap;gap:.75rem 1rem;align-items:center;justify-content:space-between;
  margin-top:1rem;padding-top:1rem;border-top:1px solid var(--rule)}
.cta-inline p{margin:0;font-size:.9rem;color:var(--muted);flex:1 1 18rem}
.modes{display:flex;flex-wrap:wrap;gap:.4rem;margin:0 0 .9rem;border:0;padding:0}
.modes label{border:1px solid var(--rule2);border-radius:999px;padding:.4rem .95rem;font-size:.9rem;cursor:pointer;color:var(--ink2)}
.modes input{position:absolute;opacity:0}
.modes label:has(input:checked){background:var(--brass);color:oklch(17% 0.03 50);border-color:var(--brass)}
.modes label:has(input:focus-visible){outline:2px solid var(--brass);outline-offset:2px}
.field{display:block;margin:0 0 .8rem}
.field span{display:block;font-size:.8rem;color:var(--muted);margin-bottom:.3rem}
.field input{width:100%}
.out{margin-top:1.2rem}
.out-head{display:flex;justify-content:space-between;align-items:center;gap:1rem;font-size:.85rem;color:var(--muted);margin-bottom:.6rem}
.chips{display:flex;flex-wrap:wrap;gap:.4rem}
.chip{background:var(--paper);border:1px solid var(--rule2);color:var(--ink2);border-radius:999px;padding:.3rem .75rem;font:inherit;font-size:.85rem;cursor:pointer}
.chip:hover{border-color:var(--brass)}
.title-list{padding:0;list-style:none;margin:0}
.title-opt{width:100%;display:grid;grid-template-columns:1fr auto;gap:.1rem 1rem;text-align:left;background:var(--paper);
  border:1px solid var(--rule);border-radius:9px;padding:.65rem .85rem;margin-bottom:.4rem;font:inherit;color:var(--ink);cursor:pointer}
.title-opt:hover{border-color:var(--brass)}
.title-opt .meta{grid-column:1;font-size:.75rem;color:var(--muted)}
.title-opt .cp{grid-row:1/3;grid-column:2;align-self:center;font-size:.8rem;color:var(--muted)}
.desc-out{min-height:16rem}
.drop{display:block;border:1.5px dashed var(--rule2);border-radius:12px;padding:1.6rem 1rem;text-align:center;cursor:pointer;background:var(--paper)}
.drop[data-over]{border-color:var(--brass)}
.drop strong{display:block;color:var(--ink);font-weight:500}
.drop small{color:var(--muted)}
.opts{display:grid;gap:.9rem;grid-template-columns:1fr;margin:1rem 0}
@media(min-width:40rem){.opts{grid-template-columns:1fr 1fr}}
.opts fieldset{border:0;padding:0;margin:0}
.opts legend{font-size:.8rem;color:var(--muted);margin-bottom:.35rem}
.progress{height:6px;background:var(--paper);border-radius:99px;overflow:hidden;margin-top:1rem}
.progress span{display:block;height:100%;width:0;background:var(--brass);transition:width .2s}
.vc-result{display:grid;gap:1rem;grid-template-columns:1fr;align-items:start;margin-top:1.2rem}
@media(min-width:40rem){.vc-result{grid-template-columns:14rem 1fr}}
.vc-result video{width:100%;max-height:26rem;background:#000;border-radius:10px}
.tools-grid{display:grid;gap:.9rem;grid-template-columns:1fr;margin:1.5rem 0 2.5rem}
@media(min-width:44rem){.tools-grid{grid-template-columns:1fr 1fr 1fr}}
.tools-grid a{display:block;border:1px solid var(--rule2);border-radius:12px;padding:1.1rem 1.2rem;text-decoration:none;background:var(--paper2)}
.tools-grid a:hover{border-color:var(--brass)}
.tools-grid strong{display:block;color:var(--ink);font-weight:500;margin-bottom:.3rem}
.tools-grid span{font-size:.88rem;color:var(--muted)}
[hidden]{display:none!important}
footer.site{border-top:1px solid var(--rule);padding:2.5rem 0;font-size:.85rem;color:var(--muted)}
footer.site a{color:var(--muted)}
footer.site .cols{display:grid;gap:1.5rem 2rem;grid-template-columns:repeat(auto-fill,minmax(11.5rem,1fr));margin-bottom:1.75rem}
footer.site .col-title{font-size:.75rem;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:var(--ink2);margin:0 0 .6rem}
footer.site ul{list-style:none;padding:0;margin:0}
footer.site li{margin:0 0 .4rem}
footer.site .more{margin:0 0 1rem;line-height:1.9}
footer.site .more a{margin-right:.9rem;white-space:nowrap}
${NAV_CSS}
${COMPONENTS_CSS}`

/* Fixed chrome strings (footer headings, breadcrumb, byline, related block)
 * per page language; the header's own strings live in seo/nav.js. Page copy
 * itself lives in the page objects. */
const I18N = {
  en: {
    home: 'Home',
    related: 'Related pages',
    by: 'By the OpenShorts team',
    published: 'Published',
    updated: 'Updated',
    footer: {
      product: 'OpenShorts',
      platforms: 'Platforms',
      useCases: 'Use cases',
      compare: 'Compare',
      tools: 'Free tools',
      guides: 'Guides',
      more: 'More comparisons',
      note: `OpenShorts self-hosted is free and open source under MIT. OpenShorts Cloud
  is the hosted service: your first video free up to 60 minutes, then 20 free
  minutes a month, paid plans from $12/month.`,
      lastUpdated: 'Last updated',
    },
  },
  es: {
    home: 'Inicio',
    related: 'Páginas relacionadas',
    by: 'Por el equipo de OpenShorts',
    published: 'Publicado',
    updated: 'Actualizado',
    footer: {
      product: 'OpenShorts',
      platforms: 'Plataformas',
      useCases: 'Casos de uso',
      compare: 'Comparativas',
      tools: 'Herramientas gratis',
      guides: 'Guías',
      more: 'Más comparativas',
      note: `OpenShorts autoalojado es gratis y de código abierto con licencia MIT.
  OpenShorts Cloud es el servicio alojado: tu primer vídeo gratis hasta 60
  minutos, después 20 minutos gratis al mes, planes de pago desde 12 $/mes.`,
      lastUpdated: 'Última actualización',
    },
  },
}
const strings = (page) => I18N[page.lang] || I18N.en

/* English/Spanish twins, for hreflang. A page can carry its own
 * `alternates: {en, es}`; this map covers the pairs that exist today. Both
 * sides of a pair must name each other, so it is keyed by either path. */
const ALTERNATE_PAIRS = [
  ['/alternatives', '/alternativas'],
  ['/terms', '/terminos'],
  ['/privacy', '/privacidad'],
  ['/legal-notice', '/aviso-legal'],
  ['/refunds', '/reembolsos'],
  ['/report-content', '/reportar-contenido'],
]
const ALTERNATES = Object.fromEntries(
  ALTERNATE_PAIRS.flatMap(([en, es]) => [[en, { en, es }], [es, { en, es }]])
)

const hreflangLinks = (page) => {
  const alt = page.alternates || ALTERNATES[page.path]
  if (page.noindex || !alt?.en || !alt?.es) return ''
  return [
    `<link rel="alternate" hreflang="en" href="${SITE.url}${alt.en}">`,
    `<link rel="alternate" hreflang="es" href="${SITE.url}${alt.es}">`,
    `<link rel="alternate" hreflang="x-default" href="${SITE.url}${alt.en}">`,
  ].join('\n') + '\n'
}

/* Tools and hubs are sites-within-the-site, not authored articles. */
const WEBSITE_PATHS = new Set(['/', '/tools'])
const ogType = (page) =>
  page.ogType || (page.tool || page.noindex || WEBSITE_PATHS.has(page.path) ? 'website' : 'article')

/* Organization is emitted once per page under a stable @id so every other node
 * can point at it instead of restating the brand. That single shared identifier
 * is what lets an engine merge these pages into one entity rather than treating
 * each URL as a separate unknown publisher. */
const orgNode = () => ({
  '@type': 'Organization',
  '@id': `${SITE.url}/#organization`,
  name: SITE.name,
  url: SITE.url,
  logo: { '@type': 'ImageObject', url: SITE.logo },
  sameAs: SAME_AS,
})

const breadcrumbNode = (page) => ({
  '@type': 'BreadcrumbList',
  '@id': `${SITE.url}${page.path}#breadcrumb`,
  itemListElement: [
    { '@type': 'ListItem', position: 1, name: strings(page).home, item: SITE.url },
    ...(page.breadcrumb || []).map((c, i) => ({
      '@type': 'ListItem',
      position: i + 2,
      name: c.name,
      item: c.path ? `${SITE.url}${c.path}` : undefined,
    })),
  ],
})

const faqNode = (page) => ({
  '@type': 'FAQPage',
  '@id': `${SITE.url}${page.path}#faq`,
  mainEntity: page.faq.map((f) => ({
    '@type': 'Question',
    name: f.q,
    acceptedAnswer: { '@type': 'Answer', text: f.a },
  })),
})

const articleNode = (page) => ({
  '@type': 'Article',
  '@id': `${SITE.url}${page.path}#article`,
  headline: page.h1,
  description: page.description,
  datePublished: page.published || SITE.published,
  dateModified: page.updated || SITE.updated,
  inLanguage: page.lang === 'es' ? 'es-ES' : 'en-US',
  mainEntityOfPage: `${SITE.url}${page.path}`,
  author: { '@id': `${SITE.url}/#organization` },
  publisher: { '@id': `${SITE.url}/#organization` },
  image: SITE.ogImage,
})

const buildGraph = (page) => {
  // Error pages are described as a plain WebPage. Emitting an Article for one
  // would assert an author and a publication date for content nobody wrote.
  const main = page.noindex || page.tool
    ? {
        '@type': 'WebPage',
        '@id': `${SITE.url}${page.path}#webpage`,
        name: page.h1,
        description: page.description,
        url: `${SITE.url}${page.path}`,
        inLanguage: page.lang === 'es' ? 'es-ES' : 'en-US',
        publisher: { '@id': `${SITE.url}/#organization` },
        ...(page.tool ? { dateModified: page.updated || SITE.updated } : {}),
      }
    : articleNode(page)

  const graph = [orgNode(), breadcrumbNode(page), main]
  if (page.faq?.length) graph.push(faqNode(page))
  if (page.extraNodes) graph.push(...page.extraNodes)
  return { '@context': 'https://schema.org', '@graph': graph }
}

/* The header is the shared site navigation (seo/nav.js), the same items and
 * targets as the homepage, translated by page language. */
const nav = (page) => `${renderNav({ lang: page.lang === 'es' ? 'es' : 'en' })}\n${NAV_SCRIPT}`

/* Footer link map, one column per topic cluster (the same clusters as the
 * nav: platforms, use cases, compare, free tools, guides). Every page on the
 * site links to the hubs and the money pages from here; the narrow comparison
 * satellites share one compact line. Labels are page names, so they stay in
 * English.
 *
 * An entry marked `optional` names a page another module adds: it is rendered
 * only when that page is in the build (`built`, passed by vite-plugin-seo.js),
 * so the footer never links to a 404. */
const FOOTER_COLUMNS = [
  ['platforms', [
    ['/platforms', 'All platforms'],
    ['/platforms/youtube', 'YouTube clip maker'],
    ['/platforms/youtube-shorts', 'YouTube Shorts maker'],
    ['/platforms/tiktok', 'TikTok clip maker'],
    ['/platforms/instagram-reels', 'AI reel generator'],
    ['/platforms/twitch', 'Twitch clip maker'],
    ['/platforms/kick', 'Kick clip maker'],
    ['/youtube-to-shorts-converter', 'YouTube to Shorts converter'],
  ]],
  ['useCases', [
    ['/use-cases', 'All use cases'],
    ['/podcast-to-shorts', 'Podcast to Shorts'],
    ['/use-cases/sermons', 'Sermon clip maker'],
    ['/gta-5-clips', 'GTA 5 clips'],
    ['/gta-6-clips', 'GTA 6 clips'],
    ['/wow-forever-clips', 'WoW Forever clips'],
    ['/auto-reframe', 'Auto reframe'],
    ['/auto-clip', 'Auto clip'],
    ['/youtube-automation', 'YouTube automation'],
    ['/ai-ugc-video-generator', 'AI UGC video generator', { optional: true }],
  ]],
  ['compare', [
    ['/alternatives', 'Alternatives compared'],
    ['/alternatives/opus-clip', 'Opus Clip alternative'],
    ['/alternatives/klap', 'Klap alternative'],
    ['/alternatives/vizard', 'Vizard alternative'],
    ['/alternatives/submagic', 'Submagic alternative'],
    ['/opus-clip-pricing', 'Opus Clip pricing'],
    ['/free-ai-clipping-tools-compared', 'Free AI clipping tools compared'],
    ['/free-ai-clip-generator', 'Free AI clip generator'],
    ['/alternativas', 'Alternativas (ES)'],
  ]],
  ['tools', [
    ['/tools', 'All free tools'],
    ['/youtube-transcript-generator', 'YouTube transcript generator'],
    ['/tools/tiktok-transcript-generator', 'TikTok transcript generator', { optional: true }],
    ['/youtube-tag-generator', 'YouTube tag and title generator'],
    ['/video-aspect-ratio-converter', 'Video to 9:16 converter'],
    ['/tools/ai-thumbnail-maker', 'AI thumbnail maker', { optional: true }],
  ]],
  ['guides', [
    ['/how-to', 'All guides'],
    ['/how-to/clip-a-youtube-video', 'How to clip a YouTube video'],
    ['/how-to/make-youtube-shorts', 'How to make YouTube Shorts'],
    ['/how-to/youtube-shorts-aspect-ratio', 'YouTube Shorts aspect ratio'],
    ['/how-to/instagram-reel-dimensions', 'Instagram Reel dimensions'],
    ['/how-to/make-a-reel-on-facebook', 'How to make a Reel on Facebook'],
    ['/how-to/best-time-to-post-youtube-shorts', 'Best time to post YouTube Shorts'],
    ['/how-openshorts-works', 'How it works'],
    ['/self-hosted-ai-clip-maker', 'Self-hosted AI clip maker'],
    ['/open-source-video-clipper', 'Open source video clipper'],
    ['/mcp', 'MCP server and API'],
    ['/automate-shorts-api', 'Automate shorts with the API'],
    ['/n8n-youtube-shorts-automation', 'n8n YouTube Shorts automation'],
  ]],
]

const FOOTER_PRODUCT = [
  [`${SITE.url}/`, 'OpenShorts'],
  ['/pricing', 'Pricing'],
  ['/open-source-ai-video-generator', 'Open source AI video generator'],
  [SITE.repo, 'Source on GitHub'],
]

const FOOTER_MORE = [
  ['/opus-ai', 'Opus AI'],
  ['/vizard-ai', 'Vizard AI'],
  ['/vizard-ai-video-to-text', 'Vizard AI video to text'],
  ['/submagic-reviews', 'Submagic review'],
  ['/alternatives/vidyo-ai', 'Vidyo.ai (Quso)'],
  ['/alternatives/2short', '2short AI'],
  ['/alternatives/sendshort', 'SendShort'],
  ['/alternatives/choppity', 'Choppity', { optional: true }],
  ['/alternatives/getmunch', 'Munch', { optional: true }],
  ['/opusclip-vs-vizard', 'OpusClip vs Vizard', { optional: true }],
  ['/alternatives/arcads', 'Arcads alternative', { optional: true }],
]

/* Optional entries only when their page is built; without a `built` set
 * (a caller that does not know the page list) they are left out. */
const present = (built) => ([href, , opts]) => !opts?.optional || Boolean(built?.has(href))

const footerLink = ([href, label]) =>
  `<a href="${esc(href)}"${href.startsWith('https://github.com') ? ' rel="noopener"' : ''}>${esc(label)}</a>`

const footer = (t, built) => `
<footer class="site"><div class="wrap">
  <div class="cols">
${FOOTER_COLUMNS.map(
  ([key, links]) => `    <div>
      <p class="col-title">${esc(t.footer[key])}</p>
      <ul>${links.filter(present(built)).map((l) => `<li>${footerLink(l)}</li>`).join('')}</ul>
    </div>`
).join('\n')}
  </div>
  <p class="more"><span class="col-title">${esc(t.footer.product)}:</span> ${FOOTER_PRODUCT.map(footerLink).join(' ')}</p>
  <p class="more"><span class="col-title">${esc(t.footer.more)}:</span> ${FOOTER_MORE.filter(present(built)).map(footerLink).join(' ')}</p>
  <p>${t.footer.note}
  ${esc(t.footer.lastUpdated)} ${esc(SITE.updated)}.</p>
</div></footer>`

/* Product analytics for the static pages.
 *
 * These pages are served straight from nginx and never hydrate React, so the
 * consent manager that boots the tracker in the app (src/lib/consent.js) never
 * runs here. Without this block their traffic is invisible: they were ranking
 * at positions 1.8 to 4 and reporting nothing at all.
 *
 * The shim, the placeholder gate and ANALYTICS_HOSTS are the same ones
 * index.html uses, on purpose — one deployment, one switch. The placeholders
 * are substituted at build time by vite-plugin-seo.js (Vite only rewrites
 * index.html, and these files are emitted as assets), so an unset
 * VITE_OPENPANEL_* still yields an inert page: no init, no script, no request.
 *
 * The call is unconditional because first-party audience measurement is exempt
 * from prior consent (AEPD/CNIL criteria) and is always on in the app too; the
 * banner only gates marketing, of which there is none on these pages. */
const ANALYTICS = `
<script>
  window.op = window.op || function () { var n = []; return new Proxy(function () { arguments.length && n.push([].slice.call(arguments)); }, { get: function (t, r) { return "q" === r ? n : function () { n.push([r].concat([].slice.call(arguments))); }; }, has: function (t, r) { return "q" === r; } }); }();
  window.__osAnalyticsInit = (function () {
    var started = false;
    return function () {
      if (started) return false;
      var apiUrl = "%VITE_OPENPANEL_API_URL%";
      var clientId = "%VITE_OPENPANEL_CLIENT_ID%";
      var unset = function (v) { return !v || v.charAt(0) === "%"; };
      if (unset(apiUrl) || unset(clientId)) return false;
      var ANALYTICS_HOSTS = /^(www\\.)?openshorts\\.app$/;
      if (!ANALYTICS_HOSTS.test(location.hostname)) return false;
      started = true;
      window.op("init", {
        apiUrl: apiUrl,
        clientId: clientId,
        trackScreenViews: true,
        trackOutgoingLinks: true,
        trackAttributes: true,
      });
      var s = document.createElement("script");
      s.src = "/op1.js";
      s.defer = true;
      s.async = true;
      document.head.appendChild(s);
      return true;
    };
  })();
  window.__osAnalyticsInit();
</script>`

/* First-touch attribution for visits that start on a static page.
 *
 * The app snapshots where a visitor came from (src/lib/attribution.js, key
 * os_attrib) and posts it at signup, but that code only runs once React
 * boots, i.e. on the homepage. A visitor who landed here from Google and then
 * clicked through arrived at the app with a same-origin referrer and
 * landing_path "/", so every signup was credited to the homepage: 7,481 of
 * 7,481 signup_attribution rows in the 30 days to 23-sep-2026. Writing the
 * same key, in the same shape, from the page the visit actually started on
 * fixes it without touching the app (first touch wins on both sides).
 * Independent of the analytics switch: it is first-party localStorage that
 * only ever reaches our own API, exactly like the app's copy. */
const ATTRIBUTION = `
<script>
  (function () {
    try {
      var K = "os_attrib";
      if (localStorage.getItem(K)) return;
      var q = new URLSearchParams(location.search);
      var r = document.referrer || "";
      if (r.indexOf(location.origin) === 0) r = "";
      localStorage.setItem(K, JSON.stringify({
        referrer: r,
        landing_path: location.pathname + location.search,
        utm_source: q.get("utm_source") || "",
        utm_medium: q.get("utm_medium") || "",
        utm_campaign: q.get("utm_campaign") || ""
      }));
    } catch (e) { /* private mode */ }
  })();
</script>`

/* The body CTA. Until now the only link to the app on these pages was the fixed
 * nav button, so a reader who finished the article had nothing to click: the
 * pages sit at positions 1.8-4 with a 0.2-1.6% CTR and no path into the
 * product. Copy is per page (`page.cta`) because the push differs — a reader
 * on the no-watermark page is arguing about watermarks, not about editing.
 *
 * Plain same-origin link on purpose: utm_* here would overwrite the visitor's
 * real acquisition source (google, reddit...) in lib/attribution.js and split
 * the OpenPanel session. The inline handler reports the click with the page
 * slug, which is what tells us whether the CTA or the nav button is doing the
 * work. */
/* The line under every "Get free clips" button: the visitor arrives from a
 * "free" or "no watermark" answer, so the button says what free means here. */
export const CTA_MICROCOPY = `First video free up to 60 min · no card · no watermark from $${EDITIONS.cloud.lowPrice}/month, or self-host free (MIT)`

const DEFAULT_CTA = {
  label: 'Try it',
  title: 'Paste a link, get vertical clips',
  body: 'Your first video free up to 60 minutes, then 20 free minutes a month, no credit card. Or self-host it free under MIT.',
  button: 'Get free clips',
}

const ctaBlock = (page) => {
  const c = { ...DEFAULT_CTA, ...(page.cta || {}) }
  // Pages about Autopilot send readers to the app tab, not the landing page.
  const target = c.href || `${SITE.url}/#app`
  const href = target
  return `
<div class="cta-box">
  <div class="copy">
    <span class="label">${esc(c.label)}</span>
    <strong>${esc(c.title)}</strong>
    <p>${esc(c.body)}</p>
  </div>
  <div class="cta-act">
    <a class="btn" href="${esc(href)}" onclick="window.op&amp;&amp;window.op('track','SeoCtaClick',{page:'${esc(page.path)}'})">${esc(c.button)}</a>
    ${c.button === DEFAULT_CTA.button ? `<small class="cta-micro">${esc(CTA_MICROCOPY)}</small>` : ''}
  </div>
</div>`
}

/* Pages a reader reaches from an AI answer (comparisons, reviews, the free
 * generator) start the first video in place: the paste-a-link form stands
 * where the CTA box was. */
const pasteLinkBlock = (page) =>
  pasteLinkWidget({
    placeholder: page.pasteLink?.placeholder || 'Paste a YouTube link, or any video link',
    label: page.pasteLink?.label || 'Video link',
    hint: CTA_MICROCOPY,
  })

/* Internal links are rendered as a visible block rather than a nav bar because
 * an engine reading the raw HTML has no way to weight a nav differently from
 * body copy — a described link is a stronger signal than a bare one. */
const relatedBlock = (related, title) =>
  !related?.length
    ? ''
    : `<h2>${esc(title)}</h2><div class="cluster">${related
        .map(
          (r) =>
            `<a href="${esc(r.path)}"><strong>${esc(r.title)}</strong><span>${esc(r.blurb)}</span></a>`
        )
        .join('')}</div>`

export function renderPage(page, related = [], { cta = true, toolScript = '', built = null } = {}) {
  const canonical = `${SITE.url}${page.path}`
  const t = strings(page)
  const showCta = cta && !page.noindex && page.cta !== false
  const crumbs = [
    `<a href="${SITE.url}/">${esc(t.home)}</a>`,
    ...(page.breadcrumb || []).map((c) =>
      c.path ? `<a href="${esc(c.path)}">${esc(c.name)}</a>` : esc(c.name)
    ),
  ].join(' <span aria-hidden="true">/</span> ')

  return `<!doctype html>
<html lang="${page.lang || 'en'}">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>${esc(page.title)}</title>
<meta name="description" content="${esc(page.description)}">
${page.noindex ? '' : `<link rel="canonical" href="${canonical}">\n`}<meta name="robots" content="${
    page.noindex ? 'noindex,follow' : 'index,follow,max-image-preview:large,max-snippet:-1'
  }">
${hreflangLinks(page)}<link rel="icon" type="image/png" href="/logo-openshorts.png">
<link rel="stylesheet" href="/fonts.css">
${ANALYTICS}
${page.noindex ? '' : ATTRIBUTION}
<meta property="og:type" content="${ogType(page)}">
<meta property="og:url" content="${canonical}">
<meta property="og:title" content="${esc(page.title)}">
<meta property="og:description" content="${esc(page.description)}">
<meta property="og:image" content="${SITE.ogImage}">
<meta property="og:site_name" content="${SITE.name}">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="${esc(page.title)}">
<meta name="twitter:description" content="${esc(page.description)}">
<meta name="twitter:image" content="${SITE.ogImage}">
<style>${CSS}</style>
<script type="application/ld+json">${JSON.stringify(buildGraph(page))}</script>
</head>
<body>
${nav(page)}
<main><div class="wrap">
<div class="crumbs">${crumbs}</div>
<h1>${esc(page.h1)}</h1>
${page.lede ? `<p class="lede">${page.lede}</p>` : ''}
${page.tool ? page.tool.html : ''}
${
  // A byline and a publication date are authorship signals, and an error page
  // is not authored content. Dating the 404 also read as a mistake to anyone
  // who landed on it.
  page.noindex
    ? ''
    : `<div class="byline">
  ${esc(t.by)}<span class="sep">·</span>
  ${esc(t.published)} <time datetime="${esc(page.published || SITE.published)}">${esc(page.published || SITE.published)}</time><span class="sep">·</span>
  ${esc(t.updated)} <time datetime="${esc(page.updated || SITE.updated)}">${esc(page.updated || SITE.updated)}</time>
</div>`
}
${
  // The short answer stays first in the HTML (it is what an engine quotes),
  // but reads as the opening of the article rather than a labelled box.
  page.tldr ? `<div class="summary">${page.tldr.map((p) => `<p>${p}</p>`).join('')}</div>` : ''
}
${page.top || ''}
${page.facts ? factTiles(page.facts) : ''}
${page.pasteLink && !page.tool ? pasteLinkBlock(page) : showCta ? ctaBlock(page) : ''}
${page.body}
${relatedBlock(related, page.relatedTitle || t.related)}
</div></main>
${footer(t, built)}
${(page.tool || page.pasteLink) && toolScript ? `<script type="module" src="${esc(toolScript)}"></script>` : ''}
</body>
</html>`
}

export { esc }
