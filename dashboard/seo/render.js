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

import { SITE, SAME_AS } from './data.js'

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
header.site{border-bottom:1px solid var(--rule);position:sticky;top:0;background:var(--paper);z-index:10}
header.site .wrap{display:flex;align-items:center;justify-content:space-between;
  gap:1rem;height:4rem}
.brand{display:flex;align-items:center;gap:.6rem;font-family:var(--display);
  font-size:1.15rem;color:var(--ink);text-transform:lowercase;text-decoration:none}
.brand img{width:26px;height:26px}
.nav{display:flex;gap:1.25rem;font-size:.875rem;text-transform:lowercase}
.nav a{text-decoration:none;color:var(--muted)}
.nav a:hover{color:var(--ink)}
.cta{background:var(--brass);color:oklch(17% 0.03 50);padding:.5rem 1rem;
  border-radius:8px;font-size:.875rem;font-weight:500;text-decoration:none;white-space:nowrap}
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
.cta-box .btn:hover{color:oklch(17% 0.03 50);filter:brightness(1.06)}
footer.site{border-top:1px solid var(--rule);padding:2.5rem 0;font-size:.85rem;color:var(--muted)}
footer.site a{color:var(--muted)}
footer.site .row{display:flex;flex-wrap:wrap;gap:1.25rem;margin-bottom:1rem}
`

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
    { '@type': 'ListItem', position: 1, name: 'Home', item: SITE.url },
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
  const main = page.noindex
    ? {
        '@type': 'WebPage',
        '@id': `${SITE.url}${page.path}#webpage`,
        name: page.h1,
        description: page.description,
        url: `${SITE.url}${page.path}`,
        inLanguage: 'en-US',
        publisher: { '@id': `${SITE.url}/#organization` },
      }
    : articleNode(page)

  const graph = [orgNode(), breadcrumbNode(page), main]
  if (page.faq?.length) graph.push(faqNode(page))
  if (page.extraNodes) graph.push(...page.extraNodes)
  return { '@context': 'https://schema.org', '@graph': graph }
}

const NAV = `
<header class="site"><div class="wrap">
  <a class="brand" href="${SITE.url}/"><img src="/logo-openshorts.png" alt="OpenShorts logo" width="26" height="26">OpenShorts</a>
  <nav class="nav">
    <a href="/free-ai-clip-generator">Clip generator</a>
    <a href="/alternatives">Alternatives</a>
    <a href="/mcp">MCP &amp; API</a>
    <a href="${SITE.repo}" rel="noopener">GitHub</a>
  </nav>
  <a class="cta" href="${SITE.url}/">Get free clips</a>
</div></header>`

const footer = (_related) => `
<footer class="site"><div class="wrap">
  <div class="row">
    <a href="${SITE.url}/">OpenShorts</a>
    <a href="${SITE.repo}" rel="noopener">Source on GitHub</a>
    <a href="/free-ai-clip-generator">Free AI clip generator</a>
    <a href="/free-ai-clip-generator-no-watermark">No-watermark clip generator</a>
    <a href="/open-source-video-clipper">Open source video clipper</a>
    <a href="/open-source-ai-video-generator">Open source AI video generator</a>
    <a href="/podcast-to-shorts">Podcast to shorts</a>
    <a href="/youtube-to-shorts-converter">YouTube to Shorts converter</a>
    <a href="/gta-5-clips">GTA 5 clips</a>
    <a href="/how-openshorts-works">How it works</a>
    <a href="/alternatives">Alternatives compared</a>
    <a href="/alternativas">Alternativas (ES)</a>
    <a href="/mcp">MCP server and API</a>
    <a href="/automate-shorts-api">Automate shorts</a>
  </div>
  <div class="row">
    <a href="/opus-clip-pricing">Opus Clip pricing</a>
    <a href="/opus-clip-free-alternative">Free Opus Clip alternative</a>
    <a href="/opus-ai">Opus AI</a>
    <a href="/opus-pro">Opus Pro</a>
    <a href="/vizard-ai-video-to-text">Vizard AI video to text</a>
    <a href="/submagic-reviews">Submagic review</a>
  </div>
  <p>OpenShorts self-hosted is free and open source under MIT. OpenShorts Cloud
  is the hosted service: 20 free minutes a month, paid plans from $12/month.
  Last updated ${esc(SITE.updated)}.</p>
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
const DEFAULT_CTA = {
  label: 'Try it',
  title: 'Paste a link, get vertical clips',
  body: '20 free minutes a month, no credit card. Or self-host it free under MIT.',
  button: 'Get free clips',
}

const ctaBlock = (page) => {
  const c = { ...DEFAULT_CTA, ...(page.cta || {}) }
  const slug = page.path.replace(/^\//, '').replace(/\//g, '-') || 'home'
  const href = `${SITE.url}/`
  return `
<div class="cta-box">
  <div class="copy">
    <span class="label">${esc(c.label)}</span>
    <strong>${esc(c.title)}</strong>
    <p>${esc(c.body)}</p>
  </div>
  <a class="btn" href="${esc(href)}" onclick="window.op&amp;&amp;window.op('track','SeoCtaClick',{page:'${esc(page.path)}'})">${esc(c.button)}</a>
</div>`
}

/* Internal links are rendered as a visible block rather than a nav bar because
 * an engine reading the raw HTML has no way to weight a nav differently from
 * body copy — a described link is a stronger signal than a bare one. */
const relatedBlock = (related) =>
  !related?.length
    ? ''
    : `<h2>Related comparisons</h2><div class="cluster">${related
        .map(
          (r) =>
            `<a href="${esc(r.path)}"><strong>${esc(r.title)}</strong><span>${esc(r.blurb)}</span></a>`
        )
        .join('')}</div>`

export function renderPage(page, related = [], { cta = true } = {}) {
  const canonical = `${SITE.url}${page.path}`
  const showCta = cta && !page.noindex
  const crumbs = [
    `<a href="${SITE.url}/">Home</a>`,
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
<link rel="icon" type="image/png" href="/logo-openshorts.png">
<link rel="stylesheet" href="/fonts.css">
${ANALYTICS}
<meta property="og:type" content="article">
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
${NAV}
<main><div class="wrap">
<div class="crumbs">${crumbs}</div>
<h1>${esc(page.h1)}</h1>
${
  // A byline and a publication date are authorship signals, and an error page
  // is not authored content. Dating the 404 also read as a mistake to anyone
  // who landed on it.
  page.noindex
    ? ''
    : `<div class="byline">
  By the OpenShorts team<span class="sep">·</span>
  Published <time datetime="${esc(page.published || SITE.published)}">${esc(page.published || SITE.published)}</time><span class="sep">·</span>
  Updated <time datetime="${esc(page.updated || SITE.updated)}">${esc(page.updated || SITE.updated)}</time>
</div>`
}
${page.tldr ? `<div class="tldr"><span class="label">TL;DR</span>${page.tldr.map((p) => `<p>${p}</p>`).join('')}</div>` : ''}
${showCta ? ctaBlock(page) : ''}
${page.body}
${relatedBlock(related)}
</div></main>
${footer(related)}
</body>
</html>`
}

export { esc }
