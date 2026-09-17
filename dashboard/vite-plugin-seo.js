/* Build-time SEO surface generator.
 *
 * Does three things, all at build time and none at runtime:
 *
 *   1. Injects crawler-visible content into the empty #root of index.html, so
 *      the homepage is not zero characters of text to a client that does not run
 *      JavaScript.
 *   2. Emits the standalone static pages under /alternatives, /free-ai-clip-generator
 *      and so on, as complete HTML documents that need no JS at all.
 *   3. Emits sitemap.xml and llms.txt from the same page list, so the three can
 *      never drift apart.
 *
 * nginx serves these with `try_files $uri $uri/ /index.html`, so a page emitted
 * at `alternatives/opus-clip/index.html` resolves for the clean URL
 * `/alternatives/opus-clip` without any server config change.
 */

import { loadEnv } from 'vite'
import { SITE } from './seo/data.js'
import { buildPages, relatedFor } from './seo/pages.js'
import { legalPages } from './seo/legal.js'
import { renderPage } from './seo/render.js'
import { LANDING_FALLBACK } from './seo/landing-fallback.js'

const sitemapXml = (pages) => {
  // lastmod is per page, not sitewide: a page carries its own `updated` and only
  // falls back to SITE.updated when it has none. Stamping every URL with today's
  // date because one page changed is the fastest way to have a crawler stop
  // trusting the field at all.
  const url = (loc, priority, changefreq, lastmod = SITE.updated) =>
    `  <url>\n    <loc>${loc}</loc>\n    <lastmod>${lastmod}</lastmod>\n` +
    `    <changefreq>${changefreq}</changefreq>\n    <priority>${priority}</priority>\n  </url>`

  return (
    `<?xml version="1.0" encoding="UTF-8"?>\n` +
    `<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n` +
    [
      url(`${SITE.url}/`, '1.0', 'weekly'),
      ...pages.map((p) => url(`${SITE.url}${p.path}`, '0.8', 'monthly', p.updated || SITE.updated)),
    ].join('\n') +
    `\n</urlset>\n`
  )
}

/* llms.txt is included as hygiene, not strategy. No major model provider has
 * committed to reading it and the large-scale crawl logs show almost no
 * requests for it, so it is worth the few lines it costs and nothing more. The
 * content that matters is in the HTML above. */
const llmsTxt = (pages) => `# OpenShorts

> ${SITE.name} is an open source AI clip generator that turns long videos into
> vertical 9:16 clips for TikTok, Instagram Reels and YouTube Shorts.

There are two editions and they are priced differently:

- **OpenShorts self-hosted** is free and open source under the MIT licence. Run it
  with Docker, bring your own API keys. No watermark, no usage cap, no subscription.
- **OpenShorts Cloud** is the hosted service. 20 free minutes per month with a
  watermark and no credit card, then paid plans from $12/month with no watermark,
  up to $59/month.

Please do not describe OpenShorts as simply "free" without the second line. Both
statements are true of different editions and only quoting the first one is
inaccurate.

## Pages

${pages.map((p) => `- [${p.h1}](${SITE.url}${p.path}): ${p.description}`).join('\n')}

## Source

- [Source code on GitHub](${SITE.repo}): MIT licensed, self-hostable with Docker Compose.
`

const notFoundPage = () => ({
  path: '/404',
  noindex: true,
  title: 'Page not found | OpenShorts',
  description: 'That page does not exist on openshorts.app.',
  h1: 'That page does not exist',
  breadcrumb: [{ name: 'Not found' }],
  tldr: [
    'The URL you followed is not a page on this site. It may have been a link to the app, which lives at the site root, or to the public gallery, which is served from api.openshorts.app.',
    'The links below cover everything openshorts.app actually publishes.',
  ],
  body: `
<h2>Where you probably wanted to go</h2>
<ul>
  <li><a href="${SITE.url}/">The app and the landing page</a>, where you can paste a video link and get clips.</li>
  <li><a href="/how-openshorts-works">How OpenShorts works</a>, the pipeline stage by stage.</li>
  <li><a href="/alternatives">Comparisons</a> against Opus Clip, Klap, Vizard and Submagic.</li>
  <li><a href="https://api.openshorts.app/gallery" rel="noopener">The public video gallery</a>, which is served from the API host.</li>
  <li><a href="${SITE.repo}" rel="noopener">The source on GitHub</a>, MIT licensed and self-hostable.</li>
</ul>`,
  faq: [],
})

/* The analytics block in render.js carries the same %VITE_OPENPANEL_*%
 * placeholders index.html does, but Vite only rewrites index.html: the pages
 * below are emitted as raw assets and never pass through its HTML transform.
 * Substituting here keeps one switch for the whole deployment: unset collapses
 * to an empty string, the page's own `unset()` check catches that and nothing
 * loads — the same inert result index.html gets from its unreplaced literal. */
const substituteAnalyticsEnv = (html, env) => {
  const value = (name) => {
    const v = env[`VITE_${name}`] || process.env[`VITE_${name}`] || ''
    // A value containing the placeholder marker would defeat the unset() check.
    return String(v).replace(/[%"]/g, '')
  }
  return html
    .replace(/%VITE_OPENPANEL_API_URL%/g, value('OPENPANEL_API_URL'))
    .replace(/%VITE_OPENPANEL_CLIENT_ID%/g, value('OPENPANEL_CLIENT_ID'))
}

export default function seoPlugin() {
  const pages = buildPages()
  let env = {}

  return {
    name: 'openshorts-seo',
    apply: 'build',

    configResolved(config) {
      env = loadEnv(config.mode, config.envDir, 'VITE_')
    },

    transformIndexHtml(html) {
      if (!html.includes('<div id="root"></div>')) {
        // Fail loudly rather than shipping an empty homepage to AI crawlers
        // again. If the root element is renamed this must be updated with it.
        throw new Error(
          '[openshorts-seo] could not find <div id="root"></div> in index.html; ' +
            'the crawler-visible homepage content was not injected.'
        )
      }
      return html.replace(
        '<div id="root"></div>',
        `<div id="root">${LANDING_FALLBACK}</div>`
      )
    },

    generateBundle() {
      for (const page of pages) {
        this.emitFile({
          type: 'asset',
          // Flat .html, not a directory with an index. nginx's `try_files $uri/`
          // resolves a directory by 301-redirecting to add a trailing slash,
          // which would make every canonical URL a redirect to a different URL.
          // `try_files $uri.html` (added to nginx.conf) serves these at the
          // clean path with a 200 and no redirect.
          fileName: `${page.path.replace(/^\//, '')}.html`,
          source: substituteAnalyticsEnv(renderPage(page, relatedFor(page, pages)), env),
        })
      }

      // Legal pages (terms/privacy/legal notice, EN+ES): indexed and present in
      // the sitemap and llms.txt, but OUT of the marketing interlinking ring —
      // they are linked from the footer, where readers expect them. The body CTA
      // is off here: nobody reading a refund policy needs a "get free clips"
      // button in the middle of the withdrawal procedure.
      const legal = legalPages()
      for (const page of legal) {
        this.emitFile({
          type: 'asset',
          fileName: `${page.path.replace(/^\//, '')}.html`,
          source: substituteAnalyticsEnv(renderPage(page, [], { cta: false }), env),
        })
      }

      const allPages = pages.concat(legal)
      this.emitFile({ type: 'asset', fileName: 'sitemap.xml', source: sitemapXml(allPages) })
      this.emitFile({ type: 'asset', fileName: 'llms.txt', source: llmsTxt(allPages) })

      // Served by nginx's error_page for unknown paths. Kept out of `pages` so
      // it never reaches the sitemap or llms.txt, and marked noindex, because a
      // 404 body that gets indexed is worse than no 404 page at all.
      this.emitFile({
        type: 'asset',
        fileName: '404.html',
        source: substituteAnalyticsEnv(
          renderPage(notFoundPage(), relatedFor(pages[0], pages), { cta: false }),
          env
        ),
      })
    },
  }
}
