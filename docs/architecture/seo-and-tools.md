# SEO surface, free tools and attribution

## Crawler-visible SPA (`dashboard/vite-plugin-seo.js`)

The dashboard is a client-rendered SPA with hash routing, so the HTML served for
`/` would otherwise be an empty `<div id="root">`. Googlebot renders JavaScript;
GPTBot, ClaudeBot and PerplexityBot do not. `vite-plugin-seo.js` fixes that at
build time:

- Injects the content of `seo/landing-fallback.js` into `#root`. React's
  `createRoot().render()` replaces it on mount, so users get the app and
  non-executing clients get the copy. **Keep it in sync with `Landing.jsx`.**
- Emits the standalone pages (the `/alternatives` cluster, the clip-generator,
  open-source, use-case and automation pages, and `/mcp`; the full list is
  `buildPages()` in `seo/pages.js`) as flat `.html` files. nginx resolves the
  clean URL through `try_files $uri $uri.html`; serving them as directories
  instead makes nginx 301 to a trailing slash and every canonical would then
  point at a redirect.
- Generates `sitemap.xml` and `llms.txt` from the same page list, so they cannot
  drift. Do not add a static `public/sitemap.xml` back.

`seo/data.js` is the single source of truth for pricing, pipeline and competitor
facts used by every generated page. When editing pricing anywhere, edit
`seo/data.js` too. Nothing on the site should say "OpenShorts is free" without
naming the Cloud price in the same breath: both are true of different editions,
and quoting only the first makes AI answers describe the paid product as free.

The public `/gta-5-clips` page states the silent-footage thresholds and ceiling
(see `clip-selection.md`), and `/gta-6-clips` and `/wow-forever-clips` state the
same thresholds and the stretch sizes; if that behaviour changes, change them in
`dashboard/seo/pages.js` too.

The gaming pages (`/gta-6-clips`, `/wow-forever-clips`) are answer-first: each
question is an H2 followed by a 40-60 word answer in bold, then the detail, and
the same question/answer pairs feed the FAQPage node (`qaSections` / `qaFaq` in
`seo/pages.js`). Console, recorder and publisher-policy facts on them carry the
vendor page and the date it was checked; re-check those before editing.

## Site navigation (`seo/nav.js`)

One header for the whole site: `seo/nav.js` holds the groups (platforms,
use cases, compare, free tools), the top links (pricing, open source, GitHub), "Log in"
(`/#app?login=1`, App.jsx opens the sign-in dialog) and the CTA (`/#app`). The
same `renderNav()` markup is used by `render.js` (every static page, labels
translated for `lang: 'es'`), by the homepage fallback and by the React landing
(`src/components/SiteNav.jsx`, which only swaps the CTA to "Open app" for a
stored session). Every item is a real `<a href>` in the static HTML; dropdowns
are `<details>` and the mobile menu a checkbox, so nothing needs JavaScript.
Add or rename a nav item there, never in one surface.

Topic clusters: `seo/hub-pages.js` builds the hubs `/platforms`, `/use-cases`
and `/how-to`. Pages created inside a cluster live under its path
(`/platforms/youtube`, `/use-cases/sermons`, `/how-to/clip-a-youtube-video`);
pages that already ranked on a flat URL (`/podcast-to-shorts`, `/gta-5-clips`,
`/auto-clip`, `/how-openshorts-works`, ...) keep it and only point their
breadcrumb at the hub. Never move an established URL; a moved page gets a 301
in `nginx.conf` for the clean path, `.html` and the trailing slash. Hub cards,
footer entries marked `optional` and llms.txt key pages render only when the
page is built, so they can name a page another module adds.

Pricing has its own static page, `/pricing` (`seo/pricing-page.js`, every number
from `data.js`); every "pricing" link on the site points there, and the homepage
keeps its plan table at `#pricing`. nginx serves `pricing.html` for `/pricing`
(utm_* included) and sends a request carrying `plan` or `src` to the plan picker
at `/#/pricing?<query>`, so the deep links in emails, API/MCP errors and the
page's own plan buttons keep opening checkout. `/pricing/` and `/pricing.html`
301 to `/pricing`.

Comparison pages open on OpenShorts' name and price (`brandLede` in
`seo/pages.js`), every `/alternatives/*` page carries a ranked "Best X
alternatives" list (`ALT_PICKS` in `data.js`, edit it with the competitor
prices), and the pages engines cite most carry the paste-a-link form in place of
the CTA box (`pasteLink: true`, script `tools/paste-link.js`). `llms-full.txt`
is generated next to `llms.txt` with the opening answer and FAQ of each key page.

Routing: `/` always renders the landing; the app is `#app` / `#app?…`. Do not
reintroduce a "returning visitor goes straight to the app" shortcut: the static
fallback is the landing, so swapping it for the app after hydration makes `/`
jump, and every logo on the site must reach the landing.

## Free tools (`/tools`, `free_tools.py`)

- The router is always mounted. `GET /api/tools/youtube-transcript` reads the
  captions a video **already has** on YouTube via yt-dlp (never the GPU, never
  media). Routes: the `STATIC_PROXY_URLS` only (anonymous, then the cookies),
  direct when there are none; **never** the per-GB `PROXY_URL`. A static route
  that answers with zero formats and zero captions is a degraded route, not a
  "no captions" verdict.
- `GET /api/tools/tiktok-transcript` reads the WebVTT captions TikTok's own
  player already serves (yt-dlp with `writesubtitles`, nothing downloaded),
  only through TikTok's extractors (`allowed_extractors`) and inside
  `ssrf_guard.guarded()`, because a vm.tiktok.com share link redirects. Routes:
  direct, then up to two statics; never `PROXY_URL`. Video links are rebuilt
  from the numeric id before they reach yt-dlp.
- `POST /api/tools/youtube-metadata` is one Gemini text call (tags / titles /
  description) with the managed key.
- Per-IP windows + a global daily cap per tool, 24 h cache (also for
  "no captions" / "unavailable").
- Pages: `seo/tools.js` (hub, 3 tools and `/tools/tiktok-transcript-generator`,
  which mounts the transcript widget with `data-platform="tiktok"`; the widget
  picks the endpoint from the pasted link) and `seo/autopilot-pages.js`
  (`/auto-clip`, `/youtube-automation`). A tool page carries `tool: {entry, html}`:
  the form is in the static HTML, the behaviour is a Vite entry in
  `dashboard/tools/*.js` (`vite.config.js` rollupOptions.input) that
  `vite-plugin-seo.js` looks up by name in the bundle. The 9:16 converter runs
  in the browser with mediabunny/WebCodecs (no upload).

## Attribution

`seo/render.js` writes the same first-touch `os_attrib` key the app writes, from
the static page the visit started on, so signups are credited to the landing page
and not to `/`. Signup, CheckoutStarted and Subscribed carry
`landing_path` / `referrer_host` / utm as analytics props (`lib/analytics.js`).

The apex→www 301 is done at the reverse-proxy layer, so the nginx redirect never
sees the apex host.
