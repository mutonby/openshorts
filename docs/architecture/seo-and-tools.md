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
(see `clip-selection.md`); if that behaviour changes, change
`dashboard/seo/pages.js` too.

## Free tools (`/tools`, `free_tools.py`)

- The router is always mounted. `GET /api/tools/youtube-transcript` reads the
  captions a video **already has** on YouTube via yt-dlp (never the GPU, never
  media). Routes: the `STATIC_PROXY_URLS` only (anonymous, then the cookies),
  direct when there are none; **never** the per-GB `PROXY_URL`. A static route
  that answers with zero formats and zero captions is a degraded route, not a
  "no captions" verdict.
- `POST /api/tools/youtube-metadata` is one Gemini text call (tags / titles /
  description) with the managed key.
- Per-IP windows + a global daily cap per tool, 24 h cache (also for
  "no captions" / "unavailable").
- Pages: `seo/tools.js` (hub + 3 tools) and `seo/autopilot-pages.js`
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
