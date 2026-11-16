/* /pricing: the plans as a page of their own.
 *
 * Until October 2026 /pricing was a 301 to the homepage table (/#pricing). A
 * redirect to an anchor cannot be cited as a pricing page, and price questions
 * ("is OpenShorts free", "cheaper than Opus Clip") are answered from pages that
 * are one. The homepage table stays; this page states the same plans in plain
 * HTML for readers and engines that never run the app.
 *
 * Every number is read from data.js (PRICE_MODELS.openshorts, EDITIONS), which
 * mirrors cloud/config.py PLAN_MINUTES and the Stripe catalog. The plan buttons
 * are the existing app deep links /pricing?plan=<plan>[&interval=year]: nginx
 * sends a /pricing request that carries `plan` or `src` to the plan picker at
 * /#/pricing?..., and serves this page for everything else (utm_* included).
 */

import { SITE, EDITIONS, PRICE_MODELS, CANONICAL_ANSWERS, COMPETITORS } from './data.js'
import { esc } from './render.js'

const PUBLISHED = '2026-10-30'
const CLOUD = EDITIONS.cloud
const PLANS = PRICE_MODELS.openshorts.plans
const plan = (name) => {
  const p = PLANS.find((x) => x.name === name)
  if (!p) throw new Error(`[pricing-page] no OpenShorts plan "${name}" in data.js PRICE_MODELS`)
  return p
}
const FREE = plan('Free')
const PAID = ['Starter', 'Creator', 'Pro'].map(plan)
if (PAID[0].monthly.price !== CLOUD.lowPrice || PAID[2].monthly.price !== CLOUD.highPrice) {
  throw new Error('[pricing-page] PRICE_MODELS.openshorts and EDITIONS.cloud disagree on the plan prices')
}
// app.FIRST_VIDEO_MAX_MINUTES, stated inside EDITIONS.cloud.summary.
const FIRST_VIDEO_MINUTES = 60
if (!CLOUD.summary.includes(`up to ${FIRST_VIDEO_MINUTES} minutes`)) {
  throw new Error('[pricing-page] EDITIONS.cloud.summary no longer says the first video is free up to 60 minutes')
}
// cloud/config.py TOPUP_LOOKUP_KEYS (topup_60, topup_200), priced in Stripe.
const TOPUPS = [
  { minutes: 60, price: 9 },
  { minutes: 200, price: 25 },
]

const money = (n) => `$${Number.isInteger(n) ? n : n.toFixed(2)}`
const slug = (p) => p.name.toLowerCase()
const APP = `${SITE.url}/#app`

const planRow = (p) => `<tr>
<td class="os"><strong>${esc(p.name)}</strong></td>
<td>${money(p.monthly.price)}/month</td>
<td>${money(p.yearly.price)}/month (${money(Math.round(p.yearly.price * 12))}/year)</td>
<td>${p.monthly.minutes}</td>
<td>No</td>
<td>Yes, plus Autopilot</td>
<td><a class="btn-primary" href="/pricing?plan=${slug(p)}" onclick="window.op&amp;&amp;window.op('track','SeoCtaClick',{page:'/pricing',plan:'${slug(p)}'})">Choose ${esc(p.name)}</a>
<br><a href="/pricing?plan=${slug(p)}&amp;interval=year">or yearly</a></td>
</tr>`

const table = `
<table class="plans">
<thead><tr><th>Plan</th><th>Monthly</th><th>Billed yearly</th><th>Source minutes / month</th><th>Watermark</th><th>Auto-posting</th><th></th></tr></thead>
<tbody>
<tr>
<td class="os"><strong>Self-hosted</strong></td><td>$0</td><td>$0</td><td>No cap</td><td>No</td><td>With your own Upload-Post key</td>
<td><a href="${SITE.repo}" rel="noopener">Source on GitHub (MIT)</a></td>
</tr>
<tr>
<td class="os"><strong>Free</strong></td><td>$0</td><td>–</td><td>${FREE.monthly.minutes} (first video up to ${FIRST_VIDEO_MINUTES})</td><td>Yes</td><td>No, download only</td>
<td><a class="btn-primary" href="${APP}" onclick="window.op&amp;&amp;window.op('track','SeoCtaClick',{page:'/pricing',plan:'free'})">Start free</a></td>
</tr>
${PAID.map(planRow).join('\n')}
</tbody>
</table>
<p>Extra minutes, bought once from the app: ${TOPUPS.map((t) => `${t.minutes} minutes for ${money(t.price)}`).join(' or ')}. They are used after the plan's minutes and do not expire at the end of the month.</p>`

const opus = COMPETITORS['opus-clip']
const vizard = COMPETITORS.vizard

const faq = [
  {
    q: 'Is OpenShorts free?',
    a: CANONICAL_ANSWERS.isItFree,
  },
  {
    q: 'How much does OpenShorts cost?',
    a: CANONICAL_ANSWERS.pricing,
  },
  {
    q: 'What does one minute pay for?',
    a: 'One minute of the source video you clip. A 40-minute video uses 40 minutes whether it yields 5 clips or 12. Trimming a clip, rewriting its hook or re-rendering it uses no minutes, and you can clip only one stretch of a long video and pay just for that stretch.',
  },
  {
    q: 'Is there a free trial?',
    a: `There is a free plan instead, with no credit card: your first video is clipped whole up to ${FIRST_VIDEO_MINUTES} minutes, then ${FREE.monthly.minutes} minutes a month, with a watermark. Upgrading removes the watermark, also from the clips you already made.`,
  },
  {
    q: 'Can OpenShorts post the clips to my socials for me?',
    a: 'Yes, on every paid plan. Connect TikTok, Instagram, YouTube, Facebook, LinkedIn, Threads and Bluesky once and post or schedule each clip to all of them from the app. With Autopilot on, every new video on your YouTube channel is clipped by itself and the best clips are scheduled one a day. The free plan is download only.',
  },
  {
    q: 'I only want to post videos I already have. Do I need OpenShorts?',
    a: CANONICAL_ANSWERS.justPosting,
  },
  {
    q: 'Do unused minutes roll over?',
    a: 'Plan minutes reset at each renewal. Extra minutes bought as a top-up do not: they stay on the account until used, also after a cancellation.',
  },
  {
    q: 'How much cheaper is yearly billing?',
    a: `About 17%: ${PAID.map((p) => `${p.name} is ${money(p.yearly.price)}/month instead of ${money(p.monthly.price)}`).join(', ')}, paid once a year.`,
  },
  {
    q: 'How does it compare with Opus Clip and Vizard?',
    a: `All three bill by the minute of source video. Opus Clip starts at ${opus.entryPrice} for 150 minutes and Vizard Creator is 600 minutes for $29/month ($14.50 billed yearly), checked ${opus.checked}. Vizard is cheaper per minute at volume; OpenShorts is cheaper to start ($${CLOUD.lowPrice}/month for ${PAID[0].monthly.minutes} minutes) and the only one you can also run yourself for free.`,
  },
  {
    q: 'Can I cancel at any time?',
    a: 'Yes, from the app. The plan stays active until the end of the period you paid for, and you can keep the subscription from the same screen before then.',
  },
]

const offer = (p, interval) => ({
  '@type': 'Offer',
  name: `OpenShorts Cloud ${p.name}${interval === 'year' ? ' (billed yearly)' : ''}`,
  price: String(interval === 'year' ? Math.round(p.yearly.price * 12) : p.monthly.price),
  priceCurrency: CLOUD.currency,
  description: `${p.monthly.minutes} minutes of source video a month, no watermark, auto-posting to TikTok, Instagram, YouTube, Facebook, LinkedIn, Threads and Bluesky, and Autopilot.`,
  url: `${SITE.url}/pricing?plan=${slug(p)}${interval === 'year' ? '&interval=year' : ''}`,
  priceSpecification: {
    '@type': 'UnitPriceSpecification',
    price: String(interval === 'year' ? Math.round(p.yearly.price * 12) : p.monthly.price),
    priceCurrency: CLOUD.currency,
    unitText: interval === 'year' ? 'YEAR' : 'MONTH',
  },
})

const appNode = {
  '@type': 'SoftwareApplication',
  '@id': `${SITE.url}/pricing#app`,
  name: 'OpenShorts',
  url: `${SITE.url}/pricing`,
  applicationCategory: 'MultimediaApplication',
  operatingSystem: 'Web browser (OpenShorts Cloud); Docker (self-hosted)',
  isAccessibleForFree: true,
  publisher: { '@id': `${SITE.url}/#organization` },
  offers: [
    {
      '@type': 'Offer',
      name: 'OpenShorts self-hosted',
      price: '0',
      priceCurrency: CLOUD.currency,
      description: 'Free and open source under MIT. No watermark, no cap; your own hardware and API keys.',
    },
    {
      '@type': 'Offer',
      name: 'OpenShorts Cloud free plan',
      price: '0',
      priceCurrency: CLOUD.currency,
      description: `First video clipped whole up to ${FIRST_VIDEO_MINUTES} minutes, then ${FREE.monthly.minutes} minutes a month, with a watermark. No credit card.`,
    },
    ...PAID.flatMap((p) => [offer(p, 'month'), offer(p, 'year')]),
  ],
}

export function pricingPage() {
  return {
    path: '/pricing',
    title: 'OpenShorts Pricing: Free to Self-Host, Cloud From $12/mo',
    description: `OpenShorts pricing: free and open source to self-host, or Cloud with a free plan and paid plans of $12, $29 and $59 a month for 100, 300 and 750 minutes.`,
    h1: `OpenShorts pricing: free to self-host, Cloud from $${CLOUD.lowPrice}/month`,
    breadcrumb: [{ name: 'Pricing' }],
    published: PUBLISHED,
    updated: PUBLISHED,
    relatedTitle: 'Compare before you choose',
    related: ['/alternatives', '/opus-clip-pricing', '/free-ai-clipping-tools-compared', '/self-hosted-ai-clip-maker'],
    cta: false,
    tldr: [
      `OpenShorts has two editions. Self-hosted is free and open source under MIT: run it with Docker, bring your own API keys, no watermark and no usage cap. OpenShorts Cloud is the hosted service: your first video is clipped free up to ${FIRST_VIDEO_MINUTES} minutes, then ${FREE.monthly.minutes} minutes a month with a watermark and no credit card; paid plans remove the watermark.`,
      esc(CANONICAL_ANSWERS.pricing),
    ],
    top: table,
    body: `
<h2>What each plan is for</h2>
<ul>
<li><strong>Self-hosted ($0)</strong>: you have a machine with Docker (an NVIDIA GPU helps) and want no meter at all, or need the video to stay on your own hardware.</li>
<li><strong>Free</strong>: to see the clips on one of your own videos before paying anything.</li>
<li><strong>Starter (${money(PAID[0].monthly.price)}/month, ${PAID[0].monthly.minutes} minutes)</strong>: one or two long videos a month without a watermark.</li>
<li><strong>Creator (${money(PAID[1].monthly.price)}/month, ${PAID[1].monthly.minutes} minutes)</strong>: a weekly podcast or show.</li>
<li><strong>Pro (${money(PAID[2].monthly.price)}/month, ${PAID[2].monthly.minutes} minutes)</strong>: daily uploads, several channels, or clients.</li>
</ul>

<h2>What every Cloud plan includes</h2>
<p>The same pipeline as the open source code, with the GPU and the API keys
covered: AI moment detection, 9:16 reframing that follows the speaker (two
speakers stacked, screen recordings kept readable), word-level captions, hook
overlays, the clip editor, the REST API and the MCP server. API and MCP jobs
use the same minutes as the dashboard, with no per-call fee.</p>

<h2>What paid plans add: auto-posting and Autopilot</h2>
<ul>
<li><strong>Auto-posting.</strong> Connect TikTok, Instagram, YouTube, Facebook, LinkedIn, Threads and Bluesky once and
send every clip straight to all of them, or schedule it, without downloading a
file. Posting runs on Upload-Post and is already set up.</li>
<li><strong>Autopilot.</strong> Connect your YouTube channel and every new video
is clipped on its own; the best clips are scheduled to your socials, one a day,
at the hour you choose. <a href="/auto-clip">How Autopilot works</a>.</li>
</ul>
<p>The free plan is download only. Self-hosted, you can post with your own
Upload-Post account; Autopilot is Cloud only.</p>

<h2>Paying by the source minute</h2>
<p>Most AI clippers bill by the minute of video you import, and so does
OpenShorts Cloud. For comparison, checked ${esc(opus.checked)}: Opus Clip starts
at ${esc(opus.entryPrice)} for 150 minutes, and Vizard at ${esc(vizard.entryPrice)}
for 600 minutes. Per minute, Vizard is the cheaper of the three at volume;
OpenShorts is the cheapest of the three to start and the only one you can run yourself. The
<a href="/alternatives">alternatives page</a> has every tool's plans.</p>

<h2>Common questions</h2><dl class="faq">${faq.map((f) => `<dt>${esc(f.q)}</dt><dd>${esc(f.a)}</dd>`).join('')}</dl>
`,
    faq,
    extraNodes: [appNode],
  }
}
