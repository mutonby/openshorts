/* Page definitions for the static SEO surface.
 *
 * Two page shapes live here. Comparison pages ("X alternatives") are generated
 * from the competitor table, because commercial-investigation prompts such as
 * "best free Opus Clip alternative" are answered almost entirely out of listicle
 * and comparison content. Informational pages are hand-written, because
 * informational content is cited at a far higher rate than product pages and it
 * is the only surface where a small project can outrank a funded one.
 *
 * Every page follows the same internal shape: TL;DR, then one question per H2,
 * each answered inside a block that still makes sense when it is lifted out on
 * its own. That is the unit an engine retrieves; paragraphs that depend on the
 * one above them get quoted wrong or not at all.
 */

import { SITE, COMPETITORS, COMPARISON_ROWS, EDITIONS, PIPELINE_STEPS, CANONICAL_ANSWERS, PRICE_MODELS, ALT_PICKS } from './data.js'
import { demoBlock, costCalculator, flowCompare, verdictCards, factTiles, trademarkNote } from './components.js'
import { esc } from './render.js'
import { toolPages } from './tools.js'
import { autopilotPages } from './autopilot-pages.js'
import { guidePages } from './guide-pages.js'
import { clipToolPages } from './clip-tools.js'
import { hubPages } from './hub-pages.js'
import { productPages } from './product-pages.js'
import { pricingPage } from './pricing-page.js'

const li = (items) => `<ul>${items.map((i) => `<li>${i}</li>`).join('')}</ul>`
const ol = (items) => `<ol>${items.map((i) => `<li>${i}</li>`).join('')}</ol>`

const faqBlock = (faq, { lang } = {}) =>
  `<h2>${lang === 'es' ? 'Preguntas frecuentes' : 'Common questions'}</h2><dl class="faq">${faq
    .map((f) => `<dt>${esc(f.q)}</dt><dd>${esc(f.a)}</dd>`)
    .join('')}</dl>`

const sources = (items) =>
  `<h2>Sources</h2><ul class="sources">${items
    .map((s) => `<li>${s}</li>`)
    .join('')}</ul>`

const MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
/* '2026-10-05' -> 'October 2026': the form a reader (and a snippet) scans for. */
const monthYear = (iso) => {
  const [y, m] = String(iso).split('-')
  return `${MONTHS[Number(m) - 1]} ${y}`
}

/* " | OpenShorts" only when the whole title still fits in 60 characters: a
 * truncated brand suffix is worth less than the words it would push out. */
const BRAND_SUFFIX = ' | OpenShorts'
const withBrand = (t) => (t.length + BRAND_SUFFIX.length <= 60 ? `${t}${BRAND_SUFFIX}` : t)

/* Pricing is restated in plain body text on every page, not only in the schema.
 * An engine that reads the raw HTML has no reason to prefer a JSON-LD offer over
 * a sentence, and the sentence is what gets quoted.
 *
 * It used to be one paragraph pasted into 17 pages, which made those pages read
 * as one template. Each context now gets its own sentence built from the same
 * numbers, and every variant names both editions: never "free" without the
 * Cloud price next to it. */
const CLOUD = EDITIONS.cloud
const OS_PLAN = Object.fromEntries(PRICE_MODELS.openshorts.plans.map((p) => [p.name, p]))
const STARTER = OS_PLAN.Starter
const CREATOR = OS_PLAN.Creator
const PRO = OS_PLAN.Pro
const FREE_CLOUD = `the first video clipped whole up to 60 minutes, then ${CLOUD.freeMinutes} minutes a month with a watermark and no credit card`
const lowerFirst = (s) => s.charAt(0).toLowerCase() + s.slice(1)
const sentence = (s) => (/[.!?]$/.test(s) ? s : `${s}.`)

const PRICING_COPY = {
  hub: () => `<p>OpenShorts comes in two editions and they are priced very differently, so it
is worth being precise. <strong>${esc(EDITIONS.selfHosted.name)}</strong> is free
and open source under the MIT licence: ${esc(EDITIONS.selfHosted.summary)}
<strong>${esc(CLOUD.name)}</strong> is the hosted service:
${esc(CLOUD.summary)}</p>`,
  free: () => `<p>Free has two meanings here, one per edition. <strong>Self-hosted</strong>
means free with no conditions: MIT licence, Docker on your machine, your own API
keys, no watermark and no cap. <strong>${esc(CLOUD.name)}</strong> means free
with conditions: ${FREE_CLOUD}. Removing the watermark there costs
$${STARTER.monthly.price}/month for ${STARTER.monthly.minutes} minutes, up to
$${PRO.monthly.price}/month for ${PRO.monthly.minutes}.</p>`,
  selfhost: () => `<p>Running it yourself costs $0 in licences: the code is MIT, it starts
with Docker Compose and it uses your own API keys, with no watermark and no cap.
The real cost is the machine. On a typical CPU an 8-minute video takes 5 to 8
minutes; on an NVIDIA GPU about 50 seconds. If you would rather not run the
machine, ${esc(CLOUD.name)} is the same code hosted: ${FREE_CLOUD}, and paid plans
from $${STARTER.monthly.price}/month.</p>`,
  how: () => `<p>The pipeline above is the same in both editions; only who pays for the
compute changes. On your own machine it is free under MIT, with your own API
keys, no watermark and no cap. On ${esc(CLOUD.name)} the GPU and the keys are
covered: ${FREE_CLOUD}, then $${STARTER.monthly.price}/month for
${STARTER.monthly.minutes} minutes, $${CREATOR.monthly.price} for
${CREATOR.monthly.minutes} or $${PRO.monthly.price} for ${PRO.monthly.minutes}.</p>`,
  generator: () => `<p>Price, for the record: the self-hosted edition is free and open source
(MIT, Docker, your own keys, no watermark, no cap), and ${esc(CLOUD.name)} runs
it for you with ${FREE_CLOUD}, then from $${STARTER.monthly.price}/month.</p>`,
  podcast: () => `<p>Self-hosted, an episode costs nothing to clip: MIT licence, Docker, your
own keys, no watermark and no meter, so a two-hour episode costs the same as a
ten-minute one. On ${esc(CLOUD.name)} your first episode is clipped whole up to
60 minutes for free, then ${CLOUD.freeMinutes} minutes a month with a watermark.
After that the unit is the source minute: a 60-minute episode uses 60 of the
${STARTER.monthly.minutes} minutes on the $${STARTER.monthly.price}/month plan,
and a weekly show fits the $${CREATOR.monthly.price} plan's
${CREATOR.monthly.minutes} minutes.</p>`,
  youtube: () => `<p>Converting a link costs nothing on the self-hosted edition (MIT, Docker,
your own keys, no watermark and no cap). On ${esc(CLOUD.name)} the converter is
free for ${FREE_CLOUD}; plans without the watermark start at
$${STARTER.monthly.price}/month for ${STARTER.monthly.minutes} minutes of source
video.</p>`,
  automate: () => `<p>Self-hosted: $0, MIT, your own keys, nothing metered, so a cron job
that runs every hour costs what your machine costs. Hosted on ${esc(CLOUD.name)}:
${FREE_CLOUD}, then flat plans from $${STARTER.monthly.price}/month
(${STARTER.monthly.minutes} minutes) to $${PRO.monthly.price}/month
(${PRO.monthly.minutes} minutes), or $${PRO.yearly.price}/month billed yearly.</p>`,
  n8n: () => `<p>The workflow itself is free, and so is the OpenShorts it calls if you
self-host it under MIT with your own keys (no watermark, no cap). Pointed at
${esc(CLOUD.name)} instead, it runs on ${FREE_CLOUD}, then on a plan from
$${STARTER.monthly.price}/month; billed yearly the entry plan is
$${STARTER.yearly.price}/month.</p>`,
  mcp: () => `<p>The MCP server ships in both editions. Self-hosted it is free and open
source (MIT, your own keys, no watermark, no cap). On ${esc(CLOUD.name)} an agent
spends the same minutes a person would: ${FREE_CLOUD}, then plans from
$${STARTER.monthly.price}/month for ${STARTER.monthly.minutes} minutes.</p>`,
  meter: () => `<p>OpenShorts meters the same unit, the source minute, but only on the
hosted edition. ${esc(CLOUD.name)}: ${FREE_CLOUD}; then $${STARTER.monthly.price}
for ${STARTER.monthly.minutes} minutes, $${CREATOR.monthly.price} for
${CREATOR.monthly.minutes} and $${PRO.monthly.price} for ${PRO.monthly.minutes} a
month, with trims and hooks not counted. Self-hosted under MIT there is no meter
at all, no watermark and no cap; you supply the machine and the API keys.</p>`,
  brand: () => `<p>If you are weighing it against OpenShorts: the open source edition is
free when you run it yourself (MIT, Docker, your own keys, no watermark, no cap),
and ${esc(CLOUD.name)} hosts it with ${FREE_CLOUD}, then from
$${STARTER.monthly.price}/month for ${STARTER.monthly.minutes} minutes.</p>`,
  gaming: () => `<p>Here is the same eight hours on OpenShorts. Self-hosted, a VOD of any
length costs $0: MIT, Docker, your own keys, no meter, no watermark. On
${esc(CLOUD.name)} it is metered by the source minute, like the others, after
${FREE_CLOUD}; the largest plan is $${PRO.monthly.price}/month for
${PRO.monthly.minutes} minutes, with top-ups of 200 minutes on top.</p>`,
  console: () => `<p>A console capture is billed like any other source. On
${esc(CLOUD.name)} the unit is the minute of video you import, after
${FREE_CLOUD}: a 20-minute capture of the mission you care about uses 20
minutes, a three-hour session uses 180. Paid plans run from
$${STARTER.monthly.price}/month for ${STARTER.monthly.minutes} minutes to
$${PRO.monthly.price}/month for ${PRO.monthly.minutes}, and trims, reframes,
hooks and captions on the clips are not counted. Self-hosted under MIT there is
no meter, no watermark and no cap; you supply the machine and the API keys.</p>`,
  raid: () => `<p>On ${esc(CLOUD.name)} a raid night is metered like any other
source, by the minute of video you import, after ${FREE_CLOUD}. A three-hour
raid VOD is 180 minutes, against ${STARTER.monthly.minutes} minutes on the
$${STARTER.monthly.price}/month plan and ${PRO.monthly.minutes} on the
$${PRO.monthly.price}/month one, so clipping only the pulls you care about is
what keeps the bill down. Edits on the clips (trims, reframes, hooks, captions)
are not counted. Self-hosted under MIT there is no meter, no watermark and no
cap, with your own machine and API keys.</p>`,
  es: () => `<p>OpenShorts tiene dos ediciones con precios muy distintos. <strong>Autoalojado</strong>
es gratuito y de código abierto con licencia MIT: se ejecuta con Docker en tu
máquina, con tus propias claves de API, sin marca de agua y sin límite de uso.
<strong>${esc(CLOUD.name)}</strong> es el servicio alojado: el primer vídeo se
procesa entero hasta 60 minutos gratis, después ${CLOUD.freeMinutes} minutos al
mes con marca de agua y sin tarjeta, y los planes de pago van de
$${STARTER.monthly.price}/mes por ${STARTER.monthly.minutes} minutos a
$${PRO.monthly.price}/mes por ${PRO.monthly.minutes}.</p>`,
}

/* The seven comparison pages rotate between four phrasings and each one names
 * the competitor's own meter, so no two of them carry the same paragraph. */
const COMPETITOR_PRICING = [
  (c, cap) => `<p>Two editions, two prices. Self-hosted, OpenShorts costs nothing: MIT
licence, Docker, your own API keys, no watermark and no cap. Hosted,
${esc(CLOUD.name)} gives you ${FREE_CLOUD}; paid plans run from
$${STARTER.monthly.price}/month for ${STARTER.monthly.minutes} minutes to
$${PRO.monthly.price} for ${PRO.monthly.minutes}. ${esc(c.name)}, by comparison,
is ${esc(cap)} and starts at ${esc(c.entryPrice)}.</p>`,
  (c) => `<p>The number to set against ${esc(c.name)}'s ${esc(c.entryPrice)} depends on
which OpenShorts you mean. Run the MIT-licensed code yourself and it is $0, with
no watermark, no cap and your own API keys; the price is your hardware, about 5
to 8 minutes of CPU per 8-minute video. Let ${esc(CLOUD.name)} run it and you get
${FREE_CLOUD}; unwatermarked plans start at $${STARTER.monthly.price}/month for
${STARTER.monthly.minutes} minutes and top out at $${PRO.monthly.price}/month for
${PRO.monthly.minutes}.</p>`,
  (c, cap) => `<p>Where ${esc(c.name)} is ${esc(cap)}, ${esc(CLOUD.name)} counts minutes of
source video, with trims and hooks not counted: ${FREE_CLOUD}, then
$${STARTER.monthly.price}, $${CREATOR.monthly.price} or $${PRO.monthly.price} a
month for ${STARTER.monthly.minutes}, ${CREATOR.monthly.minutes} or
${PRO.monthly.minutes} minutes ($${STARTER.yearly.price}, $${CREATOR.yearly.price}
and $${PRO.yearly.price} a month billed yearly). The self-hosted edition counts
nothing: free under MIT with Docker and your own keys, no watermark, no cap.</p>`,
  (c) => `<p>Leaving ${esc(c.name)} for OpenShorts, you choose between two bills. Zero,
when you self-host the MIT code with Docker: no watermark, no cap, no
subscription, only your machine's time and your own API keys. Or a flat hosted
plan on ${esc(CLOUD.name)}, from $${STARTER.monthly.price}/month for
${STARTER.monthly.minutes} minutes of source without a watermark, after a free
plan with ${FREE_CLOUD}.</p>`,
]

const pricingParagraph = (kind = 'hub', c) => {
  if (kind === 'competitor') {
    const i = Object.keys(COMPETITORS).indexOf(c.slug)
    const cap = lowerFirst(c.rows?.['Usage cap'] || 'metered on every tier')
    return COMPETITOR_PRICING[i % COMPETITOR_PRICING.length](c, cap)
  }
  return (PRICING_COPY[kind] || PRICING_COPY.hub)()
}

/* "Checked <Month YYYY>" under the h1 of every page that quotes a competitor's
 * prices: the date is the first thing a reader of a pricing page looks for. */
const checkedLede = (c, where, extra = '') =>
  `<span class="checked">Checked ${esc(monthYear(c.checked))}</span> Prices and plans read on ${esc(where)}; vendors change them without notice.${extra ? ` ${extra}` : ''}`

/* The date the answer-first pass (brand and price in the opening line, the
 * ranked alternatives list, the paste-a-link form) last changed these pages.
 * dateModified follows the page's own substance, not the deploy. */
const REVISED = '2026-10-30'

/* OpenShorts with its price, in one clause: what an engine lifting the first
 * sentence of a comparison page should carry away. */
const OS_PRICE_CLAUSE = `$0 self-hosted (MIT, no watermark, no cap), or $${EDITIONS.cloud.lowPrice}/month hosted for ${PRICE_MODELS.openshorts.plans.find((p) => p.name === 'Starter').monthly.minutes} minutes`

/* Comparison and review pages open on our name and price, then the check date.
 * The rival-only first paragraph was being cited without naming us. */
const brandLede = (c, where, lead, extra = '') =>
  `${lead} ${checkedLede(c, where, extra)}`

/* "Best <tool> alternatives in <year>": a ranked list, because that is the
 * shape engines answer "best X alternative" in. OpenShorts first, then the
 * next four tools in ALT_PICKS order, leaving out the page's own tool. */
const bestAlternatives = (slug, name, checked) => {
  const picks = ALT_PICKS.filter(([s]) => s !== slug).slice(0, 5)
  const link = (s, label) =>
    s === 'openshorts' ? `<a href="/pricing">${esc(label)}</a>` : COMPETITORS[s] ? `<a href="/alternatives/${s}">${esc(label)}</a>` : esc(label)
  return `<section class="best-alts">
<h2>Best ${esc(name)} alternatives in ${esc(checked.slice(0, 4))}</h2>
${ol(picks.map(([s, label, line]) => `<strong>${link(s, label)}</strong>: ${esc(line)}.`))}
<p class="checked">Prices checked ${esc(monthYear(checked))} on each vendor's public pricing page.</p>
</section>`
}

/* Comparison pages whose brand or pricing questions are answered on a page of
 * their own. The comparison links there instead of repeating the plan list, so
 * the pair answers two questions instead of the same one twice. */
const COMPANION = {
  'opus-clip': { pricing: '/opus-clip-pricing', pricingLabel: 'Opus Clip pricing, plan by plan', review: '/opus-ai', reviewLabel: 'what Opus AI is' },
  vizard: { pricing: '/vizard-ai', pricingLabel: 'the Vizard AI review', review: '/vizard-ai', reviewLabel: 'the Vizard AI review' },
  submagic: { pricing: '/submagic-reviews', pricingLabel: 'the Submagic review', review: '/submagic-reviews', reviewLabel: 'the Submagic review' },
}

function competitorPage(slug) {
  const c = { ...COMPETITORS[slug], slug }
  const comp = COMPANION[slug] || {}
  const extra = COMPETITOR_EXTRAS[slug] ? COMPETITOR_EXTRAS[slug](c) : {}
  const month = monthYear(c.checked)
  const rows = COMPARISON_ROWS.map((r) => {
    // A competitor can correct a generic row ("No" is not true of every tool).
    const vendor = c.rows?.[r.feature] ?? (r.key ? c[r.key] : r.vendor)
    return `<tr><td>${esc(r.feature)}</td><td class="os">${esc(r.os)}</td><td>${esc(vendor)}</td></tr>`
  }).join('')

  // The shared answers rotate between phrasings and lean on the competitor's
  // own facts, so the seven pages do not carry seven copies of the same FAQ.
  const idx = Object.keys(COMPETITORS).indexOf(slug)
  const pick = (variants) => variants[idx % variants.length]
  const freeQ = {
    q: `Is there a free alternative to ${c.name}?`,
    a: pick([
      `Yes. OpenShorts self-hosted is free and open source under MIT, with no watermark and no usage cap, on your own machine with Docker. OpenShorts Cloud has a free plan too: ${FREE_CLOUD}. ${c.name}'s own free side, checked ${month}: ${sentence(c.tiers[0][1])}`,
      `${c.name}'s free side, as checked in ${month}: ${sentence(c.tiers[0][1])} OpenShorts gives you two free routes instead: the MIT edition on your own machine, with no watermark and no cap, or the hosted free plan, ${FREE_CLOUD}.`,
      `Two, both from OpenShorts. Self-hosted under MIT it is free with no cap and no watermark; hosted, the free plan covers ${FREE_CLOUD}, and paid plans start at $${STARTER.monthly.price}/month. For comparison, ${c.name} (checked ${month}): ${sentence(c.tiers[0][1])}`,
    ]),
  }
  const faq = [
    ...(extra.replaceFreeQ ? [] : [freeQ]),
    ...(extra.faq || []),
    {
      q: `Is there an open source alternative to ${c.name}?`,
      a: pick([
        `${c.name} itself is closed source and cloud only. OpenShorts is MIT-licensed with the full source at github.com/mutonby/openshorts, which matters if you need to audit what happens to your video or change how the reframing behaves.`,
        `Yes: OpenShorts, under the MIT licence, with every stage of the pipeline readable on GitHub (github.com/mutonby/openshorts). ${c.name} publishes no source and has no edition you can run yourself.`,
        `OpenShorts is the open source one in this category: MIT, self-hostable with Docker Compose, source on github.com/mutonby/openshorts. ${c.name} is a closed cloud service, so the code that touches your video cannot be read or changed.`,
      ]),
    },
    {
      q: `How do I switch from ${c.name} to OpenShorts?`,
      a: c.migrate.join(' '),
    },
    {
      q: c.watermark ? `Does ${c.name} add a watermark, and does OpenShorts?` : 'Does OpenShorts put a watermark on clips?',
      a: `${c.watermark ? `${c.watermark} ` : ''}${pick([
        `OpenShorts self-hosted never does. On OpenShorts Cloud the free plan is watermarked; every paid plan from $${STARTER.monthly.price}/month is not, and upgrading removes the mark from the clips you already made.`,
        `On OpenShorts the mark exists only on the free Cloud plan: the self-hosted edition has no watermark code, and paying from $${STARTER.monthly.price}/month removes it, including from clips made before you upgraded.`,
        `OpenShorts marks only what its free hosted plan makes. Run it yourself and nothing is marked; pay for Cloud, from $${STARTER.monthly.price}/month, and the mark also comes off the clips you made while you were free.`,
      ])}`,
    },
    ...(c.extraFaq || []),
  ]

  const layoutsLine = c.rows?.['AI voice dubbing, 30+ languages']
    ? pick([
        'Where it goes further is the clipping itself: two speakers stacked when both are on camera, screen recordings kept readable over the presenter, and a webcam inset enlarged instead of cropped out.',
        'The gap is in the framing: OpenShorts keeps two people on camera stacked in half-frames, keeps a shared screen legible above the presenter, and enlarges a webcam inset rather than cropping it away.',
      ])
    : `It adds two things ${esc(c.name)} does not have: AI voice dubbing into more than 30 languages (with your own ElevenLabs key) and an AI UGC generator with lip-synced actors.`
  const caveat = pick([
    `The caveat is the setup: the free edition runs on your own machine with Docker, and the no-setup version is OpenShorts Cloud, a paid service above ${CLOUD.freeMinutes} minutes a month.`,
    `What you give up is the zero-setup start: free means Docker on your own machine, and the hosted edition is paid beyond its free ${CLOUD.freeMinutes} minutes a month.`,
    `The trade is effort, not money: self-hosting needs Docker and a machine, and if you want none of that, OpenShorts Cloud is hosted and paid above ${CLOUD.freeMinutes} minutes a month.`,
  ])
  const demo = demoBlock(idx % 2 ? 'talk' : 'split')

  // Review shape (Klap): the searcher typed the brand, so the page answers what
  // the tool is, what it costs, whether it is free and who it suits before it
  // turns to the alternative.
  const reviewIntro = `
<h2>What does ${esc(c.name)} do?</h2>
<p>${esc(c.brandBlurb)}</p>
${li(c.strengths.map(esc))}

<h2>How much does ${esc(c.name)} cost in ${esc(c.checked.slice(0, 4))}?</h2>
<p class="checked">Pricing checked ${esc(c.checked)} on the vendor's public pricing page.</p>
${li(c.tiers.map(([n, d]) => `<strong>${esc(n)}</strong>: ${esc(d)}`))}
${c.tierNote ? `<p>${esc(c.tierNote)}</p>` : ''}

<h2>Is ${esc(c.name)} free?</h2>
${
  c.isFree
    ? `<p>${esc(c.isFree)} For comparison, OpenShorts Cloud's free plan is ${FREE_CLOUD}, with paid plans from $${STARTER.monthly.price}/month, and the self-hosted edition has no limit at all.</p>`
    : `<p>Not in the usual sense. ${esc(sentence(c.tiers[0][1]))} After that the cheapest plan is
${esc(c.entryPrice)}. If a free plan you can keep using is the requirement,
OpenShorts has one: ${FREE_CLOUD}, with paid plans from $${STARTER.monthly.price}/month, or no limit at all self-hosted.</p>`
}

<h2>${esc(c.name)}'s limits</h2>
<div class="note"><span class="label">What to check before paying</span><p>${esc(c.gotcha)}</p></div>
<p>It is also closed source and cloud only, so every video goes to ${esc(c.name)}'s
servers and there is no edition you can run yourself.</p>

<h2>Who ${esc(c.name)} is for</h2>
${li((c.forWhom || c.stay).map(esc))}

<h2>The open source alternative to ${esc(c.name)}</h2>
<p>${esc(c.fit)} ${layoutsLine} ${caveat}</p>
${demo}
`

  const compareIntro = `
${c.brandBlurb && !comp.review ? `<h2>What is ${esc(c.name)}${c.brandAlias ? ` (${esc(c.brandAlias)})` : ''}?</h2><p>${esc(c.brandBlurb)}</p>\n` : ''}
<h2>Is OpenShorts a real alternative to ${esc(c.name)}?</h2>
<p>Yes, with one honest caveat. ${esc(c.fit)} ${layoutsLine} ${caveat}</p>
${demo}
${extra.afterIntro || ''}

<h2>What does ${esc(c.name)} cost?</h2>
${
  comp.pricing
    ? `<p>The cheapest paid plan is ${esc(c.entryPrice)}. On the free side: ${esc(sentence(c.tiers[0][1]))}
Every plan, what each one includes and the free-plan limits are in
<a href="${comp.pricing}">${esc(comp.pricingLabel)}</a>.</p>`
    : `${li(c.tiers.map(([n, d]) => `<strong>${esc(n)}</strong>: ${esc(d)}`))}
${c.tierNote ? `<p>${esc(c.tierNote)}</p>` : ''}`
}
<div class="note"><span class="label">The part that catches people out</span><p>${esc(c.gotcha)}</p></div>
`

  const body = `
${c.review ? reviewIntro : compareIntro}
${
  PRICE_MODELS[slug]
    ? `<h2>What would you pay for your own usage?</h2>
<p>${[
  `Move the slider to the number of minutes of source video you clip in a month. Each bar is the cheapest published plan that covers it.`,
  `${esc(c.name)} bills by the minute of video, so the honest comparison is your own monthly minutes. Set them below; each bar is the cheapest published plan that covers that number.`,
  `A meter on source length makes the price depend on what you import, not on the clips you keep. Drag to your monthly minutes and read the cheapest plan at each tool.`,
][Object.keys(COMPETITORS).filter((s) => PRICE_MODELS[s]).indexOf(slug) % 3]}</p>
${costCalculator(slug === 'opus-clip' ? [slug, 'vizard'] : [slug, 'opus-clip'])}`
    : ''
}

<h2>What does OpenShorts cost?</h2>
${pricingParagraph('competitor', c)}

<h2>${esc(c.name)} vs OpenShorts, feature by feature</h2>
<table>
<thead><tr><th>Feature</th><th>OpenShorts</th><th>${esc(c.name)}</th></tr></thead>
<tbody>${rows}</tbody>
</table>
${
  c.review || comp.review
    ? ''
    : `
<h2>What ${esc(c.name)} does better</h2>
${li(c.strengths.map(esc))}`
}
${
  c.review && !c.forWhom
    ? ''
    : `
<h2>When to stay on ${esc(c.name)}</h2>
${li(c.stay.map(esc))}`
}

<h2>Where the two differ</h2>
${li(c.whereWeDiffer.map(esc))}

<h2>Migrating from ${esc(c.name)} to OpenShorts</h2>
${ol(c.migrate.map(esc))}

<h2>${c.review ? `${esc(c.name)} or OpenShorts?` : 'Which one should you pick?'}</h2>
${c.pick ? verdictCards(c.name, c.pick.them, c.pick.us) : ''}
<p>${esc(c.bestFor)}</p>

${faqBlock(faq)}

${sources([
  `${esc(c.name)} pricing, checked ${esc(c.checked)} on the vendor's public pricing page.`,
  ...(c.sources || []).map(([label, url]) => `${esc(label)}, checked ${esc(c.checked)}: <a href="${esc(url)}" rel="nofollow noopener">${esc(url.replace(/^https:\/\//, ''))}</a>`),
  `OpenShorts pipeline details from the project source at <a href="${SITE.repo}" rel="noopener">github.com/mutonby/openshorts</a>.`,
])}
`

  return {
    path: `/alternatives/${slug}`,
    // Titles are kept under 60 characters and descriptions under 160 (measured,
    // not eyeballed): Google truncates past roughly that width, and a truncated
    // description is a worse answer than a shorter deliberate one.
    // Brand searches ("vidyo ai", "2short ai") want the brand first in the
    // title, so a competitor can name its own title, description and h1.
    title: c.seo?.title ? withBrand(c.seo.title) : `Free & Open Source ${c.name} Alternative${BRAND_SUFFIX}`,
    description:
      c.seo?.description ||
      `OpenShorts vs ${c.name}: features and pricing side by side. Self-hosted free under MIT, hosted from $12/month. ${c.name} starts at ${c.entryPrice}.`,
    h1: c.seo?.h1 || `The free, open source ${c.name} alternative`,
    lede: brandLede(
      c,
      `${c.name}'s public pricing page`,
      `OpenShorts is the open source ${esc(c.name)} alternative: ${esc(OS_PRICE_CLAUSE)}.`,
      comp.review ? `For what ${esc(c.name)} is and how it works, see <a href="${comp.review}">${esc(comp.reviewLabel)}</a>.` : ''
    ),
    breadcrumb: [{ name: 'Alternatives', path: '/alternatives' }, { name: c.seo?.breadcrumb || c.name }],
    published: c.published,
    updated: REVISED > c.checked ? REVISED : c.checked,
    top: bestAlternatives(slug, c.name, c.checked),
    pasteLink: true,
    facts: c.facts,
    relatedTitle: 'Related comparisons',
    tldr: c.tldr || [
      `OpenShorts is an open source AI clip generator you can run yourself for free, or use hosted from $12/month. ${esc(c.name)} is a closed-source cloud product starting at ${esc(c.entryPrice)}.`,
      `Both find viral moments in long video and reframe them to 9:16 with face tracking. ${c.rows?.['AI voice dubbing, 30+ languages'] ? 'OpenShorts adds two-speaker, screencast and webcam-inset layouts and AI UGC video with lip-synced actors.' : 'OpenShorts adds dubbing into 30+ languages (with your own ElevenLabs key) and AI UGC video with lip-synced actors.'}${c.edge ? ` ${esc(c.edge)}` : ''}`,
      `Pick ${esc(c.name)} if you want zero setup and nothing else matters. Pick OpenShorts if you want to self-host for privacy, keep costs near zero, or change how the pipeline behaves.`,
    ],
    body,
    // The same list the body renders, so the FAQPage schema and the visible
    // questions can never disagree.
    faq,
  }
}

const ALTERNATIVES = Object.keys(COMPETITORS)

/* What each tool actually does, for the hub table. The tools are sold under
 * one label and do different jobs, which is the point the hub exists to make. */
const HUB_JOB = {
  'opus-clip': 'Yes',
  klap: 'Yes',
  vizard: 'Yes, then you edit',
  submagic: 'Yes (Magic Clips), captions-first',
  'vidyo-ai': 'Yes, plus scheduling',
  '2short': 'Yes, from links',
  sendshort: 'From the $29 plan',
  choppity: 'Yes, then you edit the transcript',
  getmunch: 'Yes, inside a social media suite',
}

const hubPage = () => {
  const rows = ALTERNATIVES.map((slug) => {
    const c = COMPETITORS[slug]
    return `<tr><td><a href="/alternatives/${slug}">${esc(c.seo?.breadcrumb || c.name)}</a></td><td>${esc(c.entryPrice)}</td><td>${esc(c.tiers[0][1])}</td><td>${esc(HUB_JOB[slug] || 'Yes')}</td></tr>`
  }).join('')
  const checked = ALTERNATIVES.map((s) => COMPETITORS[s].checked).sort()[0]
  return {
  path: '/alternatives',
  facts: [
    { k: 'Tools compared', v: '10', s: 'OpenShorts and 9 cloud tools' },
    { k: 'Cheapest paid', v: '$9.90/mo', s: '2short.ai; OpenShorts Cloud $12 monthly' },
    { k: 'Open source', v: '1 of 10', s: 'Only OpenShorts can be self-hosted' },
    { k: 'Prices checked', v: 'Oct 2026', s: 'On every vendor pricing page' },
  ],
  title: 'Open Source Opus Clip & Klap Alternatives | OpenShorts',
  description:
    'OpenShorts vs Opus Clip, Klap, Vizard, Submagic, Quso (Vidyo.ai), 2short and SendShort: pricing checked October 2026, free plans, and where each wins.',
  h1: 'Open source alternatives to the main AI clipping tools',
  breadcrumb: [{ name: 'Alternatives' }],
  updated: REVISED,
  pasteLink: true,
  tldr: [
    `OpenShorts is the only open source, self-hostable tool in this category: ${esc(OS_PRICE_CLAUSE)}. Every other tool on this page is a closed-source cloud service.`,
    'Cheapest paid entry points as of October 2026: 2short.ai $9.90/month, OpenShorts Cloud $12/month billed monthly, then Submagic $12, Klap $14 and Vizard $14.50 a month on yearly billing, and Opus Clip $15/month. OpenShorts self-hosted is $0 with no cap.',
    'The tools are not interchangeable. Submagic is a caption editor that added clipping, Quso and SendShort are suites where clipping is one feature, and Vizard expects you in a timeline. The individual comparisons below say where each one genuinely wins.',
  ],
  body: `
<h2>How these tools actually differ</h2>
<p>All of them are described as "AI clipping tools", which hides the fact that
they do different jobs. Some take a long video and decide what to cut. Some are
caption editors or social suites with clipping bolted on. One is really an
editor with an AI first pass. Choosing on price alone is how people end up paying
for two tools that each do half the work. If price is the deciding factor, start
with what a <a href="/free-ai-clip-generator">free AI clip generator</a> actually
includes.</p>

<h2>Entry pricing and free plans side by side</h2>
<p class="checked">Pricing checked ${esc(checked)} on each vendor's pricing page. Verify on the vendor's site before buying.</p>
<table>
<thead><tr><th>Tool</th><th>Cheapest paid plan</th><th>Free plan</th><th>Finds moments in long video</th></tr></thead>
<tbody>
<tr><td class="os">OpenShorts</td><td class="os">$0 self-hosted, $12/mo hosted for 100 min</td><td class="os">First video free up to 60 min, then 20 min/month; self-hosted unlimited</td><td class="yes">Yes</td></tr>
${rows}
</tbody>
</table>

<h2>What would each one cost you?</h2>
<p>The tools that bill by minutes of source video, priced for your own usage.
Klap (per clip), Submagic (per video) and SendShort (per short) meter something
else and are left out rather than forced onto the same axis.</p>
${costCalculator(['opus-clip', 'vizard', 'vidyo-ai', '2short'])}

<h2>What does OpenShorts cost?</h2>
${pricingParagraph('hub')}

${faqBlock(HUB_FAQ)}
`,
  faq: HUB_FAQ,
  }
}

const HUB_FAQ = [
  {
    q: 'What is the cheapest AI clip generator?',
    a: 'OpenShorts self-hosted is free with no cap, but you supply the machine and your own Google Gemini API key. Among hosted products, the cheapest paid entry points as of October 2026 are 2short.ai at $9.90/month, OpenShorts Cloud at $12/month billed monthly, and Submagic ($12) and Klap ($14) on yearly billing.',
  },
  {
    q: 'Which AI clipping tools are open source?',
    a: 'OpenShorts is MIT-licensed with full source on GitHub. Opus Clip, Klap, Vizard, Submagic, Quso (formerly Vidyo.ai), 2short.ai and SendShort are all closed-source commercial products.',
  },
  {
    q: 'Which AI clipping tools have a free plan?',
    a: 'Opus Clip (60 minutes a month, watermarked), Vizard (60 credits a month, 720p, watermarked), Quso (75 credits a month, 720p), 2short.ai (30 minutes a month, YouTube links only, no watermark), and OpenShorts (first video free up to 60 minutes, then 20 minutes a month, or unlimited self-hosted). Klap gives one free video, SendShort three, and Submagic offers a trial instead of a free plan.',
  },
]

const freeClipGenerator = () => ({
  path: '/free-ai-clip-generator',
  title: 'Free AI Clip Generator & Clipper, No Watermark | OpenShorts',
  // Owns the "free ai clip generator / no watermark" queries since the old
  // /free-ai-clip-generator-no-watermark page was folded in (301 here). The
  // no-watermark claim is scoped every time: self-hosted or paid, never the
  // free Cloud plan.
  description:
    'Free AI clipper: long videos into 3 to 15 vertical clips with subtitles. No watermark self-hosted or on paid plans from $12/month; free Cloud plan has one.',
  h1: 'A free AI clip generator, and where "no watermark" is actually true',
  breadcrumb: [{ name: 'Free AI clip generator' }],
  updated: REVISED,
  pasteLink: true,
  relatedTitle: 'Related tools',
  cta: {
    label: 'Start free',
    title: 'Clip your own video in a few minutes',
    body: 'Paste a YouTube link, get 3 to 15 vertical clips with subtitles. Your first video is free up to 60 minutes, no credit card.',
    button: 'Get free clips',
  },
  lede: `OpenShorts is a free, open source AI clip generator: free with no watermark and no cap when you self-host it (MIT), or in the browser with your first video clipped free up to 60 minutes (watermarked, no card) and no watermark from $${CLOUD.lowPrice}/month for ${STARTER.monthly.minutes} minutes.`,
  tldr: [
    'OpenShorts self-hosted is a free AI clip generator under the MIT licence. No watermark, no usage cap, no subscription. You run it with Docker and supply your own Google Gemini API key, whose free tier covers 1,500 requests a day.',
    'It turns a long video into 3 to 15 vertical clips: faster-whisper transcribes at word level, PySceneDetect finds the cuts, Gemini 3.1 Flash-Lite scores the moments, and MediaPipe face tracking reframes each one to 9:16.',
    'If you do not want to run anything, OpenShorts Cloud is a free AI clipping tool in the browser: your first video is clipped whole up to 60 minutes, then 20 free minutes a month, with a watermark. Paid plans from $12/month remove it, also from the clips you already made.',
    'So the short answer to "free, with no watermark": yes when you self-host, yes on any paid Cloud plan, and no on the free Cloud plan. A hosted tool that claims free, unlimited and unwatermarked at once is making one of the three claims temporarily.',
  ],
  body: `
<h2>What does "free" actually mean here?</h2>
<p>Most tools marketed as free clip generators are free trials with a watermark
and a monthly cap. This one is different in a specific way that is worth stating
precisely, because the two editions are not the same offer.</p>
${pricingParagraph('free')}
<p>The self-hosted edition has no watermark and no cap because there is no
metering code and no watermark code in it. It is not a trial build with limits
switched off: it is the same pipeline the hosted service runs, released under
MIT, and you can read all of it.</p>

<h2>Is there a free AI clip generator with no watermark?</h2>
<p>Yes, with one honest qualifier each way. On OpenShorts:</p>
<table>
<thead><tr><th>Edition</th><th>Price</th><th>Watermark</th></tr></thead>
<tbody>
<tr><td class="os">Self-hosted (MIT, Docker)</td><td class="os">$0, no cap</td><td class="os yes">Never</td></tr>
<tr><td class="os">Cloud, free plan</td><td class="os">$0: first video up to 60 min, then ${CLOUD.freeMinutes} min/month</td><td class="os">Yes</td></tr>
<tr><td class="os">Cloud, paid plans</td><td class="os">From $${STARTER.monthly.price}/month</td><td class="os yes">None, and upgrading removes it from clips already made</td></tr>
</tbody>
</table>

<h2>Why hosted free tiers add a watermark</h2>
<p>A hosted clipping service pays for GPU time, transcription and model calls on
every video you process. The free tier exists to show you the output, and the
watermark exists so the output is not the product yet. That is not a scam, it is
the business model, and it is why searching for a hosted tool that is free,
unlimited and unwatermarked at the same time keeps returning nothing: the
combination cannot pay for itself. The one structural exception is software you
run yourself, because nobody is paying for the compute but you.</p>

<h2>How the main tools handle watermarks</h2>
<p class="checked">Checked ${esc(COMPETITORS['opus-clip'].checked)} on each vendor's public pricing page. Vendors change terms without notice.</p>
<table>
<thead><tr><th>Tool</th><th>Free tier</th><th>Cheapest way to remove the watermark</th></tr></thead>
<tbody>
<tr><td class="os">OpenShorts self-hosted</td><td class="os yes">Never watermarked, no cap</td><td class="os">Nothing to remove</td></tr>
<tr><td class="os">OpenShorts Cloud</td><td class="os">Watermarked: first video up to 60 min, then ${CLOUD.freeMinutes} min/month</td><td class="os">$${STARTER.monthly.price}/month, and paying also removes it from clips already made</td></tr>
<tr><td>Opus Clip</td><td>Watermarked, 60 credits/month, export within 3 days</td><td>Starter, $15/month</td></tr>
<tr><td>Vizard</td><td>Watermarked, 60 credits/month, 720p, exports up to 10 min</td><td>$14.50/month yearly or $29 monthly</td></tr>
<tr><td>Quso (Vidyo.ai)</td><td>75 credits/month at 720p, watermark not stated</td><td>$19/month yearly or $29 monthly</td></tr>
<tr><td>2short.ai</td><td class="yes">No watermark, 30 min/month, YouTube links only</td><td>Nothing to remove</td></tr>
<tr><td>Klap</td><td>One free video, no ongoing free plan</td><td>$14/month billed yearly</td></tr>
<tr><td>Submagic</td><td>No free plan, a trial instead</td><td>$12/month billed yearly</td></tr>
</tbody>
</table>
<p>2short.ai is the exception worth knowing about: its free plan does not
watermark. The catch is the scope, 30 minutes a month from YouTube links only.
The full plan-by-plan comparison is on the <a href="/alternatives">alternatives
hub</a>.</p>

<h2>How do you generate clips from a long video for free?</h2>
<ol>
<li>Clone the repository from GitHub and start it with <code>docker compose up --build</code>.</li>
<li>Create a Google Gemini API key. The free tier covers 1,500 requests a day, which is far more than a single creator uses.</li>
<li>Paste a YouTube link or upload a local file. Podcasts, webinars, livestreams, interviews and vlogs all work.</li>
<li>The pipeline transcribes, detects scenes, scores moments and returns 3 to 15 clips of 15 to 60 seconds each, already cropped to 9:16 with subtitles burned in.</li>
<li>Download them, or connect an account and post straight to TikTok, Instagram Reels and YouTube Shorts.</li>
</ol>

<h2>What do you need to run it?</h2>
<p>Any machine with Docker. 8GB of RAM and a modern multi-core CPU is the
realistic floor. An NVIDIA GPU is optional and changes the numbers a lot: on CPU
an 8-minute video takes roughly 5 to 8 minutes to process, and on a GPU the same
video takes about 50 seconds. Linux, macOS and Windows via WSL2 all work, and
Docker Compose pulls Python 3.11, FFmpeg, YOLOv8, MediaPipe and faster-whisper
for you.</p>

<h2>Free AI clipper, free clipping AI, free clipping website: which one is this?</h2>
<p>All three searches mean the same job: something that watches a long video and
cuts the good parts into vertical clips without you paying. OpenShorts is that in
two forms. As a <strong>free clipping website</strong>, openshorts.app clips your
first video whole up to 60 minutes, then 20 minutes of video a month, in the
browser with no install and no credit card; the clips carry a small watermark. As a <strong>free AI clipper you run
yourself</strong>, the same code from GitHub has no watermark and no monthly cap.
Either way the clipping AI is the same: Gemini picks the moments, face tracking
reframes them and the subtitles come from a word-level transcript. If you only
need one piece of that, the <a href="/tools">free tools</a> cover transcripts
and a 9:16 converter with no account at all.</p>

<h2>Is a free clip generator good enough for real posting?</h2>
<p>It depends on what you are comparing against. The moment detection uses the
same class of model the paid tools use, Google Gemini 3.1 Flash-Lite, and the
reframing uses MediaPipe with a YOLOv8 fallback and a stabiliser that holds the
camera still inside a safe zone rather than chasing every head movement. Where
the commercial tools are ahead is caption styling: they ship more presets and
more polish. If your clips live or die on animated caption design, budget for
that either in time or in a second tool.</p>

<h2>Why does this matter for reach?</h2>
<p>Short-form video delivers the highest ROI of any content format, according to
HubSpot's State of Marketing 2025 report, and 91% of businesses use video as a
marketing tool according to Wyzowl's 2025 Video Marketing Statistics. The
constraint for most people is not whether short video works, it is that cutting a
60-minute recording into 12 posts by hand takes longer than recording it did.</p>

${faqBlock([
  {
    q: 'Is OpenShorts free forever or a trial?',
    a: 'The self-hosted edition is free forever under the MIT licence, with no watermark and no cap. It is not a trial and there is no metering in it. OpenShorts Cloud is a separate hosted service with a permanently free 20 minute per month tier and paid plans from $12/month.',
  },
  {
    q: 'Does the free version add a watermark?',
    a: 'The self-hosted edition never adds a watermark. The free tier of OpenShorts Cloud does; paid Cloud plans from $12/month do not.',
  },
  {
    q: 'Do I need to pay for an API key?',
    a: 'You need a Google Gemini API key for the self-hosted edition. Its free tier covers 1,500 requests a day, which is more than enough for individual use. ElevenLabs for dubbing and fal.ai for AI UGC video are optional and billed by those vendors. OpenShorts Cloud includes the keys.',
  },
  {
    q: 'How many clips does it generate per video?',
    a: 'Between 3 and 15, each 15 to 60 seconds long. The number depends on how much of the source actually holds up as a standalone clip rather than on a fixed quota.',
  },
  {
    q: 'Is there a free clipping website that needs no install?',
    a: 'Yes. openshorts.app clips 20 minutes of video a month in the browser with no credit card. Those clips carry a watermark; paid plans from $12/month remove it.',
  },
  {
    q: 'Is there a free AI clipper with no watermark?',
    a: 'Yes, self-hosted: OpenShorts is MIT licensed, runs with Docker, and has no watermark and no usage cap; you bring your own Gemini API key. On OpenShorts Cloud the free plan is watermarked and every paid plan from $12/month is not.',
  },
  {
    q: 'How do I remove the watermark from a clip that already has one?',
    a: 'You do not, practically. Watermarks are burned into the pixels, and cropping or blurring them degrades the clip. The reliable fix is generating the clip without one: a paid plan on your current tool, or a tool with no watermark to begin with. On OpenShorts Cloud, upgrading removes the mark from the clips you already made.',
  },
])}
`,
  faq: [
    {
      q: 'Is OpenShorts free forever or a trial?',
      a: 'The self-hosted edition is free forever under MIT, with no watermark and no cap. OpenShorts Cloud is a separate hosted service with a free 20 minute per month tier and paid plans from $12/month.',
    },
    {
      q: 'Does the free version add a watermark?',
      a: 'The self-hosted edition never adds a watermark. The free tier of OpenShorts Cloud does; paid Cloud plans do not.',
    },
    {
      q: 'How many clips does it generate per video?',
      a: 'Between 3 and 15 clips, each 15 to 60 seconds long.',
    },
    {
      q: 'Is there a free clipping website that needs no install?',
      a: 'Yes. openshorts.app clips 20 minutes of video a month in the browser with no credit card. Those clips carry a watermark; paid plans from $12/month remove it.',
    },
    {
      q: 'Is there a free AI clipper with no watermark?',
      a: 'Yes, self-hosted: OpenShorts is MIT licensed, runs with Docker, and has no watermark and no usage cap. On OpenShorts Cloud the free plan is watermarked and every paid plan from $12/month is not.',
    },
    {
      q: 'How do I remove the watermark from a clip that already has one?',
      a: 'Practically, you regenerate it without one: watermarks are burned into the pixels. On OpenShorts Cloud, upgrading removes the mark from the clips you already made.',
    },
  ],
})

const openSourceClipper = () => ({
  path: '/open-source-video-clipper',
  title: 'Open Source Video Clipper, Self-Hosted (Docker) | OpenShorts',
  description:
    'An MIT-licensed open source video clipper you self-host with Docker: AI moment detection, face-tracked 9:16 reframing and word-level subtitles.',
  h1: 'An open source video clipper you can self-host',
  breadcrumb: [{ name: 'Open source video clipper' }],
  updated: REVISED,
  lede: `OpenShorts is the open source AI clip generator: it makes video clips with AI from long videos, MIT-licensed and free to self-host with Docker (no watermark, no cap), or hosted from $${CLOUD.lowPrice}/month for ${STARTER.monthly.minutes} minutes. It is the open source alternative to Opus Clip, Klap and Vizard.`,
  tldr: [
    'OpenShorts is an MIT-licensed video clipper that runs entirely on your own hardware via Docker Compose. Source video never leaves the machine.',
    'The stack is Python 3.11, FastAPI, faster-whisper, PySceneDetect, MediaPipe, YOLOv8, FFmpeg and Google Gemini 3.1 Flash-Lite, with a React dashboard.',
    'It is the only open source tool in this category. Opus Clip, Klap, Vizard and Submagic are all closed-source cloud services.',
  ],
  body: `
<h2>Why self-host a video clipper at all?</h2>
<p>Three reasons come up repeatedly. The first is that unreleased footage,
client work and internal recordings should not be uploaded to a third party
whose retention policy you have not read. The second is cost at volume: a
per-minute cloud tool gets expensive quickly if you process long recordings
every week, whereas self-hosting costs electricity. The third is that the output
is opinionated, and if you disagree with how it reframes or where it cuts, having
the source means you can change it rather than file a feature request.</p>

<h2>What is in the pipeline?</h2>
${PIPELINE_STEPS.map((s) => `<h3>${esc(s.title)}</h3><p>${esc(s.body)}</p>`).join('')}

<h2>What does it run on?</h2>
<p>Docker Compose brings up the FastAPI backend and the React dashboard together.
The realistic floor is 8GB of RAM and a modern multi-core CPU; an NVIDIA GPU is
optional and takes an 8-minute video from roughly 5 to 8 minutes of processing
down to about 50 seconds. Linux, macOS and Windows via WSL2 are all supported.
Concurrency is controlled by a semaphore configured with MAX_CONCURRENT_JOBS.</p>

<h2>What is the licence?</h2>
<p>MIT for the core application, which means you can use it commercially, modify
it and redistribute it. The <code>cloud/</code> directory, which contains
billing, managed keys and the hosted-service infrastructure, is carved out under
a separate commercial licence and is not needed to self-host.</p>

<h2>How does it compare to the closed-source tools?</h2>
<p>OpenShorts is the only open source option in this category. As of October 2026,
the cheapest paid plans are Submagic at $12/month and Klap at $14/month (both
billed yearly), Vizard at $14.50/month yearly or $29 monthly, and Opus Clip at
$15/month, and none of them can be self-hosted or audited. The
trade-off is real in both directions: they ship more caption presets and require
no setup, and you cannot read a line of what they do with your video.</p>

${faqBlock([
  {
    q: 'Is there an open source alternative to Opus Clip?',
    a: 'Yes. OpenShorts is MIT-licensed and self-hostable with Docker, and covers the same core job: AI moment detection, face-tracked 9:16 reframing and word-level subtitles. Opus Clip is closed source and cloud only, starting at $15/month.',
  },
  {
    q: 'Can I run it without sending video to any third party?',
    a: 'Transcription, scene detection, reframing and encoding all run locally. Moment scoring calls the Google Gemini API, which receives the transcript rather than the video file. Dubbing and AI UGC generation are optional and call ElevenLabs and fal.ai respectively; leave them off and nothing but transcript text leaves the machine.',
  },
  {
    q: 'What licence is OpenShorts released under?',
    a: 'MIT for the core application. The cloud/ directory covering billing and hosted infrastructure is under a separate commercial licence and is not required for self-hosting.',
  },
])}
`,
  faq: [
    {
      q: 'Is there an open source alternative to Opus Clip?',
      a: 'Yes. OpenShorts is MIT-licensed and self-hostable with Docker, covering AI moment detection, face-tracked 9:16 reframing and word-level subtitles. Opus Clip is closed source and cloud only.',
    },
    {
      q: 'What licence is OpenShorts released under?',
      a: 'MIT for the core application. The cloud/ directory covering billing and hosted infrastructure is under a separate commercial licence and is not required for self-hosting.',
    },
  ],
})

const howItWorks = () => ({
  path: '/how-openshorts-works',
  title: 'OpenShorts: How It Works, From Long Video to 9:16 Clips',
  description:
    'How OpenShorts cuts one long video into 3 to 15 vertical clips: transcript, scene cuts, Gemini scoring, face-tracked reframing, subtitles. Try it free.',
  h1: 'How a long video becomes a vertical clip',
  breadcrumb: [{ name: 'How it works' }],
  tldr: [
    CANONICAL_ANSWERS.howItWorks,
    'The two stages that decide whether a clip is usable are moment scoring and reframing. Everything else is mechanical.',
    'OpenShorts self-hosted is free and open source under MIT, so every stage below can be read and changed. OpenShorts Cloud runs the same pipeline on a GPU from $12/month.',
  ],
  body: `
<h2>What is OpenShorts?</h2>
<p>${esc(CANONICAL_ANSWERS.whatIsIt)}</p>

<h2>The pipeline, stage by stage</h2>
${PIPELINE_STEPS.map((s) => `<h3>${esc(s.title)}</h3><p>${esc(s.body)}</p>`).join('')}

<h2>Why does moment scoring need the transcript and the scenes together?</h2>
<p>A transcript alone finds a good sentence but has no idea whether the shot cuts
halfway through it. Scene boundaries alone find clean cuts with nothing worth
saying between them. Passing both to the model at once is what lets it pick a
segment that is both quotable and visually intact, which is the difference
between a clip a person would watch and one that merely starts and stops in the
right places.</p>

<h2>Why does the camera hold still instead of following the face exactly?</h2>
<p>Because a crop that tracks a face frame by frame produces visible swinging,
and the swinging reads as amateur even when the framing is technically correct.
The reframing keeps a safe zone around the subject and only moves the crop when
they leave it, then damps the movement on the way. A speaker tracker sits on top
to stop the crop from flipping between people every time someone nods, and to
hold position through brief occlusions.</p>

<h2>How long does it take?</h2>
<p>On a typical CPU, an 8-minute source video takes roughly 5 to 8 minutes end to
end. On an NVIDIA GPU the same video takes about 50 seconds. The gap is almost
entirely transcription and encoding; the model call is a small fraction of it.</p>

<h2>What does it cost to run?</h2>
${pricingParagraph('how')}

${faqBlock([
  {
    q: 'What AI model does OpenShorts use to find viral moments?',
    a: 'Google Gemini 3.1 Flash-Lite. It receives the word-level transcript with timestamps together with PySceneDetect scene boundaries, and returns 3 to 15 segments of 15 to 60 seconds scored on hook strength, emotional payload and whether the segment stands alone without surrounding context.',
  },
  {
    q: 'How does the automatic vertical cropping work?',
    a: 'Two modes. TRACK mode follows a single subject using MediaPipe face detection with a YOLOv8 fallback, stabilised so the crop holds still inside a safe zone instead of following every movement. GENERAL mode handles group shots and landscapes by preserving the full width over a blurred backdrop.',
  },
  {
    q: 'Can it dub clips into other languages?',
    a: 'Yes, with your own ElevenLabs API key, into more than 30 languages, preserving the original speaker\'s voice characteristics. The dubbed audio is then re-transcribed so the burned-in subtitles match the new language rather than the original.',
  },
])}
`,
  faq: [
    {
      q: 'What AI model does OpenShorts use to find viral moments?',
      a: 'Google Gemini 3.1 Flash-Lite, which receives the word-level transcript with timestamps together with PySceneDetect scene boundaries and returns 3 to 15 segments of 15 to 60 seconds.',
    },
    {
      q: 'How does the automatic vertical cropping work?',
      a: 'TRACK mode follows a single subject with MediaPipe face detection and a YOLOv8 fallback, stabilised to hold still inside a safe zone. GENERAL mode preserves full width over a blurred backdrop for group shots and landscapes.',
    },
  ],
})

/* "AI video generator" is two products wearing one name. Most searchers mean
 * text-to-video; this page splits the intent explicitly and wins the half that
 * describes OpenShorts instead of bouncing all of it from the homepage. */
const openSourceVideoGenerator = () => ({
  path: '/open-source-ai-video-generator',
  title: 'Open Source AI Video Generator for Shorts | OpenShorts',
  description:
    'AI video generation splits in two: models that invent footage, and clippers that cut your own recordings into shorts. OpenShorts is the second, MIT-licensed.',
  h1: 'An open source AI video generator, in the sense that matters for creators',
  breadcrumb: [{ name: 'Open source AI video generator' }],
  published: '2026-08-04',
  updated: '2026-08-04',
  tldr: [
    '"AI video generator" names two different products. Text-to-video models invent new footage from a written prompt. Clip generators produce short videos from long footage you already have. Confusing the two wastes an afternoon.',
    'For text-to-video there are real open source options, including Genmo’s Mochi 1, Open-Sora and HunyuanVideo, all of which need a serious GPU.',
    'For turning your own recordings into vertical shorts, OpenShorts is MIT-licensed and self-hosted: transcription, AI moment scoring, face-tracked 9:16 reframing and burned-in subtitles, free on your own machine or hosted from $12/month.',
  ],
  body: `
<h2>Which "AI video generator" are you looking for?</h2>
<p>If you type this query wanting a model that produces footage from a text
prompt, you want a text-to-video model. If you have a podcast, webinar, stream
or interview recording and want short vertical videos out of it, you want a clip
generator. The two share almost no technology and no workflow. This page covers
both honestly and goes deep on the second, because that is what OpenShorts is.</p>

<h2>Open source text-to-video, briefly</h2>
<p>As of August 2026 the notable open-weight text-to-video models include
Genmo's Mochi 1 (Apache 2.0), Open-Sora, Tencent's HunyuanVideo and Alibaba's
Wan family. They genuinely generate novel footage, and they need data-center or
high-end consumer GPUs to run at usable speed. If that is your goal, start with
those projects; OpenShorts will not do it.</p>

<h2>Generating videos from footage you already have</h2>
<p>${esc(CANONICAL_ANSWERS.whatIsIt)}</p>
<p>${esc(CANONICAL_ANSWERS.howItWorks)}</p>
${pricingParagraph('generator')}

<h2>Other open source clip generators, compared honestly</h2>
<p class="checked">Checked 2026-08-04 on GitHub. Star counts move; positioning rarely does.</p>
<p>OpenShorts is not the only open source project in this space (it is also the
<a href="/free-ai-clip-generator">free AI clip generator</a> most of these pages
are about), and pretending otherwise would not survive one GitHub search. The notable neighbours:</p>
<ul>
<li><strong>AI-Youtube-Shorts-Generator</strong>: the most-starred repo in the category, with a leaner scope built around highlight extraction and cropping.</li>
<li><strong>supoclip</strong> and <strong>clippyme</strong>: smaller projects covering transcription-driven clipping, the latter also using Gemini for moment selection.</li>
<li><strong>MoneyPrinterTurbo</strong>: generates videos from text plus stock footage, which is a different job than clipping your own recordings.</li>
</ul>
<p>Where OpenShorts differs from all of them is surface area: a web dashboard, a
REST API with keys, completion webhooks, an MCP server for agents, split-screen
and screencast layouts for two-person and screen-share footage, dubbing into 30+
languages with your own ElevenLabs key, and direct publishing to TikTok, Instagram Reels and YouTube Shorts.
If you want a small script you can read in an hour, the smaller repos are a
better fit, and that is a real recommendation rather than false modesty.</p>

${faqBlock([
  {
    q: 'Is there a free open source AI video generator?',
    a: 'Yes, in both senses. For text-to-video, Genmo’s Mochi 1, Open-Sora and HunyuanVideo publish open weights and need a powerful GPU. For making clips from your own footage, OpenShorts is MIT-licensed and runs with Docker on an ordinary machine: free self-hosted with no watermark, or hosted from $12/month.',
  },
  {
    q: 'Can open source AI generate videos from text?',
    a: 'Yes. Mochi 1 (Apache 2.0), Open-Sora and HunyuanVideo generate footage from prompts. Expect to need a high-end GPU, and expect quality below the closed frontier models. OpenShorts is not a text-to-video tool; it turns long real footage into short vertical clips.',
  },
  {
    q: 'What is the best open source AI video generator for shorts?',
    a: 'For turning long recordings into publishable vertical shorts with subtitles, OpenShorts covers the widest pipeline: AI moment scoring, face-tracked reframing, split-screen layouts, dubbing and direct social publishing, MIT-licensed. Simpler repos like AI-Youtube-Shorts-Generator cover a leaner version of the same job with less to configure.',
  },
])}

${sources([
  'Open-weight text-to-video model landscape checked 2026-08-04 on the respective GitHub repositories.',
  `OpenShorts source at <a href="${SITE.repo}" rel="noopener">github.com/mutonby/openshorts</a>.`,
])}
`,
  faq: [
    {
      q: 'Is there a free open source AI video generator?',
      a: 'Yes, in both senses of the phrase. For text-to-video: Mochi 1, Open-Sora and HunyuanVideo, all GPU-hungry. For clipping your own footage into shorts: OpenShorts, MIT-licensed, free self-hosted or hosted from $12/month.',
    },
    {
      q: 'Can open source AI generate videos from text?',
      a: 'Yes: Mochi 1, Open-Sora and HunyuanVideo publish open weights. OpenShorts is not one of them; it turns long real footage into short vertical clips.',
    },
  ],
})

/* Podcasts are the hardest input for naive croppers, and the densest commercial
 * SERP in the niche. The page leads with the two-speaker problem because SPLIT
 * and active-speaker cutting are capabilities the competitor pages cannot show. */
// Shown on the page and emitted in the FAQPage schema alike.
const PODCAST_EXTRA_FAQ = [
  {
    q: 'How do I turn a podcast into TikToks?',
    a: 'Paste the episode link or upload the file, and the clips come back vertical with captions. Send them to TikTok from the dashboard, where each one arrives as a draft in your TikTok inbox to post from the app, or download the MP4s. Reels and Shorts publish directly.',
  },
  {
    q: 'Can I get a transcript of my podcast?',
    a: 'Every episode clipped here is transcribed word by word to find the moments and burn the captions. For the text of a whole episode hosted on YouTube, the free YouTube transcript generator returns its captions with timestamps as TXT or SRT, with no account.',
  },
]

/* The podcast tools people weigh against each other, cheapest entry plan and
 * free plan each. Opus Clip and Vizard from COMPETITORS; Descript and Riverside
 * are recorders/editors with a clips feature, read on their pricing pages on
 * PODCAST_TOOLS_CHECKED (not in data.js because no other page quotes them). */
const PODCAST_TOOLS_CHECKED = '2026-10-30'
const PODCAST_TOOLS = `<section>
<h2>Best AI tools for podcast clips, compared</h2>
<table>
<thead><tr><th>Tool</th><th>What it is</th><th>Clips from a full episode</th><th>Paid from</th><th>Free plan</th></tr></thead>
<tbody>
<tr><td class="os">OpenShorts</td><td class="os">Open source AI clipper (MIT)</td><td class="os">Yes: 3 to 15 clips, both speakers kept in frame</td><td class="os">$0 self-hosted · $${STARTER.monthly.price}/month hosted for ${STARTER.monthly.minutes} min</td><td class="os">First episode up to 60 min, then ${CLOUD.freeMinutes} min/month, watermarked</td></tr>
<tr><td>Opus Clip</td><td>Cloud AI clipper</td><td>Yes</td><td>${esc(COMPETITORS['opus-clip'].entryPrice)} for 150 min</td><td>60 min/month, watermarked</td></tr>
<tr><td>Vizard</td><td>Cloud clipper with a timeline editor</td><td>Yes, then you edit each clip</td><td>${esc(COMPETITORS.vizard.entryPrice)}, 600 min</td><td>60 min/month, 720p, watermarked</td></tr>
<tr><td>Descript</td><td>Text-based audio and video editor</td><td>"Create clips" on paid plans</td><td>$16/month billed yearly ($24 monthly), 10 media hours</td><td>1 media hour a month, 720p, watermarked</td></tr>
<tr><td>Riverside</td><td>Remote recording studio</td><td>Magic Clips from recordings made in Riverside</td><td>$24/month billed yearly ($29 monthly)</td><td>Yes, watermarked</td></tr>
</tbody>
</table>
<p class="checked">Opus Clip and Vizard checked ${esc(monthYear(COMPETITORS['opus-clip'].checked))}; Descript and Riverside checked ${esc(monthYear(PODCAST_TOOLS_CHECKED))} on their public pricing pages. Pick Descript or Riverside if you also need to record or edit the episode; pick a clipper if the episode already exists and you want the clips.</p>
</section>`

const podcastToShorts = () => ({
  path: '/podcast-to-shorts',
  facts: [
    { k: 'Two speakers', v: 'Both in frame', s: 'Stacked split layout on real two-shots' },
    { k: 'Episode length', v: '1-2 hours', s: 'The design case, not the limit' },
    { k: 'Clips per episode', v: '3 to 15', s: '15 to 60 seconds each, captions burned in' },
    { k: 'Cost', v: '$0', s: 'Self-hosted; hosted from $12/month' },
  ],
  title: 'Podcast Clip Maker: AI Clips From Video Podcasts',
  description:
    'Podcast clip maker: paste an episode, get 3 to 15 vertical podcast clips with subtitles and both speakers in frame. Free self-hosted; hosted from $12/month.',
  h1: 'Podcast clip maker: AI podcast clips that keep both speakers in frame',
  relatedTitle: 'Related tools',
  breadcrumb: [{ name: 'Use cases', path: '/use-cases' }, { name: 'Podcast clips' }],
  published: '2026-08-04',
  updated: REVISED,
  lede: `OpenShorts turns a video podcast into 3 to 15 vertical clips: it transcribes every word, picks the moments that stand alone, and keeps the active speaker in frame when two people talk. Free to self-host under MIT, or hosted from $${STARTER.monthly.price}/month for ${STARTER.monthly.minutes} minutes; your first episode is clipped free up to 60 minutes.`,
  top: PODCAST_TOOLS,
  cta: {
    label: 'Clip an episode',
    title: 'Paste an episode link, get podcast clips',
    body: `First episode free up to 60 minutes, then ${CLOUD.freeMinutes} minutes a month, watermarked, no credit card. No watermark from $${STARTER.monthly.price}/month, or self-host free under MIT.`,
    button: 'Get free clips',
  },
  tldr: [
    'A two-person podcast is the hardest input an auto-clipper faces: a single centered crop shows the wrong person half the time, or an empty chair. OpenShorts detects a real two-shot and renders both speakers stacked in half-frames, so a reply never happens off screen.',
    'The rest of the pipeline is the same as for any long video: word-level transcription, scene detection, Gemini scoring the 3 to 15 strongest moments, subtitles burned in, and direct publishing to TikTok, Instagram Reels and YouTube Shorts.',
    'Cost is where podcasts punish credit-based tools: they bill the whole episode length before you see a clip. Self-hosted OpenShorts has no meter at all; hosted plans start at $12/month.',
  ],
  body: `
<h2>What a podcast clip maker does</h2>
<p>Podcast clips are the 15 to 60 second moments of an episode that stand on
their own: a sharp answer, a disagreement, a story with a punchline, cut to 9:16
with captions for TikTok, Reels and Shorts. A podcast clip maker finds those
moments in a full episode for you, reframes them and burns in the subtitles, so
one recording becomes a week of posts instead of an afternoon in an editor. The
hard part is not the cutting, it is keeping the right people on screen.</p>

<h2>Why podcasts break naive clipping tools</h2>
<p>Podcast video is a conversation, and conversations move. A tool that crops a
fixed center column out of a wide two-shot shows whoever happens to sit in the
middle, which is often nobody. A tool that follows one face loses the reaction
shots that make clips work. Reviewers of the mainstream clippers consistently
report exactly this: framing that needs manual correction on multi-person
footage. It is not carelessness, it is that a single moving crop cannot show two
people at once.</p>

<h2>How the two-speaker layout works</h2>
<p>OpenShorts detects when a scene is a genuine two-shot, meaning both faces are
visible in the same frame for at least half of the sampled frames. That test
matters: it is what separates a real side-by-side conversation from
shot/countershot editing, where a naive split would show the same person twice.
Confirmed two-shots render as a split layout, both speakers stacked in
half-frames filling the 9:16 canvas. With speaker cutting enabled the clip
instead hard-cuts to whoever is talking, with mouth activity normalised per
speaker so that lighting and contrast differences do not hand the whole scene to
one side of the table.</p>

<h2>From episode to posted clips, step by step</h2>
<ol>
<li>Paste the episode's YouTube link or upload the file. Podcasts of an hour or more are the normal case, not the limit.</li>
<li>faster-whisper transcribes with word-level timestamps, and PySceneDetect maps the cuts.</li>
<li>Google Gemini reads the transcript against the scene boundaries and returns the 3 to 15 segments that stand alone best, 15 to 60 seconds each.</li>
<li>Each segment is reframed for its content: split layout for two-shots, face tracking for single speakers, screencast layout if the episode shares a screen.</li>
<li>Subtitles are burned in from the word-level transcript, and finished clips post directly to TikTok, Instagram Reels and YouTube Shorts, or come back through the API.</li>
</ol>
<p>The same steps work for any long recording of people talking. Church
services are the closest case to a podcast: one speaker for most of an hour, and
the <a href="/use-cases/sermons">sermon clip maker</a> page covers clipping only
the sermon and keeping a preacher who walks the stage in frame.</p>

<h2>Podcast to TikTok, Reels and Shorts</h2>
<p>Video podcast clips are the same file on every platform: vertical, 15 to 60
seconds by default, captions burned in for the people who watch on mute.
Connect your accounts once and each clip publishes to YouTube Shorts and
Instagram Reels directly. Podcast to TikTok works the same way with one
difference TikTok imposes: the clip lands in your TikTok inbox as a draft, and
you post it from the app, where you also add the caption and a sound. A week of
clips can be scheduled in one go, one a day, so each episode keeps working
until the next one is out.</p>

<h2>Real podcast clips, next to their source</h2>
<p>A two-person episode filmed as one wide shot: both hosts stay in frame,
stacked, and the captions sit on the seam.</p>
${demoBlock('split')}
<p>An interview cut between host and guest instead: there is no two-shot to
stack, so the crop tracks whoever is on screen.</p>
${demoBlock('talk')}

<h2>What a full episode costs to clip</h2>
<p>Credit-metered tools bill on the length of the video you import, not on the
clips you keep. As of October 2026, a 60-minute episode costs 60 credits at Opus
Clip, Vizard or Quso whether it yields 5 usable clips or 20, and a weekly show at that
length runs past the entry plans of both. OpenShorts prices the other way
around:</p>
${pricingParagraph('podcast')}

<h3>What a weekly show costs to clip, by tool</h3>
<p>A weekly one-hour episode is about 260 minutes of source a month. Move the
slider to your own schedule:</p>
${costCalculator(['opus-clip', 'vizard', 'vidyo-ai', '2short'], { minutes: 270 })}

<h2>Podcast transcripts</h2>
<p>Every episode clipped here is transcribed first, word by word, because that
transcript is what the AI reads to find the moments and what the captions are
cut from. If what you need is the text of a whole episode rather than clips,
and the podcast is on YouTube, the free
<a href="/youtube-transcript-generator">YouTube transcript generator</a> gives
you the podcast transcript from the captions YouTube already has, with
timestamps, as TXT or SRT, with no sign-up. Show notes, quotes and an article
version of the episode all start from it.</p>

<h2>Podcast clips or audiograms: which one do you need?</h2>
<p>"Podcast clips" means two different things depending on how the show is
recorded. If you record <strong>video</strong>, even a static two-camera setup,
you want vertical 9:16 clips of the speakers with captions, usually 30 to 90
seconds, for TikTok, Reels and Shorts. That is what this page and OpenShorts are
about. If the show is <strong>audio-only</strong>, there is nothing to reframe,
and what you want is an audiogram: a waveform and captions over a still image.
Audiogram tools such as Headliner serve that case better than any video clipper.</p>

<h2>How the podcast clip makers compare</h2>
<p class="checked">Checked 2026-10-05 on each vendor's pricing page.</p>
<table>
<thead><tr><th>Tool</th><th>How it makes podcast clips</th><th>Free plan</th></tr></thead>
<tbody>
<tr><td class="os">OpenShorts</td><td class="os">Paste the episode link or upload it; AI picks 3 to 15 moments and keeps both speakers in frame</td><td class="os">First video free up to 60 min, then 20 min/month, watermarked; self-hosted unlimited and unwatermarked</td></tr>
<tr><td>Riverside</td><td>Magic Clips, inside the recording studio</td><td>Yes, with a Riverside watermark; Pro ($29/month) removes it</td></tr>
<tr><td>Opus Clip</td><td>General clipper with a podcast clip maker page</td><td>60 credits/month, watermarked</td></tr>
<tr><td>Headliner</td><td>Audiograms for audio-only shows, video clipping in beta</td><td>One unwatermarked audiogram a month, the rest watermarked</td></tr>
</tbody>
</table>
<p>If you already record in Riverside, its clips come with the recording and that
convenience is real. If you record anywhere else, or publish the full episode to
YouTube first, a clipper that starts from the link avoids exporting and
re-uploading an hour of video.</p>

${faqBlock([
  {
    q: 'How do I turn a podcast into clips for free?',
    a: 'Self-host OpenShorts: clone the MIT-licensed repo, run docker compose up, add a free-tier Google Gemini API key and paste your episode link. No watermark and no cap. If you would rather not run anything, OpenShorts Cloud clips your first episode free up to 60 minutes, then 20 minutes a month, with a watermark; paid plans start at $12/month.',
  },
  {
    q: 'What is the best podcast clip maker?',
    a: 'It depends on how you record. For a video podcast with two people on camera, pick a tool that keeps both in frame: OpenShorts stacks them in a split layout when both are visible. If you record in Riverside, its built-in Magic Clips are the most convenient. For an audio-only show, an audiogram tool such as Headliner fits better than any video clipper.',
  },
  {
    q: 'How does it handle two people talking?',
    a: 'Scenes where both faces share the frame at least half the time render as a stacked split layout so both speakers stay visible. Optionally it hard-cuts to the active speaker instead, using per-speaker normalised mouth activity to decide who is talking.',
  },
  {
    q: 'Does it work with hour-long episodes?',
    a: 'Yes, long-form is the design case. Processing time scales with length: on CPU roughly 5 to 8 minutes of processing per 8 minutes of source, on an NVIDIA GPU about a tenth of that. The AI moment scoring reads the transcript rather than the raw video, so episode length does not degrade selection quality.',
  },
  ...PODCAST_EXTRA_FAQ,
])}

${sources([
  'Competitor per-minute credit billing checked 2026-10-05 on vendor pricing and help pages.',
  `Split-layout and speaker-cut implementation in the project source at <a href="${SITE.repo}" rel="noopener">github.com/mutonby/openshorts</a>.`,
])}
`,
  faq: [
    {
      q: 'How do I turn a podcast into clips for free?',
      a: 'Self-host OpenShorts (MIT, Docker, bring a free-tier Gemini key): no watermark, no cap. Or use OpenShorts Cloud: 20 free minutes a month with a watermark, paid plans from $12/month.',
    },
    {
      q: 'How does it handle two people talking?',
      a: 'Real two-shots render as a stacked split layout keeping both speakers visible; optionally it hard-cuts to the active speaker instead.',
    },
    ...PODCAST_EXTRA_FAQ,
  ],
})

/* Built on the URL-first angle. It used to rest on Opus Clip's free plan being
 * upload-only; that stopped being true (checked 2026-10-05), so the page now
 * makes the claim that still holds: links work on every tier here, and not at
 * every competitor (2short and SendShort gate imports or clipping by plan). */
const youtubeConverter = () => ({
  path: '/youtube-to-shorts-converter',
  title: 'YouTube to Shorts Converter: Paste the Link | OpenShorts',
  description:
    'Convert a YouTube video into Shorts by pasting the link: no download-and-reupload step. AI picks the moments, crops to 9:16 and burns in the subtitles.',
  h1: 'A YouTube to Shorts converter that starts from the link',
  breadcrumb: [{ name: 'YouTube to Shorts converter' }],
  relatedTitle: 'Related tools',
  published: '2026-08-04',
  updated: REVISED,
  lede: `OpenShorts is an AI YouTube to Shorts converter: paste a YouTube link and it picks the 3 to 15 strongest moments of 15 to 60 seconds, reframes each one to 9:16 around the person speaking and burns in captions, ready for Shorts, TikTok and Reels. Your first video is free up to 60 minutes (watermarked, no card); no watermark from $${CLOUD.lowPrice}/month, or self-host it free under MIT.`,
  pasteLink: { placeholder: 'https://www.youtube.com/watch?v=...', label: 'YouTube video link' },
  tldr: [
    'Paste a YouTube link in the box below and get back 3 to 15 vertical clips with subtitles, sized for Shorts, Reels and TikTok. No downloading the source and re-uploading it first.',
    'The link-first flow is free on both editions: the self-hosted MIT edition has no cap, and the hosted free tier clips your first video whole up to 60 minutes, then 20 minutes a month. Uploads work on every tier too, for sources that are not online.',
    'Convert videos you have the rights to: your own channel, your clients’ with permission, or licensed footage.',
  ],
  body: `
<h2>How to convert a YouTube video into Shorts</h2>
<ol>
<li>Paste a YouTube link in the box on this page. OpenShorts fetches the video itself; there is no download-then-upload round trip through your machine.</li>
<li>The video is transcribed with word-level timestamps and scanned for scene boundaries.</li>
<li>Google Gemini scores the transcript against the scenes and picks the 3 to 15 segments most likely to stand alone, 15 to 60 seconds each.</li>
<li>Each segment is cropped to 9:16 with face tracking, or a split or screencast layout when the content calls for it, and subtitles are burned in.</li>
<li>Download the clips, or post them straight to YouTube Shorts, TikTok and Instagram Reels from the dashboard or the API.</li>
</ol>

<h2>Why starting from the link matters</h2>
<p class="checked">Competitor terms checked 2026-10-05 on public pricing pages.</p>
<p>Most long videos worth clipping already live on YouTube, so a converter that
only accepts uploads adds a detour: fetch the file with a downloader, wait,
re-upload gigabytes, wait again. OpenShorts accepts links and uploads on every
tier, including both free ones. Not every tool does both: as of October 2026
2short.ai takes YouTube links only on its free plan and adds Drive and URL
imports from its paid plans, and SendShort only clips long videos from its
$29/month plan up.</p>

<h2>What comes out the other end</h2>
<p>Vertical 9:16 clips of 15 to 60 seconds with word-level subtitles burned in,
each with an AI-written title and description ready for the platform. The
reframing follows the content: a single speaker is face-tracked with a
stabiliser that holds the camera still instead of chasing every head movement, a
two-person conversation renders as a stacked split layout, and screen shares
keep the screen legible instead of cropping it to ribbons.</p>

<h2>Whose videos can you convert?</h2>
<p>Yours, and those you have permission for. Your own uploads, your podcast
guests' episodes with their blessing, client channels you manage, licensed or
public-domain footage. Clipping someone else's video without permission is a
copyright question OpenShorts does not answer for you, and platforms remove
reuploads that fail it. The tool fetches what you point it at; the rights are
your call and your responsibility.</p>

<h2>What does it cost?</h2>
${pricingParagraph('youtube')}

${faqBlock([
  {
    q: 'Can I convert a YouTube video to Shorts for free?',
    a: 'Yes, two ways. Self-host OpenShorts (MIT licence, Docker, your own free-tier Gemini API key): unlimited, no watermark. Or use the hosted free tier: 20 minutes of source video a month, watermarked, no credit card. Paid hosted plans without watermark start at $12/month.',
  },
  {
    q: 'Do I need to download the video first?',
    a: 'No. Paste the URL and OpenShorts fetches the source itself on every tier, including free ones. Local file upload is also supported when the source is not online.',
  },
  {
    q: 'Can I clip a video from someone else’s channel?',
    a: 'Technically yes, legally only with rights or permission. Use it for your own content, channels you manage, or footage you have licensed. Unauthorized reuploads are copyright infringement and platforms strike them.',
  },
])}
`,
  faq: [
    {
      q: 'Can I convert a YouTube video to Shorts for free?',
      a: 'Yes: self-hosted OpenShorts is free with no cap (MIT, Docker, your own Gemini key), and the hosted free tier covers 20 watermarked minutes a month. Paid hosted plans start at $12/month.',
    },
    {
      q: 'Do I need to download the video first?',
      a: 'No. OpenShorts fetches the video from the pasted URL on every tier, free tiers included.',
    },
  ],
})

/* Recipe-shaped counterpart to /mcp: that page explains the protocol surface,
 * this one shows the working loop. Kept separate so each can rank for its own
 * intent instead of one page diluting both. */
const automateShorts = () => ({
  path: '/automate-shorts-api',
  title: 'AI Clipping API: Automate Shorts From Any Video | OpenShorts',
  description:
    'One POST starts the job, one signed webhook ends it. Automate shorts with the OpenShorts REST API, n8n or cron, with no per-call meter and no polling loop.',
  h1: 'AI clipping API: automate shorts with one request in, one webhook out',
  breadcrumb: [{ name: 'Automate shorts' }],
  published: '2026-08-04',
  updated: REVISED,
  lede: `OpenShorts has an AI clipping API: one POST with a video URL returns 3 to 15 vertical clips with captions, and a webhook tells you when they are ready. It uses the same minutes as the dashboard ($${STARTER.monthly.price}/month for ${STARTER.monthly.minutes} minutes, no per-call fee), and is unmetered when self-hosted.`,
  top: `<h2>What the clipping API costs per minute of video</h2>
<table>
<thead><tr><th>Plan</th><th>Price</th><th>Source minutes a month</th><th>Per source minute</th></tr></thead>
<tbody>
<tr><td class="os">Self-hosted (MIT)</td><td>$0</td><td>No cap</td><td>$0 (your machine and keys)</td></tr>
${[STARTER, CREATOR, PRO].map((p) => `<tr><td class="os">${esc(p.name)}</td><td>$${p.monthly.price}/month</td><td>${p.monthly.minutes}</td><td>$${(p.monthly.price / p.monthly.minutes).toFixed(3)} ($${(p.yearly.price / p.yearly.minutes).toFixed(3)} billed yearly)</td></tr>`).join('')}
</tbody>
</table>
<p>API calls, dashboard jobs and MCP tool calls all draw from the same balance; there is no separate API plan and no per-request fee. Trims, hook edits and re-renders use no minutes.</p>`,
  tldr: [
    'The whole automation loop is two HTTP messages. You POST a video URL with an API key and a webhook address; when processing ends, OpenShorts sends exactly one signed webhook carrying clip titles and durable download links. No polling loop, no timeout guessing.',
    'API calls draw from the same minute balance as the dashboard, with no separate meter and no per-call pricing. On the self-hosted edition there is no meter at all, which is what makes an always-on pipeline affordable.',
    'For agent-driven automation (Claude, ChatGPT, custom agents) the same account also exposes an MCP server; that protocol surface is documented on its own page.',
  ],
  body: `
<h2>The loop, end to end</h2>
<p>Start a job with one request:</p>
<pre><code>curl -X POST https://api.openshorts.app/api/process \\
  -H "Authorization: Bearer osk_..." -H "Content-Type: application/json" \\
  -d '{"url": "https://youtube.com/watch?v=...", "acknowledged": true,
       "webhook_url": "https://your-server.com/hooks/openshorts",
       "webhook_secret": "your-shared-secret"}'</code></pre>
<p>The response returns a job id immediately. Minutes later, when the clips are
cut, subtitled and archived, OpenShorts POSTs once to your webhook URL with the
job outcome, clip titles and download links durable enough to fetch later. A
failed job also fires the webhook, so your pipeline never hangs on silence.</p>

<h2>Verifying the webhook</h2>
<p>If you passed a <code>webhook_secret</code>, the request carries an
<code>X-OpenShorts-Signature</code> header of the form
<code>sha256=&lt;hex&gt;</code>: the HMAC-SHA256 of the raw request body under
your secret. Recompute it and compare in constant time:</p>
<pre><code>expected = "sha256=" + hmac.new(secret, raw_body, hashlib.sha256).hexdigest()
hmac.compare_digest(expected, request.headers["X-OpenShorts-Signature"])</code></pre>
<p>Reject anything that does not match and you have closed the door on forged
deliveries.</p>

<h2>Using it from n8n, Zapier or Make</h2>
<p>No dedicated node is needed: the flow is a generic HTTP Request step that
POSTs to <code>/api/process</code>, then a Webhook trigger that receives the
completion payload and fans out to whatever comes next, posting to socials,
dropping links in Slack, logging to a sheet. The two-step shape above is the whole integration, which is why generic
nodes cover it. There is now also an importable
<a href="/n8n-youtube-shorts-automation">n8n workflow</a> that wires the whole
loop, including channel watching, Telegram approval and scheduled publishing.</p>

<h2>A weekly pipeline in one cron line</h2>
<p>Because the API is one POST, scheduling is whatever scheduler you already
have. A cron job that submits the latest episode URL every Monday, a GitHub
Action on your podcast repo, or an agent that watches a feed. The webhook does
the second half, so nothing stays running in between. If you would rather not
write the curl, the CLI wraps it:</p>
<pre><code>uvx openshorts process "$EPISODE_URL" \\
  --webhook https://your-server.com/hooks/openshorts \\
  --webhook-secret "$SECRET"</code></pre>

<h2>What automation costs</h2>
<p>API and MCP calls draw from the same minute balance as the dashboard. There
is no per-call price, no separate API tier and no automation surcharge. As of
August 2026 that is not the market default: the mainstream tools meter their
APIs per source minute or per operation, on top of subscription tiers, so an
always-on pipeline runs with a taxi meter attached. Self-hosted OpenShorts has
no meter of any kind, and the hosted plans are flat:</p>
${pricingParagraph('automate')}

<h2>Agents instead of scripts</h2>
<p>If the thing driving the pipeline is an AI agent rather than a script, the
same account exposes a native MCP server with six tools covering process,
status, clips, quota, subtitles and publishing. The endpoint, the tool table and
client setup live on the <a href="/mcp">MCP server and API page</a>.</p>

${faqBlock([
  {
    q: 'Can I automate YouTube Shorts creation with an open source tool?',
    a: 'Yes. OpenShorts is MIT-licensed and its self-hosted edition serves the same REST API and MCP server as the hosted service, with no metering. One POST submits a video, a signed webhook returns the finished clips, and the publishing endpoint posts them to YouTube Shorts, TikTok and Instagram Reels.',
  },
  {
    q: 'Is there an n8n integration for OpenShorts?',
    a: 'Yes: an importable workflow at /n8n-youtube-shorts-automation watches a YouTube channel, clips each video, sends the clips to Telegram for approval and schedules the approved ones to your socials. Nothing forces you to use it, though, since the integration is just an HTTP Request node posting to /api/process plus a Webhook trigger.',
  },
  {
    q: 'Do automated API calls cost more than using the dashboard?',
    a: 'No. API and MCP usage draws from the same minute balance as the dashboard with no per-call pricing: 20 free minutes a month on the hosted free tier, flat paid plans from $12/month, and no meter at all on the self-hosted edition.',
  },
])}

${sources([
  `Webhook signing implementation in the project source at <a href="${SITE.repo}" rel="noopener">github.com/mutonby/openshorts</a>.`,
  'Competitor API metering checked 2026-08-04 on public pricing and developer documentation.',
])}
`,
  faq: [
    {
      q: 'Can I automate YouTube Shorts creation with an open source tool?',
      a: 'Yes: OpenShorts self-hosted serves the same REST API, MCP server and signed webhooks as the hosted service, MIT-licensed and unmetered. One POST in, one signed webhook out.',
    },
    {
      q: 'Do automated API calls cost more than using the dashboard?',
      a: 'No. API and MCP calls draw from the same minute balance with no per-call pricing; the self-hosted edition has no meter at all.',
    },
  ],
})

/* The n8n community's two loudest unanswered asks, checked August 2026, are
 * "is there a free/self-hostable clipper I can call from a workflow" and
 * "why is posting a video to TikTok/Instagram so painful". This page answers
 * both with a workflow that exists, and gives the Reddit/forum posts a stable
 * URL to point at that is ours rather than a raw file on GitHub. */
const n8nTemplate = () => ({
  path: '/n8n-youtube-shorts-automation',
  title: 'n8n YouTube Shorts Template (Auto-Posting) | OpenShorts',
  description:
    'A free n8n workflow that watches your YouTube channel, clips each video into shorts, sends them to Telegram for approval and drip-publishes them automatically.',
  h1: 'The n8n content machine: your channel clips itself, you approve from your phone',
  breadcrumb: [{ name: 'n8n template' }],
  published: '2026-08-21',
  updated: '2026-08-21',
  body: `
<figure class="shot">
<img src="/n8n-content-machine-workflow.png" width="950" height="750" loading="eager"
     alt="The OpenShorts content machine open in n8n: four labelled stages, from the daily channel RSS trigger through the clipping call, the Telegram approval buttons and the weekly analytics digest.">
<figcaption>The whole machine is one canvas: watch the channel, clip, approve from Telegram, drip-publish and measure.</figcaption>
</figure>
<p>This workflow runs a YouTube channel on its own without handing over the
publish button. Once a day it reads your channel feed, clips one video into
vertical shorts, and sends each finished clip to your Telegram with Publish and
Skip buttons. What you approve is scheduled one post per day to the accounts you
connected, and every Sunday it reports back what the published clips actually
did. Two credentials, no polling loop, no TikTok or Instagram OAuth app.</p>

<h2>Download the workflow</h2>
<p>The JSON lives in the project repository, not behind an email form:</p>
<ul>
<li><a href="${SITE.repo}/blob/main/examples/n8n/openshorts-content-machine.json" rel="noopener">openshorts-content-machine.json</a> — the full four-stage machine described below.</li>
<li><a href="${SITE.repo}/blob/main/examples/n8n/openshorts-clip-and-notify.json" rel="noopener">openshorts-clip-and-notify.json</a> — the minimal version: a form takes a video URL, a signed webhook returns the clips.</li>
<li><a href="${SITE.repo}/tree/main/examples/n8n" rel="noopener">Setup notes</a> — credentials, the webhook secret, and the known limits.</li>
</ul>
<p>In n8n: <strong>Workflows → Import from file</strong>, then fill in your channel
id and chat id where the sticky notes on the canvas say so.</p>

<h2>What the workflow actually does</h2>
<p>Four stages, all on one canvas:</p>
<ol>
<li><strong>Watch the channel.</strong> A daily schedule trigger reads
<code>youtube.com/feeds/videos.xml?channel_id=...</code>, which needs no YouTube
API key, and picks one video: your newest upload, or if there is nothing new,
the next unprocessed video from your back catalogue. One video a day keeps the
minute burn predictable, and a 402 (out of minutes) pauses the machine with a
Telegram notice instead of failing silently.</li>
<li><strong>Clip, then get called back.</strong> One HTTP request to
<code>/api/process</code> carrying a <code>webhook_url</code>. When the job
ends, OpenShorts POSTs once with the finished clips and durable download links.
There is no Wait node anywhere in the workflow.</li>
<li><strong>Approve from your phone.</strong> Each 9:16 clip arrives in Telegram
as a video message with Publish and Skip buttons. A human approves every post,
which is also what separates this from the fully automated pipelines that
YouTube's inauthentic-content policy targets.</li>
<li><strong>Drip-publish and measure.</strong> Approved clips take the next free
daily slot and post to the accounts you connected in OpenShorts. Every Sunday
the workflow reads back the analytics of what it published and sends you
impressions, per-platform split and your best post.</li>
</ol>

<h2>Why posting does not need TikTok or Instagram credentials</h2>
<p>The usual wall in an n8n video workflow is the publishing half: TikTok
requires an audited app to post publicly, and the Instagram Graph API refuses
anything that is not a public static URL, which is why Google Drive links fail
there. This workflow sidesteps both by posting through
<code>POST /api/social/post</code> against the networks you connected once in
your OpenShorts account. The workflow itself holds no social credentials, and
the clip file is already on durable storage, so the public-URL requirement is
satisfied before Instagram ever sees it.</p>

<h2>Scheduling that does not double-book</h2>
<p>Approving two clips seconds apart used to be enough to publish them at the
same minute: each execution read the same in-memory counter before either wrote
back. The workflow now asks the server for the queue it actually holds
(<code>GET /api/social/scheduled</code>) and picks the first free slot from it,
which is shared state and cannot race with itself that way. The same endpoint,
plus <code>DELETE /api/social/scheduled/{job_id}</code>, is how you inspect or
cancel a post before it goes out.</p>

<h2>What it costs to run</h2>
${pricingParagraph('n8n')}
<p>API calls draw from the same minute balance as the dashboard: no per-call
price, no automation surcharge, and no meter at all on the self-hosted edition.
Telegram bots are free, and n8n runs wherever you already run it.</p>

<h2>Known limits, before you find them the hard way</h2>
<ul>
<li>Telegram previews a video by URL up to about 20 MB; larger clips arrive as a
link message carrying the same approval buttons.</li>
<li>A YouTube channel RSS feed exposes only the latest 15 videos, so the
back-catalogue drip reaches back that far and no further.</li>
<li>The Telegram Trigger node needs your n8n to have a public https URL, because
Telegram registers a webhook against it. A laptop-local n8n can run every other
stage, but the approval buttons need a deployed instance or a tunnel.</li>
<li>Workflow static data, where the machine remembers which videos it already
processed, only persists on production executions. Test runs from the editor do
not advance it.</li>
</ul>

<h2>Prefer an agent to a workflow?</h2>
<p>The same account exposes an MCP server, so Claude, ChatGPT or an n8n AI Agent
node can drive the pipeline as tools instead of fixed steps. That surface is
documented on the <a href="/mcp">MCP server and API page</a>, and the raw REST
loop on the <a href="/automate-shorts-api">automation page</a>.</p>

${faqBlock([
  {
    q: 'Is there a free n8n template to turn long videos into shorts?',
    a: 'Yes. OpenShorts publishes an MIT-licensed n8n workflow that clips a YouTube channel automatically and posts the approved clips to TikTok, Instagram and YouTube. The JSON is in the project repository with no email gate, and the clipper behind it is open source, so it can run entirely on your own hardware.',
  },
  {
    q: 'How do I post a video to TikTok or Instagram from n8n?',
    a: 'Neither platform has a native n8n node, and both have hard requirements: TikTok needs an audited app for public posts and Instagram needs a public static URL for the media. Posting through the OpenShorts API avoids both, because the accounts are connected once in your OpenShorts account and the clip already lives on durable public storage.',
  },
  {
    q: 'Does the workflow poll for the clipping job to finish?',
    a: 'No. Clipping a real video takes minutes, so the workflow passes a webhook_url with the job and OpenShorts calls it exactly once when the job reaches a terminal state. Failed jobs fire the same webhook with an error field, so the flow never hangs.',
  },
  {
    q: 'Can the clips publish without me approving them?',
    a: 'They can, by connecting the posting step directly to the webhook branch, but the template ships with the approval gate on purpose: unreviewed automated publishing is what YouTube\'s inauthentic-content policy targets, and one bad clip lands on your own audience.',
  },
])}

${sources([
  `Workflow JSON and setup notes in the project repository at <a href="${SITE.repo}/tree/main/examples/n8n" rel="noopener">github.com/mutonby/openshorts</a>.`,
  'TikTok and Instagram publishing constraints checked against their developer documentation, August 2026.',
])}
`,
  faq: [
    {
      q: 'Is there a free n8n template to turn long videos into shorts?',
      a: 'Yes: OpenShorts ships an MIT-licensed n8n workflow that clips a YouTube channel automatically and posts approved clips to TikTok, Instagram and YouTube. The JSON is public in the repository with no email gate.',
    },
    {
      q: 'How do I post a video to TikTok or Instagram from n8n?',
      a: 'Post through the OpenShorts API: the social accounts are connected once in your OpenShorts account, so the workflow needs no TikTok app audit and no public CDN URL for the file.',
    },
  ],
})

const mcpAgentsPage = () => ({
  path: '/mcp',
  title: 'OpenShorts MCP Server: Clip Video From Claude or ChatGPT',
  description:
    'Connect Claude, ChatGPT, Cursor or n8n with one URL and clip, subtitle and publish videos from a chat. 8 MCP tools, a REST API and signed webhooks.',
  h1: 'Clip and publish video from an AI agent',
  breadcrumb: [{ name: 'MCP server and API' }],
  cta: {
    label: 'For agents',
    title: 'Point your agent at a real pipeline',
    body: 'Connect Claude or ChatGPT to mcp.openshorts.app/mcp, or copy an API key. Self-hosted serves the same endpoint, unmetered.',
    button: 'Get an API key',
  },
  tldr: [
    'OpenShorts has a native MCP server at mcp.openshorts.app/mcp. Connect any MCP client, Claude, ChatGPT, Cursor or a custom agent, and a prompt like "clip this podcast and schedule the best three to TikTok" becomes one instruction instead of an afternoon in an editor.',
    'Eight tools cover the whole pipeline: process_video, create_upload, get_job_status, list_clips, get_quota, add_subtitles, recut_clip and publish_clip. There is also a plain REST API with per-user keys, and completion webhooks so pipelines never poll.',
    'The difference that survives comparison shopping is the meter. Most clipping tools now have an API, and Opus Clip added an MCP server in July 2026, but they meter agent calls per source minute or per operation. OpenShorts API calls draw from the same flat minute balance as the dashboard, and the self-hosted edition, free and MIT-licensed, serves the same MCP endpoint with no meter at all.',
  ],
  body: `
<h2>What can an agent actually do with OpenShorts?</h2>
<p>Everything the dashboard does. The MCP server is not a wrapper around a
subset of features: each tool calls the same pipeline the web app uses, with the
same account, the same minutes and the same job history. An agent can take a
YouTube URL, turn it into 3 to 15 vertical clips with word-level captions,
restyle those captions, and publish or schedule the result to TikTok, Instagram
Reels and YouTube Shorts.</p>

<h2>How do I connect Claude or ChatGPT?</h2>
<p>With the URL alone. The server implements OAuth 2.1 with dynamic client
registration, which is what claude.ai and ChatGPT expect from a remote MCP
server, so there is no key to copy:</p>
<ol>
<li><strong>claude.ai:</strong> Settings, Connectors, Add custom connector, paste <code>https://mcp.openshorts.app/mcp</code>, Connect.</li>
<li><strong>ChatGPT:</strong> Settings, Connectors, Create, paste the same URL, choose OAuth.</li>
<li>Approve the access on openshorts.app (sign in if you are not). The 8 tools appear in every chat, and the connection is listed under Account, API keys, where revoking it disconnects the app.</li>
</ol>
<h2>How do I connect Claude Code, Cursor or n8n?</h2>
<p>CLI and workflow clients take an API key instead: create one in your account
page (shown once, starts with <code>osk_</code>) and pass it as a Bearer
token. With Claude Code:</p>
<pre><code>claude mcp add --transport http openshorts https://mcp.openshorts.app/mcp \\
  --header "Authorization: Bearer osk_..."</code></pre>
<p>Any client that speaks Streamable HTTP works the same way: the endpoint is
<code>https://mcp.openshorts.app/mcp</code>, the server describes itself over
the protocol, tool schemas included, and the account page has copy-ready
snippets for Claude Desktop, Cursor, n8n and curl.</p>

<h2>What tools does the MCP server expose?</h2>
<table>
<thead><tr><th>Tool</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>process_video</code></td><td>Starts clipping a video from a URL or an upload_id. Returns a job id immediately; processing takes minutes. Pass captions: false when the source already has subtitles burned in, auto_hook: false to skip the hook line (on by default).</td></tr>
<tr><td><code>create_upload</code></td><td>Reserves an upload slot for a local file: the agent PUTs the bytes to the returned URL, then processes it by upload_id.</td></tr>
<tr><td><code>get_job_status</code></td><td>Progress, recent log lines, and the clips once the job completes.</td></tr>
<tr><td><code>list_clips</code></td><td>Titles, durations, platform-ready descriptions and download URLs for a finished job.</td></tr>
<tr><td><code>get_quota</code></td><td>Plan and remaining minutes, so an agent can check before starting a large job.</td></tr>
<tr><td><code>add_subtitles</code></td><td>Restyles the burned-in captions of one clip: presets (default, hormozi, pill, lime, oneword, clean), size, word-by-word reveal, a box behind the active word or one word at a time.</td></tr>
<tr><td><code>publish_clip</code></td><td>Posts or schedules one clip to TikTok, Instagram or YouTube through the connected account.</td></tr>
</tbody>
</table>
<p>In clients that support the MCP Apps extension (ChatGPT apps, mcp-ui hosts),
<code>list_clips</code> also renders as an interactive clip picker: preview each
9:16 clip inline, select the keepers and publish them without leaving the
conversation. Clients without UI support see the same data as plain results.</p>

<h2>How does this compare to the other clipping tools' agent access?</h2>
<p class="checked">Checked 2026-08-04 on vendor developer documentation and pricing pages. This market is moving fast; verify before committing a pipeline.</p>
<p>Agent access stopped being exclusive in 2026: Opus Clip launched its own MCP
server in July, Reap ships MCP plus a CLI, and Klap, Vizard and Submagic have
REST APIs. A comparison that pretended otherwise would not deserve your trust.
What still separates the offerings is how agent calls are billed and where the
server can run:</p>
<table>
<thead><tr><th>Tool</th><th>MCP server</th><th>REST API</th><th>How agent calls are billed</th></tr></thead>
<tbody>
<tr><td class="os">OpenShorts</td><td class="os yes">Yes, hosted and self-hosted</td><td class="os yes">Yes, with signed webhooks</td><td class="os">Same flat minute balance as the dashboard; self-hosted has no meter</td></tr>
<tr><td>Opus Clip</td><td class="yes">Yes, since July 2026</td><td>Yes</td><td>Credits per source minute, expiring in 60 days</td></tr>
<tr><td>Reap</td><td class="yes">Yes, plus CLI</td><td>Yes</td><td>Subscription from $9.99/month, metered minutes</td></tr>
<tr><td>Klap</td><td>No</td><td>Yes</td><td>Per operation, roughly $0.32 to $0.48 each</td></tr>
<tr><td>Vizard</td><td>No</td><td>Yes, with webhooks</td><td>Consumes plan upload minutes</td></tr>
<tr><td>Submagic</td><td>No</td><td>Yes, Business tier ($69/month)</td><td>Metered per minute on top of the tier</td></tr>
</tbody>
</table>
<p>The consequence for an autonomous pipeline is simple: an agent loop on a
metered API runs with the bill still attached to every decision it makes. On a
flat plan the worst an agent can do is spend your minutes; on the self-hosted
edition there is nothing to spend. For the recipe-shaped version of this,
webhooks, n8n and cron, see <a href="/automate-shorts-api">automating shorts
with the API</a>.</p>

<h2>Can I use a plain REST API instead of MCP?</h2>
<p>Yes. The same <code>osk_</code> key authenticates against the REST API, and
interactive documentation lives at
<a href="https://api.openshorts.app/docs" rel="noopener">api.openshorts.app/docs</a>
with the OpenAPI spec at <code>/openapi.json</code>. A processing job is one
request:</p>
<pre><code>curl -X POST https://api.openshorts.app/api/process \\
  -H "Authorization: Bearer osk_..." -H "Content-Type: application/json" \\
  -d '{"url": "https://youtube.com/watch?v=...", "acknowledged": true,
       "webhook_url": "https://your-server.com/hooks/openshorts"}'</code></pre>

<h2>Is there a CLI?</h2>
<p>Yes, a zero-dependency one on PyPI. It talks to the same REST surface as
everything else, so the terminal, the dashboard and the agents can never
disagree about what a job did:</p>
<pre><code>pip install openshorts   # or: uvx openshorts

export OPENSHORTS_API_KEY=osk_...
openshorts process "https://youtube.com/watch?v=..." --wait
openshorts clips &lt;job_id&gt;
openshorts publish &lt;job_id&gt; 0 --platforms tiktok,youtube</code></pre>
<p>Point <code>OPENSHORTS_API_URL</code> at <code>http://localhost:8000</code>
and the same binary drives a self-hosted instance with no key.</p>

<h2>How do completion webhooks work?</h2>
<p>Pass <code>webhook_url</code> when starting a job and OpenShorts sends
exactly one POST when the job finishes or fails, with clip titles and download
links in the body. Add a <code>webhook_secret</code> and the body is signed with
HMAC-SHA256 in the <code>X-OpenShorts-Signature</code> header so your receiver
can verify the sender. This is what lets an n8n, Zapier or cron pipeline run
without a polling loop.</p>

<h2>Does this work on the self-hosted edition?</h2>
<p>Yes. The self-hosted edition serves the same <code>/mcp</code> endpoint on
your own machine, with no API key required because there is no account system:
it follows the same bring-your-own-key rules as the rest of the self-hosted app.
Point your MCP client at <code>http://localhost:8000/mcp</code> and the same six
tools appear.</p>

<h2>What does it cost?</h2>
${pricingParagraph('mcp')}
<p>API and MCP calls are not billed separately: they draw from the same minute
balance as the dashboard, so automation does not change the price of anything.</p>

${faqBlock([
  {
    q: 'Does OpenShorts have an MCP server?',
    a: 'Yes, a native one at mcp.openshorts.app/mcp using the Streamable HTTP transport. It exposes eight tools covering the full pipeline: process_video, create_upload, get_job_status, list_clips, get_quota, add_subtitles, recut_clip and publish_clip. Authentication is an API key created in the dashboard, sent as a Bearer token.',
  },
  {
    q: 'Can Claude or ChatGPT create video clips with OpenShorts?',
    a: 'Yes. Any MCP-capable client, including Claude and ChatGPT, can connect to mcp.openshorts.app/mcp with an API key and drive the whole flow: submit a video URL, wait for processing, list the generated clips and publish them to TikTok, Instagram or YouTube.',
  },
  {
    q: 'Is there an API for OpenShorts?',
    a: 'Yes, a REST API documented at api.openshorts.app/docs, authenticated with per-user osk_ keys created in the dashboard. It covers processing, status, subtitles, publishing and completion webhooks.',
  },
  {
    q: 'Do API calls cost extra?',
    a: 'No. API and MCP usage draws from the same minute balance as the dashboard: 20 free minutes a month on the hosted free tier, and paid plans from $12/month. The self-hosted edition is free under MIT and serves the same endpoints with no metering. Most competing APIs are billed per source minute or per operation on top of a subscription.',
  },
  {
    q: 'How is this different from the Opus Clip MCP server?',
    a: 'The tool surface is similar: both expose around six tools covering processing, captions and publishing. The differences are billing and deployment. Opus Clip meters MCP usage in credits per source minute, and those credits expire in 60 days; OpenShorts draws from a flat minute balance with no per-call pricing. And only OpenShorts can run the same MCP server on your own machine, unmetered, because the code is MIT-licensed.',
  },
])}

${sources([
  `MCP specification and transports at <a href="https://modelcontextprotocol.io" rel="noopener">modelcontextprotocol.io</a>.`,
  `OpenShorts server implementation in the project source at <a href="${SITE.repo}" rel="noopener">github.com/mutonby/openshorts</a>.`,
])}
`,
  faq: [
    {
      q: 'Does OpenShorts have an MCP server?',
      a: 'Yes, a native MCP server at mcp.openshorts.app/mcp with six tools covering the full pipeline, authenticated with an API key created in the dashboard.',
    },
    {
      q: 'Can Claude or ChatGPT create video clips with OpenShorts?',
      a: 'Yes. Any MCP-capable client can connect with an API key and drive the whole flow from video URL to published clip.',
    },
    {
      q: 'Do API calls cost extra?',
      a: 'No. API and MCP usage draws from the same minute balance as the dashboard: 20 free minutes a month on the hosted free tier, paid plans from $12/month, and the self-hosted edition is free under MIT.',
    },
    {
      q: 'How is this different from the Opus Clip MCP server?',
      a: 'Similar tool surface, different billing and deployment: Opus Clip meters MCP usage in credits per source minute that expire in 60 days, while OpenShorts draws from a flat minute balance, and only OpenShorts can run the same MCP server self-hosted and unmetered.',
    },
  ],
})

/* ---------------------------------------------------------------------------
 * The buying-intent cluster around Opus Clip.
 *
 * /alternatives/opus-clip holds the head term ("opus clip alternative"). These
 * three answer the commercial questions either side of it, which are searched
 * by people who are already paying a competitor or about to: what it costs
 * ("opus clips pricing", "opus clip free"), what the free plan withholds
 * ("opus clip free trial"), and what the brand names mean (opus.pro is called
 * "Opus AI" and its mid tier is "Opus Pro"; both are searched far more than
 * the product name is spelled).
 *
 * Every number that exists in data.js is read from there rather than retyped,
 * so the three hand-synced price lists cannot drift further apart.
 * ------------------------------------------------------------------------- */
const OS_LANE = {
  title: 'OpenShorts',
  steps: [
    'Paste a YouTube link or upload',
    'Word-level transcript and scene cuts',
    'Gemini scores and picks 3 to 15 moments',
    'Reframed: face tracking, two speakers, screen + presenter',
    'Captions and hook burned in, posted or sent by webhook',
  ],
}

const OPUS = COMPETITORS['opus-clip']
/* Our plan as the first row of every rival tier table: an engine that lifts
 * the table for "opus clip pricing" carries the cheaper open option with it. */
const OS_TIER_ROW = `<tr><td class="os">OpenShorts (for comparison)</td><td class="os">${esc(OS_PRICE_CLAUSE)}; first video free up to 60 minutes on the hosted free plan</td></tr>`
const opusTierTable = (filter = () => true) =>
  `<table>
<thead><tr><th>Plan</th><th>What it includes</th></tr></thead>
<tbody>${OS_TIER_ROW}${OPUS.tiers
    .filter(([n]) => filter(n))
    .map(([n, d]) => `<tr><td class="os">${esc(n)}</td><td>${esc(d)}</td></tr>`)
    .join('')}</tbody>
</table>`

const opusClipPricing = () => ({
  path: '/opus-clip-pricing',
  facts: [
    { k: 'Free plan', v: '60 min/mo', s: 'Watermarked, export within 3 days' },
    { k: 'Starter', v: '$15/mo', s: '150 minutes, monthly billing only' },
    { k: 'Pro', v: '$14.50/mo', s: 'Billed yearly ($29 monthly), 300 minutes' },
    { k: 'Free trial', v: '7 days', s: 'Of Pro, no card, still watermarked' },
  ],
  // The tiers in the title are read from OPUS.tiers (Free, Starter $15, Pro
  // $29); change both together.
  title: 'Opus Clip Pricing 2026: Free, $15, $29 Plans Explained',
  description:
    'Opus Clip pricing, October 2026: free plan with 60 credits, Starter $15/month, Pro $29 or $14.50/month yearly, the 7-day trial, and what one credit buys.',
  h1: 'Opus Clip pricing in 2026: the free plan, Starter, Pro and what a credit is',
  lede: brandLede(OPUS, 'opus.pro/pricing and the OpusClip help centre',
    `Opus Clip starts at $15/month for 150 minutes; the open source alternative, OpenShorts, is ${esc(OS_PRICE_CLAUSE)}.`),
  relatedTitle: 'Related comparisons',
  breadcrumb: [
    { name: 'Alternatives', path: '/alternatives' },
    { name: 'Opus Clip', path: '/alternatives/opus-clip' },
    { name: 'Pricing' },
  ],
  published: '2026-09-17',
  updated: REVISED,
  pasteLink: true,
  cta: {
    label: 'Before you upgrade',
    title: 'Run one of your videos here first',
    body: 'Paste a link and compare the output against your last Opus Clip export. First video free up to 60 minutes, no credit card.',
    button: 'Get free clips',
  },
  tldr: [
    `Opus Clip has four plans (checked ${esc(OPUS.checked)}): <strong>Free</strong> with 60 credits a month and a watermark, <strong>Starter</strong> at $15/month for 150 credits, <strong>Pro</strong> at $29/month or $14.50/month billed yearly for 300 credits, and a custom-priced <strong>Business</strong> plan.`,
    'One credit is one <strong>minute of source video you import</strong>, not one clip you export. A 60-minute podcast costs 60 credits whether it yields 5 clips or 20. Unused credits roll over for two months.',
    'There is a 7-day free trial of Pro with no credit card, but trial exports keep the watermark. OpenShorts prices the other way round: $0 self-hosted with no meter, or hosted from $12/month for 100 minutes with no watermark.',
  ],
  body: `
<h2>How much does Opus Clip cost?</h2>
<p class="checked">Checked ${esc(OPUS.checked)} on the vendor's public pricing and help pages. Vendors change plans without notice; verify before you buy.</p>
${opusTierTable()}
<p>Starter is sold month to month only. The yearly discount (about 50%) applies
to Pro, which is why Pro billed yearly, at $14.50/month, costs less than Starter
billed monthly, at $15/month, while giving twice the credits.</p>

<h2>How Opus Clip's credit system works</h2>
<p>The unit Opus Clip bills in is not the clip. It is the minute of video you
import. ${esc(OPUS.gotcha)}</p>
<ul>
<li>A video shorter than a minute still costs 1 credit, and partial minutes round down: a 4.5-minute video costs 4.</li>
<li>Credits on a monthly plan expire after 60 days, so unused ones roll over one month.</li>
<li>Sources can be up to 10 hours long, which at one credit a minute is 600 credits, two months of Pro.</li>
</ul>
<p>If your sources are short, the meter barely moves. If they are long (a
90-minute interview, a weekly show) the meter is the whole bill, and the number
of clips you keep never enters into it.</p>

<h2 id="free">Is Opus Clip free?</h2>
<p>Yes, within limits: there is a permanent free plan, and it watermarks every
export. It gives 60 credits a month, so 60 minutes of source video, with exports
up to 1080p. There is no editing on the free plan, the virality score is hidden,
and clips have to be exported within 3 days. It imports from YouTube links and
local files. Removing the watermark starts at Starter, $15/month.</p>

<h2>Is there an Opus Clip free trial?</h2>
<p>Yes: 7 days of Pro, no credit card. Two details matter. Trial exports still
carry the watermark, and the trial leaves out 4K, social posting, the 100 GB of
storage and team seats. When the 7 days end, the account drops to the free plan
rather than charging you. Upgrading later removes the watermark from projects you
made on the free plan or the trial.</p>
<div class="note"><span class="label">The cheapest honest test</span><p>The
watermark is the thing you are trying to evaluate past. If the question is
whether the pipeline is good enough for your footage, a self-hosted OpenShorts
run on one episode answers it at no cost and with no watermark, because the
self-hosted edition contains no metering or watermark code.</p></div>

<h2 id="pro">The Opus Pro plan: $29, or $14.50/month billed yearly</h2>
<p>"Opus Pro" means two things in searches: <code>opus.pro</code>, the website of
the OpusClip clipper (see <a href="/opus-ai">what Opus AI is</a>), and Pro, its
mid plan. The plan:</p>
${opusTierTable((n) => n === 'Starter' || n === 'Pro')}
<p>${esc(OPUS.proNote)} The pricing quirk is the yearly discount: Starter is
monthly only, so Pro billed yearly ($174 a year) is cheaper than Starter and
gives twice the credits. It only stops being the better deal if you cannot
commit to a year.</p>
<p>Who it is priced for: Opus bills one credit per minute imported, so the plan
is about input volume, not clips published. A weekly 60-minute show is roughly
260 source minutes a month, which fits inside Pro and not inside Starter. One
20-minute video a week is about 90 minutes, which Starter covers. Set your own
number: at 150 minutes or fewer Starter covers it, above that Pro does, and
yearly billing halves it.</p>
${costCalculator(['opus-clip'], { minutes: 240 })}
<p>What you would give up for the same job on OpenShorts is the caption-style
library: Opus Clip's presets are more numerous and more polished than ours. What
you gain: a self-hosted edition with no meter, so a 90-minute interview and a
9-minute one both cost nothing, and a hosted one where API and MCP usage draws
from the same flat minute balance as the dashboard.</p>

<h2>What would Opus Clip cost you?</h2>
<p>Three common creators, priced on the credit rule above (one credit per source minute):</p>
${factTiles([
  { k: 'One 20-min video a week', v: '~87 min', s: 'Fits Starter at $15/month' },
  { k: 'Weekly 60-min podcast', v: '~260 min', s: 'Needs Pro: $29/month, or $14.50 yearly' },
  { k: 'Daily 30-min stream', v: '~900 min', s: 'Past two Pro packs: Business, custom price' },
])}
<p>For any other number, the calculator in <a href="#pro">the Pro section</a>
prices your own monthly minutes plan by plan.</p>

<h2>Opus Clip vs OpenShorts on price</h2>
<table>
<thead><tr><th></th><th>Opus Clip</th><th>OpenShorts</th></tr></thead>
<tbody>
<tr><td>Free</td><td>60 credits/month, watermarked</td><td class="os">Self-hosted: unlimited, no watermark. Cloud: first video free up to 60 min, then 20 min/month, watermarked</td></tr>
<tr><td>Entry paid plan</td><td>$15/month for 150 minutes</td><td class="os">$12/month for 100 minutes, no watermark</td></tr>
<tr><td>Unit of billing</td><td>1 credit per source minute</td><td class="os">1 minute per source minute, no per-clip or per-call charge</td></tr>
<tr><td>Self-hosting</td><td>No</td><td class="os">Yes, MIT licence, Docker</td></tr>
</tbody>
</table>

<h2>What does OpenShorts cost?</h2>
${pricingParagraph('meter')}

${faqBlock(OPUS_PRICING_FAQ)}

${sources([
  `${esc(OPUS.name)} plans, trial and credit rules checked ${esc(OPUS.checked)} on opus.pro/pricing and the OpusClip help centre (how credits are consumed, free trial, watermark).`,
  `OpenShorts pricing from <a href="/alternatives/opus-clip">the full comparison</a> and the project source at <a href="${SITE.repo}" rel="noopener">github.com/mutonby/openshorts</a>.`,
])}
`,
  faq: OPUS_PRICING_FAQ,
})

const OPUS_PRICING_FAQ = [
  {
    q: 'How much does Opus Clip cost per month?',
    a: 'Starter is $15/month for 150 credits (minutes of source video). Pro is $29/month, or $14.50/month billed yearly, for 300 credits. There is a free plan with 60 credits a month and a custom-priced Business plan. Checked October 2026.',
  },
  {
    q: 'Is Opus Clip free?',
    a: 'It has a free plan: 60 minutes of source video a month, exports up to 1080p with a watermark, no editing, and clips must be exported within 3 days. There is also a 7-day Pro trial with no credit card, still watermarked.',
  },
  {
    q: 'Does Opus Clip have a free trial?',
    a: 'Yes, 7 days of Pro with no credit card. Trial exports carry the watermark and the trial excludes 4K, social posting and team seats. When it ends the account becomes a free account instead of charging you.',
  },
  {
    q: 'How much does Opus Pro cost?',
    a: '$29/month, or $14.50/month billed yearly ($174 a year), for 300 credits a month, 2 seats, 100 GB of storage, AI B-roll, Premiere and DaVinci export and limited API access. Checked October 2026.',
  },
  {
    q: 'Do I need Opus Pro or is Starter enough?',
    a: 'Starter is $15/month, monthly only, for 150 credits (minutes of source video). If you import more than 150 minutes a month, or want B-roll, seats or the API, Pro is the plan priced for you, and billed yearly it is cheaper than Starter.',
  },
  {
    q: 'What counts as a credit in Opus Clip?',
    a: 'One minute of source video you import, not one clip you export. A 60-minute episode consumes 60 credits regardless of how many clips you keep. Under a minute costs 1 credit, partial minutes round down, and credits on monthly plans expire after 60 days.',
  },
  {
    q: 'What is a cheaper alternative to Opus Clip?',
    a: 'OpenShorts: free and unlimited when self-hosted under MIT, or $12/month hosted for 100 minutes with no watermark. The hosted free plan clips your first video whole up to 60 minutes, then 20 minutes a month, with no credit card.',
  },
]

/* Page-specific sections a comparison page carries on top of the shared shape.
 * /alternatives/opus-clip absorbed the old /opus-clip-free-alternative page (now
 * a 301 here): "free opus clip alternative" is the same searcher as "opus clip
 * alternative", and two pages split the signal between them. */
const COMPETITOR_EXTRAS = {
  'opus-clip': () => ({
    replaceFreeQ: true,
    afterIntro: `
<h2>What "free" means at Opus Clip, and here</h2>
<p>Free is doing a lot of work in this category. Three different things are
being sold as free: a metered tier that watermarks, a trial that expires, and
software you run yourself that has no meter in it at all. Only the third one
stays free when your usage grows.</p>
<table>
<thead><tr><th>Route</th><th>Cost</th><th>Watermark</th><th>Cap</th></tr></thead>
<tbody>
<tr><td class="os">OpenShorts self-hosted</td><td class="os">$0</td><td class="os yes">Never</td><td class="os">None, no metering code</td></tr>
<tr><td class="os">OpenShorts Cloud free</td><td class="os">$0</td><td class="os">Yes</td><td class="os">First video up to 60 min, then ${CLOUD.freeMinutes} min/month</td></tr>
<tr><td>Opus Clip free</td><td>$0</td><td>Yes</td><td>60 min/month, export within 3 days</td></tr>
<tr><td>Opus Clip Pro trial</td><td>$0 for 7 days</td><td>Yes</td><td>Pro allowance, no 4K or posting</td></tr>
</tbody>
</table>
<p class="checked">Opus Clip terms checked ${esc(OPUS.checked)}; OpenShorts
Cloud terms are ours and current.</p>

<h3>The first free route: run it yourself</h3>
<p>Clone the repository, run <code>docker compose up --build</code>, add a
Google Gemini API key (its free tier covers 1,500 requests a day) and paste a
link. Nothing is metered because there is no metering code: the same pipeline
the hosted service runs, MIT-licensed, on your hardware. Source video never
leaves the machine, which is the other reason people choose this route. What it
costs you instead: Docker, 8GB of RAM as a realistic floor, and time. An
8-minute video takes roughly 5 to 8 minutes to process on CPU and about 50
seconds on an NVIDIA GPU. For a weekly podcast that is a coffee break; for
twenty videos a day it is a job.</p>

<h3>The second free route: the hosted free plan</h3>
<p>If you would rather not run anything, ${esc(CLOUD.name)} clips your first
video whole up to 60 minutes, then gives you ${CLOUD.freeMinutes} minutes a
month, with a watermark and no credit card. It takes a YouTube link or an
upload. Paid plans from $${CLOUD.lowPrice}/month drop the watermark, including
from the clips you already made on the free plan.</p>

<h3>When paying is the honest answer</h3>
<p>If you process hours of source video every week and have no machine to spare,
a flat plan is cheaper than either free tier is convenient. What is worth
avoiding is paying a per-minute credit meter for long sources: a 60-minute
episode consumes 60 credits at Opus Clip no matter how many clips you keep, and
a weekly show at that length runs past Starter's 150 credits by the third
episode of the month.</p>
`,
    faq: [
      {
        q: 'Is there a free alternative to Opus Clip with no watermark?',
        a: 'Yes: OpenShorts self-hosted. It is MIT-licensed, runs on your own machine with Docker, and never adds a watermark because the self-hosted edition contains no watermark code. You supply a Google Gemini API key, whose free tier covers 1,500 requests a day. Hosted, OpenShorts Cloud clips your first video free up to 60 minutes and then 20 minutes a month with a watermark; paid plans from $12/month have none.',
      },
      {
        q: 'Is Opus Clip free to use every month?',
        a: 'Yes, within its free plan: 60 minutes of source video a month, exports up to 1080p with a watermark, no editing, and clips must be exported within 3 days. The 7-day Pro trial needs no credit card, but its exports are watermarked too.',
      },
      {
        q: 'What are the best OpusClip alternatives?',
        a: `Opus Clip (also searched as Opus Clips or Opus Clip AI) has a few, each better at one thing, all checked ${monthYear(OPUS.checked)}. OpenShorts, if you want it open source and free to self-host, or hosted from $${CLOUD.lowPrice}/month. Vizard, if you edit every clip by hand in a timeline (from ${COMPETITORS.vizard.entryPrice}). Klap, for the fastest link-to-clips path, billed per clip (from ${COMPETITORS.klap.entryPrice}). Submagic, if caption design is the product (from ${COMPETITORS.submagic.entryPrice}). Choppity, if you want to trim clips by editing their transcript.`,
      },
      {
        q: 'What is the catch with the free self-hosted route?',
        a: 'Hardware and time, not a hidden fee. It needs Docker and realistically 8GB of RAM, and an 8-minute video takes 5 to 8 minutes to process on CPU (about 50 seconds on an NVIDIA GPU). There is no cap, no watermark and no subscription.',
      },
    ],
  }),
}

const opusAi = () => ({
  path: '/opus-ai',
  facts: [
    { k: 'Made by', v: 'OpusClip Inc.', s: 'At opus.pro; also makes Agent Opus' },
    { k: 'Free plan', v: '60 min/mo', s: 'Watermarked; plus a 7-day Pro trial' },
    { k: 'Paid from', v: '$15/mo', s: 'Starter; Pro $14.50/mo billed yearly' },
    { k: 'Max source', v: '10 hours', s: '25+ languages, most in beta' },
  ],
  title: 'Opus AI (OpusClip): What It Is, Pricing, Is It Free?',
  description:
    'Opus AI is OpusClip, the AI video clipper at opus.pro. Yes, it has a free plan: 60 minutes a month, watermarked. Paid from $15/month. What it does, explained.',
  h1: 'Opus AI (OpusClip): what it is, what it costs, and whether it is free',
  lede: brandLede(OPUS, 'opus.pro and its pricing page',
    `Opus AI is OpusClip, from $15/month. Cheaper tools that do the same job: OpenShorts (open source; $0 self-hosted, $${CLOUD.lowPrice}/month hosted for ${STARTER.monthly.minutes} minutes), Vizard ($14.50/month billed yearly for 600 minutes) and Klap (from $14/month billed yearly for 100 clips).`),
  relatedTitle: 'Related comparisons',
  breadcrumb: [
    { name: 'Alternatives', path: '/alternatives' },
    { name: 'Opus Clip', path: '/alternatives/opus-clip' },
    { name: 'Opus AI' },
  ],
  published: '2026-09-17',
  updated: REVISED,
  pasteLink: true,
  tldr: [
    'Opus AI is OpusClip: the AI video clipping tool at opus.pro, made by OpusClip Inc., that turns long videos into short vertical clips. It is free to start, with a free plan of 60 credits (minutes of source video) a month and watermarked exports, plus a 7-day Pro trial.',
    `Paid plans start at $15/month (Starter), billed in credits per minute of source video rather than per clip. The same company also makes Agent Opus, a separate AI video agent, and paid credits work in both.`,
    'OpenShorts does the same core job (moment detection, 9:16 reframing, word-level subtitles) and differs on two axes that matter: it is MIT-licensed and self-hostable, and it prices in flat minutes rather than per-minute credits.',
  ],
  body: `
<h2>Is Opus AI the same thing as Opus Clip?</h2>
<p>Yes, in almost every search. The clipping product is OpusClip, the company is
OpusClip Inc., and the site is <code>opus.pro</code>; "Opus AI", "Opus Clip AI"
and "Opus Clips" are the other names people type for the same clipper. The same company runs a second product under the Opus name,
<strong>Agent Opus</strong> (agent.opus.pro), an AI video agent with its own
Free, Pro and Max plans. Paid credits work in both, so if you pay for one you can
spend in the other.</p>
<div class="note"><span class="label">A third Opus</span><p>"Opus" is also the
name of an Anthropic language model. That is a different thing entirely and has
nothing to do with video clipping. This page is about the video tools at
opus.pro.</p></div>

<h2>What OpusClip does</h2>
<p>It takes a long video, finds the segments worth keeping, cuts them, reframes
them and burns in animated captions. The features its users cite:</p>
${li([
  'ClipAnything, which picks moments from visual, audio and sentiment cues, so it works on footage with little dialogue, and takes a prompt for what to look for.',
  'ReframeAnything (alpha), which tracks a subject and reframes to 9:16, 1:1 or 16:9.',
  'A virality score from 0 to 99 built from hook, flow, value and trend, shown on paid plans only.',
  'Animated captions in 20+ languages, AI B-roll, dubbing, a scheduler, and an API on Pro and Business.',
])}
<p>It is cloud only: there is no self-hosted edition and no source to read. It
accepts sources up to 10 hours long.</p>

<h2>How the workflow compares</h2>
${flowCompare(
  {
    title: 'Opus AI (OpusClip)',
    steps: [
      'Upload a file or paste a YouTube link',
      'ClipAnything proposes clips with a virality score',
      { text: 'Choose the clips worth keeping' },
      { text: 'Adjust captions, B-roll and framing' },
      'Export, or schedule to connected accounts',
    ],
  },
  OS_LANE
)}

<h2>Is Opus AI free?</h2>
<p>Yes, to a point. The free plan gives 60 credits a month, one per minute of
source video, with 1080p exports that carry a watermark, no editing and a 3-day
window to export. There is also a 7-day Pro trial with no credit card, still
watermarked. Removing the watermark starts at Starter, $15/month.</p>

<h2>What Opus AI costs</h2>
<p>In one line: Free, Starter at $15/month, Pro at $29/month or $14.50/month
billed yearly, and a custom-priced Business plan, all billed in credits per
minute of source video imported. Every plan, the trial, how a credit is rounded
and what the Pro plan adds are on the
<a href="/opus-clip-pricing">Opus Clip pricing page</a>.</p>

<h2>Where OpenShorts differs</h2>
${li([
  'OpenShorts is MIT-licensed and can be self-hosted with Docker, so the source video never leaves your machine. Opus is cloud only.',
  'OpenShorts self-hosted has no meter of any kind. Opus bills credits per minute of source imported, and monthly credits expire after 60 days.',
  'OpenShorts Cloud starts at $12/month for 100 minutes with no watermark; the hosted free plan clips your first video free up to 60 minutes, then 20 minutes a month.',
  'Opus has the larger caption-style library and a longer track record. If your clips live or die on animated caption design, that advantage is real.',
])}

<h2>What the open source alternative produces</h2>
${demoBlock()}

<h2>What does OpenShorts cost?</h2>
${pricingParagraph('brand')}

${faqBlock(OPUS_AI_FAQ)}

${sources([
  `OpusClip plans, features and limits checked ${esc(OPUS.checked)} on opus.pro, opus.pro/pricing and help.opus.pro; Agent Opus credits on the Agent Opus credits FAQ.`,
  `OpenShorts pipeline in the project source at <a href="${SITE.repo}" rel="noopener">github.com/mutonby/openshorts</a>.`,
])}
`,
  faq: OPUS_AI_FAQ,
})

const OPUS_AI_FAQ = [
  {
    q: 'Is Opus AI the same as Opus Clip?',
    a: 'Yes. OpusClip is the product, OpusClip Inc. the company and opus.pro the site; "Opus AI" is the shorter name people search. The same company also makes Agent Opus, a separate AI video agent whose paid credits are shared with OpusClip.',
  },
  {
    q: 'Is Opus AI free?',
    a: 'There is a free plan: 60 minutes of source video a month with watermarked exports, no editing, and a 3-day window to export. There is also a 7-day Pro trial with no credit card. Paid plans start at $15/month. OpenShorts self-hosted is free with no watermark and no cap; OpenShorts Cloud is hosted from $12/month.',
  },
  {
    q: 'What is Agent Opus?',
    a: 'An AI video agent from the makers of OpusClip, at agent.opus.pro, with its own Free, Pro and Max plans. Free comes with a one-time grant of 60 credits; Pro includes 300 credits a month and Max 1,500. Paid credits can be spent in either product.',
  },
  {
    q: 'Does Opus AI have an open source alternative?',
    a: 'Yes. OpenShorts is MIT-licensed, self-hostable with Docker, and covers the same core job: AI moment detection, face-tracked 9:16 reframing and word-level burned-in subtitles, plus dubbing into 30+ languages with your own ElevenLabs key.',
  },
]

/* "vizard ai" is a brand query with more volume than any non-brand term in the
 * category and almost no competition for it (KD 6, checked 2026-10-05). The
 * comparison page answers "alternative to"; this one answers the question the
 * searcher actually typed: what is it, what does it cost, is it free, and is
 * there something that fits me better. */
const vizardAi = () => {
  const c = COMPETITORS.vizard
  const r = c.reviewPage
  const brandQ = (q) => c.brandFaq.find((f) => f.q === q)
  const faq = [
    brandQ('Is Vizard AI free?'),
    ...c.reviewFaq,
    brandQ('What is Vizard AI used for?'),
    {
      q: 'How much does Vizard AI cost?',
      a: 'Creator is $29/month, or $14.50/month billed yearly, from 600 credits a month with 4K exports and no watermark. Business is $39/month, or $19.50/month billed yearly, with a shared workspace and team seats at $5/month each. One credit is one minute of uploaded video. Checked October 2026.',
    },
    {
      q: 'Does Vizard AI have an API?',
      a: 'Yes. The API is included in every paid plan and draws from the same credits as the web app, with videos up to 600 minutes and 10 GB. Above 10,000 minutes a month it is a sales conversation.',
    },
  ]
  return {
    path: '/vizard-ai',
    title: 'Vizard AI Review 2026: Pricing, Free Plan, Limits',
    description:
      'Vizard AI review, October 2026: what the clipper does, the free plan (60 credits, 720p, watermark), Creator from $14.50/month, pros, cons and the verdict.',
    h1: 'Vizard AI review: what it does, what it costs, and its limits',
    lede: brandLede(c, r.lede,
      `The main open source alternative to Vizard is OpenShorts: MIT-licensed and free to self-host with no watermark or cap, or hosted from $${CLOUD.lowPrice}/month for ${STARTER.monthly.minutes} minutes. Vizard is cheaper per minute at volume (600 minutes for $29/month, $14.50 billed yearly); OpenShorts is cheaper to start and the only one you can run on your own machine.`),
    breadcrumb: [
      { name: 'Alternatives', path: '/alternatives' },
      { name: 'Vizard', path: '/alternatives/vizard' },
      { name: 'Vizard AI review' },
    ],
    published: '2026-10-05',
    updated: REVISED,
    pasteLink: true,
    relatedTitle: 'Related comparisons',
    cta: {
      label: 'Compare on your own video',
      title: 'Clip the same video here',
      body: 'Paste the link you tried in Vizard and compare the clips. First video free up to 60 minutes, no credit card.',
      button: 'Get free clips',
    },
    tldr: [
      'Vizard AI is the browser-based video clipper at vizard.ai. It transcribes a long video, proposes the moments worth posting, reframes them to vertical and hands you a timeline to fix captions and cut points before you export.',
      'Free plan: 60 credits a month (one per uploaded minute), 720p exports up to 10 minutes, watermarked, kept 3 days. Paid: Creator at $29/month, or $14.50/month billed yearly, from 600 credits.',
      'Verdict: a good editor with fair yearly pricing, for people who want to touch every clip. If you want clips without the timeline, or to self-host, look at an unattended clipper.',
    ],
    facts: [
      { k: 'Free plan', v: '60 min/mo', s: '720p, watermark, kept 3 days' },
      { k: 'Creator', v: '$29/mo', s: '$14.50/mo yearly, from 600 credits' },
      { k: 'Workflow', v: 'Timeline', s: 'AI pass, then you edit each clip' },
      { k: 'API', v: 'Paid plans', s: 'Same credits as the web app' },
    ],
    body: `
<h2>What is Vizard AI?</h2>
<p>${esc(c.brandBlurb)}</p>
<p>The features it leads with: AI clips, auto reframe, animated subtitles with
emoji and keyword highlighting, subtitle translation, AI B-roll, post
suggestions, and publishing or scheduling to connected social accounts.</p>

<h2>How Vizard AI works</h2>
${ol(r.steps.map(esc))}
<p>The step that defines the product is the fourth one. Vizard puts a person in
the timeline by design, which is the reason to pick it and the reason not to.</p>

<h2>Vizard AI pricing in ${esc(c.checked.slice(0, 4))}</h2>
<table>
<thead><tr><th>Plan</th><th>What it includes</th></tr></thead>
<tbody>${OS_TIER_ROW}${c.tiers.map(([n, d]) => `<tr><td class="os">${esc(n)}</td><td>${esc(d)}</td></tr>`).join('')}</tbody>
</table>
<p>One credit is one minute of video uploaded, whatever comes out of it. The
yearly toggle halves the price, so the $29 Creator plan is $14.50/month on a
yearly commitment, and on yearly billing it is one of the cheapest per-minute
plans in the category up to its 600 credits.</p>

<h2>Is Vizard AI free?</h2>
<p>There is a free plan, and it is for trying the tool rather than publishing
with it: 60 minutes of source a month is one podcast episode, exports stop at
720p and 10 minutes, every export carries the watermark, and files are kept for
3 days. The transcript downloads as TXT only; SRT subtitles need a paid plan. If
the transcript is what you came for, the
<a href="/vizard-ai-video-to-text">Vizard AI video to text comparison</a> covers
that case on its own.</p>

<h2>Pros and cons</h2>
<h3>What it does well</h3>
${li(c.strengths.map(esc))}
<h3>Where it falls short</h3>
${li(r.cons.map(esc))}

<h2>Who Vizard AI is for</h2>
${li(r.for.map(esc))}
<h3>Who should look elsewhere</h3>
${li(r.notFor.map(esc))}

<h2>The verdict</h2>
<p>${esc(r.verdict)}</p>
<p>If that describes you, the <a href="/alternatives/vizard">open source Vizard
AI alternative</a> page covers the switch: a feature-by-feature table, a cost
calculator, when to stay and how to move. In short, OpenShorts runs the clipping
unattended and can be self-hosted under MIT.</p>
${pricingParagraph('brand')}

${faqBlock(faq)}

${sources([
  `Vizard plans, free-plan limits and transcript formats checked ${esc(c.checked)} on vizard.ai/pricing and vizard.ai/tools/video-to-text; API limits on docs.vizard.ai.`,
  'Pros, cons and fit are our reading of the published plans and the product as offered; we do not publish an aggregate star rating for a competitor.',
])}
`,
    faq,
  }
}

/* "vizard ai video to text" is its own intent: the searcher wants a transcript,
 * not clips, and lands on clipper pages that never answer it. The honest answer
 * is that both tools produce the transcript: OpenShorts with word-level timing
 * from faster-whisper, which is what the burned-in captions are cut from. */
const videoToText = () => {
  const c = COMPETITORS.vizard
  return {
    path: '/vizard-ai-video-to-text',
    facts: [
      { k: 'Vizard free', v: 'TXT only', s: 'SRT subtitles need a paid plan' },
      { k: 'Languages', v: '180+', s: "Vizard's claim for transcripts" },
      { k: 'OpenShorts', v: 'Word-level', s: 'A timestamp per word, free self-hosted' },
      { k: 'No sign-up', v: 'Free tool', s: 'Our YouTube transcript generator, TXT or SRT' },
    ],
    title: 'Vizard AI Video to Text: Transcripts, Compared | OpenShorts',
    description:
      'Vizard AI turns a video into text: a transcript, subtitles and clips. What that costs per minute, and how to get a word-level transcript free by self-hosting.',
    h1: 'Vizard AI video to text: what you get, and what it costs',
    breadcrumb: [
      { name: 'Alternatives', path: '/alternatives' },
      { name: 'Vizard', path: '/alternatives/vizard' },
      { name: 'Video to text' },
    ],
    published: '2026-09-17',
    updated: c.checked,
    tldr: [
      `Vizard's "video to text" is a transcript with timestamps from the same pass that finds the clips. Paste a YouTube link or upload a file; Vizard says it covers 180+ languages. The free plan downloads the transcript as TXT only; SRT needs a paid plan, from ${esc(c.entryPrice)}.`,
      'A transcript is a by-product of the transcription stage every clipper already runs, which is why no tool charges for it separately and why it is not worth choosing a tool over.',
      'OpenShorts transcribes with faster-whisper at word level and returns the transcript alongside the clips, subtitles included, free when self-hosted and from $12/month hosted.',
    ],
    cta: {
      label: 'Transcript included',
      title: 'Get the transcript and the clips',
      body: 'Word-level timestamps, burned-in captions and the clip list, from one pasted link. 20 free minutes a month.',
      button: 'Get free clips',
    },
    body: `
<h2>What "video to text" means in a clipping tool</h2>
<p>Three different outputs get described as video to text, and they are not
interchangeable. A <strong>transcript</strong> is the words with timestamps. A
<strong>subtitle file</strong> is the same words shaped for playback, cut into
readable lines. A <strong>summary or article</strong> is a written document
generated from the words, which is a language-model task sitting on top of the
transcript rather than a transcription task at all.</p>
<p>When a clipping tool advertises video to text, it means the first two. They
fall out of the transcription pass the tool has to run anyway to find the good
moments, which is why they are bundled rather than sold.</p>

<h2>What Vizard gives you</h2>
<p>Vizard runs the whole thing in the browser and treats the timeline as the
product: it transcribes the upload, lets you edit the captions and the clip
boundaries on a timeline, and exports both the clips and the text. Multi-language
subtitles and subtitle translation are among its stronger features, and its
video-to-text tool says it transcribes 180+ languages (its API documentation
lists 36 languages for clipping). The free plan gives 60 credits a month, uploads
up to 60 minutes and the transcript as TXT only; the paid plans, from
${esc(c.entryPrice)}, add SRT.</p>
<p>${esc(c.gotcha)}</p>

<div class="note"><span class="label">Only need the text?</span><p>If the video is on YouTube and already has captions,
the free <a href="/youtube-transcript-generator">YouTube transcript generator</a> gives you the transcript
as TXT or SRT in seconds, with no account and no upload.</p></div>

<h2>Doing the same thing with OpenShorts</h2>
<p>OpenShorts transcribes with faster-whisper and keeps a timestamp for every
word, not every sentence. Word-level timing is what makes the subtitle file
land on the right frame and what lets the clip boundaries fall inside a sentence
instead of at the nearest one. The transcript is returned with the finished job
next to the clips, and the burned-in captions are cut from it.</p>
<ul>
<li><strong>Self-hosted:</strong> free, MIT-licensed, no watermark, no cap. Transcription runs locally, so the audio never leaves your machine.</li>
<li><strong>Hosted:</strong> 20 free minutes a month with no credit card, then flat plans from $12/month with no watermark.</li>
<li><strong>Via API or MCP:</strong> the transcript and the clips come back from the same job, so an agent can summarise the text without a second transcription bill.</li>
</ul>

<h2>Which one should you pick?</h2>
<p>If you want to hand-correct captions on a timeline before exporting, Vizard's
editor is the better fit and it is not close. If you want the text as a
by-product of clipping at volume, or you need the transcript to stay on your own
machine, self-hosted OpenShorts is free and the transcript comes with the job.</p>

${faqBlock([
  {
    q: 'Does Vizard AI convert video to text?',
    a: `Yes. It transcribes the video and gives you the text alongside the clips and subtitles, from a YouTube link or an upload. The free plan (60 credits a month) downloads it as TXT; paid plans, from ${c.entryPrice}, add SRT.`,
  },
  {
    q: 'How do I get a free transcript from a video?',
    a: 'Self-host OpenShorts: transcription runs locally with faster-whisper at word level and the transcript comes back with the finished job, at no cost and with no watermark. The self-hosted edition needs Docker and a Google Gemini API key for the moment scoring, whose free tier covers 1,500 requests a day.',
  },
  {
    q: 'Is word-level timing important in a transcript?',
    a: 'For subtitles, yes. Sentence-level timestamps force captions to appear for the whole sentence at once, which is where the timing drifts out of sync with the speech. Word-level timestamps let each caption start and end on the word being spoken, and they are also what lets a clip boundary land mid-sentence without cutting a word in half.',
  },
])}

${sources([
  `Vizard plan limits checked ${esc(c.checked)} on the vendor's public pricing page.`,
  `OpenShorts transcription and subtitle stages in the project source at <a href="${SITE.repo}" rel="noopener">github.com/mutonby/openshorts</a>.`,
])}
`,
    faq: [
      {
        q: 'Does Vizard AI convert video to text?',
        a: `Yes, alongside the clips. The free plan gives 60 credits a month and a TXT transcript; paid plans, from ${c.entryPrice}, add SRT.`,
      },
      {
        q: 'Is there a free way to turn a video into text?',
        a: 'Yes: self-hosted OpenShorts transcribes locally with faster-whisper at word level and returns the transcript with the job, free, with no watermark and no cap.',
      },
    ],
  }
}

/* Reviews intent for a competitor. Written as product facts plus the recurring
 * themes in public reviews: no invented quotes and no star rating, because a
 * rating we cannot verify is exactly the kind of thing this site refuses to
 * publish elsewhere. */
const submagicReview = () => {
  const c = COMPETITORS.submagic
  const tierRows = c.tiers.map(([n, d]) => `<tr><td class="os">${esc(n)}</td><td>${esc(d)}</td></tr>`).join('')
  const faq = [
    {
      q: 'Is Submagic worth it?',
      a: 'For captions, yes: it is the strongest caption styler in this category, with B-roll, zooms and silence removal around it. For clipping long videos it is a newer feature (Magic Clips) inside a tool metered per video, with each video capped at 2, 5 or 30 minutes depending on the plan, so check your source length against the plan before you buy.',
    },
    {
      q: 'Can Submagic clip long videos?',
      a: 'Yes, with Magic Clips: paste a YouTube link or upload a long video and it returns short clips. Its pricing page does not publish a separate Magic Clips allowance, while every plan caps the length of each video (2 minutes on Starter, 5 on Pro, 30 on Business).',
    },
    {
      q: 'Does Submagic have a free plan?',
      a: 'No. It offers a trial with no credit card, then plans from $12/month billed yearly ($19 month to month). OpenShorts has a free plan (first video up to 60 minutes, then 20 minutes a month) and is free with no cap when self-hosted.',
    },
    {
      q: 'What is a free alternative to Submagic?',
      a: 'OpenShorts, self-hosted: MIT-licensed, free, no watermark and no cap, with moment detection, 9:16 reframing and word-level burned-in captions in the same pipeline. Its caption presets are plainer than Submagic\'s; that is the honest trade.',
    },
  ]
  return {
    path: '/submagic-reviews',
    facts: c.facts,
    title: 'Submagic Review 2026: Pricing, Captions & Alternative',
    description:
      'Submagic review, October 2026: top-tier captions, Magic Clips, no free plan, from $12/month yearly ($19 monthly), per-video caps, and an alternative.',
    h1: 'Submagic review 2026: pricing, captions, Magic Clips and the catch',
    lede: checkedLede(c, 'submagic.co/pricing'),
    relatedTitle: 'Related comparisons',
    breadcrumb: [
      { name: 'Alternatives', path: '/alternatives' },
      { name: 'Submagic', path: '/alternatives/submagic' },
      { name: 'Review' },
    ],
    published: '2026-09-17',
    updated: c.checked,
    tldr: [
      'The recurring theme across public reviews is the same one the product page leads with: the caption styling is the best in this category, and the presets are why people stay.',
      'Submagic now also clips long videos with Magic Clips. The catch is the meter: every plan counts videos and caps each one\'s length (2, 5 or 30 minutes), and there is no free plan, only a trial.',
      'OpenShorts is built the other way round, clipping-first: moment detection, 9:16 reframing with two-speaker and screencast layouts, and word-level captions, free when self-hosted and from $12/month hosted. Its caption designs are plainer than Submagic\'s, and that is the real trade.',
    ],
    body: `
<h2>What Submagic is for</h2>
<p>Submagic started as a caption styler and is still built around captions. You
bring a video, it transcribes it and burns in animated, well-designed captions
with emoji and keyword highlighting, and around that it adds B-roll, auto zooms,
hook titles and silence and bad-take removal. Since then it has added
<strong>Magic Clips</strong>, which takes a long video or a YouTube link and
returns short clips, so it is no longer only the second half of a clipping
workflow.</p>

<h2>What the reviews consistently praise</h2>
<p>The caption library and the speed on short inputs come up in almost every
public review. If caption design is the reason you are shopping, the
recommendation is straightforward and it is not ours.</p>

<h2>How the workflow compares</h2>
${flowCompare(
  {
    title: 'Submagic',
    steps: [
      'Upload a video or paste a link',
      'Captions transcribed and styled',
      'Optional: Magic Clips cuts a long video into clips',
      { text: 'Pick a template, B-roll and zooms' },
      { text: 'Export each video' },
    ],
  },
  OS_LANE
)}

<h2>Where it is weaker</h2>
<p>${esc(c.gotcha)}</p>
<p>The clipping side is younger than the captions. Submagic does not publish how
Magic Clips picks moments or how it reframes multi-person footage, and it cannot
be self-hosted, so a per-video meter is the only way to buy it.</p>

<h2>Submagic pricing in ${esc(c.checked.slice(0, 4))}</h2>
<p class="checked">Checked ${esc(c.checked)} on the vendor's public pricing page. Vendors change tiers without notice.</p>
<table>
<thead><tr><th>Plan</th><th>What it includes</th></tr></thead>
<tbody>${tierRows}</tbody>
</table>
<p>Every tier is metered in videos per month with a cap on each video's length.
For a podcast that is the number to check first: on the self-serve plans only
Business allows a 30-minute video.</p>

<h2>Is Submagic free?</h2>
<p>No. There is no free plan, only a trial with no credit card. The cheapest
plan is Starter at $12/month billed yearly, or $19 month to month, for 15 videos
of up to 2 minutes each, without a watermark.</p>

<h2>What a clipping-first tool produces</h2>
<p>For comparison, a real OpenShorts clip next to its source:</p>
${demoBlock()}

<h2>The verdict</h2>
<p>Submagic is the right buy if captions are your product and your videos are
short. If you are starting from long recordings, compare what Magic Clips does
with your footage against a clipping-first tool before committing to a yearly
plan. OpenShorts does the clipping in depth and loses on exactly the axis
Submagic wins on: caption design. The side-by-side table, when to stay and how
to move are on the <a href="/alternatives/submagic">Submagic alternative
page</a>.</p>
<div class="note"><span class="label">On star ratings</span><p>We do not publish
an aggregate score for a competitor. The numbers on the software directories move
monthly and we would be quoting a snapshot as if it were a fact. Check them at the
source if a rating is what you want; the product description above does not
depend on one.</p></div>

${faqBlock(faq)}

${sources([
  `Submagic plans and limits checked ${esc(c.checked)} on submagic.co/pricing; Magic Clips on submagic.co/features/magic-clips.`,
  `OpenShorts pipeline stages in the project source at <a href="${SITE.repo}" rel="noopener">github.com/mutonby/openshorts</a>.`,
])}
`,
    faq,
  }
}

/* Reading order for the comparison cluster, used by the Spanish index and by
 * relatedFor's blurbs. Kept next to the pages it describes so a new comparison
 * has one obvious place to register. */
const COMPARISON_INDEX = [
  {
    path: '/alternatives/opus-clip',
    title: 'Alternativa gratuita a Opus Clip',
    blurb: 'Las dos vías gratuitas frente a un plan gratuito con marca de agua, y dónde gana cada uno.',
  },
  {
    path: '/opus-clip-pricing',
    title: 'Cuánto cuesta Opus Clip',
    blurb: 'Qué es un crédito, si es gratis, qué compra el plan Pro de $29/mes y cada plan de 2026.',
  },
  {
    path: '/opus-ai',
    title: 'Opus AI (opus.pro)',
    blurb: 'Qué es realmente el nombre «Opus AI», si es gratis y qué cuesta la herramienta.',
  },
  {
    path: '/alternatives/vizard',
    title: 'Vizard, comparado',
    blurb: 'Editor en línea de tiempo tras el paso de IA, y a quién le hace falta.',
  },
  {
    path: '/vizard-ai',
    title: 'Análisis de Vizard AI',
    blurb: 'El plan gratuito (60 créditos, 720p, marca de agua), los planes de pago y la API.',
  },
  {
    path: '/vizard-ai-video-to-text',
    title: 'Vizard AI: vídeo a texto',
    blurb: 'Transcripción, subtítulos y clips: qué es cada cosa y cuánto cuesta.',
  },
  {
    path: '/alternatives/klap',
    title: 'Klap, comparado',
    blurb: 'La vía más rápida de URL a clip, y lo que se pierde por el camino.',
  },
  {
    path: '/alternatives/submagic',
    title: 'Submagic, comparado',
    blurb: 'Subtítulos primero; ahora también recorta vídeos largos con Magic Clips.',
  },
  {
    path: '/submagic-reviews',
    title: 'Análisis de Submagic',
    blurb: 'Subtítulos de primera, Magic Clips y un medidor por vídeo con límite de duración.',
  },
  {
    path: '/alternatives/vidyo-ai',
    title: 'Vidyo.ai, ahora Quso',
    blurb: 'Qué cambió con el cambio de nombre, sus planes por créditos y la alternativa abierta.',
  },
  {
    path: '/alternatives/2short',
    title: '2short AI, comparado',
    blurb: 'El más barato por hora, sin marca de agua en el gratuito, pero solo desde enlaces.',
  },
  {
    path: '/alternatives/sendshort',
    title: 'SendShort, comparado',
    blurb: 'Suite de vídeo corto: el recorte de vídeos largos empieza en el plan de $29.',
  },
  {
    path: '/alternatives/choppity',
    title: 'Análisis de Choppity',
    blurb: 'Una hora gratis al mes con su logo, planes por horas desde $24 y la alternativa abierta.',
  },
  {
    path: '/alternatives/getmunch',
    title: 'GetMunch, ahora Munch Studio',
    blurb: 'Ya no es solo un clipper: marketing en redes con IA, sin plan gratuito, desde $38/mes anual.',
  },
]

/* Spanish index of the comparison cluster. "alternativas a <tool>" is a live
 * Spanish query family and the site already publishes Spanish legal pages, so
 * this is a new surface rather than a duplicate of /alternatives: the English
 * hub is a five-tool pricing table, this one is a reading order with a line on
 * what each comparison answers. */
const alternativasIndex = () => {
  const cards = COMPARISON_INDEX.map(
    (c) =>
      `<a href="${esc(c.path)}"><strong>${esc(c.title)}</strong><span>${esc(c.blurb)}</span></a>`
  ).join('')
  return {
    path: '/alternativas',
    lang: 'es',
    facts: [
      { k: 'Herramientas', v: '10', s: 'OpenShorts y 9 herramientas en la nube' },
      { k: 'Plan de pago más barato', v: '$9,90/mes', s: '2short.ai; OpenShorts Cloud $12 al mes' },
      { k: 'Código abierto', v: '1 de 10', s: 'Solo OpenShorts se puede autoalojar' },
      { k: 'Precios comprobados', v: 'Oct 2026', s: 'En la web de precios de cada uno' },
    ],
    title: 'Alternativas a Opus Clip, Vizard y Submagic | OpenShorts',
    description:
      'OpenShorts frente a Opus Clip, Vizard, Klap, Submagic, Quso (Vidyo.ai), 2short y SendShort: precios de octubre de 2026, planes gratuitos y en qué gana cada una.',
    h1: 'Alternativas de código abierto a las herramientas de clipping',
    breadcrumb: [{ name: 'Alternativas' }],
    published: '2026-09-17',
    updated: '2026-10-05',
    tldr: [
      'OpenShorts es la única herramienta de esta categoría con código abierto y autoalojable: MIT, se ejecuta con Docker en tu propia máquina y no lleva marca de agua ni límite de uso.',
      `Los planes de pago más baratos, comprobados el ${esc(OPUS.checked)}: 2short.ai $9,90/mes, OpenShorts Cloud $12/mes con pago mensual, Submagic $12, Klap $14 y Vizard $14,50 al mes con pago anual, y Opus Clip $15/mes. OpenShorts autoalojado cuesta $0.`,
      'Las herramientas no son equivalentes: Submagic es un editor de subtítulos que añadió recortes, Quso y SendShort son suites donde recortar es una función más y Vizard espera que edites en su línea de tiempo. Cada comparativa de abajo dice dónde gana de verdad.',
    ],
    cta: {
      label: 'Pruébalo',
      title: 'Pega un enlace y mira los clips',
      body: '20 minutos gratis al mes, sin tarjeta. O autoalojado, gratis para siempre y sin marca de agua.',
      button: 'Probar gratis',
    },
    body: `
<h2>Qué comparativa leer según lo que buscas</h2>
<p>Este índice agrupa todas las comparativas del sitio. Si vienes de una búsqueda
concreta, la lista de abajo está ordenada por la pregunta que responde cada
página, no por popularidad de la herramienta.</p>
<h2>Todas las comparativas</h2>
<div class="cluster">${cards}</div>

<h2>Precios de entrada, uno al lado del otro</h2>
<p class="checked">Precios comprobados el ${esc(OPUS.checked)}. Verifica en la web del proveedor antes de comprar.</p>
<table>
<thead><tr><th>Herramienta</th><th>Precio de entrada</th><th>Código abierto</th><th>Autoalojable</th></tr></thead>
<tbody>
<tr><td class="os">OpenShorts</td><td class="os">$0 autoalojado · $12/mes alojado</td><td class="yes">Sí, MIT</td><td class="yes">Sí, Docker</td></tr>
${ALTERNATIVES.map((slug) => `<tr><td>${esc(COMPETITORS[slug].seo?.breadcrumb || COMPETITORS[slug].name)}</td><td>${esc(COMPETITORS[slug].entryPrice)}</td><td>No</td><td>No</td></tr>`).join('')}
</tbody>
</table>

<h2>Qué cuesta OpenShorts</h2>
${pricingParagraph('es')}
<p>La diferencia práctica no es solo el precio: en la edición autoalojada no hay
medidor de ningún tipo, así que un vídeo de 90 minutos cuesta lo mismo que uno de
9, y el vídeo original nunca sale de tu máquina.</p>

${faqBlock([
  {
    q: '¿Cuál es la alternativa gratuita a Opus Clip?',
    a: 'OpenShorts autoalojado: licencia MIT, se ejecuta con Docker en tu máquina, sin marca de agua y sin límite de uso. Solo necesitas una clave de Google Gemini, cuyo plan gratuito cubre 1.500 peticiones al día. Si prefieres no instalar nada, OpenShorts Cloud da 20 minutos al mes con marca de agua y planes de pago desde $12/mes.',
  },
  {
    q: '¿Qué herramienta de clipping tiene código abierto?',
    a: 'OpenShorts, con licencia MIT y el código completo en GitHub. Opus Clip, Klap, Vizard, Submagic, Quso (antes Vidyo.ai), 2short y SendShort son productos comerciales de código cerrado que solo funcionan en la nube.',
  },
  {
    q: '¿Merece la pena cambiar de herramienta?',
    a: 'Depende de lo que más te moleste hoy. Si es el precio por minuto de vídeo de origen, el ahorro es real, sobre todo con episodios largos. Si es el diseño de los subtítulos, las herramientas comerciales siguen teniendo más presets y más pulidos, y eso no lo vamos a discutir.',
  },
], { lang: 'es' })}
`,
    faq: [
      {
        q: '¿Cuál es la alternativa gratuita a Opus Clip?',
        a: 'OpenShorts autoalojado: MIT, Docker, sin marca de agua y sin límite. Hosted: 20 minutos gratis al mes y planes desde $12/mes.',
      },
      {
        q: '¿Qué herramienta de clipping es de código abierto?',
        a: 'OpenShorts (MIT). Opus Clip, Klap, Vizard, Submagic, Quso, 2short y SendShort son de código cerrado y solo en la nube.',
      },
    ],
  }
}

/* Gaming is the biggest clip-producing category on the vertical platforms and
 * the worst served by this class of tool: the source is a multi-hour VOD, the
 * gameplay fills the whole 16:9 frame, and the only face on screen is a webcam
 * box in a corner that a centre crop throws away. This page is built on the two
 * code paths that actually address that (camera_inset and WIDE) plus the one
 * arithmetic fact that decides the category: per-minute credits against a
 * source measured in hours. It is deliberately not a re-telling of the generic
 * pipeline with the word "GTA" pasted over it.
 */
const gtaClips = () => ({
  path: '/gta-5-clips',
  title: 'GTA 5 Clips: Turn Stream VODs Into Shorts | OpenShorts',
  description:
    'Turn GTA 5 and GTA RP stream VODs into shorts: gameplay keeps its full width and the facecam is enlarged, not cropped out. Free self-hosted, no meter.',
  h1: 'Turn GTA 5 and GTA RP streams into vertical clips',
  breadcrumb: [{ name: 'Use cases', path: '/use-cases' }, { name: 'GTA 5 clips' }],
  published: '2026-09-15',
  updated: '2026-09-15',
  tldr: [
    'A GTA 5 stream is four to eight hours of 16:9 gameplay with a webcam box in one corner. OpenShorts reads the whole VOD, picks the moments worth posting out of what was said, and reframes each one so the gameplay keeps its full width and the facecam is enlarged underneath it instead of cropped away.',
    'Length is what makes this expensive everywhere else. Tools in this category bill one credit per minute of source you import, so a single eight-hour stream is 480 minutes: more than the 300 minutes a $29/month Opus Clip Pro plan includes (checked 2026-07-27). Self-hosted OpenShorts has no meter at all; the hosted edition starts at $12/month.',
    'Gameplay with no commentary is handled too, and it is where most clippers stop: when a stream has no usable speech OpenShorts switches by itself to a vision pass where Gemini watches the footage and picks the moments, instead of failing on an empty transcript. The switch is automatic, and long silent captures are watched in stretches rather than rejected.',
  ],
  body: `
<h2>Why GTA clips break a normal auto-clipper</h2>
<p>Every auto-clipper in this category was designed around a talking head: one
person, centred, filling a 16:9 frame that crops cleanly to 9:16. A GTA stream
is the opposite on all three counts. The frame is gameplay, so a centre crop
keeps Los Santos and drops the minimap, the kill feed and the chat. The only
face is a small webcam box pinned to a corner, so a face tracker either ignores
it or, worse, latches onto a pedestrian NPC. And the source is not eight minutes
long, it is eight hours. Those are three different problems and each one has its
own answer below. (GTA 6, which launches on consoles only, has its own page:
<a href="/gta-6-clips">GTA 6 clips</a>.)</p>

<h2>How the webcam inset layout works</h2>
<p>The OBS layout almost every GTA streamer uses, gameplay full screen with the
camera composited into a corner, is a single video file with two things in it.
OpenShorts detects that geometrically rather than asking a model: it looks for a
subject that is <strong>small</strong>, <strong>off centre horizontally</strong>
and <strong>still between samples</strong>. All three filters are needed. A
talking head sitting high in frame is still centred, so size alone is not
enough, and a real person moves 300 pixels between samples where a pinned
webcam box moves three to eleven. On our 48-video test corpus that detector
found all five clips that had a webcam inset, with no false positives.</p>
<p>When it fires, the clip renders as INSET: the gameplay across the full width
at the top of the 9:16 frame, and the webcam box cropped out and blown up to
fill the bottom. You get the play and the reaction to the play, both legible on
a phone, instead of one of them at 100 pixels wide.</p>
<div class="note"><span class="label">Why not just ask the model</span>
<p>Offered as a fourth choice alongside the other layouts, Gemini answered
"screencast" on all five clips that had an inset, in two separate passes, and
overall layout accuracy fell from 92% to 83-85%. The geometry is a better judge
than the model here, so the detector runs after the layout decision rather than
inside it.</p></div>

<h2>When the gameplay itself is the point</h2>
<p>Not every moment has a face worth showing. A chase, a heist finale or a
five-car pileup means what it means across the whole width of the frame, and
cropping to a vertical column deletes the half that explains it. OpenShorts
measures how wide the meaningful content is and routes on that: content spanning
more than 85% of the frame renders as WIDE. A full-width strip in a vertical
frame is under a third of its height and unreadable on a phone, so WIDE zooms
about 1.3 times instead: roughly three quarters of the width, the crop slid
toward where the motion is (a still HUD or minimap never pulls it), with the
hook above the picture and the captions below it. Content that leaves room
beside it, a GTA RP scene playing out on one side of the screen for instance,
gets stacked over the presenter instead.</p>
<p>Width is the gate rather than coverage because width is what survives
measurement. A corner kill notification and a full-screen map both look "busy";
only one of them spans the frame and cannot be cropped.</p>

<h2>How to clip a GTA 5 stream, step by step</h2>
<ol>
<li>Paste the VOD link (a Twitch export, a YouTube upload) or drop the local recording in. Multi-hour sources are the normal case here, not the edge case.</li>
<li>faster-whisper transcribes with word-level timestamps and PySceneDetect maps the cuts, which is what keeps a clip from opening mid-explosion.</li>
<li>Gemini reads the transcript against those boundaries and returns the 3 to 15 segments that stand alone best, 15 to 60 seconds each.</li>
<li>Leave the vertical layout on <strong>auto</strong> (dashboard, advanced options; <code>"layouts": ["auto"]</code> on the API). Auto is what enables the screen layouts, and the inset detector is chained behind them.</li>
<li>Subtitles are burned in from the word-level transcript. On gaming feeds this is not optional polish: the clips autoplay muted.</li>
<li>Download the clips, or post them straight to TikTok, YouTube Shorts and Instagram Reels from the dashboard or the API.</li>
</ol>

<h2>What an eight-hour stream costs to clip</h2>
<p class="checked">Competitor terms checked 2026-07-27 on public pricing pages.</p>
<p>This is the arithmetic that decides the category, and it is worth doing
before you pick a tool. Credit-metered clippers bill one credit per minute of
the video you <em>import</em>, not per clip you keep. One eight-hour GTA RP
stream is 480 minutes. Opus Clip's free tier is 60 minutes a month, Starter is
150 minutes at $15/month and Pro is 300 minutes at $29/month, so a single
stream does not fit in any of them, and a streamer who goes live three times a
week is importing roughly 6,000 minutes a month. Gaming is the category where
per-minute pricing and the actual shape of the content are furthest apart.</p>
${pricingParagraph('gaming')}

<h2>What happens when nobody is talking</h2>
<p>Most of this page assumes commentary, because the default picker reads the
transcript: roleplay dialogue, heist banter and party voice chat are exactly
what it is good at, and reading words rather than frames is why an eight-hour
source costs about the same to analyse as an eight-minute one. A silent grind
has no transcript to read, so OpenShorts does not use one.</p>
<p>It switches paths on its own, and it does not need to be told to. Footage
with no audio track at all, and footage whose transcript comes back under 8
words or under 5 words per minute (music-only streams, a mic that was muted the
whole session), both trip the same branch: the video itself goes to Gemini,
which watches it and returns the same 3 to 15 moments in the same 15 to 60
second band as the transcript path. Everything downstream is identical, layouts
and inset detection included. The one difference is that the clips come out
without captions, which is correct rather than a bug: there is no speech to
caption.</p>
<p class="note"><span class="label">How long silent footage is handled</span>
This is the single stage that sends Gemini the footage instead of a handful of
frames, and Gemini counts video at roughly 300 tokens per second, so an hour of
gameplay is already around a million tokens. A long silent capture is therefore
watched in stretches of up to 40 minutes, each sent as a small low-resolution
copy, and the picks from every stretch are ranked together into one list. A job
watches up to eight stretches, a little over five hours; past that only the
first five hours are clipped. Minutes are billed on the length you import, so
for silent footage the session or segment you care about is still the cheaper
input. With commentary none of this applies, because the transcript path never
uploads the video at all.</p>

<h2>Whose footage can you clip?</h2>
<p>Yours, and footage you have permission for. Your own streams and recordings,
your RP server co-stars' VODs with their blessing, clients' channels you manage.
Two separate rights questions apply to GTA clips and they have different
answers: the <strong>recording</strong> belongs to whoever streamed it, and the
<strong>game footage</strong> is covered by Rockstar Games' policy on posting
its material. As of November 2026 that policy says Take-Two generally does not
object to non-commercial use that does not spoil the plot, defines
non-commercial as not making money through the footage, and reserves the right
to take material down. That is their policy and it can change, so check the
current version rather than taking this page's word for it. Reuploading
another streamer's clips without permission is the one case that is clearly not
fine, and the platforms strike it.</p>

${faqBlock([
  {
    q: 'How do I make GTA 5 clips for TikTok?',
    a: 'Paste the stream VOD link into OpenShorts with the vertical layout set to auto. It transcribes the whole recording, has Gemini pick the 3 to 15 strongest 15 to 60 second moments out of what was said, reframes each one to 9:16 keeping the gameplay full width with your facecam enlarged below it, and burns in word-level subtitles. Clips download or post straight to TikTok, Reels and Shorts.',
  },
  {
    q: 'Can it handle a whole eight-hour GTA RP stream?',
    a: 'Yes, long sources are the design case. Moment scoring reads the transcript rather than the raw video, so an eight-hour VOD does not degrade selection the way it degrades a frame-by-frame approach. Processing time scales with length: the GPU-backed hosted edition clips about 8 minutes of source in 50 seconds, and self-hosted on CPU it is roughly 5 to 8 minutes of processing per 8 minutes of source.',
  },
  {
    q: 'Does it keep my facecam in the clip?',
    a: 'Yes, when the layout picker is on auto. A webcam box composited into a corner is detected geometrically (small, off centre horizontally, static between samples) and the clip renders as INSET: gameplay at full width on top, the webcam cropped out and enlarged underneath, so both are legible on a phone.',
  },
  {
    q: 'Does it work on gameplay with no commentary?',
    a: 'Yes, and it switches by itself. Footage with no audio track, or whose transcript comes back under 8 words or under 5 words per minute, goes down a vision pass instead: Gemini watches the footage and returns the same 3 to 15 moments, with the same layouts and inset detection after it. The clips come out without captions, since there is no speech to caption. A long silent capture is watched in stretches of up to 40 minutes, up to a little over five hours per job; giving it only the session you care about costs fewer minutes.',
  },
  {
    q: 'Is it free for streamers?',
    a: 'Self-hosted OpenShorts is free and open source under MIT with no per-minute meter, which is the edition that makes sense when your sources are measured in hours: run it with Docker and bring your own Gemini API key. OpenShorts Cloud covers 20 minutes a month free with a watermark, and paid hosted plans start at $12/month.',
  },
])}

${sources([
  'Opus Clip tier minutes and prices checked 2026-07-27 on their public pricing page.',
  'Inset detection and layout accuracy figures are our own measurements on a 48-video internal corpus, 2026-08.',
  'Silent-footage thresholds (8 words, 5 words per minute) and the vision fallback are in <code>main.py</code>; Gemini video token rates from Google\'s published pricing.',
  `Inset, WIDE and screencast layout implementations in the project source at <a href="${SITE.repo}" rel="noopener">github.com/mutonby/openshorts</a>.`,
])}

${trademarkNote('Grand Theft Auto, GTA, Rockstar Games and Twitch', 'Rockstar Games, Take-Two Interactive or Twitch')}
`,
  faq: [
    {
      q: 'How do I make GTA 5 clips for TikTok?',
      a: 'Paste the stream VOD into OpenShorts with the vertical layout on auto: it transcribes the recording, picks the 3 to 15 strongest 15 to 60 second moments, reframes each to 9:16 keeping the gameplay full width with the facecam enlarged below, and burns in subtitles.',
    },
    {
      q: 'Can it handle a whole eight-hour GTA RP stream?',
      a: 'Yes. Moment scoring reads the transcript rather than the raw video, so multi-hour VODs are the design case. Self-hosted there is no per-minute meter, which matters when one stream is 480 minutes of source.',
    },
    {
      q: 'Does it keep my facecam in the clip?',
      a: 'Yes. A webcam box in a corner is detected geometrically and the clip renders with the gameplay full width on top and the facecam cropped out and enlarged underneath.',
    },
    {
      q: 'Does it work on gameplay with no commentary?',
      a: 'Yes. When a video has no audio track, or under 8 words of speech, OpenShorts switches automatically to a vision pass where Gemini watches the footage and picks the same 3 to 15 moments. Those clips have no captions, because there is no speech to caption.',
    },
  ],
  /* HowTo is emitted alongside the Article because the primary query here is a
   * procedure ("how to make GTA 5 clips"), and a procedure stated as steps in
   * the graph is the form an engine can lift whole. */
  extraNodes: [
    {
      '@type': 'HowTo',
      '@id': `${SITE.url}/gta-5-clips#howto`,
      name: 'How to turn a GTA 5 stream into vertical clips',
      description:
        'Turn a multi-hour GTA 5 or GTA RP stream VOD into vertical 9:16 clips for TikTok, YouTube Shorts and Instagram Reels, keeping the gameplay full width and the facecam visible.',
      totalTime: 'PT15M',
      supply: [{ '@type': 'HowToSupply', name: 'A GTA 5 stream VOD (link or local file) you have the rights to' }],
      tool: [{ '@type': 'HowToTool', name: 'OpenShorts (self-hosted with Docker, or OpenShorts Cloud)' }],
      step: [
        {
          '@type': 'HowToStep',
          name: 'Add the VOD',
          text: 'Paste the stream link or upload the local recording. Multi-hour sources are supported.',
        },
        {
          '@type': 'HowToStep',
          name: 'Set the vertical layout to auto',
          text: 'In advanced options choose the auto vertical layout, or send "layouts": ["auto"] on the API. Auto enables the screen layouts, and webcam inset detection is chained behind them.',
        },
        {
          '@type': 'HowToStep',
          name: 'Let the AI pick the moments',
          text: 'The VOD is transcribed with word-level timestamps and scanned for scene cuts, then Gemini scores the transcript and returns the 3 to 15 strongest segments of 15 to 60 seconds.',
        },
        {
          '@type': 'HowToStep',
          name: 'Review the reframed clips',
          text: 'Gameplay with a corner webcam renders as gameplay full width on top and the enlarged facecam below; full-frame action is zoomed around the motion, keeping roughly three quarters of the width. Subtitles are burned in from the word-level transcript.',
        },
        {
          '@type': 'HowToStep',
          name: 'Publish',
          text: 'Download the clips or post them directly to TikTok, YouTube Shorts and Instagram Reels from the dashboard or the API.',
        },
      ],
    },
  ],
})

/* Answer-first sections: each question as an H2 phrased the way people ask
 * it, a self-contained answer of about 40 to 60 words right under it (the
 * part an engine lifts), then the detail. The same question and answer feed
 * the FAQPage node, so the schema never says anything the page does not. */
const qaSections = (items) => items
  .map((i) => `<h2>${esc(i.q)}</h2>\n<p><strong>${esc(i.a)}</strong></p>\n${i.detail || ''}`)
  .join('\n\n')
const qaFaq = (items) => items.map(({ q, a }) => ({ q, a }))

/* Facts about consoles, capture tools and platforms quoted on the gaming
 * pages, each with the date it was read on the vendor's own page. */
const GAMING_CHECKED = '4 November 2026'
const PS5_CAPTURE_URL = 'https://www.playstation.com/en-us/support/games/capture-ps5-gameplay-screenshots/'
const PS5_APP_URL = 'https://www.playstation.com/en-us/support/games/ps5-game-captures-ps-app/'
const PS5_GALLERY_URL = 'https://www.playstation.com/en-us/support/games/ps5-media-gallery-view-screenshots/'
const XBOX_SHARE_URL = 'https://support.xbox.com/en-US/help/hardware-network/accessories/controller-share-button'
const XBOX_ONEDRIVE_URL = 'https://news.xbox.com/en-us/2023/09/27/bonus-xbox-update-for-september/'
const YT_SHORTS_URL = 'https://support.google.com/youtube/answer/12779649'
const REELS_URL = 'https://www.socialmediatoday.com/news/instagram-officially-expands-reels-length-3-minutes/737766/'

/* /gta-6-clips: written before launch (19 November 2026), so nothing on it is
 * about GTA 6 gameplay. What is new and true is where the footage comes from:
 * the game ships on PS5 and Xbox Series X|S only, so a large share of it will
 * be console captures with no facecam and often no voice, next to the usual
 * stream VODs. The page answers the questions people ask at launch (record on
 * a console, get the file off it, make it vertical, how long, will Rockstar
 * take it down), each with the vendor's own page cited and dated.
 */
/* Twitch Autopilot and the viewer-clips signal, shared by the gaming pages.
 * The rules are cloud/autopilot.py (DAILY_AUTO_LIMIT, MAX_VIDEO_AGE, is_live,
 * twitch_stretch) and twitch_signals.py (promote_hot_windows: up to a third of
 * the shortlist); if they change, change these. */
const TWITCH_AUTOPILOT_UPDATED = '2026-11-07'
const TWITCH_VIEWER_CLIPS = `When the link is a Twitch VOD, OpenShorts also
reads the clips your viewers made from that broadcast, when they were made and
how many views each got, and makes sure the stretches clipped most reach the
final pick (up to a third of the shortlist). The rest is still chosen from the
transcript, so a moment nobody clipped can still win.`
const TWITCH_AUTOPILOT_ANSWER = 'Yes, on the paid plans. Connect Twitch in the Autopilot tab and it checks for new past broadcasts about once an hour, waits until a stream has ended, and clips it: one automatic video a day, only broadcasts from after you switch it on. Instead of the first minutes, it clips the stretch your viewers clipped most.'
const TWITCH_AUTOPILOT_DETAIL = `<p>Autopilot clips up to the cap you set per
video (10 to 90 minutes, 30 by default), so a six-hour stream cannot empty your
plan. Those minutes are not the first ones, which are usually the "starting
soon" screen: Autopilot asks Twitch for the clips viewers made from the
broadcast and takes the stretch holding the most clip views, starting a minute
before the first of them. A stream with no viewer clips is clipped from the
start. Broadcasts older than 3 days are not picked up automatically, and you can
still clip up to 5 more a day by hand from the same tab.</p>
<p>The Twitch connection is read-only: Autopilot lists your past broadcasts and
never posts anything to Twitch. With autopublish on, the best clips of each
stream go out one a day on TikTok, Instagram and YouTube Shorts. The
<a href="/auto-clip">auto clip</a> page has the full rules.</p>`

/* Official trailers, embedded from the publishers' own YouTube channels
 * (uploader checked with YouTube oEmbed, embeddable and not age-gated, 7 Nov
 * 2026). youtube-nocookie and loading=lazy: nothing loads from YouTube until
 * the reader scrolls to it. */
const ytEmbed = (id, title, caption) => `<figure class="video">
<iframe src="https://www.youtube-nocookie.com/embed/${id}" title="${esc(title)}" loading="lazy"
  allow="accelerometer; encrypted-media; gyroscope; picture-in-picture; fullscreen"
  referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
<figcaption>${caption}</figcaption>
</figure>`

/* How a 16:9 game recording becomes 9:16, drawn to the real geometry on a
 * 1080x1920 frame: INSET puts the full-width screen (608 px) over the webcam
 * band (camera_inset.inset_filtergraph, 16:9 webcam = 608 px) on a blurred
 * backdrop; WIDE keeps ~77% of the width at ~41% of the height, between the
 * hook and the captions, slid toward the motion (screencast_layout.wide_crop). */
const gameLayoutDiagram = (caption) => `<figure class="diagram">
<svg viewBox="0 0 400 605" role="img" aria-labelledby="gld-t gld-d" style="font-family:var(--body)">
<title id="gld-t">How a 16:9 game recording becomes a vertical clip</title>
<desc id="gld-d">A 16:9 recording with the action in the middle, a minimap in the bottom left corner and a facecam in the top right. With a facecam, INSET puts the whole game width on top and the face enlarged below. Without one, WIDE keeps about three quarters of the width, zoomed and slid toward the action, so the minimap falls outside.</desc>
<text x="10" y="18" fill="var(--ink2)" font-size="13">Your 16:9 recording</text>
<rect x="10" y="28" width="380" height="214" rx="4" fill="var(--paper2)" stroke="var(--rule2)"/>
<rect x="176" y="81" width="131" height="83" rx="6" fill="var(--brass)" opacity=".85"/>
<text x="241" y="127" fill="var(--paper)" font-size="13" text-anchor="middle">action</text>
<rect x="22" y="182" width="47" height="47" rx="3" fill="var(--paper3)" stroke="var(--rule2)"/>
<text x="45" y="209" fill="var(--muted)" font-size="11" text-anchor="middle">map</text>
<rect x="309" y="40" width="71" height="40" rx="3" fill="var(--ok)" opacity=".8"/>
<text x="344" y="64" fill="var(--paper)" font-size="11" text-anchor="middle">facecam</text>
<rect x="98" y="29" width="291" height="212" fill="none" stroke="var(--brass)" stroke-width="2" stroke-dasharray="7 5"/>
<text x="10" y="262" fill="var(--muted)" font-size="12">Dashed: the WIDE crop, about three quarters of the width</text>
<rect x="30" y="290" width="150" height="267" rx="6" fill="var(--paper3)" stroke="var(--rule2)"/>
<rect x="30" y="339" width="150" height="84" fill="var(--paper2)"/>
<rect x="95" y="360" width="52" height="33" rx="3" fill="var(--brass)" opacity=".85"/>
<rect x="35" y="400" width="19" height="19" rx="2" fill="var(--paper3)" stroke="var(--rule2)"/>
<rect x="148" y="344" width="28" height="16" rx="2" fill="var(--ok)" opacity=".8"/>
<rect x="30" y="423" width="150" height="84" fill="var(--ok)" opacity=".8"/>
<text x="105" y="470" fill="var(--paper)" font-size="13" text-anchor="middle">face, enlarged</text>
<rect x="220" y="290" width="150" height="267" rx="6" fill="var(--paper3)" stroke="var(--rule2)"/>
<rect x="240" y="334" width="110" height="20" rx="10" fill="var(--rule2)"/>
<text x="295" y="348" fill="var(--ink2)" font-size="11" text-anchor="middle">hook</text>
<rect x="220" y="369" width="150" height="109" fill="var(--paper2)"/>
<rect x="260" y="396" width="67" height="43" rx="4" fill="var(--brass)" opacity=".85"/>
<text x="293" y="422" fill="var(--paper)" font-size="12" text-anchor="middle">action</text>
<rect x="245" y="492" width="100" height="18" rx="4" fill="var(--rule2)"/>
<text x="295" y="505" fill="var(--ink2)" font-size="11" text-anchor="middle">captions</text>
<text x="105" y="578" fill="var(--ink)" font-size="13" text-anchor="middle">INSET: with a facecam</text>
<text x="105" y="596" fill="var(--muted)" font-size="11" text-anchor="middle">whole game width, face below</text>
<text x="295" y="578" fill="var(--ink)" font-size="13" text-anchor="middle">WIDE: no facecam</text>
<text x="295" y="596" fill="var(--muted)" font-size="11" text-anchor="middle">about 1.3x, slid to the motion</text>
</svg>
<figcaption>${caption}</figcaption>
</figure>`

const GTA6_LAUNCH = '19 November 2026'
const GTA6_CHECKED = GAMING_CHECKED
const ROCKSTAR_LAUNCH_URL = 'https://www.rockstargames.com/newswire/article/ak3ak31a49a221/grand-theft-auto-vi-is-now-set-to-launch-november-19-2026'
const ROCKSTAR_TIMES_URL = 'https://support.rockstargames.com/articles/29jV3Cv2cCgvwM1XgtMO5o/grand-theft-auto-vi-regional-pre-load-and-release-times-on-playstation-5-and-xbox-series-x-or-s'
const ROCKSTAR_POLICY_URL = 'https://support.rockstargames.com/articles/7bNaeoMFTV0iUDGhStTXvz/policy-on-posting-copyrighted-rockstar-games-material'

const gta6Steps = [
  ['Bring the footage', 'Paste a Twitch VOD link (twitch.tv/videos/...), a Twitch clip of 45 seconds or more, or a YouTube link, or upload the capture from your console or capture card. Footage needs at least 45 seconds of material.'],
  ['Pick the stretch', 'For a long session, use "clip only part of it" to set a start and an end. Only that stretch is downloaded, analysed and billed.'],
  ['Leave the layout on auto', 'With the vertical layout on auto (the default; "layouts": ["auto"] on the API) a facecam in a corner becomes INSET, and full-frame action becomes WIDE, zoomed around the motion.'],
  ['Let the AI pick the moments', 'With commentary, Gemini ranks the transcript for stretches that stand alone. With almost no speech, Gemini watches the footage instead. Either way the clips are 15 to 60 seconds and 9:16.'],
  ['Check for spoilers, then post', 'Review each clip, then download it or post it to TikTok, YouTube Shorts and Instagram Reels from the dashboard or the API. The AI does not know what counts as a story spoiler.'],
]

const gta6Qa = [
  {
    q: 'How do I record GTA 6 on PS5?',
    a: 'Press the Create button while you play. Save Recent Gameplay keeps anywhere from the last 15 seconds to the last hour, and Start New Recording records up to an hour. The capture options can include your microphone and voice chat, and 4K recording with a compatible display. Then upload the file to OpenShorts.',
    detail: `<p>That is Sony's own description of the PS5 capture menu (checked
${GAMING_CHECKED}). Two settings matter for clipping. <strong>Microphone
audio</strong>: if you talk while you play, include it, because OpenShorts picks
moments from what is said; without speech it watches the footage instead, and
those clips have no captions. Sony asks you to check with the people in voice
chat before you record them. <strong>Resolution</strong>: a 4K capture is a
larger file and does not go through the PlayStation App upload described
below, so for clipping, 1080p is the easier format to move off the console.</p>`,
  },
  {
    q: 'How do I record GTA 6 on Xbox Series X|S?',
    a: 'Hold the Share button on the controller to save the gameplay that just happened; a single press takes a screenshot. Those are the default actions and you can remap them. Clip length, resolution and where captures are saved are set under Settings, Preferences, Capture and share. Then copy the file off the console and upload it to OpenShorts.',
    detail: `<p>How long one Xbox clip can be depends on the capture resolution and on
whether the console captures to its internal storage or to an external drive,
and Microsoft has changed those limits over the years, so check the current
values in the capture settings on your console rather than a number on a web
page. For a full session with commentary and a facecam, a capture card into a
PC running OBS avoids the console's limits altogether and gives you the
corner-webcam layout described further down.</p>`,
  },
  {
    q: 'How do I get GTA 6 clips from my PS5 or Xbox to my phone or PC?',
    a: 'On PS5, turn on Auto-Upload: captures up to 1080p and 3 minutes long appear in the PlayStation App for 14 days, and you download them from there. Longer or 4K captures go by USB drive from the Media Gallery. On Xbox, captures can back up to OneDrive automatically, or be copied to a USB drive.',
    detail: `<p>The PlayStation App route needs the console linked to the app and left in
rest mode with the internet connection kept on (Sony, checked
${GAMING_CHECKED}). Its 3-minute ceiling is the one that matters for GTA 6: a
whole mission or chase will not fit, so for anything longer, copy it to a USB
drive (Media Gallery, then Copy to USB Drive) and upload it from your PC. On
Xbox, the Captures app can bulk upload to OneDrive and keep future captures
backed up there (Xbox Wire, September 2023). Once the file is on your phone or
PC, upload it to OpenShorts from the browser.</p>`,
  },
  {
    q: 'How do I turn GTA 6 gameplay into TikToks?',
    a: 'Upload the capture or paste the stream VOD link into OpenShorts with the vertical layout on auto. It picks the moments worth posting, cuts each into a 15 to 60 second 9:16 clip, keeps your facecam if the footage has one, and burns in captions when there is speech. Post to TikTok, Shorts and Reels from the dashboard, or download.',
    detail: `${ol(gta6Steps.map(([n, t]) => `<strong>${esc(n)}.</strong> ${esc(t)}`))}`,
  },
  {
    q: 'How does 16:9 GTA 6 gameplay fit a vertical clip?',
    a: 'It depends on what is in the frame. With your webcam in a corner, the clip renders as INSET: the gameplay across the full width on top, your face enlarged below. With no face and action across the frame, it renders as WIDE: zoomed about 1.3 times, keeping roughly three quarters of the width, slid toward the motion.',
    detail: `<p>A centre crop of a 16:9 frame keeps a narrow slice of the map and throws
the rest away, including your face when it sits in a corner. With the vertical
layout on <strong>auto</strong>, OpenShorts looks for a webcam box by its
geometry: a subject that is small, off centre and still between samples. The
<a href="/gta-5-clips">GTA 5 clips</a> page explains how that detector was
measured. A console capture has no face at all, and the moment worth posting is
usually spread across the frame: a chase, a crash, a shootout. WIDE zooms so
the picture fills the space between the hook and the captions instead of
sitting in a thin strip, and slides the crop toward the motion. Edge HUD such as
the minimap can fall outside it; the action stays in.</p>
${gameLayoutDiagram('The two gameplay layouts, drawn to scale on a 1080 x 1920 frame. A facecam in a corner picks INSET; full-frame action with no face picks WIDE.')}`,
  },
  {
    q: 'What is the best length for a GTA 6 TikTok or Short?',
    a: 'There is no single best length. OpenShorts cuts 15 to 60 seconds by default, because most moments that stand on their own fit in that, and you can raise the maximum to 180 seconds. YouTube Shorts and Instagram Reels accept up to 3 minutes, and TikTok accepts longer uploads. Keep the chase, cut the drive to it.',
    detail: `<p>Set the range in the advanced options before the job runs: the minimum
anywhere from 5 to 175 seconds and the maximum from 10 to 180. After the job,
the clip editor trims or extends a clip on word boundaries without using
minutes. Shorts' 3-minute limit is from YouTube's help pages (checked 28
October 2026); Instagram raised Reels to 3 minutes in January 2025.</p>`,
  },
  {
    q: 'How do I clip a GTA 6 stream?',
    a: 'Paste the VOD link: the twitch.tv/videos/ address of the past broadcast, a Twitch clip of 45 seconds or more, or the YouTube link of the stream. OpenShorts reads what you said across the whole stream, picks the passages that stand alone and cuts them into vertical clips. A channel page is the live channel, not a recording.',
    detail: `<p>With commentary on, moment picking reads the transcript, so a four-hour
stream is analysed the same way as a forty-minute one. On a long stream, clip it
in parts (one job per hour, or just the stretch you know was good) and you get
more candidates for fewer minutes. ${TWITCH_VIEWER_CLIPS} The
<a href="/platforms/twitch">Twitch clip maker</a> page covers VODs, chat
overlays and long streams in more detail.</p>`,
  },
  {
    q: 'Can OpenShorts clip my GTA 6 streams automatically?',
    a: TWITCH_AUTOPILOT_ANSWER,
    detail: TWITCH_AUTOPILOT_DETAIL,
  },
  {
    q: 'What GTA 6 moments work as shorts?',
    a: 'Moments a stranger gets in a few seconds: a chase that ends badly, a heist that goes wrong, a fail, a loud reaction, an argument with the people you play with. A long drive across the map or a menu does not. Commentary gives the AI the most to work with; with none, it judges from what happens on screen.',
    detail: `<p>Nobody has GTA 6 footage before ${GTA6_LAUNCH}, so that list comes
from GTA 5 streams, where the clips that travel are the same kinds of moment. The
picker has no game-specific rules: it ranks every stretch of the transcript for
how well it would play to someone who was not there, and with no speech it
watches the footage instead. Story moments are the exception to keep out: see
the spoiler rule below.</p>`,
  },
  {
    q: 'Does it work with no commentary?',
    a: 'Yes. When the footage has no audio track, or its transcript comes back under 8 words in total or under 5 words a minute, OpenShorts switches by itself to a vision pass where Gemini watches the footage and picks the moments. Those clips have no captions, because there is no speech to caption. Layouts work the same.',
    detail: `<p>Long silent footage is watched in stretches of up to 40 minutes, and the
picks from every stretch are ranked together into one list. A job watches up to
eight stretches, a little over five hours; past that only the first five hours
are clipped. It is still worth handing it the session you care about rather than
everything you recorded: on OpenShorts Cloud minutes are billed on source
length, so three hours of free roam costs three hours whether or not anything
happened in it. Use <strong>clip only part of it</strong> to set a start and an
end, or trim the capture in the PS5 Media Gallery before you copy it off.</p>`,
  },
  {
    q: 'Will Rockstar take down my GTA 6 videos?',
    a: `It can. Rockstar's policy on posting its material (checked ${GAMING_CHECKED}) is written for occasional, non-commercial use, says videos must not spoil the story, and says footage from leaked copies posted before release is taken down. Take-Two decides case by case and can change the policy, so read the current version before you monetise anything.`,
    detail: `<p>The <a href="${ROCKSTAR_POLICY_URL}" rel="noopener">policy on posting
copyrighted Rockstar Games material</a> is one policy for all Rockstar games
rather than one written for GTA VI. Its spoiler rule covers an ending, or story
cutscenes cut together on their own. In launch week that rule is the one most
likely to matter, and the AI that picks moments does not know what counts as a
spoiler: review the clips before you post them.</p>
<p>The game footage is one rights question; the <strong>recording</strong> is
another. It belongs to whoever streamed or captured it: clip your own captures
and streams, and footage whose owner said yes, such as a streamer you edit for.
Never clip another streamer without permission; reuploads are what the platforms
strike.</p>`,
  },
  {
    q: 'Will GTA RP clips work on GTA 6?',
    a: 'The clipping side is ready: roleplay is mostly dialogue, which is what the transcript picker reads best. Whether and when roleplay servers run on GTA 6 is up to Rockstar and the server communities, and nothing has been announced that this page can point to. Until then, GTA RP clips come from GTA 5 servers.',
    detail: '<p>The <a href="/gta-5-clips">GTA 5 clips</a> page covers GTA 5 RP streams, which can be clipped today.</p>',
  },
  {
    q: 'Can I clip someone else\'s GTA 6 stream?',
    a: 'Only with their permission. The VOD belongs to whoever streamed it, and the platforms remove reuploads when the streamer reports them. Clipping for a streamer as their editor, or inside a clipping programme the streamer runs, is fine with their yes. Reposting a stranger\'s launch-day stream as your own is not.',
    detail: `<p>For a clipper working for a streamer, the workflow is the same as for
your own stream: paste their VOD link, or have them send you the file. On a long
launch-day stream, split it into parts or pick the stretch that matters, and
review every clip for story spoilers before it goes out under their name.</p>`,
  },
  {
    q: 'How much does it cost to clip GTA 6 footage?',
    a: `Self-hosted OpenShorts is free and open source under MIT, with no meter and no watermark. OpenShorts Cloud clips your first video free up to 60 minutes, then ${CLOUD.freeMinutes} minutes a month with a watermark; paid plans start at $${CLOUD.lowPrice}/month for ${STARTER.monthly.minutes} minutes of source video, and edits on the clips are not counted.`,
    detail: pricingParagraph('console'),
  },
]

const gta6Clips = () => ({
  path: '/gta-6-clips',
  title: withBrand('GTA 6 Clips: Make TikToks From Your Streams'),
  description:
    `Turn GTA 6 streams and PS5 or Xbox captures into TikToks, Shorts and Reels: facecam kept, silent gameplay handled. From $${CLOUD.lowPrice}/month, or self-host free.`,
  h1: 'Make GTA 6 clips: turn your streams and console captures into TikToks',
  lede: `OpenShorts turns GTA 6 footage into vertical clips: paste your Twitch or YouTube VOD, or upload a console capture, and it picks the moments worth posting and cuts them into 15 to 60 second 9:16 clips for TikTok, YouTube Shorts and Instagram Reels. Your first video is free up to 60 minutes (watermarked, no card); no watermark from $${CLOUD.lowPrice}/month, or self-host it free under MIT.`,
  breadcrumb: [{ name: 'Use cases', path: '/use-cases' }, { name: 'GTA 6 clips' }],
  published: '2026-11-04',
  updated: TWITCH_AUTOPILOT_UPDATED,
  pasteLink: { placeholder: 'https://www.twitch.tv/videos/...', label: 'Twitch or YouTube VOD link' },
  related: ['/gta-5-clips', '/wow-forever-clips', '/platforms/twitch', '/platforms/kick', '/auto-reframe'],
  tldr: [
    `Rockstar has set Grand Theft Auto VI to launch on ${GTA6_LAUNCH}, on PlayStation 5 and Xbox Series X|S. No PC version has been announced, so when it launches every GTA 6 clip starts on a console: a capture saved with the PS5 Create button or the Xbox Share button, a recording through a capture card, or a stream VOD.`,
    'Those sources look different and OpenShorts handles each one differently. A stream with a facecam in a corner becomes gameplay on top and your enlarged face below. Full-frame action is zoomed around the motion. A capture with nobody talking, which describes many console captures, is watched by the model instead of read from a transcript.',
  ],
  body: `
<h2>What is different about GTA 6 footage</h2>
<p>GTA 5 has been on PC for years, and a lot of its streamed footage comes
from one: the game and the webcam captured together in OBS, streamed, and
clipped from the VOD. At GTA 6's launch there is no PC version, so the footage
arrives in three shapes:</p>
<ul>
<li><strong>Console captures.</strong> The PS5 Create button and the Xbox Share button save the game picture and game audio. No facecam, and often no voice: whether your mic or party chat is recorded depends on the console's capture settings.</li>
<li><strong>Capture card recordings.</strong> The console runs into a PC through a capture card and OBS records it, often with a webcam composited into a corner. It is the usual way to stream a console with a camera.</li>
<li><strong>Stream VODs.</strong> The past broadcast on Twitch or YouTube, usually several hours long, with your commentary and, if you stream with a camera, a facecam.</li>
</ul>
${ytEmbed('VQRLujxTm3c', 'Grand Theft Auto VI Trailer 2', 'Grand Theft Auto VI Trailer 2, from the <a href="https://www.youtube.com/@RockstarGames" rel="noopener">Rockstar Games</a> YouTube channel. Wide, busy, full-frame scenes like these are what a vertical clip has to fit.')}
<p>Nothing on this page was measured on GTA 6 footage, because until
${GTA6_LAUNCH} there is none. The layouts and thresholds below were measured on
GTA 5 and other stream footage, and they do not depend on which game is on
screen.</p>

${qaSections(gta6Qa)}

<h2>What it does not do</h2>
<ul>
<li><strong>It does not clip live.</strong> It works from the VOD or the capture after the session.</li>
<li><strong>It does not publish to Twitch.</strong> Connecting Twitch to Autopilot only reads your past broadcasts.</li>
<li><strong>It does not know GTA 6.</strong> It picks moments from what is said or, with no speech, from what is shown; it has no game-specific rules.</li>
<li><strong>It does not caption silent footage.</strong> No speech means no captions.</li>
<li><strong>Free clips carry a watermark.</strong> Upgrading removes it from clips already made.</li>
</ul>

${sources([
  `Launch date: Rockstar Games Newswire, <a href="${ROCKSTAR_LAUNCH_URL}" rel="noopener">"Grand Theft Auto VI is Now Set to Launch November 19, 2026"</a>. Platforms: Rockstar Support, <a href="${ROCKSTAR_TIMES_URL}" rel="noopener">GTA VI pre-load and release times on PlayStation 5 and Xbox Series X|S</a>. Both checked ${GTA6_CHECKED}.`,
  `Video policy: Rockstar Support, <a href="${ROCKSTAR_POLICY_URL}" rel="noopener">Policy on posting copyrighted Rockstar Games material</a>, checked ${GTA6_CHECKED}.`,
  `PS5 capture: PlayStation Support, <a href="${PS5_CAPTURE_URL}" rel="noopener">How to capture gameplay and screenshots on PS5 consoles</a>, <a href="${PS5_APP_URL}" rel="noopener">How to access PS5 game captures on PS App</a> and <a href="${PS5_GALLERY_URL}" rel="noopener">How to view screenshots and video clips on PS5 consoles</a>, checked ${GAMING_CHECKED}.`,
  `Xbox capture: Xbox Support, <a href="${XBOX_SHARE_URL}" rel="noopener">What's this Share button on my controller?</a>, and Xbox Wire, <a href="${XBOX_ONEDRIVE_URL}" rel="noopener">captures to OneDrive</a> (27 September 2023), checked ${GAMING_CHECKED}.`,
  `Platform lengths: <a href="${YT_SHORTS_URL}" rel="noopener">YouTube Help: Upload YouTube Shorts</a> (up to 3 minutes), checked 28 October 2026; Instagram Reels up to 3 minutes, announced January 2025, reported by <a href="${REELS_URL}" rel="noopener">Social Media Today</a>.`,
  'Silent-footage thresholds (8 words, 5 words per minute) and the stretches of the vision pass: <code>main.py</code> and <code>clip_selection.py</code>; inset detection: <code>camera_inset.py</code>; WIDE layout: <code>screencast_layout.py</code>.',
  `OpenShorts plan minutes: <a href="/pricing">pricing</a>. Source: <a href="${SITE.repo}" rel="noopener">github.com/mutonby/openshorts</a>.`,
])}
${trademarkNote('Grand Theft Auto, GTA, Rockstar Games, PlayStation, Xbox and Twitch', 'Rockstar Games, Take-Two Interactive, Sony, Microsoft or Twitch')}
`,
  faq: qaFaq(gta6Qa),
  extraNodes: [
    {
      '@type': 'HowTo',
      '@id': `${SITE.url}/gta-6-clips#howto`,
      name: 'How to make GTA 6 clips for TikTok, Shorts and Reels',
      description:
        'Turn a GTA 6 stream VOD or a PS5, Xbox or capture card recording into vertical 9:16 clips, keeping the facecam when there is one and handling footage with no commentary.',
      totalTime: 'PT15M',
      supply: [{ '@type': 'HowToSupply', name: 'A GTA 6 stream VOD link or console capture you have the rights to' }],
      tool: [{ '@type': 'HowToTool', name: 'OpenShorts (self-hosted with Docker, or OpenShorts Cloud)' }],
      step: gta6Steps.map(([name, text]) => ({ '@type': 'HowToStep', name, text })),
    },
  ],
})

/* /wow-forever-clips: World of Warcraft: Forever launched on 4 November 2026
 * (5 November in Europe and Asia-Pacific). The searches around it are about
 * recording ("how to clip wow gameplay") and the SERP is forum threads, so the
 * page answers the recording question first, with the tools named and their
 * own pages cited, and only then what OpenShorts does with the file. Nothing
 * about Forever itself is stated without a Blizzard (or, for Hardcore timing,
 * a press) source next to it.
 */
const WOW_CHECKED = GAMING_CHECKED
const WOW_LAUNCH_URL = 'https://news.blizzard.com/en-us/article/24315820/world-of-warcraft-forever-goes-live-globally-november-4'
const WOW_ANNOUNCE_URL = 'https://worldofwarcraft.blizzard.com/en-us/news/24302093'
const WOW_WIKI_URL = 'https://warcraft.wiki.gg/wiki/Forever'
const WOW_HARDCORE_URL = 'https://www.pcgamer.com/games/world-of-warcraft/world-of-warcraft-forever-will-get-a-hardcore-mode-by-winter-say-devs-whichll-be-interesting-given-all-the-new-and-undiscovered-ways-to-die/'
const WOW_SHACK_URL = 'https://www.shacknews.com/article/150712'
const BLIZZARD_POLICY_URL = 'https://www.blizzard.com/en-us/legal/2068564f-f427-4c1c-8664-c107c90b34d5/blizzard-video-policy'
const WOW_RECORDER_URL = 'https://github.com/aza547/wow-recorder'
const OBS_REPLAY_URL = 'https://obsproject.com/forum/resources/how-to-setup-instant-replay-in-obs-studio.613/'
const AMD_REPLAY_URL = 'https://www.amd.com/en/resources/support-articles/faqs/DH3-023.html'
const GAME_BAR_URL = 'https://support.microsoft.com/en-us/help/4026387'
const MAC_RECORD_URL = 'https://support.apple.com/en-us/102618'

const wowSteps = [
  ['Bring the recording', 'Paste the twitch.tv/videos/ link of the stream, a Twitch clip of 45 seconds or more, or a YouTube link, or upload the file from OBS, Warcraft Recorder or your GPU\'s replay tool. Sources need at least 45 seconds of material.'],
  ['Pick the pulls you care about', 'On a long raid night, use "clip only part of it" to set a start and an end around the kill or the wipe. Only that stretch is downloaded, analysed and billed.'],
  ['Leave the layout on auto', 'A facecam in a corner becomes INSET (game on top, face below); a full-screen fight becomes WIDE, zoomed around the motion.'],
  ['Let the AI pick the moments', 'With voice chat or commentary, Gemini ranks the transcript for stretches that stand alone. With almost no speech, Gemini watches the footage instead. Clips are 15 to 60 seconds by default.'],
  ['Review and post', 'Fix a crop or a cut in the clip editor if needed, then post to TikTok, YouTube Shorts and Instagram Reels from the dashboard or the API, or download.'],
]

const wowQa = [
  {
    q: 'How do I record WoW Forever gameplay?',
    a: 'Use a recorder that keeps a rolling buffer, so you can save a kill after it happens: the replay buffer in OBS Studio, Instant Replay from NVIDIA or AMD, or Xbox Game Bar on Windows. On Windows, Warcraft Recorder records Forever raid encounters on its own. On a Mac, use OBS or Shift-Command-5.',
    detail: `<ul>
<li><strong>OBS Studio replay buffer.</strong> Turn on Replay Buffer under Settings, Output, and set a Save Replay hotkey under Settings, Hotkeys. OBS keeps the last stretch in memory and writes it to a file when you press the key. It is also what you stream with, so the webcam lands in the same file.</li>
<li><strong>NVIDIA and AMD Instant Replay.</strong> The GPU drivers' own recorders keep a rolling buffer too; AMD's support article puts its maximum at the last 20 minutes.</li>
<li><strong>Xbox Game Bar</strong> (built into Windows). Windows + Alt + G saves the last 30 seconds (the length is adjustable in Game Bar settings) and Windows + Alt + R starts or stops a recording.</li>
<li><strong>Warcraft Recorder.</strong> A free, open source (GPL-2.0) recorder for Windows that watches the WoW combat log and records encounters by itself. Its README lists Forever as supported for raids and manual recording (checked ${WOW_CHECKED}); Forever's raids open on 9 December 2026, so until then use its manual mode. It does not run on macOS.</li>
<li><strong>Mac.</strong> Forever runs on macOS as well as Windows. OBS runs there, and Shift-Command-5 opens the built-in screen recorder; whether it can include the game's sound depends on your macOS version, per Apple's page.</li>
</ul>
<p>Whatever records it, keep the file: an MP4 or MOV goes straight into
OpenShorts. If you want your face in the clip, put the webcam in a corner of the
OBS scene; that is the layout OpenShorts turns into INSET.</p>`,
  },
  {
    q: 'How do I turn a WoW Twitch stream into TikToks or Shorts?',
    a: 'Paste the twitch.tv/videos/ link of the past broadcast into OpenShorts, or a Twitch clip of 45 seconds or more. It reads what you and your raid said across the whole stream, picks the passages that stand alone, and cuts them into 15 to 60 second vertical clips with captions, and a corner facecam kept below the game.',
    detail: `${ol(wowSteps.map(([n, t]) => `<strong>${esc(n)}.</strong> ${esc(t)}`))}
<p>A channel page is the live channel, not a recording; the VOD is under Videos
on the channel. A raid night of three or four hours is analysed the same way as
a short stream, because moment picking reads the transcript, but one job returns
its best handful of moments, so clip a long night in parts. ${TWITCH_VIEWER_CLIPS}
The <a href="/platforms/twitch">Twitch clip maker</a> page covers long VODs and
chat overlays.</p>`,
  },
  {
    q: 'Can OpenShorts clip my WoW streams automatically?',
    a: TWITCH_AUTOPILOT_ANSWER,
    detail: TWITCH_AUTOPILOT_DETAIL,
  },
  {
    q: 'How do I make a raid boss kill or wipe vertical without losing the UI?',
    a: 'A whole WoW screen does not fit a 9:16 frame and stay readable. With a facecam in a corner, it renders as INSET: the full width of the game on top, your face enlarged below. Without one, WIDE keeps roughly three quarters of the width, slid toward the motion, so raid frames at the edges can fall outside.',
    detail: `<p>WIDE zooms about 1.3 times so the fight fills the space between the hook
and the captions, and it follows where things move: a still action bar or a
minimap never pulls the crop. That suits the boss and the players around it,
and it means interface parked at the far left and right edges is the first
thing to go. Three things help:</p>
<ul>
<li><strong>Record a cleaner UI for shorts.</strong> Move what the viewer needs (boss timers, the raid frames if they tell the story) toward the middle, or hide the interface for a cinematic pull with the game's toggle-UI key binding.</li>
<li><strong>Use the facecam.</strong> INSET keeps the full width of the game, so nothing at the edges is lost; it is just smaller.</li>
<li><strong>Fix a scene by hand.</strong> The reframe editor lets you draw the crop yourself for one scene, or split it into two stacked regions, and the rest of the clip stays automatic.</li>
</ul>
${gameLayoutDiagram('The two layouts, drawn to scale on a 1080 x 1920 frame. With a facecam, INSET keeps every edge of the UI; without one, WIDE keeps the fight and can lose the far edges.')}`,
  },
  {
    q: 'How do I clip a Hardcore death?',
    a: 'Record with a replay buffer so the death is saved after it happens and upload the file, or paste the VOD and use "clip only part of it" around the death. A death is short and often silent, so the model watches the footage when nobody speaks. Forever\'s Hardcore ruleset is not live yet; Blizzard says it comes after launch.',
    detail: `<p>Blizzard's launch post says the Hardcore ruleset becomes available
sometime after launch, and Blizzard developers have said by winter (PC Gamer;
warcraft.wiki.gg lists winter 2026/2027), checked ${WOW_CHECKED}. Until then
the same workflow applies to Classic Hardcore characters. Two practical
points: a source needs at least 45 seconds of material, so save a minute or two
around the death rather than the last ten seconds; and a clip that should end on
the death screen can be trimmed or extended in the clip editor.</p>`,
  },
  {
    q: 'What WoW moments work as shorts?',
    a: 'Moments a stranger understands in a few seconds: a first kill of a new boss, the wipe and the call that caused it, a close world PvP fight, a Hardcore death, a funny exchange on voice chat. A long trash pull or a quiet flight path does not. Commentary and voice chat give the AI the most to work with.',
    detail: `<p>The default picker reads the transcript and ranks every stretch of it for
how well it would play to someone who was not there, so a raid that talks
(callouts, reactions, the argument after a wipe) produces more candidates than a
silent one. Forever's three raids, Barrow Deeps, Hyjal Summit and Onyxia's Lair,
open on 9 December 2026 (Blizzard), so first kills on your realm start
then.</p>`,
  },
  {
    q: 'How long should a WoW clip be for TikTok, Shorts or Reels?',
    a: 'OpenShorts cuts 15 to 60 seconds by default, which fits most moments that stand alone, and you can raise the maximum to 180 seconds for a full boss kill that needs context. YouTube Shorts and Instagram Reels accept up to 3 minutes, and TikTok accepts longer uploads. Start on the action, not the pull countdown.',
    detail: `<p>Set the range in the advanced options before the job: the minimum
anywhere from 5 to 175 seconds and the maximum from 10 to 180. Trimming or
extending a clip afterwards in the editor does not use minutes. YouTube's
3-minute Shorts limit is from its help pages (checked 28 October 2026);
Instagram raised Reels to 3 minutes in January 2025.</p>`,
  },
  {
    q: 'Does it work without commentary?',
    a: 'Yes. When the recording has no audio track, or its transcript comes back under 8 words in total or under 5 words a minute, OpenShorts switches by itself to a vision pass where Gemini watches the footage and picks the moments. Those clips have no captions, because there is no speech to caption. Layouts work the same.',
    detail: `<p>Long silent footage is watched in stretches of up to 40 minutes, and the
picks from every stretch are ranked together. A job watches up to eight
stretches, a little over five hours; past that only the first five hours are
clipped. On OpenShorts Cloud minutes are billed on source length, so the pull
you care about is a cheaper input than the whole evening.</p>`,
  },
  {
    q: 'Can I post and monetize WoW videos?',
    a: `Blizzard's video policy (checked ${WOW_CHECKED}) limits use of its content to non-commercial purposes, with listed exceptions: partner programs where a video site pays you for views, and hosting on sites with paid tiers as long as viewers can also watch for free. Charging viewers or licensing a video to another company for a fee is not allowed.`,
    detail: `<p>The <a href="${BLIZZARD_POLICY_URL}" rel="noopener">Blizzard Video
Policy</a> also caps sponsor promotion at 10 seconds per video, marked
"Sponsored By", asks that videos stay within a T rating, and says Blizzard can
end your right to distribute a video at any time. The partner programs it names
are YouTube's and several older video sites; it does not mention TikTok or
Instagram by name. It is Blizzard's policy and it can change, so read the
current version before you rely on it.</p>`,
  },
  {
    q: 'Can I clip someone else\'s WoW stream?',
    a: 'Only with their permission. The VOD belongs to whoever streamed it, and the platforms remove reuploads when the streamer reports them. Clipping your guild leader\'s stream as the guild\'s editor, or as part of a clipping programme the streamer runs, is fine with their yes. Taking a stranger\'s world-first kill and posting it as yours is not.',
    detail: '',
  },
  {
    q: 'How much does it cost to clip WoW Forever footage?',
    a: `Self-hosted OpenShorts is free and open source under MIT, with no meter and no watermark. OpenShorts Cloud clips your first video free up to 60 minutes, then ${CLOUD.freeMinutes} minutes a month with a watermark; paid plans start at $${CLOUD.lowPrice}/month for ${STARTER.monthly.minutes} minutes of source video, and edits on the clips are not counted.`,
    detail: pricingParagraph('raid'),
  },
]

const wowForeverClips = () => ({
  path: '/wow-forever-clips',
  title: withBrand('WoW Forever Clips: Turn Raids and Streams Into TikToks'),
  description:
    `Record WoW Forever raids with OBS or Warcraft Recorder, then turn kills, wipes and Twitch VODs into TikToks. From $${CLOUD.lowPrice}/month, or self-host free.`,
  h1: 'WoW Forever clips: turn raids and streams into TikToks, Shorts and Reels',
  lede: `How to record World of Warcraft: Forever, and how to turn the recording or your Twitch VOD into 15 to 60 second vertical clips for TikTok, YouTube Shorts and Instagram Reels with OpenShorts, facecam kept. Your first video is free up to 60 minutes (watermarked, no card); no watermark from $${CLOUD.lowPrice}/month, or self-host it free under MIT.`,
  breadcrumb: [{ name: 'Use cases', path: '/use-cases' }, { name: 'WoW Forever clips' }],
  published: '2026-11-04',
  updated: TWITCH_AUTOPILOT_UPDATED,
  pasteLink: { placeholder: 'https://www.twitch.tv/videos/...', label: 'Twitch VOD link' },
  related: ['/platforms/twitch', '/gta-6-clips', '/auto-reframe', '/platforms/kick'],
  tldr: [
    `World of Warcraft: Forever is Blizzard's new version of the original game, announced at BlizzCon 2026 and live from 4 November 2026 (5 November in Europe, Taiwan, Korea and Australia/New Zealand). It is included with a World of Warcraft subscription or game time and runs on Windows and macOS. Its three raids open on 9 December.`,
    'A WoW recording is a busy 16:9 screen: the fight in the middle, raid frames and action bars at the edges, often a webcam in a corner. Below: how to record it, and how OpenShorts makes it vertical without losing the fight.',
  ],
  body: `
${ytEmbed('n2Wg46fwmb0', 'Launch Overview | World of Warcraft: Forever', 'Launch Overview of World of Warcraft: Forever, from the official <a href="https://www.youtube.com/@Warcraft" rel="noopener">World of Warcraft</a> YouTube channel. A WoW screen this busy is what a vertical clip has to fit.')}

${qaSections(wowQa)}

<h2>What it does not do</h2>
<ul>
<li><strong>It does not record the game.</strong> Use OBS, your GPU's replay tool, Game Bar or Warcraft Recorder, then bring the file or the VOD.</li>
<li><strong>It does not clip live.</strong> It works from the VOD once the stream has ended, pasted by you or picked up by Autopilot.</li>
<li><strong>It does not read the combat log or know WoW.</strong> It picks moments from what is said or, with no speech, from what is shown.</li>
<li><strong>It does not keep every edge of the UI</strong> in a WIDE clip; see the raid UI answer above.</li>
<li><strong>It does not caption silent footage.</strong> No speech means no captions.</li>
<li><strong>Free clips carry a watermark.</strong> Upgrading removes it from clips already made.</li>
</ul>

${sources([
  `Launch times, raids (9 December 2026) and Hardcore after launch: Blizzard News, <a href="${WOW_LAUNCH_URL}" rel="noopener">"World of Warcraft: Forever Goes Live Globally November 4!"</a>; included with a subscription or game time: Blizzard, <a href="${WOW_ANNOUNCE_URL}" rel="noopener">"Carve a New Path with World of Warcraft: Forever"</a> (12 September 2026). Both checked ${WOW_CHECKED}.`,
  `Platforms (Windows, macOS) and Hardcore in winter 2026/2027: <a href="${WOW_WIKI_URL}" rel="noopener">warcraft.wiki.gg, Forever</a>; Hardcore by winter: <a href="${WOW_HARDCORE_URL}" rel="noopener">PC Gamer</a>; BlizzCon announcement: <a href="${WOW_SHACK_URL}" rel="noopener">Shacknews</a> (12 September 2026). Checked ${WOW_CHECKED}.`,
  `Video policy: <a href="${BLIZZARD_POLICY_URL}" rel="noopener">Blizzard Video Policy</a>, checked ${WOW_CHECKED}.`,
  `Recorders: <a href="${WOW_RECORDER_URL}" rel="noopener">Warcraft Recorder README</a> (supported flavours and OS, GPL-2.0); <a href="${OBS_REPLAY_URL}" rel="noopener">OBS forum guide to the replay buffer</a>; <a href="${AMD_REPLAY_URL}" rel="noopener">AMD support article DH3-023</a> (Instant Replay); <a href="${GAME_BAR_URL}" rel="noopener">Microsoft: Xbox Game Bar shortcuts</a>; <a href="${MAC_RECORD_URL}" rel="noopener">Apple: How to record the screen on Mac</a>. All checked ${WOW_CHECKED}.`,
  `Platform lengths: <a href="${YT_SHORTS_URL}" rel="noopener">YouTube Help: Upload YouTube Shorts</a>, checked 28 October 2026; Instagram Reels up to 3 minutes, reported by <a href="${REELS_URL}" rel="noopener">Social Media Today</a>.`,
  'Silent-footage thresholds and stretches: <code>main.py</code> and <code>clip_selection.py</code>; inset detection: <code>camera_inset.py</code>; WIDE layout: <code>screencast_layout.py</code>.',
  `OpenShorts plan minutes: <a href="/pricing">pricing</a>. Source: <a href="${SITE.repo}" rel="noopener">github.com/mutonby/openshorts</a>.`,
])}
${trademarkNote('World of Warcraft, Warcraft, Blizzard and Twitch', 'Blizzard Entertainment, Microsoft or Twitch')}
`,
  faq: qaFaq(wowQa),
  extraNodes: [
    {
      '@type': 'HowTo',
      '@id': `${SITE.url}/wow-forever-clips#howto`,
      name: 'How to turn WoW Forever raids and streams into vertical clips',
      description:
        'Turn a World of Warcraft: Forever recording or Twitch VOD into 9:16 clips for TikTok, YouTube Shorts and Instagram Reels, keeping the facecam and the fight.',
      totalTime: 'PT15M',
      supply: [{ '@type': 'HowToSupply', name: 'A WoW Forever recording or Twitch VOD you have the rights to' }],
      tool: [{ '@type': 'HowToTool', name: 'OpenShorts (self-hosted with Docker, or OpenShorts Cloud)' }],
      step: wowSteps.map(([name, text]) => ({ '@type': 'HowToStep', name, text })),
    },
  ],
})

/* Hand-picked neighbours for every page in this file, by topic: a review links
 * to its comparison and the hub, a pricing page to the brand page and the
 * comparison, a use case to its sibling use cases. relatedFor shows the first
 * three that exist, so a list can name a page another module adds (the clip
 * maker and pricing pages) and fall back to the next one until it ships. */
/* /opusclip-vs-vizard: the two category leaders side by side, with OpenShorts
 * as the third column. Third-party "OpusClip vs Vizard" pages are among the
 * sources engines cite most for both brands; this one answers the same
 * question from the numbers in data.js, dated. */
const opusVsVizard = () => {
  const o = COMPETITORS['opus-clip']
  const v = COMPETITORS.vizard
  const checked = [o.checked, v.checked].sort()[0]
  const vendor = (c, r) => c.rows?.[r.feature] ?? (r.key ? c[r.key] : r.vendor)
  const rows = COMPARISON_ROWS.map((r) =>
    `<tr><td>${esc(r.feature)}</td><td>${esc(vendor(o, r))}</td><td>${esc(vendor(v, r))}</td><td class="os">${esc(r.os)}</td></tr>`
  ).join('')
  const tiers = (c) => li(c.tiers.map(([n, d]) => `<strong>${esc(n)}</strong>: ${esc(d)}`))
  const faq = [
    {
      q: 'Is OpusClip or Vizard cheaper?',
      a: `Per minute of video, Vizard: its Creator plan is 600 minutes for $29/month ($14.50 billed yearly), against 300 minutes for $29/month on OpusClip Pro ($14.50 billed yearly). OpusClip's Starter at $15/month for 150 minutes is the cheaper way in if you need fewer minutes. Checked ${monthYear(checked)}.`,
    },
    {
      q: 'Which has the better free plan?',
      a: 'Both give 60 minutes of source a month with a watermark. OpusClip exports up to 1080p and clips must be exported within 3 days; Vizard exports at 720p, up to 10 minutes per export, with 3-day storage.',
    },
    {
      q: 'Which is better for editing clips by hand?',
      a: `Vizard. ${v.edge} OpusClip has the larger caption-style library, AI B-roll and a virality score.`,
    },
    {
      q: 'Is there an open source alternative to both?',
      a: `OpenShorts: MIT-licensed, free to self-host with no watermark and no cap, or hosted from $${CLOUD.lowPrice}/month for ${STARTER.monthly.minutes} minutes. It bills the same unit (the source minute) on its hosted plans and none at all when you run it yourself.`,
    },
  ]
  return {
    path: '/opusclip-vs-vizard',
    title: 'OpusClip vs Vizard (2026): Pricing, Free Plans, Editing',
    description: `OpusClip vs Vizard, checked ${monthYear(checked)}: 300 vs 600 minutes for $29/month, free plans, editing and API. Plus OpenShorts, the open source option from $12/month.`,
    h1: 'OpusClip vs Vizard: pricing, free plans and which to pick',
    lede: brandLede(o, 'opus.pro/pricing and vizard.ai/pricing',
      `OpusClip and Vizard both bill per source minute: OpusClip Pro is 300 minutes for $29/month, Vizard Creator is 600 minutes for $29/month. Pick Vizard for volume, OpusClip for its editor and templates, or OpenShorts if you want an open source option: free to self-host, $${CLOUD.lowPrice}/month hosted.`),
    breadcrumb: [{ name: 'Alternatives', path: '/alternatives' }, { name: 'OpusClip vs Vizard' }],
    published: REVISED,
    updated: REVISED,
    relatedTitle: 'Related comparisons',
    related: ['/alternatives/opus-clip', '/alternatives/vizard', '/opus-clip-pricing', '/vizard-ai'],
    pasteLink: true,
    facts: [
      { k: 'OpusClip Pro', v: '$29/mo', s: '300 minutes; $14.50/mo billed yearly' },
      { k: 'Vizard Creator', v: '$29/mo', s: '600 minutes; $14.50/mo billed yearly' },
      { k: 'Free plans', v: '60 min/mo', s: 'Both watermarked' },
      { k: 'Open source', v: 'Neither', s: 'OpenShorts: MIT, self-hostable' },
    ],
    tldr: [
      `Same unit, different allowance: both charge one credit per minute of source video, and at $29/month Vizard gives 600 minutes where OpusClip Pro gives 300. Billed yearly both are $14.50/month. OpusClip also sells Starter, $15/month for 150 minutes, monthly only.`,
      `${esc(o.edge)} ${esc(v.edge)} Both free plans give 60 watermarked minutes a month.`,
      `Neither can be self-hosted. OpenShorts is the open source third option: $0 self-hosted with no watermark or cap, or $${CLOUD.lowPrice}/month hosted for ${STARTER.monthly.minutes} minutes; per minute it costs more than Vizard at volume and less to start.`,
    ],
    body: `
<h2>OpusClip vs Vizard vs OpenShorts, feature by feature</h2>
<table>
<thead><tr><th>Feature</th><th>OpusClip</th><th>Vizard</th><th>OpenShorts</th></tr></thead>
<tbody>${rows}</tbody>
</table>

<h2>What would each cost for your minutes?</h2>
<p>All three meter the source minute on their hosted plans. Move the slider to
the minutes of video you clip in a month; each bar is the cheapest published
plan that covers it.</p>
${costCalculator(['opus-clip', 'vizard'])}

<h2>OpusClip plans</h2>
${tiers(o)}
<p>${esc(o.gotcha)}</p>

<h2>Vizard plans</h2>
${tiers(v)}
<p>${esc(v.gotcha || '')}</p>

<h2>Which one should you pick?</h2>
<ul>
<li><strong>OpusClip</strong> if caption styling, AI B-roll and a virality score matter more than minutes, and your sources are short.</li>
<li><strong>Vizard</strong> if you clip a lot of video, pay yearly and want to fix each clip in a timeline.</li>
<li><strong>OpenShorts</strong> if you want it open source: free on your own machine with no meter, or hosted from $${CLOUD.lowPrice}/month, with an API and an MCP server for agents.</li>
</ul>

${faqBlock(faq)}

${sources([
  `OpusClip plans checked ${esc(o.checked)} on opus.pro/pricing; Vizard plans checked ${esc(v.checked)} on vizard.ai/pricing.`,
  `OpenShorts plans: <a href="/pricing">pricing</a>; source at <a href="${SITE.repo}" rel="noopener">github.com/mutonby/openshorts</a>.`,
])}
`,
    faq,
  }
}

const RELATED = {
  '/alternatives': ['/free-ai-clipping-tools-compared', '/alternatives/opus-clip', '/opus-clip-pricing', '/alternatives/klap'],
  '/alternatives/opus-clip': ['/opus-clip-pricing', '/opus-ai', '/alternatives'],
  '/alternatives/klap': ['/alternatives', '/free-ai-clipping-tools-compared', '/alternatives/opus-clip', '/alternatives/2short'],
  '/alternatives/vizard': ['/vizard-ai', '/vizard-ai-video-to-text', '/alternatives'],
  '/alternatives/submagic': ['/submagic-reviews', '/alternatives', '/alternatives/opus-clip'],
  '/alternatives/vidyo-ai': ['/alternatives', '/alternatives/opus-clip', '/alternatives/sendshort'],
  '/alternatives/2short': ['/alternatives', '/youtube-to-shorts-converter', '/alternatives/klap'],
  '/alternatives/sendshort': ['/alternatives', '/alternatives/submagic', '/alternatives/vidyo-ai'],
  '/opus-clip-pricing': ['/opus-ai', '/alternatives/opus-clip', '/alternatives'],
  '/opus-ai': ['/opus-clip-pricing', '/alternatives/opus-clip', '/alternatives'],
  '/vizard-ai': ['/alternatives/vizard', '/vizard-ai-video-to-text', '/alternatives'],
  '/vizard-ai-video-to-text': ['/vizard-ai', '/youtube-transcript-generator', '/alternatives/vizard'],
  '/submagic-reviews': ['/alternatives/submagic', '/alternatives', '/opus-clip-pricing'],
  '/alternativas': ['/alternatives', '/alternatives/opus-clip', '/opus-clip-pricing'],
  '/free-ai-clip-generator': ['/free-ai-clipping-tools-compared', '/self-hosted-ai-clip-maker', '/open-source-video-clipper', '/alternatives/opus-clip', '/tools'],
  '/open-source-video-clipper': ['/self-hosted-ai-clip-maker', '/how-openshorts-works', '/open-source-ai-video-generator', '/free-ai-clip-generator'],
  '/open-source-ai-video-generator': ['/platforms/instagram-reels', '/open-source-video-clipper', '/how-openshorts-works', '/free-ai-clip-generator'],
  '/how-openshorts-works': ['/auto-reframe', '/podcast-to-shorts', '/gta-5-clips', '/youtube-to-shorts-converter'],
  '/gta-5-clips': ['/gta-6-clips', '/platforms/twitch', '/platforms/kick', '/podcast-to-shorts', '/how-openshorts-works', '/auto-clip'],
  '/podcast-to-shorts': ['/youtube-to-shorts-converter', '/use-cases/sermons', '/how-openshorts-works', '/platforms/youtube', '/auto-clip'],
  '/youtube-to-shorts-converter': ['/how-to/clip-a-youtube-video', '/platforms/youtube-shorts', '/podcast-to-shorts', '/how-openshorts-works', '/video-aspect-ratio-converter'],
  '/mcp': ['/automate-shorts-api', '/n8n-youtube-shorts-automation', '/youtube-automation'],
  '/automate-shorts-api': ['/mcp', '/n8n-youtube-shorts-automation', '/auto-clip'],
  '/n8n-youtube-shorts-automation': ['/automate-shorts-api', '/auto-clip', '/mcp'],
  '/alternatives/choppity': ['/alternatives', '/podcast-to-shorts', '/platforms/youtube', '/alternatives/opus-clip'],
  '/alternatives/getmunch': ['/alternatives', '/platforms', '/alternatives/sendshort', '/alternatives/vidyo-ai'],
}

/* The heading over the related links, by page type. */
const GUIDES = new Set(['/how-openshorts-works', '/mcp', '/automate-shorts-api', '/n8n-youtube-shorts-automation'])
const relatedTitleFor = (path) => {
  if (path === '/alternativas') return 'Comparativas relacionadas'
  if (GUIDES.has(path)) return 'Related guides'
  if (/^\/(alternatives|opus-|vizard-|submagic-)/.test(path)) return 'Related comparisons'
  return 'Related tools'
}

export function buildPages() {
  const pages = [
    hubPage(),
    ...ALTERNATIVES.map(competitorPage),
    opusClipPricing(),
    opusAi(),
    vizardAi(),
    videoToText(),
    submagicReview(),
    alternativasIndex(),
    freeClipGenerator(),
    openSourceClipper(),
    openSourceVideoGenerator(),
    howItWorks(),
    gtaClips(),
    gta6Clips(),
    wowForeverClips(),
    podcastToShorts(),
    youtubeConverter(),
    mcpAgentsPage(),
    automateShorts(),
    n8nTemplate(),
    opusVsVizard(),
    pricingPage(),
  ].map((p) => ({
    ...p,
    related: p.related || RELATED[p.path],
    relatedTitle: p.relatedTitle || relatedTitleFor(p.path),
  }))
  // The tool and autopilot modules name their own neighbours. The topic hubs
  // (/platforms, /use-cases, /how-to) come last: they list only pages that
  // are actually built, so they are built from everything else.
  const rest = [...pages, ...guidePages(), ...autopilotPages(), ...toolPages(), ...clipToolPages(), ...productPages()]
  return [...rest, ...hubPages(rest)]
}

/* Each page links to three siblings. Small, described clusters beat a single
 * dump of every URL: the described link tells an engine what it will find. */
export function relatedFor(page, all) {
  const blurb = {
    '/alternatives': 'Nine tools compared, with entry pricing and free plans.',
    '/tools/ai-thumbnail-maker': 'Thumbnails, 10 titles and a chaptered description from your video.',
    '/ai-ugc-video-generator': 'A product URL in, a 20-25 s ad with a lip-synced AI actor out.',
    '/tools/tiktok-transcript-generator': 'Paste a TikTok link, get its captions as text, TXT or SRT.',
    '/vizard-ai': 'A review: what it does, pricing, the free plan, pros and cons.',
    '/alternatives/vidyo-ai': 'Vidyo.ai became Quso: what changed and what it costs.',
    '/alternatives/2short': 'Cheapest per hour, unwatermarked free plan, links only.',
    '/alternatives/sendshort': 'A short-video suite; long-video clipping from $29/month.',
    '/opus-clip-pricing': 'Free, Starter $15 and Pro $29 explained, and what a credit is.',
    '/opus-ai': 'What Opus AI (OpusClip) is, whether it is free, what it costs.',
    '/vizard-ai-video-to-text': 'Transcript, subtitles or clips: which one you are asking for.',
    '/submagic-reviews': 'Pricing, the captions, Magic Clips and the per-video cap.',
    '/alternativas': 'Todas las comparativas, en español, ordenadas por pregunta.',
    '/alternatives/opus-clip': 'The free routes without the watermark trap, and where each wins.',
    '/alternatives/klap': 'Klap reviewed: one free video, per-clip billing, the alternative.',
    '/alternatives/vizard': 'When to leave Vizard, when to stay, and how to move.',
    '/alternatives/submagic': 'Captions-first, now with Magic Clips, metered per video.',
    '/free-ai-clip-generator': 'What free means, and where no watermark is actually true.',
    '/open-source-video-clipper': 'Self-hosting with Docker, and the MIT licence carve-out.',
    '/open-source-ai-video-generator': 'Text-to-video or clips from your footage: which you want.',
    '/how-openshorts-works': 'The full pipeline, stage by stage.',
    '/gta-5-clips': 'Stream VODs, webcam inset kept, no per-minute meter.',
    '/gta-6-clips': 'Console captures and stream VODs, facecam kept, silent footage handled.',
    '/wow-forever-clips': 'Record raids, then turn kills, wipes and Twitch VODs into vertical clips.',
    '/podcast-to-shorts': 'Podcast clips with both speakers kept in frame.',
    '/youtube-to-shorts-converter': 'Paste a link, get 9:16 clips with subtitles.',
    '/mcp': 'Drive the whole pipeline from Claude, ChatGPT or n8n.',
    '/automate-shorts-api': 'One POST in, one signed webhook out, no polling.',
    '/n8n-youtube-shorts-automation': 'The importable workflow: channel in, approved shorts out.',
    '/auto-clip': 'Every new upload on your channel clipped on its own, with the rules.',
    '/youtube-automation': 'Autopilot, n8n or the API: automate a real channel, not a content farm.',
    '/tools': 'Free transcript, metadata and 9:16 tools, no sign-up.',
    '/youtube-transcript-generator': 'Paste a link, get the transcript with timestamps, TXT or SRT.',
    '/youtube-tag-generator': 'Tags, 10 title options or a description from a short brief.',
    '/video-aspect-ratio-converter': '16:9 to 9:16 in your browser: blur, crop or bars.',
    // Pages added by other modules; a blurb for a page that is not built is
    // simply never used.
    '/platforms/youtube': 'Paste a YouTube link, get vertical clips with captions and hooks.',
    '/platforms/youtube-shorts': 'Shorts from long videos: length, hooks, captions, posting, Autopilot.',
    '/platforms/tiktok': 'Long video to TikToks, and why they land in TikTok as drafts.',
    '/platforms/instagram-reels': 'Reels posted straight to Instagram, Facebook Reels as files.',
    '/platforms/twitch': 'VODs clipped in parts, facecam kept, silent gameplay handled.',
    '/platforms/kick': 'Kick VOD and clip links, IRL and facecam layouts, permission.',
    '/auto-reframe': 'AI tracking, stacked speakers and screens, versus a static crop.',
    '/use-cases/sermons': 'Clip only the sermon, follow the preacher, a clip a day.',
    '/how-to/clip-a-youtube-video': 'Share a moment, Studio, ffmpeg, or AI-cut Shorts: four free ways.',
    '/self-hosted-ai-clip-maker': 'Docker setup, hardware, your own keys, and the real cost vs credits.',
    '/free-ai-clipping-tools-compared': 'Every free tier side by side: minutes, resolution, who watermarks.',
    '/platforms': 'YouTube, Shorts, TikTok, Reels, Twitch and Kick, with their specs side by side.',
    '/use-cases': 'Podcasts, sermons, gaming streams and channels clipped on autopilot.',
    '/how-to': 'Guides to clip a video, self-host OpenShorts and automate Shorts.',
    '/alternatives/choppity': 'Choppity reviewed: a free hour a month, hourly plans, the alternative.',
    '/alternatives/getmunch': 'GetMunch is now Munch Studio: what changed, prices, the alternative.',
    '/how-to/instagram-reel-dimensions': '1080 x 1920 at 9:16, the grid crop of the cover, safe zones, export.',
    '/how-to/youtube-shorts-aspect-ratio': '9:16 at 1080 x 1920, square allowed, up to 3 minutes, safe zones.',
    '/how-to/make-a-reel-on-facebook': 'Phone and Meta Business Suite steps, specs, sharing from Instagram.',
    '/how-to/make-youtube-shorts': 'The app camera, a vertical upload, or Shorts cut from long videos.',
    '/how-to/best-time-to-post-youtube-shorts': 'No universal best time: read your Analytics, test slots, stay steady.',
  }
  const card = (p) => ({ path: p.path, title: p.h1, blurb: blurb[p.path] || p.description })
  // A page can name its own neighbours: the first three of its list that are
  // actually built.
  if (page.related) {
    return page.related
      .map((path) => all.find((p) => p.path === path))
      .filter((p) => p && p.path !== page.path)
      .slice(0, 3)
      .map(card)
  }
  // Walk the ring starting after this page so each page links to a different
  // three. Slicing the same head every time would leave the last pages in the
  // list with no inbound links at all.
  const i = all.findIndex((p) => p.path === page.path)
  return [1, 2, 3]
    .map((n) => all[(i + n) % all.length])
    .filter((p) => p && p.path !== page.path)
    .map(card)
}
