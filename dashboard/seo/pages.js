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

import { SITE, COMPETITORS, COMPARISON_ROWS, EDITIONS, PIPELINE_STEPS, CANONICAL_ANSWERS } from './data.js'
import { esc } from './render.js'

const li = (items) => `<ul>${items.map((i) => `<li>${i}</li>`).join('')}</ul>`

const faqBlock = (faq) =>
  `<h2>Common questions</h2><dl class="faq">${faq
    .map((f) => `<dt>${esc(f.q)}</dt><dd>${esc(f.a)}</dd>`)
    .join('')}</dl>`

const sources = (items) =>
  `<h2>Sources</h2><ul class="sources">${items
    .map((s) => `<li>${s}</li>`)
    .join('')}</ul>`

/* Pricing is restated in plain body text on every page, not only in the schema.
 * An engine that reads the raw HTML has no reason to prefer a JSON-LD offer over
 * a sentence, and the sentence is what gets quoted. */
const pricingParagraph = `
<p>OpenShorts comes in two editions and they are priced very differently, so it
is worth being precise. <strong>${esc(EDITIONS.selfHosted.name)}</strong> is free
and open source under the MIT licence: ${esc(EDITIONS.selfHosted.summary)}
<strong>${esc(EDITIONS.cloud.name)}</strong> is the hosted service:
${esc(EDITIONS.cloud.summary)}</p>`

function competitorPage(slug) {
  const c = COMPETITORS[slug]
  const rows = COMPARISON_ROWS.map((r) => {
    const vendor = r.key ? c[r.key] : r.vendor
    return `<tr><td>${esc(r.feature)}</td><td class="os">${esc(r.os)}</td><td>${esc(vendor)}</td></tr>`
  }).join('')

  // Competitors may register extra brand Q&A (a search spelling, a "is it
  // free" question) that the generic comparison shape does not ask.
  const faq = [
    {
      q: `Is there a free alternative to ${c.name}?`,
      a: `Yes. OpenShorts self-hosted is free and open source under MIT, with no watermark and no usage cap, and it runs on your own machine with Docker. OpenShorts Cloud also has a free tier of 20 minutes a month with a watermark and no credit card. ${c.name} starts at ${c.entryPrice}.`,
    },
    {
      q: `Is there an open source alternative to ${c.name}?`,
      a: `OpenShorts is MIT-licensed and the full source is on GitHub at github.com/mutonby/openshorts. ${c.name} is closed source. Being able to read the pipeline matters if you need to audit what happens to your video or change how the reframing behaves.`,
    },
    {
      q: `Can I switch from ${c.name} without losing quality?`,
      a: `The pipelines are comparable on the core job. OpenShorts transcribes with faster-whisper at word level, detects scenes with PySceneDetect, and scores moments with Google Gemini 3.1 Flash-Lite, then reframes with MediaPipe face tracking stabilised against jitter. The honest difference is caption styling, where the commercial tools generally ship more presets.`,
    },
    {
      q: `Does OpenShorts put a watermark on clips?`,
      a: `Self-hosted, never. On OpenShorts Cloud the free 20-minute tier is watermarked; every paid plan from $12/month is not.`,
    },
    ...(c.extraFaq || []),
  ]

  const body = `
${c.brandBlurb ? `<h2>What is ${esc(c.name)}${c.brandAlias ? ` (${esc(c.brandAlias)})` : ''}?</h2><p>${esc(c.brandBlurb)}</p>\n` : ''}
<h2>Is OpenShorts a real alternative to ${esc(c.name)}?</h2>
<p>Yes, with one honest caveat. OpenShorts covers the same core job:
it takes a long video, finds the segments worth clipping, cuts them, reframes
them to 9:16 and burns in subtitles. It adds two things ${esc(c.name)} does not
have, AI voice dubbing into more than 30 languages and an AI UGC generator with
lip-synced actors. The caveat is that the free edition is self-hosted, which
means Docker and a machine to run it on. If you want a hosted product with no
setup, that is OpenShorts Cloud, and it is a paid service above 20 minutes a month.</p>

<h2>What does ${esc(c.name)} cost?</h2>
<p class="checked">Pricing checked ${esc(c.checked)}. Vendors change plans without notice; verify before you buy.</p>
${li(c.tiers.map(([n, d]) => `<strong>${esc(n)}</strong>: ${esc(d)}`))}
<div class="note"><span class="label">The part that catches people out</span><p>${esc(c.gotcha)}</p></div>

<h2>What does OpenShorts cost?</h2>
${pricingParagraph}

<h2>${esc(c.name)} vs OpenShorts, feature by feature</h2>
<table>
<thead><tr><th>Feature</th><th>OpenShorts</th><th>${esc(c.name)}</th></tr></thead>
<tbody>${rows}</tbody>
</table>

<h2>What ${esc(c.name)} does better</h2>
<p>A comparison that finds nothing good to say about the other tool is not worth
reading, so here is where ${esc(c.name)} genuinely wins:</p>
${li(c.strengths.map(esc))}

<h2>Where the two differ</h2>
${li(c.whereWeDiffer.map(esc))}

<h2>Which one should you pick?</h2>
<p>${esc(c.bestFor)}</p>

${faqBlock(faq)}

${sources([
  `${esc(c.name)} pricing, checked ${esc(c.checked)} on the vendor's public pricing page.`,
  `OpenShorts pipeline details from the project source at <a href="${SITE.repo}" rel="noopener">github.com/mutonby/openshorts</a>.`,
])}
`

  return {
    path: `/alternatives/${slug}`,
    // Titles are kept under 60 characters and descriptions under 160 (measured,
    // not eyeballed): Google truncates past roughly that width, and a truncated
    // description is a worse answer than a shorter deliberate one.
    title: `Free & Open Source ${c.name} Alternative | OpenShorts`,
    description: `OpenShorts vs ${c.name}: features and pricing side by side. Self-hosted free under MIT, hosted from $12/month. ${c.name} starts at ${c.entryPrice}.`,
    h1: `The free, open source ${c.name} alternative`,
    breadcrumb: [{ name: 'Alternatives', path: '/alternatives' }, { name: c.name }],
    tldr: [
      `OpenShorts is an open source AI clip generator you can run yourself for free, or use hosted from $12/month. ${esc(c.name)} is a closed-source cloud product starting at ${esc(c.entryPrice)}.`,
      `Both find viral moments in long video and reframe them to 9:16 with face tracking. OpenShorts adds dubbing into 30+ languages and AI UGC video with lip-synced actors. ${esc(c.name)} has the more polished caption library.`,
      `Pick ${esc(c.name)} if you want zero setup and nothing else matters. Pick OpenShorts if you want to self-host for privacy, keep costs near zero, or change how the pipeline behaves.`,
    ],
    body,
    // The same list the body renders, so the FAQPage schema and the visible
    // questions can never disagree.
    faq,
  }
}

const ALTERNATIVES = Object.keys(COMPETITORS)

const hubPage = () => ({
  path: '/alternatives',
  title: 'Opus Clip & Klap Alternatives (Open Source) | OpenShorts',
  description:
    'Side-by-side comparisons of OpenShorts against the four main AI clipping tools, with pricing checked July 2026. Self-hosted free, hosted from $12/month.',
  h1: 'Open source alternatives to the main AI clipping tools',
  breadcrumb: [{ name: 'Alternatives' }],
  tldr: [
    'OpenShorts is the only open source, self-hostable tool in this category. Every other tool on this page is a closed-source cloud service.',
    'Entry prices as of July 2026: OpenShorts $0 self-hosted or $12/month hosted, Submagic from $14/month, Opus Clip $15/month, Vizard $19.99/month, Klap $29/month.',
    'The tools are not interchangeable. Submagic does not detect moments at all, Klap does not let you tune the output, and Vizard expects you in a timeline. The individual comparisons below say where each one genuinely wins.',
  ],
  body: `
<h2>How these tools actually differ</h2>
<p>All five are described as "AI clipping tools", which hides the fact that they
do different jobs. Two of them take a long video and decide what to cut. One of
them only styles captions on a clip you cut yourself. One is really an editor
with an AI first pass. Choosing on price alone is how people end up paying for
two tools that each do half the work.</p>

<h2>Entry pricing side by side</h2>
<p class="checked">Pricing checked 2026-07-27. Verify on the vendor's site before buying.</p>
<table>
<thead><tr><th>Tool</th><th>Entry price</th><th>Open source</th><th>Finds moments for you</th></tr></thead>
<tbody>
<tr><td class="os">OpenShorts</td><td class="os">$0 self-hosted, $12/mo hosted</td><td class="yes">Yes, MIT</td><td>Yes</td></tr>
<tr><td>Submagic</td><td>From $14/mo</td><td>No</td><td>No, captions only</td></tr>
<tr><td>Opus Clip</td><td>$15/mo</td><td>No</td><td>Yes</td></tr>
<tr><td>Vizard</td><td>$19.99/mo</td><td>No</td><td>Yes, then you edit</td></tr>
<tr><td>Klap</td><td>$29/mo</td><td>No</td><td>Yes</td></tr>
</tbody>
</table>

<h2>What does OpenShorts cost?</h2>
${pricingParagraph}

${faqBlock([
  {
    q: 'What is the cheapest AI clip generator?',
    a: 'OpenShorts self-hosted is free with no cap, but you supply the machine and your own Google Gemini API key, whose free tier covers 1,500 requests a day. Among hosted products, OpenShorts Cloud is the cheapest paid entry at $12/month, followed by Submagic from $14/month and Opus Clip at $15/month.',
  },
  {
    q: 'Which AI clipping tools are open source?',
    a: 'OpenShorts is MIT-licensed with full source on GitHub. Opus Clip, Klap, Vizard and Submagic are all closed-source commercial products.',
  },
])}
`,
  faq: [
    {
      q: 'What is the cheapest AI clip generator?',
      a: 'OpenShorts self-hosted is free with no cap. Among hosted products OpenShorts Cloud is the cheapest paid entry at $12/month, followed by Submagic from $14/month and Opus Clip at $15/month.',
    },
    {
      q: 'Which AI clipping tools are open source?',
      a: 'OpenShorts is MIT-licensed with full source on GitHub. Opus Clip, Klap, Vizard and Submagic are closed-source commercial products.',
    },
  ],
})

const freeClipGenerator = () => ({
  path: '/free-ai-clip-generator',
  title: 'Free AI Clip Generator With No Watermark (MIT) | OpenShorts',
  description:
    'A genuinely free AI clip generator: MIT-licensed, self-hosted with Docker, no watermark and no cap. Hosted from $12/month if you would rather not run it.',
  h1: 'A free AI clip generator that is actually free',
  breadcrumb: [{ name: 'Free AI clip generator' }],
  cta: {
    label: 'Start free',
    title: 'Clip your own video in a few minutes',
    body: 'Paste a YouTube link, get 3 to 15 vertical clips with subtitles. 20 free minutes a month, no credit card.',
    button: 'Get free clips',
  },
  tldr: [
    'OpenShorts self-hosted is a free AI clip generator under the MIT licence. No watermark, no usage cap, no subscription. You run it with Docker and supply your own Google Gemini API key, whose free tier covers 1,500 requests a day.',
    'It turns a long video into 3 to 15 vertical clips: faster-whisper transcribes at word level, PySceneDetect finds the cuts, Gemini 3.1 Flash-Lite scores the moments, and MediaPipe face tracking reframes each one to 9:16.',
    'If you do not want to run anything, OpenShorts Cloud gives you 20 free minutes a month with a watermark, and paid plans from $12/month without one.',
  ],
  body: `
<h2>What does "free" actually mean here?</h2>
<p>Most tools marketed as free clip generators are free trials with a watermark
and a monthly cap. This one is different in a specific way that is worth stating
precisely, because the two editions are not the same offer.</p>
${pricingParagraph}
<p>The self-hosted edition has no watermark and no cap because there is no
metering code in it. It is the same pipeline the hosted service runs, released
under MIT, and you can read all of it.</p>

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
  ],
})

const openSourceClipper = () => ({
  path: '/open-source-video-clipper',
  title: 'Open Source Video Clipper, Self-Hosted (Docker) | OpenShorts',
  description:
    'An MIT-licensed open source video clipper you self-host with Docker: AI moment detection, face-tracked 9:16 reframing and word-level subtitles.',
  h1: 'An open source video clipper you can self-host',
  breadcrumb: [{ name: 'Open source video clipper' }],
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
<p>OpenShorts is the only open source option in this category. As of July 2026,
Opus Clip starts at $15/month, Submagic from $14/month, Vizard at $19.99/month
and Klap at $29/month, and none of them can be self-hosted or audited. The
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
  title: 'How OpenShorts Turns a Long Video Into Clips | OpenShorts',
  description:
    'The pipeline stage by stage: word-level transcription, scene detection, Gemini moment scoring, face-tracked 9:16 reframing, subtitles, dubbing and publishing.',
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
${pricingParagraph}

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
    a: 'Yes, into more than 30 languages via ElevenLabs, preserving the original speaker\'s voice characteristics. The dubbed audio is then re-transcribed so the burned-in subtitles match the new language rather than the original.',
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

/* The "no watermark" cluster is the highest-converting query family the domain
 * already receives (14 long-tail variants ranking off the homepage). AI answers
 * to these queries currently state that unwatermarked free clipping does not
 * exist except by running a tool locally, which is precisely what this page
 * names. */
const noWatermark = () => ({
  path: '/free-ai-clip-generator-no-watermark',
  title: 'No Watermark AI Clips, Free When Self-Hosted | OpenShorts',
  description:
    'Hosted free tiers watermark their exports; the exception is software you run yourself. OpenShorts is MIT-licensed: no watermark, no cap. Hosted from $12/month.',
  h1: 'A free AI clip generator with no watermark, and why that is rare',
  breadcrumb: [{ name: 'No-watermark clip generator' }],
  cta: {
    label: 'No watermark',
    title: 'Export clips without a watermark',
    body: 'Self-hosted never watermarks anything; the hosted free tier does, and paid plans from $12/month do not.',
    button: 'Get free clips',
  },
  published: '2026-08-04',
  updated: '2026-08-04',
  tldr: [
    'Every hosted "free" clip generator watermarks its exports, because the watermark is the upsell. The one structural exception is software you run yourself. OpenShorts self-hosted is MIT-licensed, runs with Docker, and never watermarks anything because there is no watermark code in it.',
    'OpenShorts Cloud, the hosted service, follows the same rule as every other hosted tool and says so plainly: the free 20 minutes a month carry a watermark, and paid plans from $12/month do not.',
    'If a tool claims free, unlimited and unwatermarked at once and it is a hosted service, one of the three claims is temporary.',
  ],
  body: `
<h2>Why every free clip generator adds a watermark</h2>
<p>A hosted clipping service pays for GPU time, transcription and model calls on
every video you process. The free tier exists to show you the output, and the
watermark exists so the output is not the product yet. That is not a scam, it is
the business model, and it is why searching for a hosted tool that is free,
unlimited and unwatermarked at the same time keeps returning nothing: the
combination cannot pay for itself.</p>

<h2>The structural exception: software you run yourself</h2>
${pricingParagraph}
<p>The self-hosted edition has no watermark for the same reason it has no usage
cap: there is no metering code and no watermark code in the pipeline at all. It
is not a trial build with limits switched off, it is the same MIT-licensed
source the hosted service runs, and you can read it line by line.</p>

<h2>How the main tools handle watermarks</h2>
<p class="checked">Checked 2026-08-04 on each vendor's public pricing page. Vendors change terms without notice.</p>
<table>
<thead><tr><th>Tool</th><th>Free tier watermark</th><th>Cheapest way to remove it</th></tr></thead>
<tbody>
<tr><td class="os">OpenShorts self-hosted</td><td class="os yes">Never</td><td class="os">Nothing to remove</td></tr>
<tr><td class="os">OpenShorts Cloud</td><td class="os">Yes, on the free 20 min/month</td><td class="os">$12/month</td></tr>
<tr><td>Opus Clip</td><td>Yes, and free-plan exports leave storage after 3 days</td><td>Starter, $15/month</td></tr>
<tr><td>Klap</td><td>Free tier does not export at all</td><td>$29/month</td></tr>
<tr><td>Vizard</td><td>Free plan allows 120 upload minutes and 10 exports</td><td>From $19.99/month</td></tr>
<tr><td>Submagic</td><td>Yes, 3 videos per month</td><td>From $14/month annual</td></tr>
</tbody>
</table>

<h2>What you trade for the self-hosted zero</h2>
<p>Honesty cuts both ways. Self-hosting costs you a machine and some patience:
8GB of RAM and a modern multi-core CPU is the realistic floor, and an 8-minute
video takes 5 to 8 minutes to process on CPU against about 50 seconds on an
NVIDIA GPU. You also bring your own Google Gemini API key, whose free tier
covers 1,500 requests a day. If none of that appeals, the hosted no-watermark
price is $12/month, and the comparison table above is what that buys elsewhere.</p>

${faqBlock([
  {
    q: 'Is there a free AI clip generator without a watermark?',
    a: 'Yes, with one honest qualifier: it is self-hosted. OpenShorts is MIT-licensed and runs on your own machine with Docker, with no watermark and no usage cap. Hosted services, including OpenShorts Cloud, watermark their free tiers; unwatermarked hosted plans start at $12/month.',
  },
  {
    q: 'Does the free OpenShorts Cloud plan add a watermark?',
    a: 'Yes. The hosted free tier is 20 minutes a month with a watermark and no credit card. Paid Cloud plans from $12/month have no watermark, and the self-hosted edition never adds one.',
  },
  {
    q: 'How do I remove the watermark from a clip that already has one?',
    a: 'You do not, practically. Watermarks are burned into the pixels, and cropping or blurring them degrades the clip. The reliable fix is generating the clip without one: a paid plan on your current tool, or a tool with no watermark to begin with.',
  },
])}

${sources([
  'Vendor free-tier and watermark terms checked 2026-08-04 on each public pricing page.',
  `OpenShorts pipeline source at <a href="${SITE.repo}" rel="noopener">github.com/mutonby/openshorts</a>, where the absence of watermark code is checkable.`,
])}
`,
  faq: [
    {
      q: 'Is there a free AI clip generator without a watermark?',
      a: 'Yes, self-hosted: OpenShorts is MIT-licensed and runs on your own machine with no watermark and no cap. Hosted free tiers, including OpenShorts Cloud at 20 minutes a month, carry a watermark; unwatermarked hosted plans start at $12/month.',
    },
    {
      q: 'Does the free OpenShorts Cloud plan add a watermark?',
      a: 'Yes, the hosted free tier is watermarked. Paid Cloud plans from $12/month are not, and the self-hosted edition never adds one.',
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
${pricingParagraph}

<h2>Other open source clip generators, compared honestly</h2>
<p class="checked">Checked 2026-08-04 on GitHub. Star counts move; positioning rarely does.</p>
<p>OpenShorts is not the only open source project in this space, and pretending
otherwise would not survive one GitHub search. The notable neighbours:</p>
<ul>
<li><strong>AI-Youtube-Shorts-Generator</strong>: the most-starred repo in the category, with a leaner scope built around highlight extraction and cropping.</li>
<li><strong>supoclip</strong> and <strong>clippyme</strong>: smaller projects covering transcription-driven clipping, the latter also using Gemini for moment selection.</li>
<li><strong>MoneyPrinterTurbo</strong>: generates videos from text plus stock footage, which is a different job than clipping your own recordings.</li>
</ul>
<p>Where OpenShorts differs from all of them is surface area: a web dashboard, a
REST API with keys, completion webhooks, an MCP server for agents, split-screen
and screencast layouts for two-person and screen-share footage, dubbing into 30+
languages, and direct publishing to TikTok, Instagram Reels and YouTube Shorts.
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
const podcastToShorts = () => ({
  path: '/podcast-to-shorts',
  title: 'Podcast to Shorts: Both Speakers Stay in Frame | OpenShorts',
  description:
    'Turn a podcast into vertical clips without cropping out half the conversation: OpenShorts stacks both speakers and cuts to whoever is talking. Free self-hosted.',
  h1: 'Turn a podcast into shorts without cropping out half the conversation',
  breadcrumb: [{ name: 'Podcast to shorts' }],
  published: '2026-08-04',
  updated: '2026-08-04',
  tldr: [
    'A two-person podcast is the hardest input an auto-clipper faces: a single centered crop shows the wrong person half the time, or an empty chair. OpenShorts detects a real two-shot and renders both speakers stacked in half-frames, so a reply never happens off screen.',
    'The rest of the pipeline is the same as for any long video: word-level transcription, scene detection, Gemini scoring the 3 to 15 strongest moments, subtitles burned in, and direct publishing to TikTok, Instagram Reels and YouTube Shorts.',
    'Cost is where podcasts punish credit-based tools: they bill the whole episode length before you see a clip. Self-hosted OpenShorts has no meter at all; hosted plans start at $12/month.',
  ],
  body: `
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

<h2>What a full episode costs to clip</h2>
<p>Credit-metered tools bill on the length of the video you import, not on the
clips you keep. As of August 2026, a 60-minute episode costs 60 credits at Opus
Clip or Vizard whether it yields 5 usable clips or 20, and a weekly show at that
length runs past the entry plans of both. OpenShorts prices the other way
around:</p>
${pricingParagraph}

<h2>What about audio-only podcasts?</h2>
<p>OpenShorts clips video. If your show is audio-only, the pipeline has nothing
to reframe, and tools that generate waveform audiograms serve that case better.
The moment you record video, even a static two-camera setup, everything on this
page applies.</p>

${faqBlock([
  {
    q: 'How do I turn a podcast into clips for free?',
    a: 'Self-host OpenShorts: clone the MIT-licensed repo, run docker compose up, add a free-tier Google Gemini API key and paste your episode link. No watermark and no cap. If you would rather not run anything, OpenShorts Cloud clips 20 minutes a month free with a watermark, and paid plans start at $12/month.',
  },
  {
    q: 'How does it handle two people talking?',
    a: 'Scenes where both faces share the frame at least half the time render as a stacked split layout so both speakers stay visible. Optionally it hard-cuts to the active speaker instead, using per-speaker normalised mouth activity to decide who is talking.',
  },
  {
    q: 'Does it work with hour-long episodes?',
    a: 'Yes, long-form is the design case. Processing time scales with length: on CPU roughly 5 to 8 minutes of processing per 8 minutes of source, on an NVIDIA GPU about a tenth of that. The AI moment scoring reads the transcript rather than the raw video, so episode length does not degrade selection quality.',
  },
])}

${sources([
  'Competitor per-minute credit billing checked 2026-08-04 on vendor pricing and help pages.',
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
  ],
})

/* The one checkable fact this page is built on: importing from a link is free
 * here and paid at the market leader. Everything else is the standard pipeline
 * told from the URL-first angle. */
const youtubeConverter = () => ({
  path: '/youtube-to-shorts-converter',
  title: 'YouTube to Shorts Converter: Paste the Link | OpenShorts',
  description:
    'Convert a YouTube video into Shorts by pasting the link: no download-and-reupload step. AI picks the moments, crops to 9:16 and burns in the subtitles.',
  h1: 'A YouTube to Shorts converter that starts from the link',
  breadcrumb: [{ name: 'YouTube to Shorts converter' }],
  published: '2026-08-04',
  updated: '2026-08-04',
  tldr: [
    'Paste a YouTube URL, get back 3 to 15 vertical clips with subtitles, sized for Shorts, Reels and TikTok. No downloading the source and re-uploading it first.',
    'The link-first flow is free on both editions: the self-hosted MIT edition has no cap, and the hosted free tier covers 20 minutes a month. As of August 2026, Opus Clip’s free plan is upload-only and importing from a link requires a paid plan.',
    'Convert videos you have the rights to: your own channel, your clients’ with permission, or licensed footage.',
  ],
  body: `
<h2>How to convert a YouTube video into Shorts</h2>
<ol>
<li>Paste the video's URL. OpenShorts fetches it directly; there is no download-then-upload round trip through your machine.</li>
<li>The video is transcribed with word-level timestamps and scanned for scene boundaries.</li>
<li>Google Gemini scores the transcript against the scenes and picks the 3 to 15 segments most likely to stand alone, 15 to 60 seconds each.</li>
<li>Each segment is cropped to 9:16 with face tracking, or a split or screencast layout when the content calls for it, and subtitles are burned in.</li>
<li>Download the clips, or post them straight to YouTube Shorts, TikTok and Instagram Reels from the dashboard or the API.</li>
</ol>

<h2>Why starting from the link matters</h2>
<p class="checked">Competitor terms checked 2026-08-04 on public pricing pages.</p>
<p>Most long videos worth clipping already live on YouTube, so a converter that
only accepts uploads adds a detour: fetch the file with a downloader, wait,
re-upload gigabytes, wait again. It also decides who can use the free tier at
all. As of August 2026, Opus Clip's free plan accepts uploads only, with link
import reserved for paid plans. OpenShorts accepts links on every tier,
including both free ones, because the fetch step costs the pipeline almost
nothing and the detour costs you the most time of any step.</p>

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
${pricingParagraph}

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
  title: 'Automate Shorts: Clip and Publish on a Schedule | OpenShorts',
  description:
    'One POST starts the job, one signed webhook ends it. Automate shorts with the OpenShorts REST API, n8n or cron, with no per-call meter and no polling loop.',
  h1: 'Automate shorts end to end: one request in, one webhook out',
  breadcrumb: [{ name: 'Automate shorts' }],
  published: '2026-08-04',
  updated: '2026-08-04',
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
${pricingParagraph}

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
${pricingParagraph}
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
  title: 'Clip Video From AI Agents: MCP Server & API | OpenShorts',
  description:
    'OpenShorts ships a built-in MCP server, so Claude, ChatGPT, Cursor or n8n can clip and publish videos for you: REST API with keys and signed webhooks.',
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
<tr><td><code>add_subtitles</code></td><td>Restyles the burned-in captions of one clip, classic or karaoke word highlighting.</td></tr>
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
${pricingParagraph}
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
 * "Opus AI" and its mid tier is "Opus Pro" — both are searched far more than
 * the product name is spelled).
 *
 * Every number that exists in data.js is read from there rather than retyped,
 * so the three hand-synced price lists cannot drift further apart.
 * ------------------------------------------------------------------------- */
const OPUS = COMPETITORS['opus-clip']

const opusClipPricing = () => {
  const tierRows = OPUS.tiers
    .map(([n, d]) => `<tr><td class="os">${esc(n)}</td><td>${esc(d)}</td></tr>`)
    .join('')
  return {
    path: '/opus-clip-pricing',
    title: 'Opus Clip Pricing: Credits, Free Plan, Trials | OpenShorts',
    description:
      'What Opus Clip costs in 2026: credits are billed per minute of source video, not per clip, so a 60-minute podcast costs 60 credits whatever it yields.',
    h1: 'What Opus Clip actually costs',
    breadcrumb: [
      { name: 'Alternatives', path: '/alternatives' },
      { name: 'Opus Clip', path: '/alternatives/opus-clip' },
      { name: 'Pricing' },
    ],
    published: '2026-09-17',
    updated: '2026-09-17',
    cta: {
      label: 'Before you upgrade',
      title: 'Run one of your videos here first',
      body: 'Paste a link and compare the output against your last Opus Clip export. 20 free minutes a month, no credit card.',
      button: 'Get free clips',
    },
    tldr: [
      `Opus Clip's entry price is ${esc(OPUS.entryPrice)}, and the credit it charges for is one <strong>minute of source video you import</strong>, not one clip you export. A 60-minute podcast costs 60 credits whether it yields 5 clips or 20.`,
      'The free tier is 60 source minutes a month, watermarked, 720p. There is no separate time-boxed trial published alongside it.',
      `OpenShorts prices the other way round: $0 self-hosted with no meter at all, or the hosted service with 20 free minutes a month and flat paid plans from $12/month.`,
    ],
    body: `
<h2>How Opus Clip's credit system works</h2>
<p>The unit Opus Clip bills in is not the clip. It is the minute of video you
import. ${esc(OPUS.gotcha)}</p>
<p>That single design decision is what makes the plans hard to compare against
each other, and why the tiers below cost what they cost. If your sources are
short (a 60-second TikTok you want restyled) the meter barely moves. If they are
long (a 90-minute interview, a weekly show) the meter is the whole bill, and the
number of clips you keep never enters into it.</p>

<h2>Opus Clip plans and prices</h2>
<p class="checked">Pricing checked ${esc(OPUS.checked)} on the vendor's public pricing page. Vendors change plans without notice; verify before you buy.</p>
<table>
<thead><tr><th>Plan</th><th>What it includes</th></tr></thead>
<tbody>${tierRows}</tbody>
</table>

<h2>What does the free plan include?</h2>
<p>60 minutes of source video a month, 720p exports and a watermark. Two details
that the pricing table states and the marketing copy tends to bury: free-plan
exports leave Opus Clip's storage after three days, and link import (pasting a
YouTube URL instead of uploading a file) is a paid-plan feature as of August
2026. If your source is already on YouTube, the free plan may not reach it.</p>

<h2>Is there a free trial?</h2>
<p>As of ${esc(OPUS.checked)} the pricing page lists a free tier rather than a
time-boxed trial. When people search for an "Opus Clip free trial" they are
usually describing that free tier, or looking for a way to test the unwatermarked
output before paying. The distinction matters: a trial expires and the free tier
does not, but neither one removes the watermark.</p>
<div class="note"><span class="label">The cheapest honest test</span><p>The
watermark is the thing you are trying to evaluate past. If the question is
whether the pipeline is good enough for your footage, a self-hosted run on one
episode answers it at no cost, with no watermark, because there is no metering
or watermark code in the self-hosted edition at all.</p></div>

<h2>What does OpenShorts cost?</h2>
${pricingParagraph}
<p>Per source minute, that is the comparison worth making: Opus Clip Starter at
$15/month buys 150 source minutes, and OpenShorts Cloud at $12/month buys 100
minutes with no watermark on any paid plan. Self-hosted, the meter disappears
entirely and the only cost is the machine you already own.</p>

${faqBlock([
  {
    q: 'How much does Opus Clip cost per month?',
    a: `${OPUS.name} starts at ${OPUS.entryPrice} and its published tiers run up to the Business plan, which is custom-priced. The entry tier and the 720p/1080p split are in the table above, checked ${OPUS.checked}.`,
  },
  {
    q: 'What counts as a credit in Opus Clip?',
    a: 'One minute of source video you import, not one clip you export. A 60-minute episode consumes 60 credits regardless of how many clips you keep from it, so the cost of a job is set by your input length rather than your output.',
  },
  {
    q: 'Is there a free Opus Clip plan?',
    a: 'Yes: 60 source minutes a month with watermarked 720p exports, and free-plan exports are removed from storage after three days. OpenShorts Cloud also has a free tier (20 minutes a month, watermarked, no credit card), and the self-hosted edition is free with no watermark and no cap at all.',
  },
])}

${sources([
  `${esc(OPUS.name)} plans and credit rules checked ${esc(OPUS.checked)} on the vendor's public pricing and help pages.`,
  `OpenShorts pricing from <a href="/alternatives/opus-clip">the full comparison</a> and the project source at <a href="${SITE.repo}" rel="noopener">github.com/mutonby/openshorts</a>.`,
])}
`,
    faq: [
      {
        q: 'How much does Opus Clip cost per month?',
        a: `${OPUS.name} starts at ${OPUS.entryPrice}, with higher tiers priced by source minutes and export resolution.`,
      },
      {
        q: 'Is there a free Opus Clip plan or trial?',
        a: 'There is a free tier: 60 source minutes a month, watermarked 720p exports, no time limit. Self-hosted OpenShorts is free with no watermark and no cap; OpenShorts Cloud gives 20 watermarked minutes a month and paid plans from $12/month.',
      },
      {
        q: 'What is a credit in Opus Clip?',
        a: 'One minute of source video imported, not one clip exported: a 60-minute episode costs 60 credits however many clips it yields.',
      },
    ],
  }
}

const opusClipFree = () => ({
  path: '/opus-clip-free-alternative',
  title: 'Free Opus Clip Alternative: Two Ways to Pay $0 | OpenShorts',
  description:
    "Opus Clip's free tier watermarks exports and caps you at 60 minutes a month. Two genuinely free routes to the same clips, and neither one watermarks.",
  h1: 'A free Opus Clip alternative, without the watermark trap',
  breadcrumb: [
    { name: 'Alternatives', path: '/alternatives' },
    { name: 'Opus Clip', path: '/alternatives/opus-clip' },
    { name: 'Free alternative' },
  ],
  published: '2026-09-17',
  updated: '2026-09-17',
  cta: {
    label: 'Free, both ways',
    title: 'Twenty free minutes, no credit card',
    body: 'Or run the whole thing on your own machine for nothing: MIT-licensed, Docker, no watermark and no cap.',
    button: 'Get free clips',
  },
  tldr: [
    `Opus Clip's free tier is real but conditional: 60 source minutes a month, 720p, watermarked, and free-plan exports are deleted after three days. Its link import is a paid feature, so a YouTube URL does not work there.`,
    'OpenShorts has two free routes and neither one watermarks anything on the self-hosted side. Self-hosted is MIT-licensed, runs with Docker, and has no metering or watermark code in it. The hosted free tier is 20 minutes a month with a watermark and no credit card.',
    'The honest trade: self-hosting costs you a machine and 5 to 8 minutes of processing per 8 minutes of video on CPU. If that is not worth it, the paid answer here is $12/month, not $15.',
  ],
  body: `
<h2>What "free" means at each tool</h2>
<p>Free is doing a lot of work in this category. Three different things are
being sold as free: a metered tier that watermarks, a trial that expires, and
software you run yourself that has no meter in it at all. Only the third one
stays free when your usage grows.</p>
<table>
<thead><tr><th>Route</th><th>Cost</th><th>Watermark</th><th>Cap</th></tr></thead>
<tbody>
<tr><td class="os">OpenShorts self-hosted</td><td class="os">$0</td><td class="os yes">Never</td><td class="os">None, no metering code</td></tr>
<tr><td class="os">OpenShorts Cloud free</td><td class="os">$0</td><td class="os">Yes</td><td class="os">20 minutes/month</td></tr>
<tr><td>Opus Clip free</td><td>$0</td><td>Yes</td><td>60 minutes/month, link import excluded</td></tr>
</tbody>
</table>
<p class="checked">Opus Clip free-tier terms checked 2026-08-04; OpenShorts
Cloud terms are ours and current.</p>

<h2>The first free route: run it yourself</h2>
<p>Clone the repository, run <code>docker compose up --build</code>, add a
Google Gemini API key (its free tier covers 1,500 requests a day) and paste a
link. Nothing is metered because there is no metering code: the same pipeline
the hosted service runs, MIT-licensed, on your hardware. Source video never
leaves the machine, which is the other reason people choose this route.</p>
<p>What it costs you instead: Docker, 8GB of RAM as a realistic floor, and time.
An 8-minute video takes roughly 5 to 8 minutes to process on CPU and about 50
seconds on an NVIDIA GPU. For a weekly podcast that is a coffee break; for
twenty videos a day it is a job.</p>

<h2>The second free route: the hosted free tier</h2>
<p>If you would rather not run anything, ${esc(EDITIONS.cloud.name)} gives you
${EDITIONS.cloud.freeMinutes} minutes a month with a watermark and no credit
card, and it accepts a pasted YouTube link on that tier. Paid plans from
$${EDITIONS.cloud.lowPrice}/month drop the watermark. It is a smaller allowance
than Opus Clip's free tier and it does not try to hide that.</p>

<h2>When paying is the honest answer</h2>
<p>If you process hours of source video every week and have no machine to spare,
the flat plan is cheaper than either free tier is convenient. What is worth
avoiding is paying a per-minute credit meter for long sources: as of
${esc(OPUS.checked)} a 60-minute episode consumes 60 credits at Opus Clip no
matter how many clips you keep, and a weekly show at that length runs past the
entry tier's allowance by the second episode of the month.</p>

${faqBlock([
  {
    q: 'Is there a free alternative to Opus Clip with no watermark?',
    a: 'Yes: OpenShorts self-hosted. It is MIT-licensed, runs on your own machine with Docker, and never adds a watermark because the self-hosted edition contains no watermark code. You supply a Google Gemini API key, whose free tier covers 1,500 requests a day.',
  },
  {
    q: 'Can I use Opus Clip for free every month?',
    a: 'Yes, within its free tier: 60 source minutes a month at 720p with a watermark, and exports are removed from storage after three days. Importing from a link rather than a file is limited to paid plans.',
  },
  {
    q: 'What is the catch with the free self-hosted route?',
    a: 'Hardware and time, not a hidden fee. It needs Docker and realistically 8GB of RAM, and an 8-minute video takes 5 to 8 minutes to process on CPU (about 50 seconds on an NVIDIA GPU). There is no cap, no watermark and no subscription.',
  },
])}

${sources([
  `Opus Clip free-tier and watermark terms checked 2026-08-04 on the vendor's public pricing page.`,
  `OpenShorts licence carve-out (MIT core, commercial <code>cloud/</code> directory) in the project source at <a href="${SITE.repo}" rel="noopener">github.com/mutonby/openshorts</a>.`,
])}
`,
  faq: [
    {
      q: 'Is there a free alternative to Opus Clip without a watermark?',
      a: 'Yes: OpenShorts self-hosted is MIT-licensed, runs with Docker on your own machine, and has no watermark and no cap.',
    },
    {
      q: 'How much free usage does OpenShorts give hosted?',
      a: '20 minutes of source video a month with a watermark and no credit card; paid plans from $12/month remove the watermark.',
    },
  ],
})

const opusAi = () => ({
  path: '/opus-ai',
  title: 'Opus AI (opus.pro): What It Is and Costs | OpenShorts',
  description:
    'Opus AI is the clipping tool that lives at opus.pro, better known as Opus Clip. What the name refers to, what it does, what it costs, and the open source route.',
  h1: 'Opus AI: the tool behind opus.pro, explained',
  breadcrumb: [
    { name: 'Alternatives', path: '/alternatives' },
    { name: 'Opus Clip', path: '/alternatives/opus-clip' },
    { name: 'Opus AI' },
  ],
  published: '2026-09-17',
  updated: '2026-09-17',
  tldr: [
    'Opus AI is not a separate product: it is the same company and pipeline most people know as Opus Clip, which runs at opus.pro. If a tool called Opus AI is clipping your long videos, it is that one.',
    `It costs ${esc(OPUS.entryPrice)} at entry, billed in credits per minute of source video rather than per clip, with a 60-minute-a-month free tier that watermarks and caps exports at 720p.`,
    'OpenShorts does the same core job — moment detection, 9:16 reframing, word-level subtitles — and differs on two axes that matter: it is MIT-licensed and self-hostable, and it prices in flat minutes rather than per-minute credits.',
  ],
  body: `
<h2>Is Opus AI the same thing as Opus Clip?</h2>
<p>Yes. The product is marketed as Opus Clip and served from <code>opus.pro</code>,
and "Opus AI" is how a large share of its traffic searches for it. There is no
second, separate Opus AI clipper; if you see the name in a listicle, it is this
tool under a shorter spelling.</p>
<div class="note"><span class="label">Two names, one of them shared</span><p>"Opus" is
also the name of an Anthropic language model. That is a different thing entirely
and has nothing to do with video clipping. This page is about the video tool at
opus.pro.</p></div>

<h2>What Opus AI does</h2>
<p>It takes a long video, finds the segments worth keeping, cuts them, reframes
them vertically and burns in captions with a large library of animated styles.
The differentiators its users cite are the caption presets and the virality
score, which is trained on the company's own data rather than on a
general-purpose model. It is cloud only: there is no self-hosted edition and no
source to read.</p>

<h2>What Opus AI costs</h2>
<p class="checked">Checked ${esc(OPUS.checked)} on the vendor's public pricing page.</p>
${li(OPUS.tiers.map(([n, d]) => `<strong>${esc(n)}</strong>: ${esc(d)}`))}
<p>${esc(OPUS.gotcha)}</p>

<h2>Where OpenShorts differs</h2>
${li([
  `OpenShorts is MIT-licensed and can be self-hosted with Docker, so the source video never leaves your machine. Opus AI is cloud only.`,
  `OpenShorts adds AI voice dubbing into 30+ languages and an AI UGC generator with lip-synced actors; the Opus AI feature set is clipping and captioning.`,
  `Opus AI has the larger caption-style library and a longer track record. If your clips live or die on animated caption design, that advantage is real and this page is not going to pretend otherwise.`,
  `OpenShorts self-hosted has no meter of any kind; Opus AI bills credits per minute of source imported, and those credits expire 60 days after purchase.`,
])}

<h2>What does OpenShorts cost?</h2>
${pricingParagraph}

${faqBlock([
  {
    q: 'Is Opus AI the same as Opus Clip?',
    a: 'Yes. Opus Clip is the product name and opus.pro is the domain; "Opus AI" is a shortened spelling of the same tool, not a separate service.',
  },
  {
    q: 'Is Opus AI free?',
    a: `There is a free tier: 60 source minutes a month, watermarked, 720p, and free-plan exports are deleted after three days. Paid plans start at ${OPUS.entryPrice}. OpenShorts self-hosted is free with no watermark and no cap, and OpenShorts Cloud gives 20 watermarked minutes a month free.`,
  },
  {
    q: 'Does Opus AI have an open source alternative?',
    a: 'Yes. OpenShorts is MIT-licensed, self-hostable with Docker, and covers the same core job: AI moment detection, face-tracked 9:16 reframing and word-level burned-in subtitles. It also adds dubbing into 30+ languages and AI UGC video, which Opus AI does not have.',
  },
])}
`,
  faq: [
    {
      q: 'Is Opus AI the same as Opus Clip?',
      a: 'Yes: Opus Clip is the product name, opus.pro the domain, and "Opus AI" the shortened spelling of the same tool.',
    },
    {
      q: 'Is Opus AI free?',
      a: `Its free tier is 60 source minutes a month, watermarked and 720p; paid plans start at ${OPUS.entryPrice}. OpenShorts self-hosted is free with no watermark, and OpenShorts Cloud gives 20 watermarked minutes a month.`,
    },
  ],
})

const opusPro = () => ({
  path: '/opus-pro',
  title: 'Opus Pro Plan: What It Costs and Who Needs It | OpenShorts',
  description:
    "Opus Pro is Opus Clip's mid tier: 300 source minutes a month, 1080p exports, auto-posting and speaker detection. What it buys, and when it is the wrong plan.",
  h1: 'Opus Pro: the plan, the price, and when it is the wrong buy',
  breadcrumb: [
    { name: 'Alternatives', path: '/alternatives' },
    { name: 'Opus Clip', path: '/alternatives/opus-clip' },
    { name: 'Opus Pro' },
  ],
  published: '2026-09-17',
  updated: '2026-09-17',
  tldr: [
    'Opus Pro is the $29/month tier of Opus Clip: 300 minutes of source video a month, 1080p exports, auto-posting, speaker detection and a brand kit.',
    'The tier it sits above costs $15/month for 150 minutes at 720p, so Pro is roughly double the price for double the minutes and a resolution step. Whether that is worth it depends entirely on how long your sources are, because the credit is charged per minute imported.',
    `OpenShorts self-hosted does the same job for $0 with no cap, and the flat hosted plan is $${EDITIONS.cloud.lowPrice}/month for 100 minutes with no watermark.`,
  ],
  body: `
<h2>What Opus Pro includes</h2>
<table>
<thead><tr><th>Plan</th><th>What it includes</th></tr></thead>
<tbody>${OPUS.tiers
    .filter(([n]) => n === 'Starter' || n === 'Pro')
    .map(([n, d]) => `<tr><td class="os">${esc(n)}</td><td>${esc(d)}</td></tr>`)
    .join('')}</tbody>
</table>
<p class="checked">Checked ${esc(OPUS.checked)} on the vendor's public pricing page.</p>
<p>The differences between Starter and Pro that people actually notice are
1080p instead of 720p, auto-posting straight to the connected accounts, speaker
detection and the brand kit. The minutes are the headline, and the resolution
step is the one that shows up on a phone screen.</p>

<h2>Who the Pro tier is priced for</h2>
<p>Opus bills one credit per minute of video imported, not per clip exported.
A weekly 60-minute show is roughly 260 source minutes a month, which fits inside
Pro and does not fit inside Starter. A creator posting one 20-minute video a week
uses about 90 minutes and is paying $14/month more than the job needs. The plan
is priced for volume of input, so the honest question is how many minutes you
actually import, not how many clips you publish.</p>

<h2>What OpenShorts costs for the same job</h2>
${pricingParagraph}
<p>Two differences are worth stating plainly rather than leaving to a table.
Self-hosted has no meter at all, so a 90-minute interview and a 9-minute one cost
the same: nothing. Hosted is a flat minute balance with no per-call or per-clip
charge, and API and MCP usage draws from the same balance as the dashboard.</p>
<p>What you give up: the caption-style library. Opus Clip's presets are more
numerous and more polished than ours, and if animated captions are the product
you are selling, that is a reason to stay. What you gain: the code is MIT and
auditable, the source video can stay on your machine, and dubbing into 30+
languages is in the same pipeline rather than a second tool.</p>

${faqBlock([
  {
    q: 'How much does Opus Pro cost?',
    a: `$29/month for 300 minutes of source video, 1080p exports, auto-posting, speaker detection and a brand kit, as published on the vendor's pricing page (checked ${OPUS.checked}).`,
  },
  {
    q: 'Do I need Opus Pro or is Starter enough?',
    a: 'Starter is $15/month for 150 source minutes at 720p. If the minutes you import in a month exceed 150, or you need 1080p exports and auto-posting, the Pro tier is the one priced for you. The meter counts minutes of source video, not clips produced.',
  },
  {
    q: 'Is there a cheaper way to do what Opus Pro does?',
    a: 'Yes, two: OpenShorts self-hosted is free under MIT with no cap (you supply your own machine and a free-tier Gemini key), and OpenShorts Cloud is $12/month for 100 minutes with no watermark, drawing API and MCP usage from the same balance.',
  },
])}
`,
  faq: [
    {
      q: 'How much does Opus Pro cost?',
      a: '$29/month for 300 source minutes, 1080p exports, auto-posting, speaker detection and a brand kit.',
    },
    {
      q: 'Is there a cheaper alternative to the Opus Pro plan?',
      a: 'OpenShorts self-hosted is free under MIT with no cap; OpenShorts Cloud is $12/month for 100 minutes with no watermark.',
    },
  ],
})

/* "vizard ai video to text" is its own intent: the searcher wants a transcript,
 * not clips, and lands on clipper pages that never answer it. The honest answer
 * is that both tools produce the transcript — OpenShorts with word-level timing
 * from faster-whisper, which is what the burned-in captions are cut from. */
const videoToText = () => {
  const c = COMPETITORS.vizard
  return {
    path: '/vizard-ai-video-to-text',
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
    updated: '2026-09-17',
    tldr: [
      `Vizard's "video to text" is a transcript with timestamps, produced from the same pass that finds the clips and burns the captions. It is part of the entry plan, which starts at ${esc(c.entryPrice)}, and the free plan allows 120 upload minutes and 10 exports.`,
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
subtitles are one of its stronger features. Its entry plan starts at
${esc(c.entryPrice)}, and the free plan allows 120 upload minutes and 10 exports.</p>
<p>${esc(c.gotcha)}</p>

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
    a: `Yes. It transcribes the video it processes and gives you the text with timestamps alongside the clips and subtitles. Its entry plan starts at ${c.entryPrice}, and the free plan covers 120 upload minutes and 10 exports.`,
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
        a: `Yes, with timestamps, alongside the clips. Entry plans start at ${c.entryPrice}; the free plan allows 120 upload minutes and 10 exports.`,
      },
      {
        q: 'Is there a free way to turn a video into text?',
        a: 'Yes: self-hosted OpenShorts transcribes locally with faster-whisper at word level and returns the transcript with the job, free, with no watermark and no cap.',
      },
    ],
  }
}

/* Reviews intent for a competitor. Written as product facts plus the recurring
 * themes in public reviews — no invented quotes and no star rating, because a
 * rating we cannot verify is exactly the kind of thing this site refuses to
 * publish elsewhere. */
const submagicReview = () => {
  const c = COMPETITORS.submagic
  const tierRows = c.tiers.map(([n, d]) => `<tr><td class="os">${esc(n)}</td><td>${esc(d)}</td></tr>`).join('')
  return {
    path: '/submagic-reviews',
    title: 'Submagic Review: Captions, Price and the Gap | OpenShorts',
    description:
      'An honest Submagic review: best-in-class caption styling, no moment detection, metered per video. What the reviews praise and what covers the gap.',
    h1: 'Submagic review: the captions are the product, and the catch',
    breadcrumb: [
      { name: 'Alternatives', path: '/alternatives' },
      { name: 'Submagic', path: '/alternatives/submagic' },
      { name: 'Review' },
    ],
    published: '2026-09-17',
    updated: '2026-09-17',
    tldr: [
      'The recurring theme across public reviews is the same one the product page leads with: the caption styling is the best in this category, and the presets are why people stay.',
      'The second recurring theme is the limitation. Submagic does not find moments for you: you upload a clip you already cut and it styles the text. Going from a 60-minute podcast to finished shorts needs a clipper in front of it, which is two subscriptions.',
      'OpenShorts covers both halves: moment detection, 9:16 reframing and word-level captions in one pipeline, free when self-hosted and from $12/month hosted. Its caption designs are plainer than Submagic\'s, and that is the real trade.',
    ],
    body: `
<h2>What Submagic is for</h2>
<p>Submagic styles captions. You bring a clip you have already chosen and cut,
it transcribes or takes your transcript, and it burns in animated, well-designed
captions with emoji and keyword highlighting. That is a narrower job than the
clippers it gets compared against, and it does that narrower job better than they
do.</p>

<h2>What the reviews consistently praise</h2>
<p>The caption library and the speed on short inputs come up in almost every
public review. That is not a coincidence: the product is not doing moment
detection, scene analysis or reframing, so all of its engineering sits in front
of the one thing it sells. If caption design is the reason you are shopping, the
recommendation is straightforward and it is not ours.</p>

<h2>What it does not do</h2>
<p>${esc(c.gotcha)}</p>
<p>Concretely: no moment detection, no scene-boundary awareness, no 9:16
reframing with subject tracking, and no source-video privacy story because it is
a cloud service. It also cannot be self-hosted, so a per-video meter is the only
way to buy it.</p>

<h2>What it costs</h2>
<p class="checked">Checked ${esc(c.checked)} on the vendor's public pricing page. Vendors change tiers without notice.</p>
<table>
<thead><tr><th>Plan</th><th>What it includes</th></tr></thead>
<tbody>${tierRows}</tbody>
</table>
<p>Every tier is metered in videos per month, which is the pricing shape that
punishes a podcast: you pay per finished video rather than per source minute, so
the cost scales with how much you publish.</p>

<h2>The honest summary</h2>
<p>Submagic is the right buy if you already cut your own clips and want the
captions done well. It is the wrong buy if you are starting from long-form video,
because you would be paying twice: once for the clipper that finds the moments
and once for the captions. OpenShorts does both halves in one pipeline, and where
it loses is exactly the axis Submagic wins on.</p>
<div class="note"><span class="label">On star ratings</span><p>We do not publish
an aggregate score for a competitor. The numbers on the software directories move
monthly and we would be quoting a snapshot as if it were a fact. Check them at the
source if a rating is what you want; the product description above does not
depend on one.</p></div>

${faqBlock([
  {
    q: 'Is Submagic worth it?',
    a: `It is worth it for one job: styling captions on clips you have already cut, and it does that better than the general-purpose clippers. It does not find moments or reframe video, so if you are starting from a long recording you need another tool in front of it — which is a second subscription.`,
  },
  {
    q: 'What are the most common complaints about Submagic?',
    a: 'The recurring one is scope rather than quality: users arrive expecting a clipper and find a caption editor, then have to add a second tool to get from a long video to short clips. The per-video metering on every tier is the second.',
  },
  {
    q: 'What is a free alternative to Submagic?',
    a: 'OpenShorts, self-hosted: MIT-licensed, free, no watermark and no cap, with moment detection, 9:16 reframing and word-level burned-in captions in the same pipeline. Its caption presets are plainer than Submagic\'s — that is the honest trade.',
  },
])}

${sources([
  `Submagic plans and metering checked ${esc(c.checked)} on the vendor's public pricing page.`,
  `OpenShorts pipeline stages in the project source at <a href="${SITE.repo}" rel="noopener">github.com/mutonby/openshorts</a>.`,
])}
`,
    faq: [
      {
        q: 'Is Submagic worth it?',
        a: 'For caption styling on clips you already cut, yes — it is the strongest in the category. For going from a long video to shorts it is half a pipeline.',
      },
      {
        q: 'What is a free alternative to Submagic?',
        a: 'OpenShorts self-hosted: free under MIT, no watermark, no cap, with moment detection and captions in one pipeline. Caption presets are plainer than Submagic\'s.',
      },
    ],
  }
}

/* Reading order for the comparison cluster, used by the Spanish index and by
 * relatedFor's blurbs. Kept next to the pages it describes so a new comparison
 * has one obvious place to register. */
const COMPARISON_INDEX = [
  {
    path: '/alternatives/opus-clip',
    title: 'Opus Clip, comparado',
    blurb: 'Créditos por minuto de vídeo de origen, 720p frente a 1080p y dónde gana cada uno.',
  },
  {
    path: '/opus-clip-pricing',
    title: 'Cuánto cuesta Opus Clip',
    blurb: 'Qué es un crédito, qué incluye el plan gratuito y qué planes existen en 2026.',
  },
  {
    path: '/opus-clip-free-alternative',
    title: 'Alternativa gratuita a Opus Clip',
    blurb: 'Las dos vías realmente gratuitas frente al plan gratuito con marca de agua.',
  },
  {
    path: '/opus-ai',
    title: 'Opus AI (opus.pro)',
    blurb: 'Qué es realmente el nombre «Opus AI» y qué cuesta la herramienta.',
  },
  {
    path: '/opus-pro',
    title: 'El plan Opus Pro',
    blurb: 'Qué compra el plan de $29/mes y cuándo es el plan equivocado.',
  },
  {
    path: '/alternatives/vizard',
    title: 'Vizard, comparado',
    blurb: 'Editor en línea de tiempo tras el paso de IA, y a quién le hace falta.',
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
    blurb: 'Solo subtítulos: no sustituye a un clipper, lo complementa.',
  },
  {
    path: '/submagic-reviews',
    title: 'Análisis de Submagic',
    blurb: 'Lo que destacan los análisis públicos y el hueco que deja.',
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
    title: 'Alternativas a Opus Clip, Vizard y Submagic | OpenShorts',
    description:
      'Comparativas de OpenShorts frente a Opus Clip, Vizard, Klap y Submagic: precios reales, qué incluye el plan gratuito y en qué gana cada herramienta.',
    h1: 'Alternativas de código abierto a las herramientas de clipping',
    breadcrumb: [{ name: 'Alternativas' }],
    published: '2026-09-17',
    updated: '2026-09-17',
    tldr: [
      'OpenShorts es la única herramienta de esta categoría con código abierto y autoalojable: MIT, se ejecuta con Docker en tu propia máquina y no lleva marca de agua ni límite de uso.',
      `Los precios de entrada, comprobados el ${esc(OPUS.checked)}: OpenShorts $0 autoalojado o $12/mes alojado, Submagic ${esc(COMPETITORS.submagic.entryPrice)}, Opus Clip ${esc(OPUS.entryPrice)}, Vizard ${esc(COMPETITORS.vizard.entryPrice)} y Klap ${esc(COMPETITORS.klap.entryPrice)}.`,
      'Las herramientas no son equivalentes: Submagic no detecta momentos, Klap no deja ajustar la salida y Vizard espera que edites en su línea de tiempo. Cada comparativa de abajo dice dónde gana de verdad.',
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
<tr><td>Submagic</td><td>${esc(COMPETITORS.submagic.entryPrice)}</td><td>No</td><td>No</td></tr>
<tr><td>Opus Clip</td><td>${esc(OPUS.entryPrice)}</td><td>No</td><td>No</td></tr>
<tr><td>Vizard</td><td>${esc(COMPETITORS.vizard.entryPrice)}</td><td>No</td><td>No</td></tr>
<tr><td>Klap</td><td>${esc(COMPETITORS.klap.entryPrice)}</td><td>No</td><td>No</td></tr>
</tbody>
</table>

<h2>Qué cuesta OpenShorts</h2>
${pricingParagraph}
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
    a: 'OpenShorts, con licencia MIT y el código completo en GitHub. Opus Clip, Klap, Vizard y Submagic son productos comerciales de código cerrado que solo funcionan en la nube.',
  },
  {
    q: '¿Merece la pena cambiar de herramienta?',
    a: 'Depende de lo que más te moleste hoy. Si es el precio por minuto de vídeo de origen, el ahorro es real, sobre todo con episodios largos. Si es el diseño de los subtítulos, las herramientas comerciales siguen teniendo más presets y más pulidos, y eso no lo vamos a discutir.',
  },
])}
`,
    faq: [
      {
        q: '¿Cuál es la alternativa gratuita a Opus Clip?',
        a: 'OpenShorts autoalojado: MIT, Docker, sin marca de agua y sin límite. Hosted: 20 minutos gratis al mes y planes desde $12/mes.',
      },
      {
        q: '¿Qué herramienta de clipping es de código abierto?',
        a: 'OpenShorts (MIT). Opus Clip, Klap, Vizard y Submagic son de código cerrado y solo en la nube.',
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
  breadcrumb: [{ name: 'GTA 5 clips' }],
  published: '2026-09-15',
  updated: '2026-09-15',
  tldr: [
    'A GTA 5 stream is four to eight hours of 16:9 gameplay with a webcam box in one corner. OpenShorts reads the whole VOD, picks the moments worth posting out of what was said, and reframes each one so the gameplay keeps its full width and the facecam is enlarged underneath it instead of cropped away.',
    'Length is what makes this expensive everywhere else. Tools in this category bill one credit per minute of source you import, so a single eight-hour stream is 480 minutes: more than the 300 minutes a $29/month Opus Clip Pro plan includes (checked 2026-07-27). Self-hosted OpenShorts has no meter at all; the hosted edition starts at $12/month.',
    'Gameplay with no commentary is handled too, and it is where most clippers stop: when a stream has no usable speech OpenShorts switches by itself to a vision pass where Gemini watches the footage and picks the moments, instead of failing on an empty transcript. The switch is automatic, with one practical ceiling noted below.',
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
own answer below.</p>

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
more than 85% of the frame renders as WIDE, which keeps the full width intact
over a blurred backdrop rather than side-cropping it. Content that leaves room
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
${pricingParagraph}

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
<p class="note"><span class="label">The one ceiling worth knowing</span>
This is the single stage that sends Gemini the footage instead of a handful of
frames, and Gemini bills video at roughly 300 tokens per second. An hour of
gameplay is around 1.08 million tokens, which does not fit a 1 million token
context window, and there is no length guard in front of it: a silent eight-hour
VOD will fail at the model rather than politely. So for silent footage, hand it
the session or the segment you care about rather than the full stream. With
commentary the ceiling does not exist, because the transcript path never uploads
the video at all.</p>

<h2>Whose footage can you clip?</h2>
<p>Yours, and footage you have permission for. Your own streams and recordings,
your RP server co-stars' VODs with their blessing, clients' channels you manage.
Two separate rights questions apply to GTA clips and they have different
answers: the <strong>recording</strong> belongs to whoever streamed it, and the
<strong>game footage</strong> is covered by Rockstar Games' own policy on fan
videos, which has historically permitted gameplay videos monetised through the
platforms' standard ad programs. That is their policy and it can change, so
check the current version rather than taking this page's word for it. Reuploading
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
    a: 'Yes, and it switches by itself. Footage with no audio track, or whose transcript comes back under 8 words or under 5 words per minute, goes down a vision pass instead: Gemini watches the footage and returns the same 3 to 15 moments, with the same layouts and inset detection after it. The clips come out without captions, since there is no speech to caption. The practical limit is length, because that pass sends Gemini the video rather than a few frames: give it the session you care about, not a silent eight-hour VOD.',
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
          text: 'Gameplay with a corner webcam renders as gameplay full width on top and the enlarged facecam below; full-frame action keeps its full width over a blurred backdrop. Subtitles are burned in from the word-level transcript.',
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

export function buildPages() {
  // Ring order matters: relatedFor links each page to the next three, so
  // neighbours are chosen to be topically adjacent.
  return [
    hubPage(),
    ...ALTERNATIVES.map(competitorPage),
    opusClipPricing(),
    opusClipFree(),
    opusAi(),
    opusPro(),
    videoToText(),
    submagicReview(),
    alternativasIndex(),
    freeClipGenerator(),
    noWatermark(),
    openSourceClipper(),
    openSourceVideoGenerator(),
    howItWorks(),
    gtaClips(),
    podcastToShorts(),
    youtubeConverter(),
    mcpAgentsPage(),
    automateShorts(),
    n8nTemplate(),
  ]
}

/* Each page links to three siblings. Small, described clusters beat a single
 * dump of every URL: the described link tells an engine what it will find. */
export function relatedFor(page, all) {
  const blurb = {
    '/alternatives': 'All four tools compared, with entry pricing.',
    '/opus-clip-pricing': 'What a credit is, what the free tier includes, and every plan.',
    '/opus-clip-free-alternative': 'The two genuinely free routes, against a watermarked free tier.',
    '/opus-ai': 'What the name refers to, what it does and what it costs.',
    '/opus-pro': 'What the $29 tier buys, and when it is the wrong plan.',
    '/vizard-ai-video-to-text': 'Transcript, subtitles or clips: which one you are asking for.',
    '/submagic-reviews': 'What the reviews praise, and the half of the job it does not do.',
    '/alternativas': 'Todas las comparativas, en español, ordenadas por pregunta.',
    '/alternatives/opus-clip': 'Per-minute credits, 720p vs 1080p, and where each one wins.',
    '/alternatives/klap': 'Fastest URL-to-clip path, and what you give up for it.',
    '/alternatives/vizard': 'Timeline editing after the AI pass, and who needs it.',
    '/alternatives/submagic': 'Captions only, so it does not replace a clipper.',
    '/free-ai-clip-generator': 'What free means when there is no metering code.',
    '/free-ai-clip-generator-no-watermark': 'Why free tools watermark, and the structural exception.',
    '/open-source-video-clipper': 'Self-hosting with Docker, and the MIT licence carve-out.',
    '/open-source-ai-video-generator': 'Text-to-video or clips from your footage: which you want.',
    '/how-openshorts-works': 'The full pipeline, stage by stage.',
    '/gta-5-clips': 'Stream VODs, webcam inset kept, no per-minute meter.',
    '/podcast-to-shorts': 'Two-speaker episodes without cropping anyone out.',
    '/youtube-to-shorts-converter': 'Paste a link, get 9:16 clips with subtitles.',
    '/mcp': 'Drive the whole pipeline from Claude, ChatGPT or n8n.',
    '/automate-shorts-api': 'One POST in, one signed webhook out, no polling.',
    '/n8n-youtube-shorts-automation': 'The importable workflow: channel in, approved shorts out.',
  }
  // Walk the ring starting after this page so each page links to a different
  // three. Slicing the same head every time would leave the last pages in the
  // list with no inbound links at all.
  const i = all.findIndex((p) => p.path === page.path)
  return [1, 2, 3]
    .map((n) => all[(i + n) % all.length])
    .filter((p) => p && p.path !== page.path)
    .map((p) => ({ path: p.path, title: p.h1, blurb: blurb[p.path] || p.description }))
}
