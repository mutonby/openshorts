/* Topic hubs: /platforms, /use-cases and /how-to.
 *
 * Each hub is the parent of a cluster. The platform pages and the sermon page
 * live under their hub's path (/platforms/youtube, /use-cases/sermons, ...);
 * the pages that were already ranking on a flat URL (/podcast-to-shorts,
 * /gta-5-clips, /auto-clip, /how-openshorts-works, ...) stay where they are and
 * are only listed here, with a breadcrumb pointing back at the hub.
 *
 * A card is rendered only when a page with that path is actually built (the
 * same rule llms.txt uses for its key pages), so a hub can name a page another
 * module adds and never links to a 404 before that page ships.
 *
 * Same rules as the rest of the SEO surface: nothing invented, the free plan is
 * always stated with its watermark and the paid price next to it, no em dashes.
 */

import { SITE, EDITIONS } from './data.js'
import { esc } from './render.js'

const PUBLISHED = '2026-10-29'
const CLOUD = EDITIONS.cloud
// Checked against data.js like clip-tools.js does, so the hubs cannot drift.
const FIRST_VIDEO_MINUTES = 60
if (!CLOUD.summary.includes(`up to ${FIRST_VIDEO_MINUTES} minutes`)) {
  throw new Error('[hub-pages] EDITIONS.cloud.summary no longer says the first video is free up to ' +
    `${FIRST_VIDEO_MINUTES} minutes; update FIRST_VIDEO_MINUTES in seo/hub-pages.js`)
}
const PRICE_LINE =
  `OpenShorts Cloud clips your first video free up to ${FIRST_VIDEO_MINUTES} minutes, then ${CLOUD.freeMinutes} minutes a month, ` +
  `with a watermark and no credit card; paid plans start at $${CLOUD.lowPrice}/month with no watermark ` +
  '(<a href="/pricing">all plans</a>). The self-hosted edition is free under MIT.'

/* Cards for the entries that exist in `built` (path -> page), in list order. */
const cardsFor = (entries, built) => entries.filter((e) => built.has(e.path))

const cardGrid = (cards) => `
<div class="tools-grid">
${cards.map((c) => `<a href="${c.path}"><strong>${esc(c.name)}</strong><span>${esc(c.blurb)}</span></a>`).join('\n')}
</div>`

/* The hub's children as an ItemList, next to the Article the renderer emits. */
const itemList = (path, name, cards) => ({
  '@type': 'ItemList',
  '@id': `${SITE.url}${path}#list`,
  name,
  itemListElement: cards.map((c, i) => ({
    '@type': 'ListItem',
    position: i + 1,
    name: c.name,
    url: `${SITE.url}${c.path}`,
  })),
})

const hub = (path, extra) => ({
  path,
  published: PUBLISHED,
  updated: PUBLISHED,
  ogType: 'website',
  relatedTitle: 'More on this site',
  faq: [],
  ...extra,
})

/* ------------------------------------------------------------------ /platforms */

export const PLATFORM_CHILDREN = [
  { path: '/platforms/youtube', name: 'YouTube clip maker', blurb: 'Paste a YouTube link, get vertical clips with captions and a hook line.' },
  { path: '/platforms/youtube-shorts', name: 'YouTube Shorts maker', blurb: 'Shorts from your long videos, posted or scheduled to your channel.' },
  { path: '/platforms/tiktok', name: 'TikTok clip maker', blurb: 'A long video cut into TikToks, sent to your TikTok inbox as drafts.' },
  { path: '/platforms/instagram-reels', name: 'AI reel generator', blurb: 'Instagram Reels posted straight to Instagram, Facebook Reels as files.' },
  { path: '/platforms/twitch', name: 'Twitch clip maker', blurb: 'Hours-long VODs clipped in parts, with the facecam kept.' },
  { path: '/platforms/kick', name: 'Kick clip maker', blurb: 'Kick VOD and clip links, with IRL and facecam layouts.' },
]

const PLATFORM_EXTRAS = [
  { path: '/youtube-to-shorts-converter', name: 'YouTube to Shorts converter', blurb: 'The short answer for one link: paste it, get 9:16 clips with subtitles.' },
  { path: '/tools/tiktok-transcript-generator', name: 'TikTok transcript generator', blurb: 'The words of a TikTok as text, free and without an account.' },
]

/* Destination and source specs. Lengths are the ones the platform pages state
 * with their sources (YouTube help, Instagram's announcement, TikTok's
 * Content Posting API), checked 28 October 2026. The caption line is what the
 * renderer does on every clip (subtitles.SAFE_MARGIN_V). */
const SPECS_CHECKED = '28 October 2026'
const CAPTIONS = 'Burned in about 15% of the frame height above the bottom edge, above the caption and music bar'
const PLATFORM_SPECS = [
  ['YouTube Shorts', 'Destination', 'Up to 3 minutes', 'Vertical or square; clips are 9:16', CAPTIONS, 'Published to your channel, or scheduled'],
  ['TikTok', 'Destination', 'Longer uploads accepted; clips here go up to 180 s', '9:16', CAPTIONS, 'Sent to your TikTok inbox as a draft, finished in the app'],
  ['Instagram Reels', 'Destination', 'Up to 3 minutes', '9:16', CAPTIONS, 'Published to Instagram, or scheduled'],
  ['Facebook Reels', 'Destination', 'Clips up to 180 s', '9:16', CAPTIONS, 'Download the MP4 and post it yourself'],
  ['YouTube', 'Source', 'Any public video; clipped from your minute balance', '16:9 in, 9:16 out', 'Word-level, from the transcript', 'n/a (one video link at a time)'],
  ['Twitch and Kick', 'Source', 'Hours-long VODs, clipped in parts', '16:9 in, facecam stacked under gameplay', 'Word-level, from the transcript', 'Not posted back; clips go to Shorts, TikTok or Reels'],
]

const specsTable = () => `
<table>
<thead><tr><th>Platform</th><th>Role</th><th>Max length</th><th>Aspect</th><th>Captions safe zone</th><th>Posting from OpenShorts</th></tr></thead>
<tbody>
${PLATFORM_SPECS.map(([name, ...cells]) => `<tr><td class="os">${esc(name)}</td>${cells.map((c) => `<td>${esc(c)}</td>`).join('')}</tr>`).join('\n')}
</tbody>
</table>`

const platformsHub = (built) => {
  const children = cardsFor(PLATFORM_CHILDREN, built)
  const extras = cardsFor(PLATFORM_EXTRAS, built)
  return hub('/platforms', {
    title: 'AI Clips for YouTube, TikTok, Reels and More | OpenShorts',
    description: `One AI clip generator for YouTube, Shorts, TikTok, Instagram Reels, Twitch and Kick: what each platform accepts and how clips get there. First video free up to ${FIRST_VIDEO_MINUTES} min.`,
    h1: 'AI clips for every platform: YouTube, TikTok, Reels, Twitch and Kick',
    lede: 'The clip generator is the same everywhere: it reads the whole video, picks the moments that stand alone and cuts them to 9:16 with captions. What changes from one platform to the next is where the video comes from, how long a clip may run and how it gets posted.',
    breadcrumb: [{ name: 'Platforms' }],
    related: ['/use-cases', '/how-to', '/tools'],
    extraNodes: [itemList('/platforms', 'OpenShorts clip makers by platform', [...children, ...extras])],
    body: `
<h2>Pick the platform</h2>
<p>Each page below covers one source or destination: the links it accepts, the
layout it gets, and what happens after the clip is cut.</p>
${cardGrid(children)}

<h2>Platform specs side by side</h2>
<p class="checked">Platform limits checked ${esc(SPECS_CHECKED)} on each platform's help pages; sources are listed on each platform page.</p>
${specsTable()}
<p>Clips are 15 to 60 seconds by default, because most moments that stand on
their own run that long; the maximum can be raised to 180 seconds in the
advanced options. Every clip is at least 1080 pixels wide, so a 1080p source
gives 1080x1920. Captions sit above the bottom strip where TikTok, Reels and
Shorts draw the username, the description and the music ticker, so the exported
file and the posted clip read the same.</p>

<h2>Sources and destinations are different jobs</h2>
<p>YouTube, Twitch and Kick are mostly where the long video lives. A podcast
episode, a stream VOD or an interview is read end to end and cut into several
clips. YouTube Shorts, TikTok and Instagram Reels are where those clips go. The
same clip can be posted to all three from the dashboard, scheduled across the
week, or downloaded as an MP4 for anywhere else.</p>
${extras.length ? `<div class="cluster">${extras.map((c) => `<a href="${c.path}"><strong>${esc(c.name)}</strong><span>${esc(c.blurb)}</span></a>`).join('')}</div>` : ''}

<h2>What it costs</h2>
<p>${PRICE_LINE}</p>
`,
  })
}

/* ------------------------------------------------------------------ /use-cases */

export const USE_CASE_CHILDREN = [
  { path: '/podcast-to-shorts', name: 'Podcast clip maker', blurb: 'Podcast clips that keep both speakers in frame.' },
  { path: '/use-cases/sermons', name: 'Sermon clip maker', blurb: 'Only the sermon clipped out of a service, with the preacher followed.' },
  { path: '/gta-5-clips', name: 'GTA 5 and gaming clips', blurb: 'Stream VODs with the webcam inset kept, and quiet gameplay still handled.' },
  { path: '/gta-6-clips', name: 'GTA 6 clips', blurb: 'Console captures and stream VODs into TikToks, facecam kept, silent footage handled.' },
  { path: '/wow-forever-clips', name: 'WoW Forever clips', blurb: 'Record raids, then turn kills, wipes and Twitch VODs into vertical clips with the facecam kept.' },
  { path: '/auto-reframe', name: 'Auto reframe 9:16', blurb: 'Horizontal to vertical by tracking the speaker, stacking two, or keeping the screen.' },
  { path: '/auto-clip', name: 'Auto clip', blurb: 'Every new upload on your channel clipped on its own, with your rules.' },
  { path: '/youtube-automation', name: 'YouTube automation', blurb: 'Autopilot, n8n or the API: automate a real channel, not a content farm.' },
  { path: '/ai-ugc-video-generator', name: 'AI UGC video generator', blurb: 'Short product videos with an AI presenter, for ads and social.' },
]

const caseHub = (built) => {
  const children = cardsFor(USE_CASE_CHILDREN, built)
  return hub('/use-cases', {
    title: 'AI Clips by Use Case: Podcasts, Sermons, Gaming | OpenShorts',
    description: `How OpenShorts clips podcasts, sermons, gaming streams and whole channels: the layouts, the reframing and the automation behind each. First video free up to ${FIRST_VIDEO_MINUTES} min.`,
    h1: 'AI clips by use case: podcasts, sermons, gaming and channels on autopilot',
    lede: 'A two-person podcast, a church service and a GTA stream need different things from a clip: both faces kept, only the sermon, the facecam stacked under the game. These pages explain what the clip generator does for each, and how to run it without opening the app at all.',
    breadcrumb: [{ name: 'Use cases' }],
    related: ['/platforms', '/how-to', '/how-openshorts-works'],
    extraNodes: [itemList('/use-cases', 'OpenShorts use cases', children)],
    body: `
<h2>Pick the use case</h2>
${cardGrid(children)}

<h2>What changes from one use case to the next</h2>
<ul>
<li><strong>Who is in the frame.</strong> A podcast keeps both speakers, stacked when they talk over each other. A sermon follows one preacher across the stage. A stream keeps the facecam under the gameplay.</li>
<li><strong>What counts as a moment.</strong> The transcript is scored for stretches that stand on their own. Gameplay with little speech is still handled, so a quiet stream does not come back empty.</li>
<li><strong>How often it runs.</strong> One video at a time from the dashboard, or every new upload on a channel with Autopilot, n8n or the API.</li>
</ul>
<p>The reframing behind all of them is described on <a href="/auto-reframe">the
auto reframe page</a>, and the whole pipeline stage by stage on
<a href="/how-openshorts-works">how OpenShorts works</a>.</p>

<h2>What it costs</h2>
<p>${PRICE_LINE}</p>
`,
  })
}

/* --------------------------------------------------------------------- /how-to */

export const GUIDE_CHILDREN = [
  { path: '/how-to/clip-a-youtube-video', name: 'How to clip a YouTube video', blurb: 'Four free ways, from the Clip button to AI-cut vertical Shorts.' },
  { path: '/how-to/make-youtube-shorts', name: 'How to make YouTube Shorts', blurb: 'The app camera, a vertical upload, or Shorts cut from your long videos.' },
  { path: '/how-to/youtube-shorts-aspect-ratio', name: 'YouTube Shorts aspect ratio', blurb: '9:16 at 1080 x 1920, square allowed, up to 3 minutes, and the safe zones.' },
  { path: '/how-to/instagram-reel-dimensions', name: 'Instagram Reel dimensions', blurb: '1080 x 1920, the 3:4 grid crop of the cover, safe zones and export settings.' },
  { path: '/how-to/make-a-reel-on-facebook', name: 'How to make a Reel on Facebook', blurb: 'The app, Meta Business Suite, specs, and sharing Reels from Instagram.' },
  { path: '/how-to/best-time-to-post-youtube-shorts', name: 'Best time to post YouTube Shorts', blurb: 'No universal answer: find your viewers\' hours in Analytics and test them.' },
  { path: '/how-openshorts-works', name: 'How OpenShorts works', blurb: 'The full pipeline, from the link to a captioned 9:16 clip.' },
  { path: '/self-hosted-ai-clip-maker', name: 'Self-host the clip maker', blurb: 'Docker setup, hardware, your own keys, and the real cost against credits.' },
  { path: '/n8n-youtube-shorts-automation', name: 'Automate Shorts with n8n', blurb: 'The importable workflow: channel in, approved Shorts out.' },
  { path: '/automate-shorts-api', name: 'Automate Shorts with the API', blurb: 'One POST in, one signed webhook out, no polling.' },
  { path: '/mcp', name: 'Clip from an AI agent (MCP)', blurb: 'Drive the whole pipeline from Claude, ChatGPT or n8n.' },
]

const guidesHub = (built) => {
  const children = cardsFor(GUIDE_CHILDREN, built)
  return hub('/how-to', {
    title: 'OpenShorts Guides: Clip, Self-Host and Automate',
    description: 'Step-by-step guides: clip a YouTube video, see how the pipeline works, self-host OpenShorts with Docker, and automate Shorts with n8n, the API or an AI agent.',
    h1: 'OpenShorts guides: clip, self-host and automate',
    lede: 'Guides for the three things people do with OpenShorts after the first clip: understand what it did, run it on their own machine, and make it run without them.',
    breadcrumb: [{ name: 'Guides' }],
    related: ['/platforms', '/use-cases', '/tools'],
    extraNodes: [itemList('/how-to', 'OpenShorts guides', children)],
    body: `
<h2>All guides</h2>
${cardGrid(children)}

<h2>Where to start</h2>
<ul>
<li><strong>New to clipping?</strong> <a href="/how-to/clip-a-youtube-video">How to clip a YouTube video</a> compares the free ways to do it by hand with letting the AI cut the Shorts.</li>
<li><strong>Need the platform specs or steps?</strong> <a href="/how-to/youtube-shorts-aspect-ratio">YouTube Shorts aspect ratio</a> and <a href="/how-to/instagram-reel-dimensions">Instagram Reel dimensions</a> give the sizes and safe zones; <a href="/how-to/make-youtube-shorts">how to make YouTube Shorts</a> and <a href="/how-to/make-a-reel-on-facebook">how to make a Reel on Facebook</a> give the steps; <a href="/how-to/best-time-to-post-youtube-shorts">the best time to post Shorts</a> shows how to find yours.</li>
<li><strong>Want to know what happens to your video?</strong> <a href="/how-openshorts-works">How OpenShorts works</a> walks through transcription, moment picking, reframing and captions.</li>
<li><strong>Want it on your own machine?</strong> <a href="/self-hosted-ai-clip-maker">The self-hosting guide</a> covers Docker, hardware and keys. The code is MIT licensed.</li>
<li><strong>Want it to run on its own?</strong> Start with <a href="/n8n-youtube-shorts-automation">the n8n workflow</a> for a channel, or <a href="/automate-shorts-api">the API</a> and <a href="/mcp">the MCP server</a> for your own code or agent.</li>
</ul>

<h2>What it costs</h2>
<p>${PRICE_LINE}</p>
`,
  })
}

/* `all` is every other built page; the hubs only link what is in it. */
export function hubPages(all) {
  const built = new Set(all.map((p) => p.path))
  return [platformsHub(built), caseHub(built), guidesHub(built)]
}
