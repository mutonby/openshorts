/* /auto-clip and /youtube-automation: the two pages about running a channel's
 * clipping without opening the app.
 *
 * Every rule stated here is a constant in cloud/autopilot.py (checked
 * 23-sep-2026); if one changes there, change it here. Screenshots are the real
 * Autopilot tab in production with the connected channel's name, handle,
 * thumbnails and video titles blurred: that channel belongs to someone who did
 * not agree to be on a marketing page. The usage numbers are aggregates over
 * all accounts from usage_ledger, nothing per user.
 */

import { SITE } from './data.js'
import { esc } from './render.js'

const PUBLISHED = '2026-09-23'

const faqBlock = (faq) =>
  `<h2>Common questions</h2><dl class="faq">${faq
    .map((f) => `<dt>${esc(f.q)}</dt><dd>${esc(f.a)}</dd>`)
    .join('')}</dl>`

const sources = (items) =>
  `<h2>Sources</h2><ul class="sources">${items.map((s) => `<li>${s}</li>`).join('')}</ul>`

/* Checked against GET https://api.openshorts.app/api/billing/plans on
 * 23-sep-2026. Autopilot is in every paid plan and in none of the free ones. */
const PLANS = [
  ['Starter', '$12/month', '100 minutes'],
  ['Creator', '$29/month', '300 minutes'],
  ['Pro', '$59/month', '750 minutes'],
]

/* usage_ledger, job_type "process", status committed, 24-aug to 23-sep-2026. */
const USAGE = { jobs: '4,641', minutes: '56,590', median: 12, window: '24 August to 23 September 2026' }

const plansTable = `
<table>
<thead><tr><th>Plan</th><th>Price</th><th>Source video per month</th><th>Autopilot</th></tr></thead>
<tbody>
<tr><td>Free</td><td>$0</td><td>20 minutes, watermarked</td><td>No (clip by hand)</td></tr>
${PLANS.map(([n, p, m]) => `<tr><td class="os">${n}</td><td>${p}</td><td>${m}</td><td class="yes">Yes</td></tr>`).join('\n')}
</tbody>
</table>
<p class="sources">Prices from the OpenShorts billing API, checked 23 September 2026. Yearly plans cost ten times the monthly price.</p>`

const autoClip = () => ({
  path: '/auto-clip',
  title: 'Auto Clip Your YouTube Videos Into Shorts | OpenShorts',
  description:
    'Autopilot auto clips every new video on your YouTube channel into vertical shorts with subtitles and can post the best ones daily. Rules, limits, price.',
  h1: 'Auto clip: every new YouTube video turned into shorts on its own',
  breadcrumb: [{ name: 'Use cases', path: '/use-cases' }, { name: 'Auto clip' }],
  published: PUBLISHED,
  updated: PUBLISHED,
  relatedTitle: 'Related',
  related: ['/youtube-automation', '/n8n-youtube-shorts-automation', '/how-openshorts-works'],
  cta: {
    label: 'Autopilot',
    title: 'Connect your channel once',
    body: 'Included in every paid plan from $12/month. On the free plan you can still clip any video by hand.',
    button: 'Open Autopilot',
    href: `${SITE.url}/#app?tab=autopilot`,
  },
  tldr: [
    'Automatic clipping means software watches a long video, picks the moments that stand on their own and cuts them into short vertical clips. OpenShorts does that for any video you paste, and Autopilot does it for every new upload on your YouTube channel without you opening anything.',
    'Autopilot checks your channel about once an hour, clips each new video (only videos published after you switch it on, at most one a day, the first 10 to 90 minutes of each), emails you when the clips are ready and, if you turn on autopublish, schedules the best 1 to 5 of them one a day on TikTok, Instagram and YouTube Shorts.',
  ],
  body: `
<h2>What "auto clip" does, step by step</h2>
<ol>
<li><strong>Transcribe.</strong> The audio is transcribed with word-level timestamps, so every clip can start and end on a word.</li>
<li><strong>Find the cuts.</strong> Scene detection marks where the picture changes.</li>
<li><strong>Score the moments.</strong> Google Gemini reads the transcript and rates stretches of 15 to 60 seconds on whether they make sense alone and hold attention, then keeps the 3 to 15 best.</li>
<li><strong>Reframe to 9:16.</strong> Face tracking keeps the speaker in the frame; screen recordings and two-person shots get their own layouts.</li>
<li><strong>Subtitle and title.</strong> Captions are burned in from the transcript and each clip gets a hook and a title.</li>
</ol>
<p>That pipeline is the same whether you paste a link by hand or Autopilot starts
it. The details are on <a href="/how-openshorts-works">how OpenShorts works</a>.
Across all accounts, OpenShorts Cloud ran ${USAGE.jobs} clipping jobs on
${USAGE.minutes} minutes of source video between ${USAGE.window}; the median
source video was ${USAGE.median} minutes long.</p>

<h2>Autopilot: the channel clips itself</h2>
<figure class="shot">
<img src="/screens/autopilot-settings.webp" width="920" height="672" loading="eager"
     alt="The Autopilot tab in OpenShorts: a connected YouTube channel, the Autopilot switch turned on with the first 10 minutes of each video selected, and the autopublish switch off.">
<figcaption>The Autopilot tab in production, 23 September 2026. The channel name is blurred because it belongs to a user.</figcaption>
</figure>
<p>You connect your YouTube channel once. From then on Autopilot reads your
channel's uploads through the YouTube account you connected, so it also sees
videos you upload straight to YouTube, and it clips them through the normal
pipeline. The rules are deliberately conservative, because it spends your
plan's minutes while you are not looking:</p>
<ul>
<li><strong>Only new videos.</strong> Switching it on never clips your back catalogue: only videos published after that moment, and only while they are at most 3 days old.</li>
<li><strong>One automatic video a day.</strong> If you publish three videos in a day, the others wait for you. You can still clip up to 5 more a day by hand from the same tab.</li>
<li><strong>A cap per video.</strong> Autopilot clips the first 10, 20, 30, 45, 60 or 90 minutes of each video (30 by default), so a three-hour stream cannot empty your plan.</li>
<li><strong>Shorts are skipped.</strong> A video that is already a Short is not clipped again.</li>
<li><strong>You get an email</strong> when the clips of each video are ready.</li>
</ul>

<figure class="shot">
<img src="/screens/autopilot-channel-videos.webp" width="920" height="454" loading="lazy"
     alt="The latest videos list in the Autopilot tab: each upload with its date, a clips count or not clipped, and open or clip it buttons.">
<figcaption>Your latest uploads, what Autopilot did with each, and a "clip it" button for any of them. Titles and thumbnails blurred.</figcaption>
</figure>

<h2>Autopublish: the best clips go out one a day</h2>
<p>Off by default. Turn it on and, for each video, Autopilot takes the 1 to 5
clips with the highest predicted score (3 by default) and schedules them one per
day at the hour you choose, in your own time zone, on the accounts you pick:
TikTok, Instagram and YouTube Shorts. TikTok clips arrive as drafts in your
TikTok inbox, so you open the app to post them. Leave autopublish off and you
review and post the clips yourself.</p>

<h2>What it costs</h2>
<p>Autopilot runs on your plan's minutes, so it is part of the paid plans and not
the free one. A 30-minute cap per video uses at most 30 minutes of your plan for
that video.</p>
${plansTable}
<p>The free plan still includes the clip generator itself: 20 minutes a month,
with a watermark, clipped by hand. Or self-host the
<a href="/open-source-video-clipper">open source video clipper</a> and wire the
same automation yourself with the
<a href="/n8n-youtube-shorts-automation">n8n workflow</a>.</p>

<h2>What it does not do</h2>
<ul>
<li>It watches YouTube and Twitch. Twitch is read-only: Autopilot lists your past broadcasts, waits until a stream has ended, clips the stretch your viewers clipped most instead of the first minutes, and never posts to Twitch. Podcasts, Kick VODs or files on your drive have to be pasted or sent through <a href="/automate-shorts-api">the API</a>.</li>
<li>It does not guarantee views. It picks the moments the model scores highest; some weeks those will be the wrong ones, which is why autopublish is off until you turn it on.</li>
<li>It is new: Autopilot launched on 22 September 2026.</li>
</ul>

${faqBlock([
  {
    q: 'What is auto clipping?',
    a: 'Software that watches a long video, finds the moments that work on their own and cuts them into short vertical clips with subtitles, instead of you scrubbing the timeline by hand. OpenShorts returns 3 to 15 clips of 15 to 60 seconds per video.',
  },
  {
    q: 'Can OpenShorts clip my new YouTube videos automatically?',
    a: 'Yes. Autopilot checks your connected channel about once an hour and clips each new video on its own: only videos published after you switch it on, at most one a day, up to the minute cap you set.',
  },
  {
    q: 'Does Autopilot post the clips for me?',
    a: 'Only if you turn on autopublish. Then it schedules the best 1 to 5 clips of each video one a day on TikTok, Instagram and YouTube Shorts. TikTok clips arrive as drafts you post from the app.',
  },
  {
    q: 'Is automatic clipping free?',
    a: 'Clipping a video you paste is free for 20 minutes a month on OpenShorts Cloud, or unlimited if you self-host. Autopilot, which runs on its own, is included in the paid plans from $12/month.',
  },
  {
    q: 'Will it clip my whole back catalogue and burn my minutes?',
    a: 'No. It only picks up videos published after you switch it on, at most 3 days old, one a day, and only the first 10 to 90 minutes of each.',
  },
  {
    q: 'Can Autopilot clip my Twitch streams?',
    a: 'Yes. Connect Twitch in the Autopilot tab (read-only) and each new past broadcast is clipped once the stream has ended, under the same rules. Instead of the first minutes, it clips the stretch your viewers clipped most, because the start of a stream is usually a "starting soon" screen.',
  },
])}

${sources([
  `Autopilot rules: <code>cloud/autopilot.py</code> in the <a href="${SITE.repo}" rel="noopener">OpenShorts source</a>, checked 23 September 2026.`,
  'Usage figures: OpenShorts Cloud billing ledger, all accounts aggregated, 24 August to 23 September 2026.',
])}
`,
  faq: [
    { q: 'What is auto clipping?', a: 'Software that finds the moments of a long video that work on their own and cuts them into short vertical clips with subtitles. OpenShorts returns 3 to 15 clips of 15 to 60 seconds per video.' },
    { q: 'Can OpenShorts clip my new YouTube videos automatically?', a: 'Yes. Autopilot checks your connected channel about once an hour and clips each new video: only videos published after you switch it on, at most one a day.' },
    { q: 'Does Autopilot post the clips for me?', a: 'Only if you turn on autopublish. Then it schedules the best 1 to 5 clips of each video one a day on TikTok, Instagram and YouTube Shorts.' },
    { q: 'Is automatic clipping free?', a: 'Clipping a pasted video is free for 20 minutes a month, or unlimited self-hosted. Autopilot is included in the paid plans from $12/month.' },
    { q: 'Can Autopilot clip my Twitch streams?', a: 'Yes. Connect Twitch in the Autopilot tab (read-only) and each new past broadcast is clipped once the stream has ended, from the stretch your viewers clipped most instead of the first minutes.' },
  ],
})

const youtubeAutomation = () => ({
  path: '/youtube-automation',
  title: 'YouTube Automation for Shorts, Step by Step | OpenShorts',
  description:
    'Automate the repetitive part of a real YouTube channel: clip every upload into Shorts, schedule them, or wire it to n8n, the API or Claude over MCP.',
  h1: 'YouTube automation: hand the repetitive work to software, keep the channel yours',
  breadcrumb: [{ name: 'Use cases', path: '/use-cases' }, { name: 'YouTube automation' }],
  published: PUBLISHED,
  updated: PUBLISHED,
  relatedTitle: 'Related',
  related: ['/auto-clip', '/n8n-youtube-shorts-automation', '/mcp'],
  cta: {
    label: 'Start with no code',
    title: 'Turn on Autopilot for your channel',
    body: 'Every new upload clipped on its own, the best clips scheduled one a day if you want. In every paid plan from $12/month.',
    button: 'Open Autopilot',
    href: `${SITE.url}/#app?tab=autopilot`,
  },
  tldr: [
    '"YouTube automation" means two different things. One is faceless channels that mass-produce videos from templates; YouTube demonetises that as inauthentic content. The other is taking the repetitive work off a real channel: cutting every long video into Shorts, writing the metadata, scheduling the posts. This page is about the second.',
    'With OpenShorts there are three ways to automate it: Autopilot in the app (no code), the n8n template (you approve each clip from Telegram), or the REST API and MCP server for your own scripts and for agents like Claude and ChatGPT.',
  ],
  body: `
<h2>What is worth automating on a YouTube channel</h2>
<p>The work that does not need your judgement, every single week:</p>
<ul>
<li><strong>Cutting Shorts from each long video.</strong> Finding the moments, reframing to 9:16, adding subtitles. This is the slowest part to do by hand and the easiest to automate well.</li>
<li><strong>Posting them on a schedule</strong> to YouTube Shorts, TikTok and Instagram, one a day instead of all at once.</li>
<li><strong>The first draft of titles, descriptions and tags.</strong> You still edit them; the free <a href="/youtube-tag-generator">tag, title and description generator</a> writes the draft.</li>
<li><strong>Transcripts</strong> for show notes and blog posts, from the <a href="/youtube-transcript-generator">transcript generator</a>.</li>
</ul>
<p>What should stay yours: the long video itself, and the decision to publish.
YouTube's monetisation policy was changed on 15 July 2025 to name "inauthentic
content", which includes "AI-generated content made with generic or unoriginal
templates giving the impression of mass production". Automating the chores of a
channel you actually make is not that. Generating the channel is.</p>

<h2>Three ways to automate it, from no code to full control</h2>
<table>
<thead><tr><th>Route</th><th>Who it is for</th><th>You approve each post?</th><th>What you need</th></tr></thead>
<tbody>
<tr><td class="os">Autopilot</td><td>Creators who want it done, no setup</td><td>Your choice: autopublish on or off</td><td>A paid plan, your YouTube channel connected</td></tr>
<tr><td class="os">n8n template</td><td>People who already run n8n</td><td>Yes, from Telegram</td><td>n8n, an OpenShorts API key, a Telegram bot</td></tr>
<tr><td class="os">API and MCP</td><td>Developers and AI agents</td><td>Whatever your code decides</td><td>An API key, or a claude.ai / ChatGPT connector</td></tr>
</tbody>
</table>

<h2>Route 1: Autopilot, no code</h2>
<ol>
<li>Sign in to OpenShorts and open the <strong>Autopilot</strong> tab.</li>
<li>Connect your YouTube channel.</li>
<li>Switch Autopilot on, confirm you own the content, and choose how many minutes of each video to clip (10 to 90).</li>
<li>Optionally switch on autopublish: pick TikTok, Instagram and/or YouTube Shorts, how many clips per video (1 to 5) and the hour to post.</li>
</ol>
<figure class="shot">
<img src="/screens/autopilot-settings.webp" width="920" height="672" loading="eager"
     alt="The Autopilot settings in OpenShorts: connected channel, Autopilot on with a 10 minute cap, autopublish off.">
<figcaption>Autopilot in production, 23 September 2026. Channel name blurred.</figcaption>
</figure>
<p>From then on Autopilot looks at your channel about once an hour, starts a
job for each new upload it finds and emails you when the clips are ready. The exact rules (one automatic
video a day, only new videos, Shorts skipped) are on the
<a href="/auto-clip">auto clip page</a>.</p>

<h2>Route 2: the n8n template, with a human in the loop</h2>
<figure class="shot">
<img src="/n8n-content-machine-workflow.png" width="950" height="750" loading="lazy"
     alt="The OpenShorts content machine workflow open in n8n, from the channel trigger to Telegram approval and weekly analytics.">
<figcaption>The importable n8n workflow: watch the channel, clip, approve from Telegram, drip-publish, report weekly.</figcaption>
</figure>
<p>If you run n8n, the free <a href="/n8n-youtube-shorts-automation">n8n YouTube
Shorts template</a> does the same job with one difference that matters: every
clip reaches your Telegram with Publish and Skip buttons, and nothing goes out
until you tap Publish. It reads your channel's public RSS feed, so it needs no
YouTube API key, and a self-hosted OpenShorts works too: change the base URL in
the workflow.</p>

<h2>Route 3: the API and MCP, for your own code and for agents</h2>
<p>Everything the app does is one HTTP call away. <code>POST /api/process</code>
takes a video URL and a <code>webhook_url</code>, and OpenShorts calls you back
once with the finished clips; the details and a working example are on
<a href="/automate-shorts-api">automate shorts with the API</a>. The same
pipeline is exposed as an MCP server, so Claude, ChatGPT, Cursor or n8n's agent
node can clip and publish when you ask them in plain language.</p>
<figure class="shot">
<img src="/screens/connect-an-agent.webp" width="684" height="366" loading="lazy"
     alt="The connect an agent card in the OpenShorts account page, with setup steps for claude.ai and the MCP server URL https://mcp.openshorts.app/mcp.">
<figcaption>Connecting Claude takes one URL: <code>https://mcp.openshorts.app/mcp</code>. The card also has tabs for ChatGPT, Claude Code, Claude Desktop, Cursor, n8n and curl.</figcaption>
</figure>
<p>More on the tools the server exposes on the <a href="/mcp">MCP server page</a>.</p>

<h2>What it costs</h2>
<p>Autopilot, the API and MCP all use the same minutes as the app: 20 free
minutes a month on the free plan (Autopilot itself needs a paid plan), then
$12/month for 100 minutes, $29 for 300 or $59 for 750, as of 23 September 2026.
Self-hosting the MIT licensed code is free with no minute cap; you pay for your
own hardware and API keys.</p>

${faqBlock([
  {
    q: 'Is YouTube automation allowed?',
    a: 'Automating chores on a channel you create is fine. What YouTube does not monetise is inauthentic content: mass-produced, generic or repetitive videos, including AI-generated content made from generic templates. Clipping your own long videos into Shorts is not that.',
  },
  {
    q: 'Can I automate YouTube Shorts from my long videos?',
    a: 'Yes. OpenShorts Autopilot clips each new upload on your channel into vertical Shorts with subtitles and can schedule the best ones one a day. The n8n template and the API do the same for people who want to build their own flow.',
  },
  {
    q: 'Do I need to code?',
    a: 'No. Autopilot is a switch in the app. The n8n template needs n8n; the API and the MCP server are for developers and AI agents.',
  },
  {
    q: 'Can an AI agent run my channel with OpenShorts?',
    a: 'An agent such as Claude or ChatGPT can call OpenShorts over MCP to clip a video, list the clips and publish them. Whether it publishes without asking you is up to how you set it up; the n8n template keeps a human approval step on purpose.',
  },
])}

${sources([
  '<a href="https://support.google.com/youtube/answer/1311392" rel="noopener">YouTube channel monetization policies</a> (inauthentic content, renamed from repetitious content on 15 July 2025), checked 23 September 2026.',
  `Autopilot rules: <code>cloud/autopilot.py</code> in the <a href="${SITE.repo}" rel="noopener">OpenShorts source</a>; plan prices from the OpenShorts billing API, both checked 23 September 2026.`,
])}
`,
  faq: [
    { q: 'Is YouTube automation allowed?', a: 'Automating chores on a channel you create is fine. YouTube does not monetise inauthentic content: mass-produced, generic or repetitive videos, including AI-generated content from generic templates.' },
    { q: 'Can I automate YouTube Shorts from my long videos?', a: 'Yes. OpenShorts Autopilot clips each new upload into vertical Shorts with subtitles and can schedule the best ones one a day.' },
    { q: 'Do I need to code?', a: 'No. Autopilot is a switch in the app. The n8n template needs n8n; the API and MCP server are for developers and AI agents.' },
  ],
})

export function autopilotPages() {
  return [autoClip(), youtubeAutomation()]
}
