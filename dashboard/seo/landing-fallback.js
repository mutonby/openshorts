/* Crawler-visible homepage content, injected into #root at build time.
 *
 * The app is a client-rendered SPA, which means the HTML served for / contains
 * an empty <div id="root"> and nothing else. Googlebot renders JavaScript and so
 * sees the real landing page; GPTBot, ClaudeBot and PerplexityBot do not, and
 * measure the homepage as zero characters of text. That is why the site ranks in
 * Google and is absent from AI answers.
 *
 * React's createRoot().render() replaces the contents of #root on mount, so this
 * block is what a non-executing client reads and what a browser shows for the
 * few hundred milliseconds before hydration. It is neither hidden nor cloaked:
 * it is the same content the rendered page shows, in the same order, and it is
 * styled well enough that seeing it briefly is not jarring.
 *
 * Keep it in sync with Landing.jsx when the substance of the page changes. It is
 * duplicated on purpose: a build-time React SSR pass would remove the
 * duplication but would also put hydration on the critical path of a production
 * app that currently has none.
 */

import { EDITIONS, CANONICAL_ANSWERS } from './data.js'
import { renderNav, NAV_CSS, NAV_SCRIPT } from './nav.js'

const S = {
  wrap: 'max-width:72rem;margin:0 auto;padding:0 1.5rem',
  h1: 'font-family:\'Instrument Serif\',ui-serif,Georgia,serif;font-weight:400;font-size:clamp(2.2rem,5vw,3.6rem);line-height:1.06;letter-spacing:-.03em;color:oklch(96% 0.006 262);margin:0 0 1.25rem;text-transform:lowercase',
  h2: 'font-family:\'Instrument Serif\',ui-serif,Georgia,serif;font-weight:400;font-size:clamp(1.5rem,2.6vw,2rem);line-height:1.15;letter-spacing:-.02em;color:oklch(96% 0.006 262);margin:0 0 1rem;text-transform:lowercase',
  h3: 'font-size:1rem;font-weight:600;color:oklch(96% 0.006 262);margin:0 0 .35rem',
  p: 'margin:0 0 1rem;color:oklch(86% 0.01 262);line-height:1.65',
  muted: 'margin:0 0 1rem;color:oklch(64% 0.012 262);line-height:1.65',
  eyebrow:
    'font-family:"JetBrains Mono",ui-monospace,monospace;font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;color:oklch(76% 0.17 50);margin:0 0 1rem',
  section: 'padding:3.5rem 0;border-bottom:1px solid oklch(96% 0.006 262 / .08)',
  grid: 'display:grid;gap:1.25rem;grid-template-columns:repeat(auto-fit,minmax(15rem,1fr))',
  card:
    'border:1px solid oklch(96% 0.006 262 / .08);border-radius:10px;padding:1.1rem 1.25rem;background:oklch(16.5% 0.015 265)',
  a: 'color:oklch(86% 0.01 262);text-underline-offset:3px',
}

const FEATURES = [
  ['AI viral moment detection', 'Google Gemini 3.1 Flash-Lite scores the transcript and scene data to find the 3 to 15 strongest moments, so there is no manual scrubbing.'],
  ['Smart 9:16 vertical cropping', 'Dual-mode reframing with MediaPipe face tracking and a YOLOv8 fallback, stabilised so the camera holds still instead of swinging.'],
  ['Word-level automatic subtitles', 'faster-whisper produces a timestamp for every word, and the subtitles are burned in with FFmpeg.'],
  ['AI voice dubbing in 30+ languages', 'With your own ElevenLabs key, in the cloud or self-hosted: the audio is translated keeping the speaker\'s voice, then re-transcribed so subtitles match.'],
  ['Hook text overlays', 'AI-written hook titles covering the first seconds, which is where a short-form viewer decides whether to stay.'],
  ['AI video effects', 'Gemini-generated FFmpeg filter chains for colour grading, transitions and visual clean-up.'],
  ['Local video upload', 'Podcasts, webinars, livestreams, interviews and vlogs, at full resolution, from a file or a YouTube link.'],
  ['Self-hosted and private', 'Run it with Docker on your own machine and the source video never leaves your infrastructure.'],
  ['Free AI YouTube studio', 'AI thumbnail generator, 10 viral title suggestions and auto-written descriptions with chapter timestamps.'],
  ['Direct social publishing', 'Post to TikTok, Instagram Reels and YouTube Shorts from the dashboard.'],
  ['MCP server, API and CLI for AI agents', 'Connect Claude, ChatGPT or n8n to an always-on endpoint (mcp.openshorts.app/mcp) and automate clipping end to end, with a REST API, per-user keys, completion webhooks and a zero-dependency CLI (pip install openshorts). Guide at /mcp.'],
  ['AI UGC video generator', 'The AI writes a script and generates a lip-synced avatar video for any product or business, from $0.65 per video.'],
  ['AI actors with lip-sync', 'Pick an AI actor or upload a photo to get a talking-head video with matched lip movement.'],
]

const STEPS = [
  ['Upload a long-form video', 'Any video you own: podcasts, webinars, livestreams, interviews, or a YouTube link.'],
  ['AI finds the best moments', 'Gemini 3.1 Flash-Lite returns 3 to 15 candidate clips of 15 to 60 seconds each.'],
  ['Smart cropping to vertical 9:16', 'Face-tracked reframing keeps the subject centred without the camera swinging.'],
  ['Subtitles, hooks and effects', 'Word-level subtitles, an AI hook overlay, optional effects and dubbing into 30+ languages with your own ElevenLabs key.'],
  ['Download or post directly', 'Export the clips or publish straight to TikTok, Instagram Reels and YouTube Shorts.'],
]

// The first seven follow the order of the faqs array in Landing.jsx; the new
// buyer questions (watermark, card, clip count, rights) are also in the
// FAQPage JSON-LD in index.html, with the same answers.
const FAQ = [
  [
    'Is OpenShorts really free?',
    'Self-hosting is free under the MIT licence, with no watermark and no cap; OpenShorts Cloud has a free plan (your first video up to 60 minutes, then 20 minutes a month, watermarked) and paid plans from $12/month.' +
      ' Self-hosted costs you hardware and time: on a typical CPU an 8-minute video takes 5 to 8 minutes to process, and you supply your own Google Gemini key, whose free tier covers 1,500 requests a day. On OpenShorts Cloud an NVIDIA GPU clips that same video in about 50 seconds, the Gemini key is included, auto-posting to TikTok, Instagram and YouTube is already wired up, and your clips stay retrievable from any browser.',
  ],
  [
    'Do the clips have a watermark?',
    'Only on the free Cloud plan. Clips made on the free plan of OpenShorts Cloud carry a small watermark; every paid plan, from $12/month, has none, and upgrading removes the mark from the clips you already made. The self-hosted edition, free under MIT, never adds one.',
  ],
  [
    'Do I need a credit card to start?',
    'No. The free plan of OpenShorts Cloud needs no credit card: your first video is clipped free up to 60 minutes, then you get 20 free minutes every month, with a watermark. You only add a card if you choose a paid plan, from $12/month.',
  ],
  [
    'How many clips do I get from one video?',
    'Between 3 and 15, each 15 to 60 seconds long. The AI scores every moment of the video and keeps the ones that hold up as standalone clips, so the number depends on the source, not on a fixed quota.',
  ],
  [
    'Can I clip someone else\'s YouTube video?',
    'Only if you own it or have the rights or permission to use it. Before a video is processed, OpenShorts asks you to confirm you own the content or have the rights to it, and you are responsible for what you submit. Fair use depends on the country and on how the clip is used, so it is never guaranteed; reuploading someone else\'s video without permission is copyright infringement, and platforms remove it and strike the account. Clip your own videos, channels you manage, or footage you have licensed.',
  ],
  ['What is OpenShorts and how does it work?', CANONICAL_ANSWERS.whatIsIt + ' ' + CANONICAL_ANSWERS.howItWorks],
  [
    'How does OpenShorts compare to Opus Clip?',
    'Both do AI viral moment detection and smart vertical cropping. OpenShorts is MIT-licensed and can be self-hosted, so the source video never leaves your machine, and it adds voice dubbing into 30+ languages (with your own ElevenLabs key) plus an AI UGC generator with lip-synced actors. Opus Clip is closed source and cloud only, starting at $15/month as of October 2026, and it ships a larger caption-style library. OpenShorts Cloud starts at $12/month, and self-hosting is free. Full comparison at /alternatives/opus-clip.',
  ],
  [
    'How does the smart vertical cropping work?',
    'TRACK mode follows a single subject with MediaPipe face detection and a YOLOv8 fallback, damped so the crop holds still inside a safe zone rather than chasing every head movement. GENERAL mode handles group shots and landscapes by preserving the full width over a blurred backdrop. A speaker tracker prevents the crop from flipping between people and holds position through brief occlusions.',
  ],
  [
    'Can OpenShorts translate and dub videos?',
    'Yes, with your own ElevenLabs API key, into more than 30 languages, preserving the original speaker\'s voice characteristics. After dubbing, the audio is re-transcribed so the burned-in subtitles are in the target language rather than the original.',
  ],
  [
    'What is the AI UGC video generator?',
    'It generates marketing videos with AI actors for any product or business. You describe the business or paste a website URL, and the AI writes a script, generates a lip-synced actor with voiceover, adds b-roll and subtitles, with the hook spoken in the first seconds. Low Cost mode is about $0.65 per video and Premium is about $2.00. It works for restaurants, e-commerce, coaching, local businesses and apps, not only software.',
  ],
  [
    'Can I automate OpenShorts from Claude, ChatGPT or n8n?',
    'Yes. OpenShorts has a native MCP server at mcp.openshorts.app/mcp plus a REST API with per-user keys and completion webhooks, so an agent can submit a video URL, wait for processing, list the clips and publish them. The hosted endpoint is always on, with the API key created in your account page; the self-hosted edition serves the same /mcp endpoint while your machine is running. A zero-dependency CLI is on PyPI as openshorts. Full guide at /mcp.',
  ],
  [
    'What are the system requirements to self-host?',
    'Any machine with Docker, realistically 8GB of RAM and a modern multi-core CPU. An NVIDIA GPU is optional and cuts processing time by roughly an order of magnitude. Docker Compose pulls Python 3.11, FFmpeg, YOLOv8, MediaPipe, faster-whisper and the React dashboard. Linux, macOS and Windows via WSL2 are supported.',
  ],
]

/* The hero's paste box, usable before React has mounted (slow devices) and
 * visible to crawlers that never run the bundle. On submit it does exactly what
 * Landing.jsx's handleHeroSubmit + main.jsx's handleLaunchApp do: park the link
 * in os_pending_url (MediaInput preloads it), skip the landing next time and
 * open #app. main.jsx resolves its first view from the hash, so a submit made
 * before hydration lands in the app; after hydration this markup is gone and
 * Landing.jsx's own form takes over (it starts from whatever was typed here).
 * The site sends no CSP header and index.html already runs inline scripts. */
const HERO_FORM = `<form id="seo-hero-form" action="/#app" method="get" style="display:flex;flex-wrap:wrap;gap:.6rem;max-width:44rem;margin:0 0 1rem">
    <label for="seo-hero-url" style="position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0)">Video link</label>
    <input id="seo-hero-url" type="url" name="url" placeholder="Paste a YouTube link" autocomplete="off" style="flex:1 1 18rem;min-width:0;padding:.8rem 1rem;border-radius:10px;border:1px solid oklch(96% 0.006 262 / .16);background:oklch(16.5% 0.015 265);color:oklch(96% 0.006 262);font:inherit;font-size:1rem">
    <button type="submit" style="padding:.8rem 1.3rem;border-radius:10px;border:0;background:oklch(76% 0.17 50);color:oklch(13% 0.014 265);font:inherit;font-weight:600;cursor:pointer">Get free clips &rarr;</button>
  </form>
  <script>(function(){var f=document.getElementById('seo-hero-form');if(!f)return;f.addEventListener('submit',function(e){e.preventDefault();var u=(f.elements.url.value||'').trim();try{if(u)localStorage.setItem('os_pending_url',u);localStorage.setItem('openshorts_skip_landing','1')}catch(_){}location.hash='#app'})})()</script>`

/* Per-platform links, the same ones and in the same order as WORKS_WITH,
 * PLATFORMS and SWITCHING in Landing.jsx. */
const WORKS_WITH = [
  ['/platforms/youtube', 'YouTube'],
  ['/platforms/twitch', 'Twitch'],
  ['/platforms/kick', 'Kick'],
  ['/platforms/tiktok', 'TikTok'],
  ['/podcast-to-shorts', 'podcasts'],
]

const PLATFORMS = [
  ['YouTube clips', 'Paste a YouTube link and get vertical clips back, Shorts for your own channel, or the whole video turned into 9:16 pieces.',
    [['/platforms/youtube', 'YouTube clip maker'], ['/platforms/youtube-shorts', 'YouTube Shorts maker'], ['/youtube-to-shorts-converter', 'YouTube to Shorts converter']]],
  ['TikTok &amp; Reels', 'The same clips, captioned and with a hook, sent to TikTok and Instagram Reels from the dashboard.',
    [['/platforms/tiktok', 'TikTok clip maker'], ['/platforms/instagram-reels', 'AI reel generator']]],
  ['Streams', 'Hours-long Twitch and Kick VODs clipped in parts, with the facecam kept and quiet gameplay still handled.',
    [['/platforms/twitch', 'Twitch clip maker'], ['/platforms/kick', 'Kick clip maker'], ['/gta-5-clips', 'GTA 5 clips'], ['/gta-6-clips', 'GTA 6 clips'], ['/wow-forever-clips', 'WoW Forever clips']]],
  ['Podcasts &amp; talks', 'Both podcast hosts kept in frame, only the sermon clipped out of a service, and a speaker who walks the stage tracked.',
    [['/podcast-to-shorts', 'Podcast clip maker'], ['/use-cases/sermons', 'Sermon clip maker'], ['/auto-reframe', 'Auto reframe']]],
]

const SWITCHING = [
  ['/alternatives/opus-clip', 'vs Opus Clip'],
  ['/alternatives/klap', 'vs Klap'],
  ['/alternatives/vizard', 'vs Vizard'],
  ['/alternatives/submagic', 'vs Submagic'],
  ['/free-ai-clipping-tools-compared', 'Free plans compared'],
  ['/alternatives', 'All alternatives'],
]

const link = ([href, label]) => `<a style="${S.a}" href="${href}">${label}</a>`

const platformCard = ([title, body, links]) =>
  `<div style="${S.card}"><h3 style="${S.h3}">${title}</h3><p style="margin:0 0 .6rem;color:oklch(64% 0.012 262);font-size:.9rem;line-height:1.6">${body}</p><p style="margin:0;font-size:.9rem;line-height:1.8">${links.map(([h, l]) => link([h, `${l} &rarr;`])).join('<br>')}</p></div>`

const card = (title, body) =>
  `<div style="${S.card}"><h3 style="${S.h3}">${title}</h3><p style="margin:0;color:oklch(64% 0.012 262);font-size:.9rem;line-height:1.6">${body}</p></div>`

export const LANDING_FALLBACK = `<div id="seo-content" style="background:oklch(13% 0.014 265);color:oklch(86% 0.01 262);font-family:'Geist',ui-sans-serif,system-ui,sans-serif;line-height:1.65">
<style>${NAV_CSS}</style>
${renderNav()}
${NAV_SCRIPT}

<section style="${S.section};padding-top:4rem"><div style="${S.wrap}">
  <p style="${S.eyebrow}">Free AI clip generator &middot; open source</p>
  <h1 style="${S.h1}">Free AI clip maker: long video in, viral shorts out</h1>
  <p style="${S.p};max-width:44rem;font-size:1.05rem">Paste a YouTube link or upload a podcast, stream or interview. The AI picks the 3 to 15 moments worth posting, reframes them to 9:16 around the people talking, and burns in the captions, ready for TikTok, Reels and Shorts in minutes. Prefer to run it yourself? It is <a href="/free-ai-clip-generator" style="${S.a}">free and open source</a>. Claude, ChatGPT and n8n can drive it too, over <a href="/mcp" style="${S.a}">MCP</a>.</p>
  ${HERO_FORM}
  <p style="${S.muted};max-width:44rem">45,000+ accounts created &middot; 32,900+ videos clipped &middot; 8,300+ hours of video processed &middot; 7.5k GitHub stars.</p>
  <p style="${S.muted};max-width:44rem"><strong style="color:oklch(75% 0.11 150)">No credit card required.</strong> Your first video is free up to 60 minutes, then 20 free minutes every month. Paid plans from $12/month without a watermark. Prefer to run it yourself? <a style="${S.a}" href="https://github.com/mutonby/openshorts" rel="noopener">Self-host free on GitHub</a>.</p>
  <p style="${S.muted};max-width:44rem">Works with links from ${WORKS_WITH.map(link).join(' &middot; ')}.</p>
</div></section>

<section id="platforms" style="${S.section}"><div style="${S.wrap}">
  <h2 style="${S.h2}">Made for every platform: YouTube, TikTok, streams and podcasts</h2>
  <p style="${S.muted};max-width:48rem">One clip generator, tuned to where the video comes from and where the clips are going.</p>
  <div style="${S.grid}">${PLATFORMS.map(platformCard).join('')}</div>
  <p style="${S.p};margin-top:1.25rem">${link(['/platforms', 'Every platform and its specs &rarr;'])}</p>
</div></section>

<section style="${S.section}"><div style="${S.wrap}">
  <h2 style="${S.h2}">what openshorts is</h2>
  <p style="${S.p};max-width:48rem">${CANONICAL_ANSWERS.whatIsIt}</p>
  <p style="${S.p};max-width:48rem">${CANONICAL_ANSWERS.howItWorks}</p>
</div></section>

<section id="pricing" style="${S.section}"><div style="${S.wrap}">
  <h2 style="${S.h2}">Free plan vs paid: what you actually get</h2>
  <div style="${S.grid};max-width:56rem">
    ${card('OpenShorts self-hosted &middot; $0', EDITIONS.selfHosted.summary)}
    ${card('OpenShorts Cloud &middot; free tier, then from $12/month', EDITIONS.cloud.summary)}
  </div>
</div></section>

<section id="features" style="${S.section}"><div style="${S.wrap}">
  <h2 style="${S.h2}">AI clipping that finds the moments worth posting</h2>
  <div style="${S.grid}">${FEATURES.map(([t, b]) => card(t, b)).join('')}</div>
</div></section>

<section id="how-it-works" style="${S.section}"><div style="${S.wrap}">
  <h2 style="${S.h2}">How to turn a long video into Shorts with AI (5 steps)</h2>
  <ol style="max-width:48rem;padding-left:1.25rem;margin:0">
    ${STEPS.map(([t, b]) => `<li style="margin-bottom:.75rem"><strong style="color:oklch(96% 0.006 262)">${t}.</strong> <span style="color:oklch(64% 0.012 262)">${b}</span></li>`).join('')}
  </ol>
</div></section>

<section id="comparison" style="${S.section}"><div style="${S.wrap}">
  <h2 style="${S.h2}">OpenShorts vs Opus Clip, Klap and Vizard</h2>
  <p style="${S.muted};max-width:48rem">Entry pricing checked 5 October 2026. OpenShorts is $0 self-hosted or $12/month hosted without a watermark, billed monthly. 2short.ai starts at $9.90/month, Submagic at $12, Klap at $14 and Vizard at $14.50 a month on yearly billing, Opus Clip at $15/month and Quso (formerly Vidyo.ai) at $19/month yearly. OpenShorts is the only open source and self-hostable option.</p>
  <h3 style="${S.h3};margin-top:1.5rem">Switching from another tool?</h3>
  <p style="${S.muted}">${SWITCHING.map(link).join(' &middot; ')}</p>
  <p style="${S.muted}">
    <a style="${S.a}" href="/alternatives/opus-clip">Opus Clip alternative</a> &middot;
    <a style="${S.a}" href="/alternatives/klap">Klap alternative</a> &middot;
    <a style="${S.a}" href="/alternatives/vizard">Vizard alternative</a> &middot;
    <a style="${S.a}" href="/alternatives/submagic">Submagic alternative</a> &middot;
    <a style="${S.a}" href="/free-ai-clip-generator">Free AI clip generator</a> &middot;
    <a style="${S.a}" href="/tools">Free tools</a> &middot;
    <a style="${S.a}" href="/youtube-transcript-generator">YouTube transcript generator</a> &middot;
    <a style="${S.a}" href="/youtube-tag-generator">YouTube tag generator</a> &middot;
    <a style="${S.a}" href="/video-aspect-ratio-converter">Video to 9:16 converter</a> &middot;
    <a style="${S.a}" href="/auto-clip">Auto clip</a> &middot;
    <a style="${S.a}" href="/youtube-automation">YouTube automation</a> &middot;
    <a style="${S.a}" href="/platforms">All platforms</a> &middot;
    <a style="${S.a}" href="/use-cases">Use cases</a> &middot;
    <a style="${S.a}" href="/how-to">Guides</a> &middot;
    <a style="${S.a}" href="/platforms/youtube">YouTube clip maker</a> &middot;
    <a style="${S.a}" href="/platforms/youtube-shorts">YouTube Shorts maker</a> &middot;
    <a style="${S.a}" href="/platforms/tiktok">TikTok clip maker</a> &middot;
    <a style="${S.a}" href="/platforms/instagram-reels">AI reel generator</a> &middot;
    <a style="${S.a}" href="/platforms/twitch">Twitch clip maker</a> &middot;
    <a style="${S.a}" href="/platforms/kick">Kick clip maker</a> &middot;
    <a style="${S.a}" href="/auto-reframe">Auto reframe</a> &middot;
    <a style="${S.a}" href="/use-cases/sermons">Sermon clip maker</a> &middot;
    <a style="${S.a}" href="/how-to/clip-a-youtube-video">How to clip a YouTube video</a> &middot;
    <a style="${S.a}" href="/pricing">Pricing</a> &middot;
    <a style="${S.a}" href="/open-source-video-clipper">Open source video clipper</a> &middot;
    <a style="${S.a}" href="/self-hosted-ai-clip-maker">Self-hosted AI clip maker</a> &middot;
    <a style="${S.a}" href="/open-source-ai-video-generator">Open source AI video generator</a> &middot;
    <a style="${S.a}" href="/podcast-to-shorts">Podcast clips</a> &middot;
    <a style="${S.a}" href="/youtube-to-shorts-converter">YouTube to Shorts converter</a> &middot;
    <a style="${S.a}" href="/how-openshorts-works">How it works</a> &middot;
    <a style="${S.a}" href="/alternatives">All alternatives compared</a> &middot;
    <a style="${S.a}" href="/free-ai-clipping-tools-compared">Free AI clipping tools compared</a> &middot;
    <a style="${S.a}" href="/alternativas">Alternativas (ES)</a> &middot;
    <a style="${S.a}" href="/opus-clip-pricing">Opus Clip pricing</a> &middot;
    <a style="${S.a}" href="/opus-ai">Opus AI</a> &middot;
    <a style="${S.a}" href="/vizard-ai">Vizard AI</a> &middot;
    <a style="${S.a}" href="/vizard-ai-video-to-text">Vizard AI video to text</a> &middot;
    <a style="${S.a}" href="/alternatives/vidyo-ai">Vidyo.ai (Quso) alternative</a> &middot;
    <a style="${S.a}" href="/alternatives/2short">2short AI alternative</a> &middot;
    <a style="${S.a}" href="/alternatives/sendshort">SendShort alternative</a> &middot;
    <a style="${S.a}" href="/submagic-reviews">Submagic review</a> &middot;
    <a style="${S.a}" href="/mcp">MCP server &amp; API</a> &middot;
    <a style="${S.a}" href="/automate-shorts-api">Automate shorts via API</a> &middot;
    <a style="${S.a}" href="/n8n-youtube-shorts-automation">n8n template: channel on autopilot</a>
  </p>
</div></section>

<section id="faq" style="${S.section}"><div style="${S.wrap}">
  <h2 style="${S.h2}">FAQ: free AI clipping, watermarks and limits</h2>
  <dl style="max-width:48rem;margin:0">
    ${FAQ.map(([q, a]) => `<dt style="${S.h3};margin-top:1.5rem">${q}</dt><dd style="margin:.4rem 0 0;color:oklch(64% 0.012 262)">${a}</dd>`).join('')}
  </dl>
</div></section>

<section style="${S.section};border-bottom:none"><div style="${S.wrap}">
  <h2 style="${S.h2}">Open source: run the same pipeline on your own hardware</h2>
  <p style="${S.p};max-width:48rem">${EDITIONS.selfHosted.summary} It needs Docker and realistically 8GB of RAM; an NVIDIA GPU is optional and cuts processing time by roughly an order of magnitude. <a style="${S.a}" href="https://github.com/mutonby/openshorts" rel="noopener">Source and setup on GitHub</a>.</p>
</div></section>

</div>`
