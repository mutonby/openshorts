/* Product pages for the two app features that are not the clipper:
 *
 *   /tools/ai-thumbnail-maker   Thumbnail Studio (the "YouTube Studio" tab):
 *                               titles, thumbnails, description, publish.
 *   /ai-ugc-video-generator     AI Shorts: UGC-style product videos with
 *                               lip-synced AI actors.
 *
 * Neither page has an in-page widget: both features need an account (and the
 * second one the reader's own fal.ai and ElevenLabs keys), so the honest call
 * to action is the app itself, not a form that pretends to be free.
 *
 * Every number on these pages comes from the code or from data.js, and the
 * source is named in the comment above it. When a limit changes in
 * cloud/config.py, thumbnail.py or saasshorts.py, change it here too. Same
 * house rules as clip-tools.js: nothing invented, the free plan always stated
 * with its watermark and the paid price next to it, no em dashes.
 */

import { SITE, EDITIONS, PRICE_MODELS, COMPARISON_ROWS, COMPETITORS } from './data.js'
import { esc } from './render.js'

const PUBLISHED = '2026-10-29'
const CHECKED = '29 October 2026'

const CLOUD = EDITIONS.cloud
const PLANS = PRICE_MODELS.openshorts.plans
const plan = (name) => {
  const p = PLANS.find((x) => x.name === name)
  if (!p) throw new Error(`[product-pages] no OpenShorts plan "${name}" in data.js PRICE_MODELS`)
  return p
}
const FREE = plan('Free')
const STARTER = plan('Starter')
const CREATOR = plan('Creator')
const PRO = plan('Pro')
if (STARTER.monthly.price !== CLOUD.lowPrice || PRO.monthly.price !== CLOUD.highPrice) {
  throw new Error('[product-pages] PRICE_MODELS.openshorts and EDITIONS.cloud disagree on the plan prices')
}

// app.FIRST_VIDEO_MAX_MINUTES, stated inside EDITIONS.cloud.summary.
const FIRST_VIDEO_MINUTES = 60
if (!CLOUD.summary.includes(`up to ${FIRST_VIDEO_MINUTES} minutes`)) {
  throw new Error('[product-pages] EDITIONS.cloud.summary no longer says the first video is free up to ' +
    `${FIRST_VIDEO_MINUTES} minutes; update FIRST_VIDEO_MINUTES in seo/product-pages.js`)
}

const PRICING = '/pricing'

/* Thumbnail Studio limits (cloud/config.py, thumbnail.py, app.py). */
const THUMB = {
  batchMinutes: 3, //            cloud/config.THUMBNAIL_MINUTES
  analysisMinutes: 1, //         cloud/config.MANAGED_ANALYSIS_MINUTES
  transcribeMin: 2, //           cloud/config.TRANSCRIBE_MINUTES
  sourcePerCharged: 10, //       cloud/config.THUMBNAIL_SOURCE_MINUTES_PER_CHARGED
  maxSourceMinutes: 180, //      cloud/config.THUMBNAIL_MAX_SOURCE_MINUTES
  refinesPerSession: 40, //      cloud/config.THUMBNAIL_REFINES_PER_SESSION
  textCallsPerHour: 60, //       cloud/config.THUMBNAIL_TEXT_CALLS_PER_HOUR
  uiCounts: '1, 2, 3 or 4', //   ThumbnailStudio.jsx COUNT control (default 3)
  titles: 10, //                 thumbnail.analyze_video_for_titles (critic keeps 10)
  brainstorm: 25, //             same, first call
  refined: 8, //                 thumbnail.refine_titles
  mobileChars: 50, //            thumbnail.TITLE_MOBILE_CHARS
  maxTitleChars: 65, //          thumbnail.TITLE_MAX_CHARS
  frames: 5, //                  thumbnail.extract_face_frames n
}

/* AI Shorts (saasshorts.py, SaaShortsTab.jsx). The headline per-video price is
 * the one data.js already publishes; the breakdowns are the in-app estimates. */
const UGC_FROM = (COMPARISON_ROWS.find((r) => r.feature.startsWith('AI UGC video'))?.os.match(/\$[\d.]+/) || [''])[0]
if (UGC_FROM !== '$0.65') {
  throw new Error(`[product-pages] COMPARISON_ROWS now says AI UGC from ${UGC_FROM}; the Low Cost breakdown on ` +
    '/ai-ugc-video-generator (SaaShortsTab.jsx cost card) must be checked against it')
}
const UGC = {
  lowCost: UGC_FROM,
  premium: '$2', //            Landing FAQ / PricingSection "~$0.65-2 per video"
  premiumUi: '$2.50', //       SaaShortsTab.jsx cost card for Premium
  portraits: '$0.06', //       SaaShortsTab.jsx "Generate 3 new actors"
  maxScripts: 10, //           cloud/config.SAAS_MAX_SCRIPTS (the UI offers 1, 2, 3 or 5)
  descriptionChars: '10,000', // cloud/config.SAAS_DESCRIPTION_MAX_CHARS
  subpages: 3, //              saasshorts._scrape_website
}
const SENDSHORT = COMPETITORS.sendshort
const SENDSHORT_PRO = (SENDSHORT.tiers.find((t) => t[0] === 'Professional') || [])[1]?.split(':')[0]
if (!SENDSHORT_PRO || !SENDSHORT.rows['AI UGC video with lip-synced actors']) {
  throw new Error('[product-pages] SendShort Professional tier or avatar row missing from data.js')
}

const MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
  'September', 'October', 'November', 'December']
const longDate = (iso) => {
  const [y, m, d] = iso.split('-').map(Number)
  return `${d} ${MONTHS[m - 1]} ${y}`
}

/* ------------------------------------------------------------------------ */
/* Shared blocks                                                             */
/* ------------------------------------------------------------------------ */

const faqBlock = (faq) =>
  `<h2>Common questions</h2><dl class="faq">${faq
    .map((f) => `<dt>${esc(f.q)}</dt><dd>${esc(f.a)}</dd>`)
    .join('')}</dl>`

const sources = (items) =>
  `<h2>Sources</h2><ul class="sources">${items.map((s) => `<li>${s}</li>`).join('')}</ul>`

/* The three steps are shown as H3s and emitted as HowTo with the same text. */
const stepsBlock = (heading, steps) => `
<h2>${esc(heading)}</h2>
${steps.map(([n, text], i) => `<h3>${i + 1}. ${esc(n)}</h3>\n<p>${esc(text)}</p>`).join('\n')}`

const howToNode = (path, name, description, steps) => ({
  '@type': 'HowTo',
  '@id': `${SITE.url}${path}#howto`,
  name,
  description,
  tool: [{ '@type': 'HowToTool', name: 'OpenShorts (OpenShorts Cloud in the browser, or self-hosted with Docker)' }],
  step: steps.map(([n, text], i) => ({ '@type': 'HowToStep', position: i + 1, name: n, text })),
})

const monthly = (p, description) => ({
  '@type': 'Offer',
  name: `OpenShorts Cloud ${p.name}`,
  price: String(p.monthly.price),
  priceCurrency: CLOUD.currency,
  description,
  priceSpecification: {
    '@type': 'UnitPriceSpecification',
    price: String(p.monthly.price),
    priceCurrency: CLOUD.currency,
    unitText: 'MONTH',
  },
})

/* Real offers only, and never a lone price of 0 (see the note in index.html). */
const appNode = (path, name, description, offers) => ({
  '@type': 'SoftwareApplication',
  '@id': `${SITE.url}${path}#app`,
  name,
  description,
  url: `${SITE.url}${path}`,
  applicationCategory: 'MultimediaApplication',
  operatingSystem: 'Web browser (OpenShorts Cloud); Docker (self-hosted)',
  publisher: { '@id': `${SITE.url}/#organization` },
  offers,
})

const selfHostedOffer = (description) => ({
  '@type': 'Offer',
  name: 'OpenShorts self-hosted (MIT)',
  price: '0',
  priceCurrency: CLOUD.currency,
  description,
})

const planTable = () => `
<table>
<thead><tr><th>Plan</th><th>Price</th><th>Minutes a month</th></tr></thead>
<tbody>
<tr><td>Cloud Free</td><td>$0, no card</td><td>First video up to ${FIRST_VIDEO_MINUTES} min, then ${FREE.monthly.minutes} min. Clips watermarked</td></tr>
<tr><td>Cloud ${esc(STARTER.name)}</td><td>$${STARTER.monthly.price}/month</td><td>${STARTER.monthly.minutes}, no watermark</td></tr>
<tr><td>Cloud ${esc(CREATOR.name)}</td><td>$${CREATOR.monthly.price}/month</td><td>${CREATOR.monthly.minutes}, no watermark</td></tr>
<tr><td>Cloud ${esc(PRO.name)}</td><td>$${PRO.monthly.price}/month</td><td>${PRO.monthly.minutes}, no watermark</td></tr>
<tr><td>Self-hosted</td><td>$0, MIT</td><td>No meter; your hardware and your own API keys</td></tr>
</tbody>
</table>`

/* ------------------------------------------------------------------------ */
/* /tools/ai-thumbnail-maker                                                 */
/* ------------------------------------------------------------------------ */

const THUMB_PATH = '/tools/ai-thumbnail-maker'

const thumbSteps = [
  ['Upload the video, or type the title', `Open the YouTube Studio tab in the app and drop in the video file. It is transcribed while you wait, ten frames are sampled across it, and Gemini writes ${THUMB.brainstorm} candidate titles before a second pass keeps the best ${THUMB.titles}, each paired with a 1 to 4 word thumbnail text. No video yet? Type your own title instead and go straight to the thumbnails.`],
  ['Pick a title and generate thumbnails', `Choose a title (or refine the list in the chat), then pick a frame of your face from the video or upload a photo, add optional instructions and a background, and generate ${THUMB.uiCounts} thumbnails at once. Each one is a different concept, cropped to 1280x720 and saved as a JPEG under 2 MB.`],
  ['Check it on a phone, then download or publish', 'The phone preview shows the thumbnail at the size most viewers will see it. Download the one you want, or carry on: the studio writes a description with chapters from the transcript and can publish the video to YouTube with that title, description and thumbnail.'],
]

const thumbFaq = [
  { q: 'Is this AI thumbnail maker free?', a: `Not on the hosted free plan. On OpenShorts Cloud, generating thumbnail images is reserved for paid plans, from $${STARTER.monthly.price}/month for ${STARTER.monthly.minutes} minutes, and each batch uses ${THUMB.batchMinutes} of those minutes. The free plan (your first video up to ${FIRST_VIDEO_MINUTES} minutes, then ${FREE.monthly.minutes} minutes a month, watermarked clips, no card) can analyse a video and get titles, but not images. The self-hosted edition is free under MIT and includes the studio; you bring your own Google Gemini key and Google bills the image calls.` },
  { q: 'What size are the thumbnails?', a: 'Exactly 1280x720 pixels, 16:9, saved as JPEG. The file is re-compressed in steps until it is under 2 MB, which is the limit YouTube puts on a custom thumbnail.' },
  { q: 'Can it put my face on the thumbnail?', a: `Yes. After a video is analysed the studio scans sampled frames for a large, sharp face and offers up to ${THUMB.frames} of them, spread across the runtime. Pick one and it is used as the person reference. You can also upload a photo, which takes priority over the frame.` },
  { q: 'Is the text on the thumbnail spelled correctly?', a: 'With the default Crisp setting, yes: the image model leaves one side empty and the text is set afterwards in the Anton typeface with a black stroke, from the exact words you chose. With AI painted, the image model draws the text itself, which can look more integrated but can also misspell it.' },
  { q: 'How many thumbnails do I get per batch?', a: `You choose ${THUMB.uiCounts} in the studio (3 by default). They are different concepts, not one prompt repeated: a text call designs each one (hook text, which side the text sits on, palette, scene) before the images are generated in parallel. A batch costs ${THUMB.batchMinutes} minutes on Cloud whatever the count.` },
  { q: 'Does it write YouTube titles too?', a: `Yes, that is the first step. It brainstorms ${THUMB.brainstorm} titles across seven styles, then a critic scores them and keeps ${THUMB.titles}, with the payoff inside the first ${THUMB.mobileChars} characters and the main keyword in the first three words. A chat lets you ask for changes ("shorter", "less clickbait", "in Spanish") and returns ${THUMB.refined} new options. Refining titles does not use minutes.` },
  { q: 'Can I use the thumbnail generator without a video?', a: 'Yes. Choose "Write your own", type the title (up to 70 characters) and go straight to thumbnail generation. Without a video there are no face frames to pick from and no transcript, so the description is yours to write and publishing from the studio is not available.' },
  { q: 'How is it different from a free online thumbnail maker?', a: 'Most thumbnail makers are template editors: you pick a layout and drop in a photo and text. This one generates the image with AI from your own video, with your face taken from its frames and the title written from its transcript, and it outputs a finished 1280x720 JPEG. The trade is that image generation is a paid feature on Cloud, and there is no layer editor for moving elements by hand.' },
  { q: 'Does it change the thumbnail of a video already on my channel?', a: 'No. Publishing from the studio uploads the video you analysed as a new public YouTube video, with the title, description and thumbnail you chose, through your connected YouTube account. To change an existing video, download the JPEG and upload it in YouTube Studio.' },
  { q: 'Does it make Shorts thumbnails?', a: 'It makes 16:9 thumbnails for regular YouTube videos. Shorts are vertical and YouTube handles their cover differently, so for Shorts the clip generator is the relevant tool, not the thumbnail studio.' },
  { q: 'What languages does it work in?', a: 'Titles, thumbnail text and descriptions are written in the language of the video\'s transcript, never translated unless you ask. In the title chat you can ask for another language and the new titles and thumbnail texts come back in it.' },
  { q: 'Are the thumbnails marked as AI-generated?', a: 'Yes, in the file metadata. Every generated JPEG carries the IPTC "trainedAlgorithmicMedia" source type in XMP and an EXIF Software tag naming OpenShorts. Nothing is drawn on the image itself.' },
]

const thumbnailMaker = () => ({
  path: THUMB_PATH,
  published: PUBLISHED,
  updated: PUBLISHED,
  title: 'AI YouTube Thumbnail Maker & Title Generator | OpenShorts',
  description: `Make YouTube thumbnails with AI from your video: 1280x720, under 2 MB, your face, crisp text, plus ${THUMB.titles} titles and chapters. Paid plans from $${STARTER.monthly.price}/month.`,
  h1: 'AI YouTube thumbnail maker: thumbnails and titles from your own video',
  breadcrumb: [{ name: 'Free tools', path: '/tools' }, { name: 'AI thumbnail maker' }],
  relatedTitle: 'More YouTube tools',
  // First three that are built win; the fallbacks cover a build where the
  // /platforms pages do not exist yet.
  related: ['/youtube-tag-generator', '/platforms/youtube-shorts', '/platforms/youtube',
    '/youtube-transcript-generator', '/youtube-shorts-maker'],
  cta: {
    label: 'YouTube Studio',
    title: 'Titles, thumbnails and a description from one video',
    body: `Image generation is on paid plans, from $${STARTER.monthly.price}/month (${THUMB.batchMinutes} minutes a batch). The free plan gets titles, and self-hosting is free under MIT with your own Gemini key.`,
    button: 'Open the thumbnail studio',
    href: `${SITE.url}/#app?tab=thumbnails`,
  },
  tldr: [
    `OpenShorts makes YouTube thumbnails from the video they are for. It reads the transcript and a spread of frames, proposes ${THUMB.titles} titles each paired with a short thumbnail text, finds frames where your face is large and sharp, and generates ${THUMB.uiCounts} different thumbnail concepts at once, each 1280x720 and under YouTube's 2 MB limit.`,
    `It is not a free web toy: it lives in the OpenShorts app (the YouTube Studio tab). On the hosted Cloud, thumbnail images are a paid-plan feature from $${STARTER.monthly.price}/month, ${THUMB.batchMinutes} minutes of the plan per batch. Titles and descriptions work on the free plan too, and the self-hosted edition includes everything under MIT with your own Gemini key.`,
  ],
  body: `
<h2>What this AI thumbnail maker does</h2>
<p>In the app the feature is the <strong>YouTube Studio</strong> tab, and it is
built around one idea: a thumbnail and a title are one package, so they should
be made together and from the actual video rather than from a blank prompt. It
runs in five steps, and you can stop after any of them.</p>
<ol>
<li><strong>Input.</strong> Upload the video file, or type a title if you only want the thumbnail.</li>
<li><strong>Titles.</strong> ${THUMB.titles} titles written from the transcript and frames, the strongest two marked as top picks, each with a 1 to 4 word thumbnail text that adds to the title rather than repeating it.</li>
<li><strong>Generate.</strong> Thumbnails built on that title and text, with your face, an optional background and your own instructions.</li>
<li><strong>Description.</strong> A YouTube description with a short hook, calls to action, chapters with timestamps and 5 to 10 hashtags.</li>
<li><strong>Publish.</strong> Upload the video to YouTube with the chosen title, description and thumbnail, through your connected account.</li>
</ol>
<p>Everything it produces is editable before it is used: the title field, the
thumbnail text, the instructions for the image and the description are all plain
text boxes, and nothing is published until you press the button.</p>

${stepsBlock('How to make a YouTube thumbnail with AI, in three steps', thumbSteps)}

<h2>What the YouTube thumbnail generator produces</h2>
<table>
<thead><tr><th>Property</th><th>What the studio produces</th></tr></thead>
<tbody>
<tr><td>Size</td><td>1280x720 pixels, cover-cropped to 16:9 whatever the image model returns</td></tr>
<tr><td>File</td><td>JPEG, re-compressed in steps until it is under 2 MB</td></tr>
<tr><td>Per batch</td><td>${THUMB.uiCounts} thumbnails, each a different concept</td></tr>
<tr><td>Text</td><td>Set in Anton with a black stroke (Crisp), or painted by the model (AI painted)</td></tr>
<tr><td>Person</td><td>A frame of your face from the video, or an uploaded photo</td></tr>
<tr><td>Labelling</td><td>IPTC "trainedAlgorithmicMedia" in XMP plus an EXIF Software tag</td></tr>
</tbody>
</table>
<p>Each batch is designed before it is drawn. A text call plans every concept
separately: the hook text, the side of the frame it sits on, the palette and the
scene. Then one image call per concept runs in parallel, so three thumbnails are
three ideas you can choose between, not three near-copies of the same picture.
If none of them is right, change the instructions ("red and black, dramatic
lighting, a laptop on fire") and press Regenerate.</p>

<h2>Thumbnail maker or thumbnail generator: what you get here</h2>
<p>A thumbnail maker in the usual sense is a canvas: templates, stickers, a
background remover, and you do the design. An AI YouTube thumbnail generator
does the design from a description. This studio sits closer to the second, with
one difference that matters: the description is your video. The title, the
words on the image and the face all come from it, so the result looks like the
video it is for, and you steer it with plain-text instructions instead of
dragging layers.</p>

<h2>YouTube thumbnail size and rules</h2>
<p>YouTube's own guidance for custom thumbnails, checked ${esc(CHECKED)}: a
resolution of 1280x720 with a minimum width of 640 pixels, a 16:9 aspect ratio,
an image format such as JPG, GIF or PNG, and a file under 2 MB. Custom thumbnails
also need a verified channel; until your account is verified, YouTube offers
only frames it picks from the video.</p>
<p>The studio meets the technical part by construction: every output is
1280x720, 16:9 and a JPEG under 2 MB, so it uploads without a resize. What no
tool can do for you is the part that actually decides clicks, which is whether
the image still reads when it is shrunk. A thumbnail is designed at 1280 pixels
wide and mostly seen at a fraction of that, in a phone feed, a search result
or the suggested column beside another video. The image prompts the studio sends
ask for a picture that stays readable at 168x94 pixels: one subject, strong
contrast, no clutter, no small details. The phone preview next to the results
shows each candidate at that kind of size before you choose.</p>

<h2>Crisp text or AI painted text</h2>
<p>Image models are good at scenes and unreliable at letters. Ask one for a
thumbnail that says "I QUIT MY JOB" and you will often get "I QUIT MY JOP", or
the right words in the wrong language's accents. So the default, <strong>Crisp</strong>,
splits the job: the model paints the scene and leaves a clean band (about 45%
of the width, or 40% of the height) with no lettering at all, and the text is
set afterwards in the Anton typeface with a heavy black stroke and a soft shadow.
The words are exactly the ones you approved, spelled right, with accents intact.</p>
<p><strong>AI painted</strong> lets the image model draw the text itself. When
it works the lettering can sit inside the scene (on a sign, in neon, in the
sky) in a way overlaid text cannot. When it does not, you get a typo in the
largest type on your channel. It is there for the cases where the integration
is worth a check.</p>

<h2>Your face, taken from the video</h2>
<p>Thumbnails with a recognisable creator tend to look like a channel rather
than stock art, and the creator is already in the video. After analysis the
studio samples frames, scores each one by how large and how sharp the face is
(MediaPipe face detection plus a Laplacian sharpness measure), and keeps up to
${THUMB.frames} spread across the runtime, so you get different moments rather
than five copies of one second. Click one and it becomes the person reference
for the batch. An uploaded photo of you, or of a guest who agreed to it, always
takes priority over the frame. You can also upload a background image for the
scene.</p>

<h2>Titles written for the thumbnail, not next to it</h2>
<p>The title pass reads the transcript plus ten frames at 1024 pixels, never the
whole video. It runs twice. The first call brainstorms ${THUMB.brainstorm}
titles across seven styles: a specific result, a curiosity gap that the video
pays off, a contrarian take, a how-to with a concrete promise, a number or list,
the question the viewer is already asking, and a story or confession. The second
call acts as a critic: it scores each candidate for specificity, curiosity,
clarity and honesty, drops near-duplicates of the same angle, and returns the
best ${THUMB.titles}.</p>
<p>The rules baked into that pass are the ones that hold on a phone. The payoff
has to land inside the first ${THUMB.mobileChars} characters, because that is
roughly where mobile layouts cut a title, and nothing runs past ${THUMB.maxTitleChars}.
The main subject goes in the first three words. Titles stay in the language of
the transcript. And each one comes with a thumbnail text of 1 to 4 words that
complements it: if the title says what happened, the image says how it felt or
what it cost, so the two together carry more than either repeats.</p>
<p>If none of the ${THUMB.titles} fits, the chat beside the list takes plain
requests ("more specific", "less clickbait", "mention the price", "write them in
Spanish") and returns ${THUMB.refined} new titles with their thumbnail texts.
Up to the last 20 turns of that conversation are sent back each time, so it
remembers what you already rejected.</p>

<h2>Thumbnail tips that move click-through rate</h2>
<p>Click-through rate is what a thumbnail is for, and most of what moves it is
editorial rather than technical. The studio is built around these, but they
apply whatever tool you use.</p>
<ul>
<li><strong>One subject, one idea.</strong> A face, an object or a result. Three competing elements at 168 pixels wide become a blur.</li>
<li><strong>Few words, and not the title.</strong> One to four words that add something: the stakes, the number, the emotion. Repeating the title wastes the space.</li>
<li><strong>Contrast over detail.</strong> A bright subject on a dark ground, or the reverse. Fine textures and thin fonts disappear at small sizes.</li>
<li><strong>A readable face.</strong> Large, sharp and showing an expression that matches the video. Which is why the studio scores frames by face size and sharpness first.</li>
<li><strong>Keep the bottom-right corner clear.</strong> YouTube draws the video length there, over whatever you put underneath.</li>
<li><strong>Promise what the video delivers.</strong> A thumbnail that over-promises can win the click and lose the viewer in the first seconds, which shows up in retention and in what YouTube recommends next.</li>
<li><strong>Look like your channel.</strong> A consistent colour or layout lets returning viewers recognise your videos in a busy feed.</li>
<li><strong>Test, then decide.</strong> Where YouTube Studio's own thumbnail testing is available on your channel, upload two or three of the concepts from one batch and let the views decide, rather than guessing.</li>
</ul>

<h2>Which plan includes it, and what it costs</h2>
<p>The studio is the same code in both editions; what differs is who pays for
the Google calls behind it.</p>
<p><strong>OpenShorts Cloud.</strong> Thumbnail image generation is a paid-plan
feature: a free account that presses Generate is asked to upgrade. Every
action is metered from the same minute balance your clips use, and the meter is
deliberately light:</p>
<table>
<thead><tr><th>Action</th><th>Minutes on Cloud</th></tr></thead>
<tbody>
<tr><td>Analyse a video (transcription)</td><td>1 per ${THUMB.sourcePerCharged} minutes of source, never less than ${THUMB.transcribeMin}</td></tr>
<tr><td>Title analysis on top of it</td><td>${THUMB.analysisMinutes}</td></tr>
<tr><td>Refine titles in the chat</td><td>0</td></tr>
<tr><td>Generate a batch of thumbnails</td><td>${THUMB.batchMinutes} per batch, paid plans only</td></tr>
<tr><td>Write the description</td><td>0</td></tr>
</tbody>
</table>
<p>So a 20-minute video costs ${THUMB.transcribeMin + THUMB.analysisMinutes} minutes to
analyse and ${THUMB.batchMinutes} per batch of thumbnails after that. The free
plan can run the analysis and get titles and a description from its own
minutes; the images need a paid plan. The plans, from the
<a href="${PRICING}">pricing table</a>:</p>
${planTable()}
<p>Fair-use limits apply on Cloud: sources up to ${THUMB.maxSourceMinutes}
minutes long, ${THUMB.refinesPerSession} title refinements per session and
${THUMB.textCallsPerHour} title or description requests per hour per account.
A free account that has run out of minutes can keep refining a session it
already paid for, but cannot start a new one.</p>
<p><strong>Self-hosted.</strong> The studio ships in the open source edition,
free under MIT, with no minutes and no plan check. It needs your own Google
Gemini key: titles and descriptions are text calls, thumbnails are image calls,
and Google bills those to your key at its own rates. Publishing needs your own
Upload-Post key and a connected YouTube profile.</p>

<h2>Descriptions with chapters, and publishing</h2>
<p>From the analysed video, the Description step writes a ready-to-paste YouTube
description: a two or three sentence hook, the usual calls to subscribe, like
and comment, a chapter list built from the transcript's timestamps (starting at
0:00, at least three chapters, each at least ten seconds apart, which is the
format YouTube recognises as chapters), and 5 to 10 hashtags at the end, all
under YouTube's 5,000-character limit and in the video's language. You can edit
it in place or regenerate it. Writing it does not use minutes.</p>
<p>Publish then uploads that same video to YouTube as a new public video, with
your title, the description and the thumbnail you picked, through your connected
YouTube account. It runs in the background and the studio shows when the upload
is done. If you only wanted the image, download it from the Generate step and
stop there.</p>

<h2>What it does not do</h2>
<ul>
<li><strong>It does not edit a thumbnail by hand.</strong> There is no layer editor; you steer it with instructions and regenerate, or open the JPEG in an image editor.</li>
<li><strong>It does not replace the thumbnail of a video already on YouTube.</strong> Publishing creates a new upload. For an existing video, download the file and set it in YouTube Studio.</li>
<li><strong>It does not take a YouTube link in the app.</strong> The studio works from an uploaded file (or from a title you type). Download your own video from YouTube Studio first if it is already online.</li>
<li><strong>It does not make vertical Shorts covers.</strong> Every output is 16:9 for regular videos.</li>
<li><strong>It does not generate images on the free Cloud plan.</strong> Titles and descriptions, yes; thumbnails need a paid plan or the self-hosted edition.</li>
<li><strong>It does not predict your CTR.</strong> The critic scores titles for clarity and honesty, but only your audience decides what gets clicked.</li>
<li><strong>It does not clear the rights to a face.</strong> Use your own face, or a guest's with their permission.</li>
</ul>

<h2>More tools for the same video</h2>
<p>A thumbnail is one part of packaging a video. The free
<a href="/youtube-tag-generator">YouTube tag and title generator</a> writes tags,
title options and a description from a short brief, with no sign-up and no
video needed. And once the long video is out, the same footage can feed your
Shorts: <a href="/platforms/youtube-shorts">turn it into YouTube Shorts</a> with
the clip generator, or see <a href="/platforms/youtube">everything OpenShorts does
with a YouTube video</a>.</p>

${faqBlock(thumbFaq)}

${sources([
  `<a href="https://support.google.com/youtube/answer/72431" rel="noopener">YouTube Help: Add video thumbnails</a> (1280x720, minimum width 640, 16:9, JPG/GIF/PNG, under 2 MB; custom thumbnails need a verified account), checked ${esc(CHECKED)}.`,
  `Studio behaviour and limits: <code>thumbnail.py</code>, the <code>/api/thumbnail/*</code> routes in <code>app.py</code>, <code>cloud/config.py</code> and <code>dashboard/src/components/ThumbnailStudio.jsx</code>. Source: <a href="${SITE.repo}" rel="noopener">github.com/mutonby/openshorts</a>.`,
  `Plan prices and minutes: the <a href="${PRICING}">pricing table</a>, checked ${esc(longDate(PRICE_MODELS.openshorts.checked))}.`,
])}
`,
  faq: thumbFaq,
  extraNodes: [
    howToNode(THUMB_PATH, 'How to make a YouTube thumbnail with AI',
      'Generate 1280x720 YouTube thumbnails, titles and a description from your own video with the OpenShorts thumbnail studio.', thumbSteps),
    appNode(THUMB_PATH, 'OpenShorts AI YouTube thumbnail maker',
      'AI YouTube thumbnails (1280x720, under 2 MB) with matching titles and a chaptered description, made from your own video.',
      [
        monthly(STARTER, `${STARTER.monthly.minutes} minutes a month; thumbnail image generation included at ${THUMB.batchMinutes} minutes per batch.`),
        monthly(CREATOR, `${CREATOR.monthly.minutes} minutes a month; thumbnail image generation included at ${THUMB.batchMinutes} minutes per batch.`),
        monthly(PRO, `${PRO.monthly.minutes} minutes a month; thumbnail image generation included at ${THUMB.batchMinutes} minutes per batch.`),
        selfHostedOffer('Free and open source; the studio runs with your own Google Gemini key, billed by Google.'),
      ]),
  ],
})

/* ------------------------------------------------------------------------ */
/* /ai-ugc-video-generator                                                   */
/* ------------------------------------------------------------------------ */

const UGC_PATH = '/ai-ugc-video-generator'

const ugcSteps = [
  ['Describe the product or paste its website', `Open the AI Shorts tab in the app. Paste the product's URL (the page and up to ${UGC.subpages} of its subpages are read, and the product is researched on the web) or describe the business in your own words. Pick the language, the actor's gender, a script style and how many scripts you want.`],
  ['Choose a script, an actor and a voice', 'Each script is a 20 to 25 second ad in five beats: hook, problem, solution, demo, call to action. Pick one, edit the narration if you like, then choose an actor (from the gallery, from three new AI portraits, or your own photo) and a voice from your ElevenLabs account.'],
  ['Generate, then download or post', `Choose Low Cost (about ${UGC.lowCost} a video) or Premium (about ${UGC.premium}), billed by fal.ai and ElevenLabs on your keys. You get a 1080x1920 MP4 with the lip-synced actor, two b-roll inserts and burned-in captions, ready to download or post to TikTok, Instagram, YouTube, Facebook, LinkedIn, Threads and Bluesky.`],
]

const ugcFaq = [
  { q: 'How much does an AI UGC video cost?', a: `About ${UGC.lowCost} a video in Low Cost mode and about ${UGC.premium} in Premium (the in-app estimate shows up to ${UGC.premiumUi} for Premium), paid to fal.ai and ElevenLabs on your own keys. On OpenShorts Cloud the script and orchestration come out of your plan's minutes (the free plan included); plans run from $0 to $${CLOUD.highPrice}/month. Self-hosted, OpenShorts itself is free under MIT and you also bring a Gemini key.` },
  { q: 'Is there a free AI UGC video generator?', a: `Not in the sense of free videos. On OpenShorts Cloud the research and the scripts run on the free plan, from its minutes, so you can get UGC scripts for nothing; the video itself (actor, lip-sync, voice, b-roll) is always billed by fal.ai and ElevenLabs on your own keys, about ${UGC.lowCost} a video in Low Cost mode. Paid Cloud plans start at $${STARTER.monthly.price}/month; self-hosted, OpenShorts is free under MIT and you pay only those providers and your own Gemini key.` },
  { q: 'Do I need my own API keys?', a: 'Yes, for the video. Generation always uses your fal.ai key (actor image, talking-head video, lip-sync, b-roll) and your ElevenLabs key (voice), on Cloud and self-hosted alike, and those providers bill you directly. On Cloud the Gemini part (research and scripts) and posting run on the managed keys; self-hosted, you supply a Gemini key too.' },
  { q: 'What do I need to give it?', a: `Either the product's website URL or a written description of up to ${UGC.descriptionChars} characters, or both (the description then adds context). With a URL, the page and up to ${UGC.subpages} subpages are read and the product is researched on the web before the scripts are written.` },
  { q: 'How long are the videos?', a: 'About 20 to 25 seconds, by design. Every script follows a fixed five-part structure (hook, problem, solution, demo, call to action), which is the length and shape of a typical short-form product ad.' },
  { q: 'What format is the video?', a: 'A vertical 9:16 MP4 at 1080x1920, H.264 video with AAC audio, captions burned in. It fits TikTok, Instagram Reels and YouTube Shorts without any reframing.' },
  { q: 'Can I use my own face or a real person as the actor?', a: 'Yes: upload a photo and it is animated and lip-synced to the voiceover. Only use a photo of yourself or of someone who has agreed to appear in your ads; animating a real person without consent can break the law and the platforms\' rules.' },
  { q: 'Which languages are supported?', a: 'English and Spanish. Scripts are written natively in the chosen language (casual US English, or Spanish), and the voice comes from ElevenLabs\' multilingual model, so it is not a translation of an English script.' },
  { q: 'What is the difference between Low Cost and Premium?', a: `Low Cost animates the actor image with MiniMax Hailuo 2.3 Fast and then lip-syncs it with VEED Lipsync, for about ${UGC.lowCost} a video. Premium generates the talking head with Kling Avatar v2 from the image and the audio in one step, for about ${UGC.premium}, with more natural movement. Both use the same script, voice, b-roll and captions.` },
  { q: 'Is it an AI avatar video generator?', a: 'In the sense that it makes talking-head videos of an AI-generated person, yes. It is built for one job, short vertical product ads, rather than long presenter videos or training content, and it does not train a persistent avatar of you: each video is generated from a single image.' },
  { q: 'Can I post the videos straight to social media?', a: 'Yes, to TikTok, Instagram, YouTube, Facebook, LinkedIn, Threads and Bluesky, now or scheduled, from the result screen. TikTok receives it as a draft you finish in the TikTok app, where you can add a trending sound. Each script also comes with a suggested caption.' },
  { q: 'Do I have to label it as AI?', a: 'Rules differ by platform and by country, and they change. TikTok, YouTube and Instagram all have their own labels for realistic AI-generated content, and advertising rules generally forbid presenting an actor as a real customer. Check the current rules of the platform you post on; OpenShorts does not add a label for you.' },
]

const ugcGenerator = () => ({
  path: UGC_PATH,
  published: PUBLISHED,
  updated: PUBLISHED,
  title: 'AI UGC Video Generator: AI Actors for Ads | OpenShorts',
  description: `Turn a product URL or description into a 20-25 s vertical UGC ad with a lip-synced AI actor, b-roll and captions. About ${UGC.lowCost} a video on your own keys.`,
  h1: 'AI UGC video generator: product ads with lip-synced AI actors',
  breadcrumb: [{ name: 'AI UGC video generator' }],
  relatedTitle: 'Related tools',
  related: ['/alternatives/arcads', '/platforms/tiktok', '/platforms/instagram-reels', THUMB_PATH],
  cta: {
    label: 'AI Shorts',
    title: 'A product link in, a UGC-style ad out',
    body: `Open AI Shorts in the app and add your fal.ai and ElevenLabs keys: about ${UGC.lowCost} a video billed by them. The script runs on your plan, free plan included; paid plans from $${STARTER.monthly.price}/month.`,
    button: 'Make a UGC video',
    href: `${SITE.url}/#app?tab=saasshorts`,
  },
  tldr: [
    `OpenShorts' AI Shorts turns a product URL or a short description into a vertical UGC-style ad: Gemini researches the product and writes 20 to 25 second scripts (hook, problem, solution, demo, call to action), an AI actor says it with lip-synced voice, two b-roll inserts cover the problem and the product, and the captions are burned in. Output is a 1080x1920 MP4 for TikTok, Reels and Shorts.`,
    `The video generation runs on your own fal.ai and ElevenLabs keys: about ${UGC.lowCost} a video in Low Cost mode and about ${UGC.premium} in Premium, billed by those providers. On OpenShorts Cloud the scripts come out of your plan's minutes, the free plan included; self-hosted, OpenShorts is free under MIT and you add a Gemini key.`,
  ],
  body: `
<h2>What AI UGC is, and what this generator makes</h2>
<p>UGC, user-generated content, is the style of ad that looks like a person
talking to their phone camera about something they use, rather than a produced
commercial. It works on TikTok and Reels because it looks like the posts around
it. The AI Shorts tab makes that format without a person or a camera: an
AI-generated actor delivers a script written for your product, with real
lip movement, cut with b-roll and captioned the way short-form video is.</p>
<p>Every video follows the same five-beat structure, because that is the shape a
20-second product ad has to have:</p>
<table>
<thead><tr><th>Beat</th><th>Time</th><th>On screen</th></tr></thead>
<tbody>
<tr><td>Hook</td><td>0 to 5 s</td><td>The actor, talking to camera</td></tr>
<tr><td>Problem</td><td>5 to 9 s</td><td>B-roll of the pain point, voice over it</td></tr>
<tr><td>Solution</td><td>9 to 16 s</td><td>The actor presents the product</td></tr>
<tr><td>Demo</td><td>16 to 21 s</td><td>B-roll of the product in use</td></tr>
<tr><td>Call to action</td><td>21 to 25 s</td><td>The actor, pointing to the link in bio</td></tr>
</tbody>
</table>
<p>The result is a vertical 9:16 MP4 at 1080x1920, H.264 with AAC audio, with
captions burned in a few words at a time, the way short-form viewers expect
to read them. Each script also comes with a suggested caption for the post.</p>
<p>Put another way, it is a UGC AI video generator for one format only: the
short vertical ad. The AI UGC scripts are written first and shown as text, so you
choose the angle before anything is generated or paid for.</p>

${stepsBlock('How to make an AI UGC video, in three steps', ugcSteps)}

<h2>What you put in</h2>
<ul>
<li><strong>A website URL, a description, or both.</strong> With a URL, the page and up to ${UGC.subpages} subpages are read and Gemini researches the product on the web, then extracts the value proposition, the audience, the pain points and the angles worth trying. With only a description (up to ${UGC.descriptionChars} characters), the scripts are written from what you say.</li>
<li><strong>Language.</strong> English or Spanish, written natively in each rather than translated.</li>
<li><strong>Actor.</strong> A woman or a man.</li>
<li><strong>Style.</strong> UGC natural (authentic, talking to camera), educational, shock or discovery (a surprising opener), storytelling (a mini narrative arc) or before and after.</li>
<li><strong>How many scripts.</strong> 1, 2, 3 or 5 per run (the API accepts up to ${UGC.maxScripts}). Each script takes a different pain point or angle, which is the point: you test angles, not one ad.</li>
</ul>
<p>It is not limited to software. The product analysis works from whatever the
page or the description says, so a restaurant, a shop, an online course, a
coaching practice or a local service gets scripts about its own customers'
problems.</p>

<h2>The AI actor and the voice</h2>
<p>After you pick a script you choose who says it. There are three ways:</p>
<ul>
<li><strong>The actor gallery:</strong> portraits generated for earlier videos, ready to reuse so a series of ads keeps the same face.</li>
<li><strong>Three new portraits:</strong> describe the person or keep the script's suggestion, and three options are generated (about ${UGC.portraits} for the three).</li>
<li><strong>Your own photo:</strong> upload a picture and that face is animated and lip-synced. Use only your own face or someone who has agreed to it.</li>
</ul>
<p>The voice is any voice in your ElevenLabs account, previewable before you
generate, including voices you created there. The narration is shown as editable
text, so you can fix a product name or change a line before anything is paid
for. Speech is generated with ElevenLabs' multilingual model, and the captions
are then transcribed from that audio so they match what is said word for word.</p>

<h2>Low Cost or Premium</h2>
<p>The two modes share the script, the voice, the b-roll and the captions; they
differ in how the talking head is made. The estimates below are the ones the app
shows before you generate, and the providers' own prices decide the final bill.</p>
<table>
<thead><tr><th>Step</th><th>Low Cost</th><th>Premium</th></tr></thead>
<tbody>
<tr><td>Actor image (Flux 2 Pro)</td><td>$0.05</td><td>$0.05</td></tr>
<tr><td>Voice (ElevenLabs)</td><td>about $0.10</td><td>about $0.10</td></tr>
<tr><td>Talking head</td><td>Hailuo 2.3 Fast image-to-video, $0.19, then VEED Lipsync, $0.20</td><td>Kling Avatar v2 from image and audio, about $1.69</td></tr>
<tr><td>Two b-roll inserts</td><td>about $0.10</td><td>about $0.70</td></tr>
<tr><td>Compositing and captions</td><td>$0 (FFmpeg)</td><td>$0 (FFmpeg)</td></tr>
<tr><td><strong>Per video</strong></td><td><strong>about ${UGC.lowCost}</strong></td><td><strong>about ${UGC.premium} to ${UGC.premiumUi}</strong></td></tr>
</tbody>
</table>
<p>Low Cost is the one to test angles with: three Low Cost videos cost about what
one Premium video does. Premium generates the face and the mouth together, so head
movement and expression look more natural, and it is worth it for the script that
won the test. The b-roll in both modes is an AI still image animated with a slow
zoom and pan, which reads as motion in a feed without the price of generated
video. A Premium talking head takes a few minutes to render; failed steps can be
retried and reuse the parts already generated, so a failure late in the pipeline
does not pay for the image and the voice twice.</p>

<h2>What it costs, and who bills you</h2>
<p>There are two bills, and keeping them apart is the honest way to describe it.</p>
<p><strong>The video generation</strong> (actor image, talking head, lip-sync,
b-roll, voice) is billed by fal.ai and ElevenLabs, to the keys you connect in
Settings. That is true on OpenShorts Cloud and self-hosted alike: no plan
includes it yet. The keys are kept in your browser and sent with each request.</p>
<p><strong>The research, the scripts and the posting</strong> are OpenShorts. On
Cloud they run on the managed Gemini and posting keys and each analysis run
uses 1 minute of your plan's balance, which the free plan (your first video up
to ${FIRST_VIDEO_MINUTES} minutes, then ${FREE.monthly.minutes} minutes a month,
watermarked clips, no card) covers too. Paid plans, from the
<a href="${PRICING}">pricing table</a>:</p>
${planTable()}
<p>Self-hosted, OpenShorts is free under MIT with no meter; you add your own
Gemini key for the research and scripts, plus
the same fal.ai and ElevenLabs keys.</p>

<h2>AI UGC or a human creator</h2>
<p>Hiring a UGC creator is usually quoted somewhere between $50 and $500 a video,
depending on the creator and the usage rights. An AI UGC video here costs about
${UGC.lowCost} to ${UGC.premium} in generation, takes minutes instead of a
brief, a shipment and a wait, and can be made in five variants to see which
angle works before spending more.</p>
<p>What the AI version cannot give you is the thing that makes UGC persuasive
in the first place: a real person who really used the product. The actor has
never held it, the b-roll is illustrative rather than footage of your product,
and viewers who look closely can tell. The sensible split is to use AI UGC to
find the hook and the angle cheaply, then put a real customer or creator on the
script that won, and to never present an AI actor as a real customer.</p>

<h2>Use cases</h2>
<ul>
<li><strong>Testing ad angles.</strong> Five scripts from five pain points, generated in Low Cost, posted or run as ads; keep the one that holds attention.</li>
<li><strong>Apps and software.</strong> The original use: a URL in, ads about the problems the product solves.</li>
<li><strong>E-commerce products.</strong> A product page becomes a hook, a problem and a reason to tap the link.</li>
<li><strong>Restaurants and local businesses.</strong> A description of the place and what it is known for, in English or Spanish.</li>
<li><strong>Coaches and course creators.</strong> Talking-head promos for a programme, without filming another one.</li>
<li><strong>Agencies.</strong> First drafts of UGC concepts for a client, before a creator is booked.</li>
</ul>

<h2>Vertical formats, and where to post them</h2>
<p>Every video is 9:16 at 1080x1920, the native shape of
<a href="/platforms/tiktok">TikTok</a>, <a href="/platforms/instagram-reels">Instagram Reels</a>
and <a href="/platforms/youtube-shorts">YouTube Shorts</a>, and about 20 to 25
seconds, well inside the limits of all three. From the result screen you can
post to TikTok, Instagram, YouTube, Facebook, LinkedIn, Threads and Bluesky at once or schedule it for a date and time.
TikTok receives it as a draft, which you finish in the app; that is also where
you add a trending sound, which posting through an API cannot. If you already
have long videos of your own, the clip generator is the other half: it cuts real
footage into Shorts and TikToks, while AI Shorts makes videos when there is no
footage at all. For the thumbnail of the long video itself, see the
<a href="${THUMB_PATH}">AI thumbnail maker</a>.</p>

<h2>Disclosure: saying it is AI</h2>
<p>An AI actor is a synthetic person, and the platforms and advertising rules
treat that as something viewers may need to know. TikTok, YouTube and Instagram
each have their own label for realistic AI-generated content, with their own
rules for when it is required, and advertising rules in many countries forbid
presenting a paid or invented endorsement as a genuine customer review. Those
rules change and differ by market, so check the current ones for where you post.
OpenShorts does not add a label or a mark to the video; whether and how to
disclose is your decision, and labelling it is rarely a disadvantage with an
audience that can usually tell anyway.</p>

<h2>Rights and what you are responsible for</h2>
<ul>
<li><strong>Faces.</strong> Generated actors are not real people. An uploaded photo must be you or someone who consented to being animated in an ad.</li>
<li><strong>Claims.</strong> The script is written from your page and from web research. Read it before generating: you are the advertiser, and the claims in the ad are yours.</li>
<li><strong>Providers' terms.</strong> The image, video and voice come from fal.ai's models and ElevenLabs, under your accounts and their terms, including what they allow for commercial use.</li>
<li><strong>The public gallery is opt-in.</strong> A checkbox lets you share a finished video in the OpenShorts gallery; when ticked, the video, the product name and the script are public. It is off unless you tick it.</li>
</ul>

<h2>Compared with avatar video subscriptions (Arcads, SendShort)</h2>
<p>Most AI avatar video tools sell a monthly subscription with a quota of videos
or minutes. SendShort, for example, includes AI avatar videos from its
Professional plan, ${esc(SENDSHORT_PRO)} (checked ${esc(longDate(SENDSHORT.checked))}). OpenShorts works the other way
round: the code is open source, there is no avatar quota, and you pay the
generation providers per video on your own keys, so a month with two ads costs
two ads. The trade-off is setup: two provider accounts and keys instead of one
subscription, and a narrower product, built only for short vertical product ads.
Weighing it against Arcads specifically? See <a href="/alternatives/arcads">the Arcads alternative</a>.</p>

<h2>What it does not do</h2>
<ul>
<li><strong>It does not make long videos.</strong> Scripts are 20 to 25 seconds, by design.</li>
<li><strong>It does not cover the generation cost on any plan yet.</strong> fal.ai and ElevenLabs bill you directly.</li>
<li><strong>It does not write in languages other than English and Spanish.</strong></li>
<li><strong>It does not add background music.</strong> Add a sound when you finish the post, as TikTok drafts invite you to.</li>
<li><strong>It does not draw a separate hook headline on screen.</strong> The hook is spoken in the first seconds and appears in the captions.</li>
<li><strong>It does not film your product.</strong> The b-roll is AI imagery of the problem and the product category, not your real interface or packaging.</li>
<li><strong>It does not train a persistent avatar.</strong> Each video starts from one image; reuse a gallery actor to keep the same face across a series.</li>
</ul>

${faqBlock(ugcFaq)}

${sources([
  `Pipeline, models and per-step estimates: <code>saasshorts.py</code> and <code>dashboard/src/components/SaaShortsTab.jsx</code>; limits in <code>cloud/config.py</code>. Source: <a href="${SITE.repo}" rel="noopener">github.com/mutonby/openshorts</a>.`,
  `Provider prices are the estimates the app shows, ${esc(CHECKED)}; fal.ai and ElevenLabs set the final price on their own pricing pages.`,
  `SendShort plans: our comparison data, checked ${esc(longDate(SENDSHORT.checked))}. See <a href="/alternatives/sendshort">SendShort compared</a>.`,
  `OpenShorts plan prices and minutes: the <a href="${PRICING}">pricing table</a>.`,
])}
`,
  faq: ugcFaq,
  extraNodes: [
    howToNode(UGC_PATH, 'How to make an AI UGC video with an AI actor',
      'Turn a product URL or description into a vertical UGC-style ad with a lip-synced AI actor, b-roll and captions, using OpenShorts AI Shorts.', ugcSteps),
    appNode(UGC_PATH, 'OpenShorts AI UGC video generator',
      `Vertical UGC-style product ads with lip-synced AI actors, from a URL or a description. Video generation is billed by fal.ai and ElevenLabs on your own keys, about ${UGC.lowCost} to ${UGC.premium} a video.`,
      [
        {
          '@type': 'Offer',
          name: 'OpenShorts Cloud free plan',
          price: '0',
          priceCurrency: CLOUD.currency,
          description: `Scripts and orchestration from the free minutes (first video up to ${FIRST_VIDEO_MINUTES} minutes, then ${FREE.monthly.minutes} a month; clips watermarked). Video generation billed by fal.ai and ElevenLabs on your keys.`,
        },
        monthly(STARTER, `${STARTER.monthly.minutes} minutes a month, no watermark. Video generation billed by fal.ai and ElevenLabs on your keys.`),
        selfHostedOffer('Free and open source; bring your own Gemini, fal.ai and ElevenLabs keys.'),
      ]),
  ],
})

/* ------------------------------------------------------------------------ */
/* /alternatives/arcads                                                      */
/* ------------------------------------------------------------------------ */

/* Arcads, read on arcads.ai and its help centre on ARCADS.checked. It publishes
 * no plan prices on its site (they are shown inside the app), so this page
 * states what it does publish and never invents a dollar figure for it. */
const ARCADS = {
  name: 'Arcads',
  checked: '2026-10-30',
  plans: 'Starter (8,000 credits a month), Creator (16,000 credits a month), then Pro, Scale and Business',
  trial: '72 hours with 1,000 generation credits; a card is required at sign-up and the trial rolls into the Starter plan unless cancelled',
}
const ARCADS_PATH = '/alternatives/arcads'

const arcadsFaq = [
  { q: 'What is a cheaper alternative to Arcads?', a: `OpenShorts AI Shorts, if you would rather pay per video than per month: about ${UGC.lowCost} a video in Low Cost mode (about ${UGC.premium} in Premium), billed by fal.ai and ElevenLabs on your own keys, with no avatar quota. OpenShorts itself is free to self-host under MIT, or from $${STARTER.monthly.price}/month hosted.` },
  { q: 'How much does Arcads cost?', a: `Arcads does not publish its prices on its website (checked ${longDate(ARCADS.checked)}). Its help centre names the plans: ${ARCADS.plans}. The free trial is ${ARCADS.trial}.` },
  { q: 'Is there a free Arcads alternative?', a: `Not for the video itself: lip-synced AI actors cost compute on every tool. With OpenShorts the scripts and product research run on the free plan, and the video is billed per piece by fal.ai and ElevenLabs on your keys, about ${UGC.lowCost} in Low Cost mode. Self-hosted, OpenShorts is free under MIT and you add a Gemini key.` },
  { q: 'What does OpenShorts AI Shorts make?', a: 'A 20 to 25 second vertical UGC-style ad from a product URL or description: hook, problem, solution, demo and call to action, said by a lip-synced AI actor, with two b-roll inserts and burned-in captions, as a 1080x1920 MP4. English and Spanish.' },
  { q: 'Where is Arcads better?', a: 'It is a dedicated UGC ad product with one subscription and no API keys to manage, a larger actor library, and more ad formats and workflows than a single 20-second structure. OpenShorts AI Shorts is narrower: one ad format, your own provider keys, open source code.' },
]

const arcadsAlternative = () => ({
  path: ARCADS_PATH,
  published: '2026-10-30',
  updated: '2026-10-30',
  title: 'Arcads Alternative: AI UGC Ads From ~$0.65 a Video',
  description: `An open source alternative to Arcads: product URL in, 20-25 s UGC ad with a lip-synced AI actor out, about ${UGC.lowCost} a video on your own keys.`,
  h1: 'A pay-per-video alternative to Arcads for AI UGC ads',
  lede: `The cheaper-to-start alternative to Arcads is OpenShorts AI Shorts: it turns a product URL into a 20 to 25 second UGC-style ad with a lip-synced AI actor, b-roll and captions for about ${UGC.lowCost} a video in Low Cost mode (about ${UGC.premium} in Premium), billed by fal.ai and ElevenLabs on your own keys. OpenShorts itself is free to self-host under MIT, or from $${STARTER.monthly.price}/month hosted. <span class="checked">Checked ${esc(longDate(ARCADS.checked))}</span> Arcads terms read on arcads.ai and its help centre.`,
  breadcrumb: [{ name: 'Alternatives', path: '/alternatives' }, { name: 'Arcads' }],
  relatedTitle: 'Related tools',
  related: [UGC_PATH, '/alternatives/sendshort', '/platforms/tiktok', '/alternatives'],
  cta: {
    label: 'AI Shorts',
    title: 'A product link in, a UGC-style ad out',
    body: `Open AI Shorts in the app and add your fal.ai and ElevenLabs keys: about ${UGC.lowCost} a video billed by them. Paid plans from $${STARTER.monthly.price}/month; self-host free under MIT.`,
    button: 'Make a UGC video',
    href: `${SITE.url}/#app?tab=saasshorts`,
  },
  facts: [
    { k: 'OpenShorts per video', v: `~${UGC.lowCost}`, s: `Low Cost mode; ~${UGC.premium} Premium, on your keys` },
    { k: 'Arcads pricing', v: 'In the app', s: 'Not published on arcads.ai' },
    { k: 'Arcads trial', v: '72 hours', s: '1,000 credits, card required' },
    { k: 'Open source', v: 'OpenShorts', s: 'MIT; Arcads is closed' },
  ],
  tldr: [
    `Arcads is a closed-source subscription for AI UGC ads with AI actors. It does not publish its prices on its website; its help centre lists ${ARCADS.plans}, and a ${ARCADS.trial}.`,
    `OpenShorts AI Shorts does one format of the same job (a 20 to 25 second product ad with a lip-synced AI actor) and bills it per video on your own fal.ai and ElevenLabs keys: about ${UGC.lowCost} in Low Cost mode, so two ads in a month cost two ads, not a plan.`,
  ],
  body: `
<h2>Arcads vs OpenShorts AI Shorts</h2>
<table>
<thead><tr><th></th><th>OpenShorts AI Shorts</th><th>Arcads</th></tr></thead>
<tbody>
<tr><td>Price</td><td class="os">About ${UGC.lowCost} a video (Low Cost) or ${UGC.premium} (Premium), paid to fal.ai and ElevenLabs; OpenShorts $0 self-hosted or from $${STARTER.monthly.price}/month hosted</td><td>Subscription in credits; prices shown in the app, not on arcads.ai</td></tr>
<tr><td>Trial</td><td class="os">Free plan, no card: scripts and research run on its minutes</td><td>${esc(ARCADS.trial)}</td></tr>
<tr><td>Input</td><td class="os">Product URL (page + up to ${UGC.subpages} subpages) or a description</td><td>Script, product and actor chosen in the app</td></tr>
<tr><td>Output</td><td class="os">20-25 s vertical MP4, 1080x1920, lip-synced actor, 2 b-roll inserts, captions</td><td>UGC-style ads with AI actors in several formats</td></tr>
<tr><td>Your own face</td><td class="os">Yes, from a photo (with consent)</td><td>See arcads.ai</td></tr>
<tr><td>Open source</td><td class="os">Yes, MIT</td><td>No</td></tr>
</tbody>
</table>

<h2>How AI Shorts makes an ad</h2>
${ugcSteps.map(([n, text], i) => `<h3>${i + 1}. ${esc(n)}</h3>\n<p>${esc(text)}</p>`).join('\n')}
<p>The full walkthrough, the formats and the limits are on the
<a href="${UGC_PATH}">AI UGC video generator</a> page.</p>

<h2>When Arcads is the better choice</h2>
<ul>
<li>You want one subscription and no provider accounts or API keys to manage.</li>
<li>You need more than one ad format, or a larger library of actors.</li>
<li>Your team produces ads at a volume where a credit plan is simpler to budget than per-video provider bills.</li>
</ul>

${faqBlock(arcadsFaq)}

${sources([
  `Arcads plans and free trial: help.arcads.ai ("How to buy more credits", "How the free trial works"), checked ${esc(longDate(ARCADS.checked))}. arcads.ai showed no public price list on that date.`,
  `OpenShorts AI Shorts estimates: <code>dashboard/src/components/SaaShortsTab.jsx</code>, ${esc(CHECKED)}. Source: <a href="${SITE.repo}" rel="noopener">github.com/mutonby/openshorts</a>.`,
])}
`,
  faq: arcadsFaq,
})

export function productPages() {
  return [thumbnailMaker(), ugcGenerator(), arcadsAlternative()]
}
