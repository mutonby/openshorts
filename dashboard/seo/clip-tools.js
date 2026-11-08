/* Clip tool pages: one per source or destination the clip generator serves
 * (/platforms/youtube, /platforms/youtube-shorts, /platforms/tiktok,
 * /platforms/instagram-reels, /platforms/twitch, /platforms/kick, /auto-reframe,
 * /use-cases/sermons).
 *
 * Each page opens with the paste-a-link widget right under the H1 (static
 * HTML here, behaviour in dashboard/tools/paste-link.js), the same hand-off
 * to the app the homepage hero does. The copy is written per platform on
 * purpose: the sections, the examples and the limits differ because the
 * sources differ, and a template with the platform name swapped in is exactly
 * what these pages must not be.
 *
 * Rules that apply to every line (same as tools.js): nothing invented, every
 * competitor number comes from data.js with its check date, the free plan is
 * always stated with its watermark and the paid price next to it, no em
 * dashes.
 */

import { SITE, EDITIONS, COMPETITORS, PRICE_MODELS, COMPARISON_ROWS } from './data.js'
import { demoBlock, pasteLinkWidget, trademarkNote } from './components.js'
import { esc } from './render.js'

const PUBLISHED = '2026-10-28'
const SPECS_CHECKED = '28 October 2026'

const CLOUD = EDITIONS.cloud
// app.FIRST_VIDEO_MAX_MINUTES. data.js states it inside the Cloud summary
// rather than as a field, so check the two still agree instead of trusting a
// copy of the number.
const FIRST_VIDEO_MINUTES = 60
if (!CLOUD.summary.includes(`up to ${FIRST_VIDEO_MINUTES} minutes`)) {
  throw new Error('[clip-tools] EDITIONS.cloud.summary no longer says the first video is free up to ' +
    `${FIRST_VIDEO_MINUTES} minutes; update FIRST_VIDEO_MINUTES in seo/clip-tools.js`)
}
const STARTER = PRICE_MODELS.openshorts.plans.find((p) => p.name === 'Starter')

const PRICE_LINE =
  `your first video clipped whole up to ${FIRST_VIDEO_MINUTES} minutes, then ${CLOUD.freeMinutes} minutes a month, ` +
  `with a watermark and no credit card. Paid plans start at $${CLOUD.lowPrice}/month with no watermark, ` +
  'and the self-hosted edition is free under MIT.'

const FREE_SHORT =
  `First video free up to ${FIRST_VIDEO_MINUTES} min, then ${CLOUD.freeMinutes} min a month, watermarked, no card. ` +
  `No watermark from $${CLOUD.lowPrice}/month, or self-host free under MIT.`

/* ------------------------------------------------------------------------ */
/* Shared blocks                                                             */
/* ------------------------------------------------------------------------ */

const widget = (placeholder, label) => pasteLinkWidget({ placeholder, label, hint: FREE_SHORT })

const faqBlock = (faq) =>
  `<h2>Common questions</h2><dl class="faq">${faq
    .map((f) => `<dt>${esc(f.q)}</dt><dd>${esc(f.a)}</dd>`)
    .join('')}</dl>`

const sources = (items) =>
  `<h2>Sources</h2><ul class="sources">${items.map((s) => `<li>${s}</li>`).join('')}</ul>`

const MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
  'September', 'October', 'November', 'December']
const longDate = (iso) => {
  const [y, m, d] = iso.split('-').map(Number)
  return `${d} ${MONTHS[m - 1]} ${y}`
}

/* Free tiers side by side. Every competitor cell is read from data.js, and the
 * check date printed above the table is the one stored with those facts. */
const freeTable = (slugs, heading = 'Free tier compared') => {
  const rows = slugs.map((slug) => {
    const c = COMPETITORS[slug]
    if (!c) throw new Error(`[clip-tools] no competitor "${slug}" in data.js`)
    const free = c.facts.find((f) => f.k === 'Free plan')
    return { name: c.name, free: free ? `${free.v}. ${free.s}` : c.tiers[0][1], paid: c.entryPrice, checked: c.checked }
  })
  const dates = [...new Set(rows.map((r) => r.checked))].map(longDate).join(', ')
  return `
<h2>${esc(heading)}</h2>
<p class="checked">Competitor plans checked ${esc(dates)} on their public pricing pages.</p>
<table>
<thead><tr><th>Tool</th><th>Free plan</th><th>Paid from</th></tr></thead>
<tbody>
<tr><td class="os">OpenShorts Cloud</td><td class="os">First video up to ${FIRST_VIDEO_MINUTES} min, then ${CLOUD.freeMinutes} min/month. Watermark, no credit card</td><td class="os">$${CLOUD.lowPrice}/month, ${STARTER.monthly.minutes} min, no watermark</td></tr>
<tr><td class="os">OpenShorts self-hosted</td><td class="os">Free under MIT. No watermark, no cap; your hardware and API keys</td><td class="os">$0</td></tr>
${rows.map((r) => `<tr><td>${esc(r.name)}</td><td>${esc(r.free)}</td><td>${esc(r.paid)}</td></tr>`).join('\n')}
</tbody>
</table>`
}

const howToNode = (path, name, description, steps) => ({
  '@type': 'HowTo',
  '@id': `${SITE.url}${path}#howto`,
  name,
  description,
  tool: [{ '@type': 'HowToTool', name: 'OpenShorts (OpenShorts Cloud in the browser, or self-hosted with Docker)' }],
  step: steps.map(([n, text], i) => ({ '@type': 'HowToStep', position: i + 1, name: n, text })),
})

/* Two offers, not one: a lone price of 0 is what made engines describe the
 * whole product as free (see the note in index.html). */
const appNode = (path, name, description) => ({
  '@type': 'SoftwareApplication',
  '@id': `${SITE.url}${path}#app`,
  name,
  description,
  url: `${SITE.url}${path}`,
  applicationCategory: 'MultimediaApplication',
  operatingSystem: 'Web browser (OpenShorts Cloud); Docker (self-hosted)',
  isAccessibleForFree: true,
  publisher: { '@id': `${SITE.url}/#organization` },
  offers: [
    {
      '@type': 'Offer',
      name: 'OpenShorts Cloud free plan',
      price: '0',
      priceCurrency: CLOUD.currency,
      description: `First video clipped whole up to ${FIRST_VIDEO_MINUTES} minutes, then ${CLOUD.freeMinutes} minutes a month, with a watermark. No credit card.`,
    },
    {
      '@type': 'Offer',
      name: `OpenShorts Cloud ${STARTER.name}`,
      price: String(CLOUD.lowPrice),
      priceCurrency: CLOUD.currency,
      description: `${STARTER.monthly.minutes} minutes a month, no watermark.`,
      priceSpecification: {
        '@type': 'UnitPriceSpecification',
        price: String(CLOUD.lowPrice),
        priceCurrency: CLOUD.currency,
        unitText: 'MONTH',
      },
    },
  ],
})

/* The three steps are shown as H3s and emitted as HowTo with the same text. */
const stepsBlock = (heading, steps) => `
<h2>${esc(heading)}</h2>
${steps.map(([n, text], i) => `<h3>${i + 1}. ${esc(n)}</h3>\n<p>${esc(text)}</p>`).join('\n')}`

/* The one-sentence answer under the H1, before the form: what AI Overviews and
 * Gemini lift for "YouTube clip maker" style questions. Pages that carry one
 * were revised on ANSWER_REVISED. */
const ANSWER_REVISED = '2026-10-30'
const ANSWER_PRICE = `Your first video is free up to ${FIRST_VIDEO_MINUTES} minutes (watermarked, no card); no watermark from $${CLOUD.lowPrice}/month, or self-host it free under MIT.`

const common = (path, extra) => ({
  path,
  published: PUBLISHED,
  updated: PUBLISHED,
  relatedTitle: 'More clip tools',
  ...extra,
})

// The sites the app lists as accepted (MediaInput SUPPORTED_PLATFORMS) plus
// Kick, which the download path handles like Twitch.
const SITES_LIST = 'YouTube, Vimeo, TikTok, X (Twitter), Twitch, Kick, Facebook, Instagram, Dailymotion, Reddit and Streamable'

/* ------------------------------------------------------------------------ */
/* /platforms/youtube                                                       */
/* ------------------------------------------------------------------------ */

const ytClipSteps = [
  ['Paste the YouTube link', 'Copy the address of the video (a watch link, a youtu.be share link, a Shorts link or the replay of a finished livestream) and paste it in the box. OpenShorts fetches the video itself, so there is nothing to download first.'],
  ['The AI finds the moments', 'The audio is transcribed word by word and the video is scanned for cuts. Google Gemini then scores the whole transcript in windows of about 90 seconds, shortlists the strongest and cuts each moment so it opens on a complete thought, 15 to 60 seconds long unless you set other limits.'],
  ['Reframe, caption, export or post', 'Each clip is reframed to 9:16 so the speaker stays in the frame, captions are burned in from the word-level transcript and a hook line is drawn over the opening seconds. Download the MP4s, or post them to YouTube Shorts, TikTok and Instagram from the dashboard.'],
]

const ytClipFaq = [
  { q: 'Is this YouTube clip maker free?', a: `On OpenShorts Cloud the free plan clips your first video whole up to ${FIRST_VIDEO_MINUTES} minutes, then ${CLOUD.freeMinutes} minutes of source video a month, with a watermark and no credit card. Paid plans start at $${CLOUD.lowPrice}/month and remove the watermark. The self-hosted edition is free and open source under MIT, with no watermark and no cap, on your own hardware and API keys.` },
  { q: 'Do free clips have a watermark?', a: 'Yes, on the hosted free plan. Upgrading to a paid plan removes it, including from the clips you already made: the clean file is kept alongside the marked copy, so nothing is rendered again. The self-hosted edition never adds one.' },
  { q: 'Do I need a credit card to start?', a: 'No. You sign in with Google or an email link and the free minutes are there. A card is only asked for when you choose a paid plan.' },
  { q: 'How many clips does one YouTube video give?', a: 'It depends on the length and on how much of it stands alone. For anything longer than about 12 minutes the AI is asked for 6 to 12 clips, and it returns fewer rather than padding when the material does not hold them. You can pin an exact number from 1 to 15 in the advanced options.' },
  { q: 'How long can the YouTube video be?', a: `There is no fixed maximum on the link itself; the limit that matters is your minute balance on Cloud. On the free plan a source longer than your balance is clipped from the start up to what you have left (your first video is clipped whole up to ${FIRST_VIDEO_MINUTES} minutes). The minimum is 45 seconds of material. You can also pick just a stretch of a long video and pay only for that stretch.` },
  { q: 'How long is each clip?', a: 'Between 15 and 60 seconds by default. In the advanced options you can set the minimum anywhere from 5 to 175 seconds and the maximum from 10 to 180 seconds, which covers the 3-minute ceiling of a YouTube Short.' },
  { q: 'What format are the clips?', a: 'MP4 files in 9:16, at least 1080 pixels wide (1080x1920 from a 1080p source), with captions and the hook burned in. Each clip also comes with an AI-written title and description.' },
  { q: 'Which YouTube links work?', a: 'Any link to one public video: watch pages, youtu.be links, Shorts, embeds and the replays of finished livestreams. Playlists, channels and search pages are refused because they are not one video; open the video and paste its own link. Private and members-only videos cannot be fetched.' },
  { q: 'Can I clip someone else\'s YouTube video?', a: 'Only with the rights to it: your own channel, a client\'s channel you manage, a guest\'s episode with their permission, or licensed and public-domain footage. Every job asks you to confirm that. Reposting other people\'s videos without permission is copyright infringement and the platforms remove it.' },
  { q: 'Can I use it as a YouTube trimmer to cut one part of a video?', a: 'Partly. After you paste the link, "clip only part of it" sets a start and an end on two sliders in 5-second steps (the stretch has to be at least a minute long), and only that stretch is analysed and counted against your minutes. The AI then cuts its clips from inside it, and the clip editor lets you trim or extend each clip on word boundaries. What it does not do is export one exact, hand-picked segment as it is; for that, the guide on how to clip a YouTube video covers YouTube Studio and ffmpeg.' },
  { q: 'Do I need to download the YouTube video first?', a: 'No. Paste the link and OpenShorts fetches the video itself, so there is no downloader or converter step. Only for private or members-only videos do you download the file from YouTube Studio and upload it instead.' },
  { q: 'How is it different from Opus Clip or Vizard?', a: `All three take a YouTube link and return vertical clips. Opus Clip and Vizard are closed cloud products with free plans of 60 credits a month (checked ${longDate(COMPETITORS['opus-clip'].checked)}); OpenShorts is open source and can be self-hosted, and its hosted plans are billed monthly from $${CLOUD.lowPrice}. Vizard has the more complete hand-editing timeline, and Opus Clip the larger caption-style library.` },
]

const youtubeClipMaker = () => common('/platforms/youtube', {
  // "clip a youtube video" and "youtube clipper" outweigh "youtube clip maker"
  // (which the H1 keeps); 58 characters with the brand.
  title: 'YouTube Clipper: Clip a YouTube Video With AI | OpenShorts',
  description: `Clip a YouTube video from its link: AI picks the 15-60 s moments, reframes them to 9:16 and adds captions. First video free up to ${FIRST_VIDEO_MINUTES} min, watermarked.`,
  h1: 'YouTube clip maker: clip a YouTube video from its link',
  breadcrumb: [{ name: 'Platforms', path: '/platforms' }, { name: 'YouTube clip maker' }],
  updated: ANSWER_REVISED,
  lede: `OpenShorts is an AI YouTube clip maker: paste a YouTube link and it picks the 3 to 15 strongest moments of 15 to 60 seconds, reframes each one to 9:16 around the person speaking and burns in captions, ready for Shorts, TikTok and Reels. ${ANSWER_PRICE}`,
  tool: { entry: 'tool-paste-link', html: widget('https://www.youtube.com/watch?v=...', 'YouTube video link') },
  cta: {
    label: 'Clip a video',
    title: 'One link in, a set of vertical clips out',
    body: `Free plan: ${PRICE_LINE}`,
  },
  related: ['/how-to/clip-a-youtube-video', '/platforms/youtube-shorts', '/auto-reframe', '/platforms/tiktok', '/alternatives/opus-clip'],
  tldr: [
    'Paste the link of a 40-minute interview and OpenShorts reads all of it: a transcript with a timestamp on every word, a map of the camera cuts, and a score for each 90-second stretch. The ten best stretches go to a second pass that cuts the clips, and for a source that long it is asked for 6 to 12 of them, each 15 to 60 seconds, each opening on a sentence rather than halfway through one.',
    'Every clip comes back vertical, captioned and with a hook line over the first seconds, ready to post as a Short, a TikTok or a Reel. You never download the YouTube video or upload it again.',
  ],
  body: `
${stepsBlock('How the YouTube clip maker works', ytClipSteps)}

<h2>What one real job looks like</h2>
<p>The clip below came out of a 39-minute interview pasted as a YouTube link,
untouched. Nobody chose the moment, set a crop or typed a caption: Gemini
picked the passage from the transcript, the face tracker kept the speaker in a
frame a third the width of the original, and the words on screen are the
transcript's own, timed word by word.</p>
${demoBlock('talk')}
<p>That is the default path. Everything in it can be overridden afterwards: the
clip editor shows which stretches of the source a clip was cut from and lets
you trim, extend, split or reorder them, snapped to word boundaries; the
reframe editor lets you move the crop scene by scene where the tracker chose
wrong; captions can be restyled and the hook rewritten.</p>

<h2>YouTube clips versus clips made from a YouTube video</h2>
<p>YouTube has its own <strong>Clip</strong> button, and it is worth knowing
what YouTube clips are before choosing a tool. A YouTube clip is a 5 to 60 second
selection that stays on YouTube: it plays from the original video, it can only
be watched by people who can watch that video, and you share it as a link. It
is a bookmark with a start and an end, not a file. It stays horizontal, has no
burned-in captions and cannot be posted to TikTok or Instagram.</p>
<p>A video clip maker for YouTube produces new videos. What comes out of OpenShorts is a set of
separate MP4 files in 9:16, each one reframed, captioned and titled, that you
keep like any other export and can upload anywhere. When people search for a
"YouTube video clip converter" this is usually what they mean: turn the long
horizontal upload into short vertical pieces, not save a timestamp.</p>
<p>If you only need one moment and would rather cut it by hand, <a
href="/how-to/clip-a-youtube-video">how to clip a YouTube video</a> walks
through the four free ways: the Clip button, YouTube Studio, ffmpeg, and an AI
clipper for Shorts.</p>

<h2>Cutting one part of a YouTube video: trimmer, cutter or clipper</h2>
<p>People who search for a YouTube trimmer or a YouTube video cutter usually
want one thing: a specific stretch out of a long video, as a file. A YouTube
clipper like this one starts from the other end, finding the moments for you,
but you can point it at the stretch you mean. On OpenShorts Cloud, once you
paste the link the app shows how long the video is and how many minutes it will
use, and <strong>clip only part of it</strong> opens two sliders, from and to,
in 5-second steps. The stretch must run at least a minute, and only that stretch
is downloaded, analysed and counted against your minutes.</p>
<p>The AI then cuts its clips from inside that stretch, and the advanced
options set how many (an exact number from 1 to 15) and how long. Each clip can
be trimmed, extended or split afterwards in the clip editor, on word boundaries,
so it starts and ends on a whole sentence. What OpenShorts does not do is export
a frame-exact segment you marked by hand, untouched and horizontal: YouTube
Studio's trim tool or a one-line ffmpeg command does that, and the <a
href="/how-to/clip-a-youtube-video">guide to clipping a YouTube video</a> covers
both.</p>

<h2>Which links it takes</h2>
<p>Anything that points at <strong>one</strong> public video works: a
<code>watch?v=</code> page, a <code>youtu.be</code> share link (with or
without a timestamp), a Shorts link, an embed link, or the replay of a
livestream once it has ended. The box above checks the shape before sending you
on, because the server refuses the rest anyway: a playlist, a channel, an
<code>@handle</code> page or a search results page is not one video, and
walking through every video behind it is never what you meant.</p>
<p>What cannot be fetched: private videos, members-only videos and anything
YouTube puts behind a sign-in, such as age-restricted uploads. For those,
download the file from YouTube Studio if it is yours and upload it in the app
instead. Uploads work on every plan, the free one included.</p>

<h2>Works with links from other sites too</h2>
<p>The same box takes public video links from ${SITES_LIST}. Some of those have
their own page here because they behave differently
(<a href="/platforms/twitch">Twitch</a> and <a href="/platforms/kick">Kick</a>
VODs run for hours, <a href="/platforms/tiktok">TikTok</a> and
<a href="/platforms/instagram-reels">Instagram</a> are usually destinations rather than
sources). The rest work exactly like a YouTube link:</p>
<ul>
<li><strong>Vimeo:</strong> public videos and links to a single video in a showcase. Password-protected Vimeo links cannot be fetched; download the file and upload it.</li>
<li><strong>X (Twitter):</strong> the link of a post that carries a video. Long uploads are where clipping pays off.</li>
<li><strong>Facebook:</strong> public videos and the replays of public Facebook Lives, which is where many churches, local newsrooms and webinar hosts stream.</li>
<li><strong>Reddit:</strong> the link of a video post, including <code>v.redd.it</code> links.</li>
<li><strong>Dailymotion</strong> and <strong>Streamable:</strong> any public video link.</li>
</ul>
<p>If a site is not on that list the box tells you so and still lets you try,
because the downloader behind it understands far more sites than the ones we
test. When a link fails, uploading the file is the fallback that always works.</p>

<h2>Choosing how many clips, and how long</h2>
<p>By default the length of the video decides. The scoring pass shortlists
roughly one window per 90 seconds of source, capped at ten, and the cutting
pass is asked for a number of clips that grows with that shortlist: three to
six for a two-minute video, four to ten for five minutes, six to twelve for
anything past about twelve minutes. Those are targets for the model, and an
empty answer is accepted where a padded one would not be. The floor is
enforced in code rather than left to the prompt, so a long video does not
collapse to a single clip.</p>
<p>Open <strong>advanced options</strong> to override any of it: an exact
number of clips from 1 to 15, a minimum length from 5 to 175 seconds and a
maximum from 10 to 180. And for a long video where you only care about one
part, <strong>clip only part of it</strong> lets you pick the start and the end,
so only that stretch is analysed and only that stretch counts against your
minutes.</p>

${freeTable(['opus-clip', 'vizard', 'klap', '2short'])}
<p>The meter is what to compare, more than the free allowance. Opus Clip and
Vizard count every minute of the video you import, whatever number of clips
you keep; Klap counts clips; 2short.ai counts hours analysed and takes YouTube
links only on its free plan. OpenShorts Cloud also counts source minutes, which
is why picking a stretch of a long video matters there, and the self-hosted
edition counts nothing.</p>

<h2>What it does not do</h2>
<ul>
<li><strong>It does not clip a stream while it is live.</strong> Paste the replay once the broadcast has ended.</li>
<li><strong>It does not fetch private, members-only or sign-in-only videos.</strong> Upload the file instead.</li>
<li><strong>It does not take sources under 45 seconds.</strong> There has to be enough material to cut from; a video that is already a Short is better posted as it is.</li>
<li><strong>It does not add stock B-roll or music.</strong> The clip is your footage, reframed and captioned.</li>
<li><strong>It does not guarantee a clip count.</strong> A video with three good minutes gives fewer clips than one with thirty; the AI returns what stands alone.</li>
<li><strong>The free plan is watermarked.</strong> The mark goes when you upgrade, from new and existing clips alike.</li>
</ul>

<h2>Whose videos you can clip</h2>
<p>Your own, and those you have the rights to: your channel, the channels of
clients you manage, a guest's episode with their blessing, licensed or
public-domain footage. Each job asks you to confirm that before it starts.
OpenShorts fetches what you point it at; it does not decide whether you are
allowed to repost it, and YouTube, TikTok and Instagram all remove reuploads
of other people's work when the owner complains. Channels that post famous
creators' moments do it with the creator's permission, often through a
clipping programme the creator runs, or at the risk of strikes.</p>

${faqBlock(ytClipFaq)}

${sources([
  '<a href="https://support.google.com/youtube/answer/10332730" rel="noopener">YouTube Help: Create and share clips</a> (clips are 5 to 60 seconds and play from the original video), checked 28 October 2026.',
  `Competitor free plans and entry prices: our comparison data, checked ${longDate(COMPETITORS['opus-clip'].checked)}. See <a href="/alternatives">the full comparison</a>.`,
  `Clip counts, window size and the clip-count floor are in <code>clip_selection.py</code>; the length and count limits are the ones the API enforces. Source: <a href="${SITE.repo}" rel="noopener">github.com/mutonby/openshorts</a>.`,
])}
`,
  faq: ytClipFaq,
  extraNodes: [
    howToNode('/platforms/youtube', 'How to clip a YouTube video into vertical clips with AI',
      'Turn a YouTube video into 9:16 clips with captions by pasting its link into OpenShorts.', ytClipSteps),
    appNode('/platforms/youtube', 'OpenShorts YouTube clip maker',
      'Paste a YouTube link and get vertical 9:16 clips with captions and hooks, picked by AI from the whole video.'),
  ],
})

/* ------------------------------------------------------------------------ */
/* /platforms/youtube-shorts                                                     */
/* ------------------------------------------------------------------------ */

const UGC_PRICE = (COMPARISON_ROWS.find((r) => r.feature.startsWith('AI UGC video'))?.os.match(/\$[\d.]+/) || [''])[0]

const shortsSteps = [
  ['Bring the long video', 'Paste a link to the long video (YouTube, or any of the other sites listed below) or upload the file. A podcast episode, a talk, a tutorial or a livestream replay all work; the source needs at least 45 seconds of material.'],
  ['Let the AI pick and cut the Shorts', 'The video is transcribed word by word, Gemini scores every stretch of it for how well it would stand alone, and the best ones are cut on sentence boundaries, 15 to 60 seconds by default or up to 3 minutes if you raise the maximum.'],
  ['Post them as Shorts', 'Each Short is reframed to 9:16, captioned, given a hook over the first seconds and a title. Publish it to your YouTube channel from the dashboard, schedule a week of them, or download the files.'],
]

const shortsFaq = [
  { q: 'Is this YouTube Shorts maker free?', a: `Yes on two terms. OpenShorts Cloud clips your first video whole up to ${FIRST_VIDEO_MINUTES} minutes, then ${CLOUD.freeMinutes} minutes a month, with a watermark and no credit card; paid plans start at $${CLOUD.lowPrice}/month with no watermark. The self-hosted edition is free under MIT with no watermark and no cap.` },
  { q: 'Will my Shorts have a watermark?', a: 'On the hosted free plan, yes. A paid plan removes it from new Shorts and from the ones you already made. The self-hosted edition never adds one.' },
  { q: 'How long can a YouTube Short be?', a: `Up to 3 minutes, in a vertical or square aspect ratio, according to YouTube's help pages (checked ${SPECS_CHECKED}). OpenShorts cuts 15 to 60 seconds by default; set the maximum up to 180 seconds in the advanced options when a longer Short suits the moment.` },
  { q: 'Can it post straight to my YouTube channel?', a: 'Yes. Connect the channel once in the dashboard and publish a Short with its title and description, or schedule a week of them. TikTok and Instagram can be posted to from the same screen.' },
  { q: 'Can it make Shorts from every new upload automatically?', a: 'Yes, with Autopilot on a paid plan. Every new video on your connected channel is clipped on its own, and if you turn on publishing the best Shorts are scheduled one a day at the hour you choose. It never touches your back catalogue and skips videos that are already Shorts.' },
  { q: 'How many Shorts does one video give?', a: 'For a video longer than about 12 minutes the AI is asked for 6 to 12, fewer when the material does not hold them. Shorter videos give fewer. You can set an exact number from 1 to 15.' },
  { q: 'Does it write the titles?', a: 'Yes. Every clip comes with an AI-written title and description meant for a Short, and the hook text burned over the opening seconds is generated from what is said in that clip.' },
  { q: 'What is the difference from the YouTube to Shorts converter page?', a: 'Same pipeline, different question. The converter page is about the link-first flow: paste a YouTube URL, get clips. This page is about the Shorts themselves: length, hooks, captions, titles, posting and doing it on a schedule.' },
  { q: 'What aspect ratio does a YouTube Short need?', a: `9:16 vertical, 1080x1920 pixels for a 1080p video, up to 3 minutes long; YouTube also takes square uploads as Shorts (its help pages, checked ${SPECS_CHECKED}). Every Short made here is 9:16 and at least 1080 pixels wide.` },
  { q: 'Can it turn a long video into short videos automatically?', a: 'Yes, that is the whole job: paste the long video (or upload it) and the AI picks the moments and cuts them into vertical Shorts with captions, hooks and titles. With Autopilot on a paid plan it also happens on its own for every new upload on your connected channel.' },
  { q: 'Is it an AI YouTube Shorts generator or a clipper?', a: `A clipper: the Shorts are cut from your own footage, so they sound like you and point back to your channel. If you have no long video at all, OpenShorts also has an AI Shorts tab that makes UGC-style videos with AI actors, from ${UGC_PRICE} a video.` },
  { q: 'How does it compare with Opus Clip for Shorts?', a: `Both cut Shorts from long videos and post them to YouTube. Opus Clip has a free plan of 60 credits a month with a watermark and paid plans from ${COMPETITORS['opus-clip'].entryPrice} (checked ${longDate(COMPETITORS['opus-clip'].checked)}), and a larger caption-style library. OpenShorts is open source and self-hostable, and hosted from $${CLOUD.lowPrice}/month.` },
]

const youtubeShortsMaker = () => common('/platforms/youtube-shorts', {
  title: 'YouTube Shorts Maker: AI Shorts From Long Videos, Free',
  description: `Make YouTube Shorts from any long video: AI picks the moments, adds hooks and captions, and posts to your channel. First video free up to ${FIRST_VIDEO_MINUTES} min, watermarked.`,
  h1: 'YouTube Shorts maker: AI Shorts from your long videos',
  breadcrumb: [{ name: 'Platforms', path: '/platforms' }, { name: 'YouTube Shorts maker' }],
  updated: ANSWER_REVISED,
  lede: `OpenShorts is an AI YouTube Shorts maker: paste the link of a long video and it turns it into 3 to 15 Shorts automatically, picking moments of 15 to 60 seconds, reframing each to 9:16 around the person speaking and burning in captions and a hook. ${ANSWER_PRICE}`,
  tool: { entry: 'tool-paste-link', html: widget('Paste a link to the long video', 'Long video link') },
  cta: {
    label: 'Make Shorts',
    title: 'Your next week of Shorts is already in last week\'s video',
    body: `Free plan: ${PRICE_LINE}`,
  },
  related: ['/platforms/youtube', '/how-to/youtube-shorts-aspect-ratio', '/platforms/instagram-reels', '/how-to/clip-a-youtube-video', '/platforms/tiktok', '/opus-clip-free-alternative'],
  tldr: [
    'A one-hour podcast episode is, as far as YouTube Shorts are concerned, a quarry. OpenShorts reads the whole of it, ranks every stretch for how well it would play to someone who has never heard of you, and cuts the best 6 to 12 into vertical Shorts with captions, a hook over the opening seconds and a title written for the Shorts feed.',
    'Then it can post them to your channel for you, one a day if you like, so the long video you already made keeps feeding the Shorts shelf for a week or more.',
  ],
  body: `
${stepsBlock('How to make YouTube Shorts from a long video', shortsSteps)}

<h2>What YouTube needs from a Short, and how each part is handled</h2>
<p>A Short is not just a short video. YouTube treats an upload as a Short when it
is vertical or square and runs up to 3 minutes (its help pages, checked
${SPECS_CHECKED}), and it is watched in a feed where the viewer decides in the
first second or two whether to keep watching. Each part of a Short that
OpenShorts makes answers one of those constraints:</p>
<ul>
<li><strong>Aspect ratio.</strong> The horizontal source is reframed to 9:16 rather than letterboxed, so the Short fills the phone. A single speaker is face-tracked; two people talking are stacked; a screen recording keeps the screen readable. A source that is already vertical is passed through untouched.</li>
<li><strong>Length.</strong> 15 to 60 seconds by default, because most moments that stand alone are that long. Raise the maximum to as much as 180 seconds in the advanced options when you clip a story that needs room.</li>
<li><strong>The first seconds.</strong> A hook line, written from what is said in that clip, is drawn over the opening. Its height is chosen per clip: six frames are checked for faces and the text goes where it covers the least of anyone's face, inside the area the Shorts player does not cover.</li>
<li><strong>Sound off.</strong> Captions are burned in from the word-level transcript, placed about 15% above the bottom edge so the title, channel name and buttons of the Shorts player do not sit on top of them.</li>
<li><strong>The title.</strong> Each clip gets its own title and description, written for a Short rather than copied from the long video.</li>
</ul>

<h2>YouTube Shorts aspect ratio and size</h2>
<p>The short answer: 9:16, vertical, at 1080x1920 pixels, and up to 3 minutes
long. YouTube also accepts square uploads as Shorts, but a vertical frame is the
one that fills the phone. Every Short this maker exports is 9:16 and at least
1080 pixels wide, so a 1080p source gives exactly 1080x1920, and nothing needs
resizing before upload. A horizontal source is reframed rather than
letterboxed, as described above. For the specs in full, see the guide to the
<a href="/how-to/youtube-shorts-aspect-ratio">YouTube Shorts aspect
ratio</a>.</p>

<h2>Long to short: why one long video beats a blank page</h2>
<p>Most Shorts makers start from nothing: a template, stock footage, a script
for an AI voice. A long to short workflow starts from something you already
made and already know works, so every Short is your own voice and points back
to the full video. That is the job of this AI Shorts maker: it reads the long
video, finds the passages that hold up alone, and turns each into a vertical
Short. As a YouTube short creator it suits anyone who already records long
(podcasts, tutorials, talks, streams), and turning one long video into short
videos this way is how a channel that publishes once a week ends up with
something new on the Shorts shelf every day.</p>

<h2>Picking the moments: why the whole video is read</h2>
<p>The cheap way to make Shorts is to cut the first minute, or a random one. The
useful way is to find the passages that make sense with no context, and that
takes reading all of it. OpenShorts transcribes the video with a timestamp on
every word, splits the transcript into windows of about 90 seconds and has
Gemini score every window on hook strength, emotional payload and how well it stands alone. The
score is global, so the best ten windows of a long episode win wherever they
sit, then a second pass cuts each clip so it starts on a sentence and ends on
the payoff.</p>
<p>When a video has almost no speech (gameplay, music, a silent demo), the
transcript is useless, so the video itself is sent to Gemini to watch instead,
and the Shorts come back the same way, just without captions.</p>
${demoBlock('talk', 'A 39-minute interview cut into a Short: the moment was picked from the transcript, the crop follows the speaker and the captions are timed word by word.')}

<h2>Posting to YouTube from the same place</h2>
<p>Connect your YouTube channel once in the dashboard. From then on each Short
has a publish button that sends it to the channel with its title and
description, and <strong>schedule a week</strong> spreads a batch across the
coming days at the time you choose, which is how most channels use a long
video: one Short a day pointing back at it. TikTok and Instagram sit on the
same screen if you post there too. Posting works on the free plan as well, with
a daily cap.</p>
<p>One thing YouTube lets you do that is worth doing by hand: in YouTube Studio
a Short can link to a related video, which puts a link to the full episode
under the Short. OpenShorts does not set that field for you.</p>

<h2>Shorts on autopilot</h2>
<p>If you publish long videos regularly, <a href="/auto-clip">Autopilot</a> does
the whole loop without you. On a paid plan it watches your connected channel,
clips every new upload on its own (the first 10 to 90 minutes of each, your
choice, 30 by default) and, if you turn publishing on, schedules the best of
them, up to five per video, one a day at your hour and in your time zone. It
only clips videos published after you switched it on, so the back catalogue
is never billed by surprise, and it skips uploads that are already Shorts.</p>

<h2>Not the same page as the YouTube to Shorts converter</h2>
<p>The <a href="/youtube-to-shorts-converter">YouTube to Shorts converter</a>
covers the one-off job: you have a YouTube link and want it turned into
vertical clips without downloading anything. This page is about running a
Shorts shelf: deciding the length, the hook, the captions and the titles, and
posting on a schedule. Same pipeline underneath; use whichever question you
came with. And if all you want is a single moment trimmed by hand, <a
href="/how-to/clip-a-youtube-video">how to clip a YouTube video</a> compares the
free manual routes with the AI one.</p>

<h2>Works with links from other sites too</h2>
<p>The long video does not have to be on YouTube. The box above takes public
video links from ${SITES_LIST}, so a webinar on Vimeo, a Facebook Live replay,
a video post on X or Reddit, or a Dailymotion or Streamable upload can all
become YouTube Shorts. Private and sign-in-only videos cannot be fetched; for
those, and for files on your disk, upload the file in the app.</p>

${freeTable(['opus-clip', '2short', 'submagic', 'sendshort'])}
<p>Two things the free allowances hide. Submagic and SendShort have no ongoing
free plan, and SendShort only clips long videos from its $29/month tier up.
2short.ai has the one free plan here with no watermark, and takes YouTube links
only on it. OpenShorts' free plan is watermarked and takes links and uploads.</p>

<h2>What it does not do</h2>
<ul>
<li><strong>It does not invent footage.</strong> A Short is cut from your video. For videos made from nothing, see the AI Shorts tab and its AI actors.</li>
<li><strong>It does not pick music or trending sounds.</strong> The audio is the source's own.</li>
<li><strong>It does not set the related-video link</strong> in YouTube Studio, or your thumbnail frame for Shorts.</li>
<li><strong>Autopilot is a paid feature</strong> and works from a connected YouTube channel, not from other sites.</li>
<li><strong>It does not promise views.</strong> It finds the moments most likely to stand alone; the feed decides the rest.</li>
<li><strong>The free plan is watermarked.</strong> Upgrading removes the mark from existing Shorts too.</li>
</ul>

<h2>Whose videos can become Shorts</h2>
<p>Yours, and videos you have the rights to. Shorts cut from someone else's
upload without permission get claimed or removed like any reupload. Every job
asks you to confirm the rights before it runs.</p>

${faqBlock(shortsFaq)}

${sources([
  '<a href="https://support.google.com/youtube/answer/12779649" rel="noopener">YouTube Help: Upload YouTube Shorts</a> (up to 3 minutes, square or vertical), checked 28 October 2026.',
  `Competitor plans: our comparison data, checked ${longDate(COMPETITORS.submagic.checked)}. See <a href="/alternatives">all alternatives</a>.`,
  `Hook placement, caption margin and Autopilot rules are in <code>hook_placement.py</code>, <code>subtitles.py</code> and <code>cloud/autopilot.py</code> at <a href="${SITE.repo}" rel="noopener">github.com/mutonby/openshorts</a>.`,
])}
`,
  faq: shortsFaq,
  extraNodes: [
    howToNode('/platforms/youtube-shorts', 'How to make YouTube Shorts from a long video with AI',
      'Cut a long video into vertical YouTube Shorts with captions, hooks and titles, then post them to your channel.', shortsSteps),
    appNode('/platforms/youtube-shorts', 'OpenShorts YouTube Shorts maker',
      'AI YouTube Shorts maker: cuts long videos into vertical Shorts with captions, hooks and titles and posts them to YouTube.'),
  ],
})

/* ------------------------------------------------------------------------ */
/* /platforms/tiktok                                                        */
/* ------------------------------------------------------------------------ */

const tiktokSteps = [
  ['Paste the long video', 'A YouTube link is the usual start, but any public video link from the sites listed on this page works, a long TikTok included. Or upload the recording.'],
  ['AI cuts the TikToks', 'Gemini reads the word-level transcript, ranks every stretch for how well it would hold a stranger, and cuts the best ones on sentence boundaries, 15 to 60 seconds each by default.'],
  ['Send them to TikTok', 'Each clip comes back 9:16 with captions and a hook. Send it to your TikTok account from the dashboard, where it lands as a draft for you to finish with a sound and hashtags, or download the MP4.'],
]

const tiktokFaq = [
  { q: 'Is this TikTok clip maker free?', a: `The hosted free plan clips your first video whole up to ${FIRST_VIDEO_MINUTES} minutes, then ${CLOUD.freeMinutes} minutes a month, with a watermark and no credit card. Paid plans start at $${CLOUD.lowPrice}/month without the watermark. Self-hosted OpenShorts is free under MIT with no watermark and no cap.` },
  { q: 'Is there a watermark on the TikToks?', a: 'On the hosted free plan, yes; paying removes it from new and existing clips.' },
  { q: 'Why does my clip show up as a draft on TikTok?', a: 'Because TikTok\'s API delivers videos sent from other apps to your TikTok inbox as drafts. You get a notification in the app, then add the caption, a sound and hashtags there and post it. The title and description written for the clip are not carried onto a TikTok draft, so you paste or write them in the app.' },
  { q: 'How do I turn a YouTube video into TikToks?', a: 'Paste the YouTube link above. The video is fetched, transcribed and cut into vertical clips with captions and hooks, which you can send to TikTok or download. You do not need to download the YouTube video first.' },
  { q: 'Can I clip a TikTok video or a TikTok LIVE?', a: 'A public TikTok video link works as a source if it has at least 45 seconds of material, which mostly means long uploads. A LIVE cannot be clipped while it is on air; if you saved the replay, upload that file.' },
  { q: 'How long should a TikTok clip be?', a: 'The default here is 15 to 60 seconds, because that is how long most moments that stand alone run. TikTok accepts far longer uploads; raise the maximum up to 180 seconds in the advanced options for a story that needs it.' },
  { q: 'How many TikToks does one video give?', a: 'A video longer than about 12 minutes is asked for 6 to 12 clips, and fewer come back when fewer moments stand alone. You can ask for an exact number from 1 to 15.' },
  { q: 'Can it make TikTok clips from a long video without me picking the moments?', a: 'Yes. The AI reads the whole transcript and picks the moments itself; you can override any of it afterwards in the clip editor, or mark one stretch with "clip only part of it" before the job starts so it only looks there.' },
  { q: 'Is it an AI TikTok generator?', a: `It is a clipper: every TikTok is cut from your own footage. For videos made from a script with no footage at all, the AI Shorts tab makes UGC-style videos with AI actors, from ${UGC_PRICE} a video.` },
  { q: 'Can I schedule TikToks?', a: 'Yes. Schedule a week spreads a batch of clips over the coming days. On TikTok each one still arrives as a draft at its scheduled time, so you finish it in the app.' },
  { q: 'How does it compare with Klap or Opus Clip for TikTok?', a: `All three cut long videos into TikToks. Klap is the fastest path from a link to a first clip but gives one free video and bills clips (from ${COMPETITORS.klap.entryPrice}); Opus Clip has a watermarked free plan of 60 credits a month and paid plans from ${COMPETITORS['opus-clip'].entryPrice}. Both checked ${longDate(COMPETITORS.klap.checked)}. OpenShorts is open source, self-hostable, and hosted from $${CLOUD.lowPrice}/month billed monthly.` },
]

const tiktokClipMaker = () => common('/platforms/tiktok', {
  title: 'TikTok Clip Maker: Long Video to TikToks With AI',
  description: `Turn a long video into TikTok clips with AI: moments picked, reframed to 9:16, captioned and sent to TikTok. First video free up to ${FIRST_VIDEO_MINUTES} min, then watermarked.`,
  h1: 'TikTok clip maker: turn long videos into TikToks',
  breadcrumb: [{ name: 'Platforms', path: '/platforms' }, { name: 'TikTok clip maker' }],
  tool: { entry: 'tool-paste-link', html: widget('Paste a YouTube or other video link', 'Video link to turn into TikToks') },
  cta: {
    label: 'Make TikToks',
    title: 'From one long video to a week of TikTok drafts',
    body: `Free plan: ${PRICE_LINE}`,
  },
  related: ['/platforms/instagram-reels', '/platforms/youtube-shorts', '/platforms/twitch', '/alternatives/klap'],
  tldr: [
    'Take a 30-minute YouTube video that did well. Pasted here, it comes back as a stack of vertical clips, 6 to 12 of them for a source that long, each 15 to 60 seconds, captioned for the muted autoplay most TikToks start in and opened by a hook line written from what is said in that clip.',
    'Send them to your TikTok account from the dashboard and they wait in your TikTok inbox as drafts, where you add the sound, the hashtags and the caption the way TikTok rewards. Or download the files and post them however you like.',
  ],
  body: `
${stepsBlock('How to turn a long video into TikTok clips', tiktokSteps)}

<h2>YouTube to TikTok without the download step</h2>
<p>The usual way to get a YouTube video onto TikTok is a chain of tools: a
downloader, an editor to find the moment and crop it to vertical, a caption
app, and then the upload. Each step loses quality or time. Here the YouTube
link is the only input: OpenShorts fetches the source at the best quality it
offers, does the cutting, reframing and captioning in one pass, and hands you
finished vertical files, at least 1080 pixels wide.</p>
<p>That also works for the videos you never posted to YouTube. A Zoom or
Riverside recording, a webinar, a podcast episode recorded locally: upload
the file instead of pasting a link and the rest is the same. Long video to
TikTok is one job here whatever the source, not a chain of four apps.</p>

<h2>What TikTok clips need that a YouTube video does not have</h2>
<p>A horizontal video dropped onto TikTok as it is plays as a thin strip with
black bars, and people scroll past it. Three things turn a moment from a long
video into something that works in the For You feed:</p>
<h3>Vertical framing that keeps the person</h3>
<p>Cropping 16:9 to 9:16 throws away two thirds of the width. A fixed centre
crop loses anyone who is not sitting in the middle, so each clip is reframed
by what is in it: the speaker is tracked with face detection and a stabiliser
that holds still instead of chasing every head movement, two people in the
same shot are stacked one above the other, and a screen share keeps the
screen readable with the presenter below. See
<a href="/auto-reframe">auto reframe</a> for how each layout is chosen.</p>
<h3>Captions you can read on mute</h3>
<p>Captions are burned in from a transcript with a timestamp on every word, so
they appear as the word is spoken. They sit about 15% above the bottom edge,
which keeps them clear of TikTok's own caption, username and music line at
the bottom of the screen. Styles can be changed after the job; the dubbing
option (with your own ElevenLabs key) can also translate a clip into more than
30 languages and re-caption it in the new language.</p>
<h3>A reason to stay in the first second</h3>
<p>A hook line is generated for each clip from what is said in it and drawn
over the opening seconds. Its height is chosen per clip so it covers the least
of anyone's face, inside the part of the screen the app does not cover.</p>

<h2>Why clips arrive on TikTok as drafts</h2>
<p>This surprises people, so it is worth explaining before you connect your
account. When OpenShorts sends a clip to TikTok it goes through TikTok's
upload API, and that API delivers the video to your TikTok inbox as a draft,
not as a live post. You get a notification in the TikTok app, open the draft,
and post it from there. TikTok's draft upload carries the video only, so the
title and description written for the clip do not come with it: you write the
caption in the app.</p>
<p>That is less of a limitation than it sounds. Finishing the post inside
TikTok is where you pick a trending sound, add effects, tag people and write
hashtags, which a post made outside the app cannot do. Instagram and YouTube,
by contrast, are published directly. Scheduling a week of clips works for
TikTok too; each draft lands at its scheduled time.</p>

<h2>Using TikTok videos as the source</h2>
<p>The box above also takes TikTok links, which is useful the other way round:
a long TikTok upload with a great 40 seconds in the middle can be cut down for
Reels or Shorts. Two limits apply. The source needs at least 45 seconds of
material, so most TikToks are already as short as they should be. And a TikTok
LIVE cannot be clipped while it is on air; save the replay and upload it.</p>

${freeTable(['klap', 'opus-clip', 'vidyo-ai', 'submagic'])}
<p>For TikTok, what you are billed for matters more than the headline price.
Klap bills the clips it generates and says its algorithm leans on speech, so a
gameplay or music-heavy source gives it little to work with. Opus Clip and
Quso bill the minutes you import. Submagic is captions-first and caps the
length of each video by plan. OpenShorts Cloud bills source minutes too, and
lets you clip only part of a long video to spend fewer of them.</p>

<h2>What it does not do</h2>
<ul>
<li><strong>It does not post TikToks live.</strong> They arrive as drafts, by TikTok's design, and you finish them in the app.</li>
<li><strong>It does not add trending sounds or TikTok effects.</strong> Those are added in TikTok, on the draft.</li>
<li><strong>It does not clip a TikTok LIVE in real time</strong>, and it cannot fetch private TikToks.</li>
<li><strong>It does not take sources under 45 seconds.</strong></li>
<li><strong>It does not write hashtags into the draft</strong>, since TikTok drops the description on drafts.</li>
<li><strong>Free clips carry a watermark.</strong> Paid plans remove it, from earlier clips as well.</li>
</ul>

<h2>Whose videos can you turn into TikToks</h2>
<p>Your own and those you have permission for. Reposting a creator's video to
TikTok without permission is a copyright claim waiting to happen, and TikTok
removes reuploads when the owner reports them. Every OpenShorts job asks you to
confirm you have the rights before it runs.</p>

${faqBlock(tiktokFaq)}

${sources([
  'TikTok draft behaviour: TikTok for Developers, Content Posting API (videos uploaded to the inbox arrive as drafts to be posted from the app), checked 28 October 2026.',
  `Competitor plans: our comparison data, checked ${longDate(COMPETITORS.klap.checked)}. See <a href="/alternatives/klap">OpenShorts vs Klap</a>.`,
  `Caption margin and hook placement: <code>subtitles.py</code> and <code>hook_placement.py</code> at <a href="${SITE.repo}" rel="noopener">github.com/mutonby/openshorts</a>.`,
])}
`,
  faq: tiktokFaq,
  extraNodes: [
    howToNode('/platforms/tiktok', 'How to turn a long video into TikTok clips with AI',
      'Cut a long video, from a YouTube link or an upload, into vertical TikTok clips with captions and hooks.', tiktokSteps),
    appNode('/platforms/tiktok', 'OpenShorts TikTok clip maker',
      'AI TikTok clip maker: turns long videos into vertical TikToks with captions and hooks and sends them to TikTok as drafts.'),
  ],
})

/* ------------------------------------------------------------------------ */
/* /platforms/instagram-reels                                                        */
/* ------------------------------------------------------------------------ */

const reelSteps = [
  ['Add the long video', 'Paste a public link (YouTube, Vimeo, a Facebook video or Live replay, an Instagram post, and the other sites listed below) or upload the recording: a podcast, a webinar, a talk, a class.'],
  ['The AI picks the Reels', 'The recording is transcribed word by word and Gemini ranks every stretch of it for how well it would play on its own. The strongest are cut on sentence boundaries, 15 to 60 seconds each unless you set other limits.'],
  ['Publish to Instagram or download', 'Each Reel comes back 9:16 with captions, a hook and a description written for Instagram. Publish it to your Instagram account as a Reel, schedule a week of them, or download the MP4 for Facebook.'],
]

const reelFaq = [
  { q: 'Is this a free reels maker?', a: `Yes, with an honest asterisk: the hosted free plan clips your first video whole up to ${FIRST_VIDEO_MINUTES} minutes, then ${CLOUD.freeMinutes} minutes a month, and those Reels carry a watermark. No credit card is needed. Paid plans start at $${CLOUD.lowPrice}/month and remove the watermark; the self-hosted edition is free under MIT with no watermark.` },
  { q: 'Does it post Reels to Instagram directly?', a: 'Yes. Connect your Instagram account in the dashboard and each clip can be published as a Reel with its own description, or scheduled across a week.' },
  { q: 'Is there a reels maker app to install?', a: 'No app store app: OpenShorts Cloud runs in the web browser and the self-hosted edition runs on your own machine with Docker. Either way the Reels you make are MP4 files you can post from the Instagram app, or publish to Instagram straight from the dashboard.' },
  { q: 'What size is an Instagram Reel made here?', a: 'Vertical 9:16, at least 1080 pixels wide: 1080x1920 from a 1080p source, with captions and the hook burned in. The guide to Instagram Reel dimensions covers the rest of the specs.' },
  { q: 'Does it work for Facebook Reels?', a: 'The clips are the right shape for Facebook Reels, but OpenShorts does not publish to Facebook. Download the MP4 and post it from the Facebook app or Meta Business Suite.' },
  { q: 'How long can an Instagram Reel be?', a: 'Instagram raised the Reels limit to 3 minutes in January 2025. Clips here are 15 to 60 seconds by default; set the maximum up to 180 seconds in the advanced options for longer Reels.' },
  { q: 'Will the captions be covered by the Instagram buttons?', a: 'They are placed about 15% above the bottom edge, which keeps them clear of the caption text and account name Instagram draws at the bottom of a Reel.' },
  { q: 'How many Reels does one video give?', a: 'A video longer than about 12 minutes is asked for 6 to 12 Reels, fewer when fewer moments stand alone. You can ask for an exact number from 1 to 15.' },
  { q: 'Can I make Reels from an Instagram Live or a Facebook Live?', a: 'From the replay, once it has ended: paste the link of a public Facebook Live replay, or upload the file of an Instagram Live you downloaded. Nothing is clipped while a broadcast is on air.' },
  { q: 'Can the Reels be in another language?', a: 'Yes, with your own ElevenLabs API key: dubbing translates a clip into more than 30 languages, keeps the speaker\'s voice and re-captions it in the new language.' },
  { q: 'Is the reel maker AI picking the moments, or me?', a: 'The AI picks them, from the transcript. You can then trim, extend or reorder any clip in the clip editor, move the crop scene by scene, restyle the captions and rewrite the hook.' },
  { q: 'How does it compare with Opus Clip, Vizard or Quso?', a: `All of them cut long videos into vertical clips. Their free plans, checked ${longDate(COMPETITORS.vizard.checked)}: Opus Clip and Vizard 60 credits a month with a watermark, Quso 75 credits at 720p. Quso adds a content planner and analytics we do not have; Vizard has the stronger hand-editing timeline. OpenShorts is open source, self-hostable and hosted from $${CLOUD.lowPrice}/month billed monthly.` },
]

const aiReelGenerator = () => common('/platforms/instagram-reels', {
  // "reels maker" and "ai reels maker" together outweigh "ai reel generator"
  // (kept in the steps heading, the FAQ and the app node); 56 characters.
  title: 'AI Reels Maker: Turn Long Videos Into Reels | OpenShorts',
  description: `AI reels maker and reel generator: a long video in, captioned 9:16 Reels out for Instagram and Facebook. First video free up to ${FIRST_VIDEO_MINUTES} min, watermarked.`,
  h1: 'AI reels maker: turn long videos into Instagram and Facebook Reels',
  breadcrumb: [{ name: 'Platforms', path: '/platforms' }, { name: 'AI reels maker' }],
  tool: { entry: 'tool-paste-link', html: widget('Paste a video link (YouTube, Facebook, Instagram...)', 'Video link to turn into Reels') },
  cta: {
    label: 'Make Reels',
    title: 'Reels cut from what you already recorded',
    body: `Free plan: ${PRICE_LINE}`,
  },
  related: ['/how-to/instagram-reel-dimensions', '/how-to/make-a-reel-on-facebook', '/platforms/tiktok', '/platforms/youtube-shorts', '/use-cases/sermons', '/alternatives/vidyo-ai'],
  tldr: [
    'A one-hour webinar, recorded on Zoom and sitting in a folder, holds a month of Reels for most businesses. Upload it here and the AI reads all of it, finds the 6 to 12 passages that make sense on their own, and returns each one as a vertical Reel with captions, a hook over the first seconds and a description written for Instagram.',
    'Publish them to Instagram as Reels from the same screen, one a day if you schedule a week, or download them for Facebook. Nothing is generated from stock footage: every Reel is you, or your guest, saying the thing.',
  ],
  body: `
${stepsBlock('How the AI reel generator works', reelSteps)}

<h2>Reels from recordings, not from templates</h2>
<p>Search for a reel maker and most of what comes back is a template editor:
pick a layout, drop in photos and a song, and it makes a slideshow. That is a
different job. An AI reel generator in the sense this page means takes a
recording you already have (an interview, a class, a podcast, a product demo,
a sermon) and turns its best minutes into Reels. The material is real, the
voice is yours and the work saved is the part that takes longest by hand:
watching the whole thing to find the moments.</p>
<p>That search is what the AI does here. The recording is transcribed with a
timestamp on every word, the transcript is split into windows of about 90
seconds, and Gemini scores every window on hook strength, emotional payload
and how well it stands alone. The best windows go to a second pass that cuts
each Reel on sentence boundaries. For anything over about 12 minutes it is
asked for 6 to 12 Reels, and returns fewer rather than padding.</p>

<h2>What makes a clip work as a Reel</h2>
<h3>The full screen, with the person in it</h3>
<p>Reels are 9:16 and fill the phone; every Reel here is at least 1080 pixels
wide, 1080x1920 from a 1080p source (the
<a href="/how-to/instagram-reel-dimensions">Instagram Reel dimensions</a> guide
has the full specs). A horizontal recording is reframed rather
than shrunk: one speaker is face-tracked with a stabiliser that holds the
frame steady, two people on camera together are stacked, and a screen share
keeps the slide or the demo readable with the presenter beside it. If you
recorded vertical in the first place, the video is left exactly as it is.</p>
<h3>Captions for a muted feed</h3>
<p>Reels autoplay in a feed people scroll with the sound off. Captions are
burned in from the word-level transcript, so each word appears as it is said,
and placed about 15% above the bottom edge, clear of the caption and account
name Instagram draws there. Change the style afterwards if your brand has one.</p>
<h3>An opening that earns the second second</h3>
<p>Each Reel gets a hook line written from what is said in it, drawn over the
opening seconds at the height that covers the least of anyone's face.</p>

<h2>Instagram Reels maker: posting directly</h2>
<p>Connect your Instagram account once in the dashboard. Each clip then has a
publish button that posts it to Instagram as a Reel, with the description
written for Instagram (edit it first if you like), and
<strong>schedule a week</strong> spreads a batch over the following days at
the hour you pick. Instagram raised the Reels length limit to 3 minutes in
January 2025, so you can raise the clip maximum to 180 seconds when a longer
Reel suits the moment; the default of 15 to 60 seconds is where most moments
that stand alone end.</p>
<p>The same screen posts to TikTok and YouTube Shorts, so one recording can
fill all three. Posting is available on the free plan as well, with a daily
cap.</p>

<h2>Facebook Reels maker: same file, posted by you</h2>
<p>A Reel made here is the right shape for Facebook: vertical, captioned, with
the hook burned in. What OpenShorts does not do is publish to Facebook, so
download the MP4 and post it from the Facebook app or Meta Business Suite.
Going the other way works: if your source is on Facebook (a public video, or
the replay of a public Facebook Live), paste its link above and it is fetched
like a YouTube link. Facebook Live is where many churches, local businesses and
community groups already stream, which makes it one of the most common sources
for Reels. For posting by hand, the guide on <a
href="/how-to/make-a-reel-on-facebook">how to make a reel on Facebook</a> walks
through the Facebook side.</p>

<h2>Reels in other languages</h2>
<p>An account with followers in several countries can post the same Reel more
than once. With your own ElevenLabs API key, the dubbing option translates a
clip into more than 30 languages while keeping the speaker's voice, and the
dubbed audio is transcribed again so the captions match the new language. It
works on Cloud and self-hosted alike.</p>

<h2>Works with links from</h2>
<p>${SITES_LIST}, as public links, plus file uploads on every plan. Instagram
posts and Reels can be used as a source too, though most are already short;
the source needs at least 45 seconds of material.</p>

${freeTable(['opus-clip', 'vizard', 'vidyo-ai', 'sendshort'])}
<p>Quso (the renamed Vidyo.ai) adds a content planner and analytics around the
clipper, which OpenShorts does not have. SendShort has no free plan and starts
clipping long videos at $29/month. If a Reel without a watermark on day one
matters more than anything else, the self-hosted edition is the free route that
gives you that.</p>

<h2>What it does not do</h2>
<ul>
<li><strong>It does not publish to Facebook.</strong> Download the Reel and post it there yourself.</li>
<li><strong>It does not make Reels from photos or templates.</strong> It cuts them from a video recording.</li>
<li><strong>It does not add licensed music.</strong> Add a sound from Instagram's library when you post, if you want one.</li>
<li><strong>It does not fetch private or sign-in-only posts</strong>, and it does not clip a Live while it is on air.</li>
<li><strong>It does not pick the cover frame</strong> Instagram shows on your grid.</li>
<li><strong>Free Reels carry a watermark.</strong> Upgrading removes it, also from Reels you already made.</li>
</ul>

<h2>Whose recordings you can turn into Reels</h2>
<p>Yours, your clients' with their permission, and anything you have licensed.
A Reel cut from someone else's video is a reupload, and Instagram and Facebook
take those down when the owner reports them. Each job asks you to confirm the
rights first.</p>

${faqBlock(reelFaq)}

${sources([
  'Instagram Reels up to 3 minutes: announced by Instagram in January 2025, reported by <a href="https://www.socialmediatoday.com/news/instagram-officially-expands-reels-length-3-minutes/737766/" rel="noopener">Social Media Today</a>, checked 28 October 2026.',
  `Competitor plans: our comparison data, checked ${longDate(COMPETITORS['vidyo-ai'].checked)}. See <a href="/alternatives/vidyo-ai">Vidyo.ai (Quso)</a>.`,
  `Instagram posting as Reels, caption margin and dubbing: <code>app.py</code>, <code>subtitles.py</code> and <code>translate.py</code> at <a href="${SITE.repo}" rel="noopener">github.com/mutonby/openshorts</a>.`,
])}
`,
  faq: reelFaq,
  extraNodes: [
    howToNode('/platforms/instagram-reels', 'How to turn a long video into Instagram Reels with AI',
      'Cut a long recording into vertical Reels with captions and hooks, then publish them to Instagram.', reelSteps),
    appNode('/platforms/instagram-reels', 'OpenShorts AI reels maker',
      'AI reels maker and Instagram reel generator: turns long videos into captioned 9:16 Reels for Instagram and Facebook, and publishes them to Instagram.'),
  ],
})

/* ------------------------------------------------------------------------ */
/* /platforms/twitch                                                        */
/* ------------------------------------------------------------------------ */

const PLAN_MIN = (name) => PRICE_MODELS.openshorts.plans.find((p) => p.name === name)

const twitchSteps = [
  ['Paste the VOD link', 'Open the past broadcast on Twitch and copy its address (twitch.tv/videos/...), or the link of a Twitch clip of 45 seconds or more. A channel page is the live channel, not a recording, so the box asks for the VOD instead.'],
  ['The AI finds the moments in the stream', 'The VOD is transcribed with a timestamp on every word and scanned for scene cuts, and Gemini ranks every stretch of the talk for how well it would play to someone who was not watching live. Streams with almost no speech go to a vision pass instead.'],
  ['Get vertical clips with your facecam', 'Each clip is reframed to 9:16: gameplay with a webcam box becomes gameplay on top and your enlarged facecam below, just-chatting segments follow your face. Captions and a hook are burned in; post to TikTok, YouTube Shorts and Instagram or download.'],
]

const twitchFaq = [
  { q: 'Is this Twitch clip maker free?', a: `The hosted free plan clips your first video whole up to ${FIRST_VIDEO_MINUTES} minutes, then ${CLOUD.freeMinutes} minutes a month, watermarked, no credit card. A multi-hour VOD is longer than that, so on the free plan pick the stretch you want. Paid plans start at $${CLOUD.lowPrice}/month. Self-hosted OpenShorts is free under MIT with no meter, which is the edition that suits streamers with hours of VODs.` },
  { q: 'Does it work with a whole Twitch VOD?', a: 'Yes. Paste the twitch.tv/videos/ link of the past broadcast. Moment scoring reads the transcript rather than every frame, so a long VOD is analysed the same way as a short one. One job returns the best 6 to 12 moments of whatever it is given, so for a long stream it usually pays to clip it in parts.' },
  { q: 'Can it clip my stream while I am live?', a: 'No. It works from recordings: paste the VOD once the broadcast has ended, before Twitch removes the past broadcast.' },
  { q: 'Will my facecam stay in the clip?', a: 'Yes, with the vertical layout on auto. A webcam box composited into a corner is detected by its geometry (small, off centre, still) and the clip renders with the gameplay full width on top and the facecam cropped out and enlarged below.' },
  { q: 'What about gameplay with no commentary?', a: 'When the transcript comes back with under 8 words, or under 5 words a minute, OpenShorts switches to a vision pass where Gemini watches the footage and picks the moments. Those clips have no captions. A long silent VOD is watched in stretches of up to 40 minutes and the best picks of all of them are kept, up to a little over five hours per job.' },
  { q: 'Can I turn Twitch clips into TikToks?', a: 'A Twitch clip of 45 seconds or more can be pasted as a source and comes back reframed to 9:16 with captions. Shorter clips are under the minimum; for those a static crop in the free video to 9:16 converter is the quicker route.' },
  { q: 'How many clips do I get from a stream?', a: 'Each job returns up to about a dozen: for any source longer than 12 minutes the AI is asked for 6 to 12. A four-hour stream clipped in four one-hour parts gives four times as many candidates as the same stream clipped in one job.' },
  { q: 'Does it post to TikTok and YouTube Shorts?', a: 'Yes, from the dashboard: YouTube Shorts and Instagram Reels are published directly, and TikTok receives the clip as a draft in your TikTok inbox to finish in the app.' },
  { q: 'Can it clip my new Twitch streams automatically?', a: 'Yes, on the paid plans. Connect Twitch in the Autopilot tab and it checks for new past broadcasts about once an hour, waits until the stream has ended and clips it, one automatic video a day. Instead of the first minutes it clips the stretch your viewers clipped most, up to the per-video cap you set.' },
  { q: 'Does it use the clips my viewers made?', a: 'Yes, as a signal. For a Twitch VOD link it reads the clips viewers made from that broadcast, with their time and views, and makes sure the stretches clipped most reach the final pick. The rest is chosen from the transcript, so a moment nobody clipped can still win.' },
  { q: 'Can it manage the clips in my Twitch account?', a: 'No. The Twitch connection Autopilot uses is read-only and only lists your past broadcasts, so it cannot edit or delete the clips under your channel\'s Clips page and never posts to Twitch. It makes new vertical clips from a VOD, or from a single Twitch clip of 45 seconds or more.' },
  { q: 'Is there an editor for the Twitch clips it makes?', a: 'Yes. Each clip opens in a clip editor where you can trim, extend, split or reorder its stretches on word boundaries, move the crop scene by scene, restyle the captions and rewrite the hook. Edits on clips do not use minutes.' },
  { q: 'Can I clip another streamer\'s VOD?', a: 'Only with their permission, for example as an editor or as part of a clipping programme they run. The VOD belongs to whoever streamed it, and the platforms remove reuploads when the streamer reports them.' },
  { q: 'How does it compare with Opus Clip or Klap for Twitch?', a: `Both bill in ways that hurt on long streams: Opus Clip one credit per imported minute, from ${COMPETITORS['opus-clip'].entryPrice}, and Klap per clip, from ${COMPETITORS.klap.entryPrice}, with a speech-reliant picker (checked ${longDate(COMPETITORS.klap.checked)}). OpenShorts Cloud also bills source minutes, but lets you clip just a stretch, and the self-hosted edition has no meter at all.` },
]

const twitchClipMaker = () => common('/platforms/twitch', {
  // "twitch clips" (and "clips for twitch") is the bigger search; 56 characters.
  title: 'Twitch Clip Maker: AI Twitch Clips From Streams and VODs',
  description: `Twitch clip maker and AI stream clipper: paste a VOD, get 9:16 clips with your facecam kept and captions. First video free up to ${FIRST_VIDEO_MINUTES} min, then watermarked.`,
  h1: 'Twitch clip maker: AI clips from your streams and VODs',
  breadcrumb: [{ name: 'Platforms', path: '/platforms' }, { name: 'Twitch clip maker' }],
  updated: ANSWER_REVISED,
  lede: `OpenShorts is an AI tool that clips Twitch streams into TikToks, Shorts and Reels: paste a VOD link and it picks the moments that stand alone, cuts them into 15 to 60 second vertical clips with your facecam kept and captions burned in. ${ANSWER_PRICE} Prices of the other stream clippers are in the table further down.`,
  tool: { entry: 'tool-paste-link', html: widget('https://www.twitch.tv/videos/...', 'Twitch VOD link') },
  cta: {
    label: 'Clip a stream',
    title: 'Last night\'s stream, as a week of vertical clips',
    body: `Free plan: ${PRICE_LINE}`,
  },
  related: ['/platforms/kick', '/platforms/tiktok', '/auto-reframe', '/alternatives/klap'],
  tldr: [
    'A three-hour Twitch stream is mostly waiting, travelling and loading screens, with a handful of moments worth posting. Paste the VOD link and OpenShorts reads everything you said across those three hours, picks the passages that stand alone, and cuts each one into a vertical clip: gameplay across the top, your facecam enlarged underneath, captions burned in.',
    'One job returns the best 6 to 12 moments of what it is given. For a long stream, clip it in parts (one job per hour, or just the segment you know was good) and you get more candidates and spend fewer minutes.',
  ],
  body: `
${stepsBlock('How the Twitch stream clipper works', twitchSteps)}

<h2>Twitch clips from the Clip button versus an AI stream clipper</h2>
<p>Twitch already lets you and your viewers clip. Those clips are short,
horizontal, made by someone who was watching at the right second, and they live
on Twitch. They are great for the moment a chat explodes, and useless for the
ninety minutes nobody happened to clip. A stream clipper starts from the other
end: the whole VOD, read from start to finish by a model that has no favourite
moment, cut into vertical clips sized for TikTok, Shorts and Reels. Making
clips for Twitch channels that way is the job editors and clippers do by hand,
scrubbing through hours of VOD; here the model does the scrubbing.</p>
<p>The two are not exclusive. For a Twitch VOD link, OpenShorts asks Twitch
for the clips viewers made from that broadcast, with when they were made and
how many views each got, and makes sure the stretches clipped most reach the
final pick (up to a third of it). The rest is chosen from the transcript, so the
ninety minutes nobody clipped still get read.</p>
<p>With <a href="/auto-clip">Autopilot</a> on a paid plan you do not paste
anything: connect Twitch once (read-only, nothing is ever posted to Twitch) and
each new past broadcast is clipped after the stream ends, one automatic video a
day, from the stretch your viewers clipped most rather than the "starting soon"
screen.</p>
<p>That is also why the box asks for the VOD link rather than your channel. A
<code>twitch.tv/yourname</code> link is the live channel; the recording is at
<code>twitch.tv/videos/</code> followed by a number, which you find under
Videos on your channel page. A Twitch clip link works too, as long as the clip
is at least 45 seconds long.</p>

<h2>Long VODs: how a multi-hour stream is handled</h2>
<p>Moment picking reads the transcript, not the frames, so a four-hour stream
costs about the same to analyse as a forty-minute one, and selection does not
degrade with length. The transcript is split into windows of about 90 seconds,
every window is scored, and the best ten go on to be cut. That cap is the thing
to plan around: whether the job is one hour or six, it returns its best 6 to 12
moments.</p>
<p>So for a long stream, clip it in parts. <strong>Clip only part of it</strong>
lets you set a start and an end, and only that stretch is downloaded, analysed
and billed. Run the first hour, the boss fight and the end-of-stream chat as
three jobs and you get three sets of candidates. On OpenShorts Cloud the meter
is minutes of source, so a full four-hour VOD is 240 minutes: more than the
${PLAN_MIN('Starter').monthly.minutes} minutes of the $${PLAN_MIN('Starter').monthly.price} plan and inside the
${PLAN_MIN('Creator').monthly.minutes} of the $${PLAN_MIN('Creator').monthly.price} one. Self-hosted, there is no meter at all,
which is the edition that makes sense for a channel with a VOD every day.</p>
<p>One practical point about Twitch itself: past broadcasts are not kept
forever. How long Twitch stores them depends on your account type, and the
"store past broadcasts" setting has to be on for a VOD to exist at all. Clip
a stream in the days after it, or export it, before it lapses.</p>

<h2>Facecam, gameplay and just chatting: the layouts</h2>
<p>A stream is not a talking head, and a centre crop of one keeps the middle of
the map and loses you. With the vertical layout on <strong>auto</strong> (the
default in advanced options), each clip is laid out by what is in it:</p>
<ul>
<li><strong>Gameplay with a webcam box.</strong> The OBS layout most streamers use, gameplay full screen and a camera in a corner, is detected geometrically: a subject that is small, off centre and still between samples. The clip renders as INSET: the gameplay across the full width on top, the webcam box cropped out and enlarged to fill the bottom. Both stay readable on a phone.</li>
<li><strong>Full-frame action.</strong> When the meaningful content spans the whole width (a chase, a team fight, a map), the clip is zoomed around the motion, keeping roughly three quarters of the width instead of cropping half of it away.</li>
<li><strong>Just chatting and IRL.</strong> A face-filled frame is face-tracked, with a stabiliser that holds the frame still inside a safe zone instead of following every lean towards the mic.</li>
<li><strong>Two people on camera.</strong> A co-stream or a guest in the same shot is stacked, one person in each half.</li>
</ul>
<p>If the tracker gets a scene wrong, the reframe editor lets you draw the crop
yourself for that scene, or split it into two stacked regions, and the rest of
the clip stays automatic. The rest of the Twitch clip editor works the same way:
trim or extend a clip on word boundaries, restyle the captions, rewrite the hook. The <a href="/gta-5-clips">GTA 5 clips</a> page goes
deeper into the inset layout and how it was measured, and the
<a href="/gta-6-clips">GTA 6 clips</a> page covers console captures with no
facecam and no commentary. For World of Warcraft raids, where the interface
sits at the edges of the screen, see <a href="/wow-forever-clips">WoW Forever
clips</a>.</p>

<h2>Silent gameplay and music-only streams</h2>
<p>The default picker reads what you say, which is exactly right for
commentary. A grind with the mic muted, or a stream that is mostly music, has
nothing to read, so OpenShorts does not try. When the transcript comes back
under 8 words in total, or under 5 words per minute, the video itself is sent to
Gemini, which watches it and returns moments in the same 15 to 60 second band.
Layouts and facecam detection work the same; the clips come out without
captions, because there is no speech to caption.</p>
<p>That vision pass sends Gemini the footage itself, which it counts at
roughly 300 tokens a second, so an hour is already around a million tokens. A
long silent VOD is therefore watched in stretches of up to 40 minutes, each as
a small low-resolution copy, and the picks from every stretch are ranked
together. Up to eight stretches are watched per job, a little over five hours;
past that only the first five hours are clipped, so for a longer silent VOD
pick the part you care about with "clip only part of it".</p>

<h2>Chat, alerts and overlays</h2>
<p>A Twitch VOD has no chat in it unless your overlay drew the chat onto the
stream. If it did, the chat box is just part of the picture: in a facecam
INSET layout it is usually cropped away with the rest of the corner, and in a
full-width layout it stays where it was. Alerts and on-screen text behave the
same way. Subtitles from the stream itself are noticed: when burned-in text
sits along the bottom in at least 40% of the sampled frames, that band is trimmed
before OpenShorts adds its own captions, so you do not get two layers.</p>

${freeTable(['opus-clip', 'klap', 'vidyo-ai', 'vizard'])}
<p>For streamers the billing unit decides everything. A four-hour VOD is 240
credits on any per-minute meter, which is more than every free plan above and
most entry plans. Klap's meter is clips rather than minutes, but it says its
algorithm leans on speech, so quiet gameplay gives it little to work with.</p>

<h2>What it does not do</h2>
<ul>
<li><strong>It does not clip live.</strong> It works from the VOD after the stream.</li>
<li><strong>It does not read Twitch chat</strong> to find hype moments; it reads what is said and, for silent footage, what is shown.</li>
<li><strong>It does not return dozens of clips from one long job.</strong> Each job returns its best 6 to 12; clip long streams in parts.</li>
<li><strong>It does not watch more than about five hours of a silent VOD</strong> in one job; pick the part you want.</li>
<li><strong>It does not fetch subscriber-only VODs.</strong> Download and upload them instead.</li>
<li><strong>Free clips carry a watermark.</strong> Upgrading removes it from clips already made.</li>
</ul>

<h2>Whose streams you can clip, and the music question</h2>
<p>Your own VODs, and streams whose owner said yes: editors working for a
streamer, members of a clipping programme the streamer runs, co-streamers with
permission. The VOD belongs to whoever streamed it, and the game footage is
covered by the publisher's own policy on videos of its game. Music is the trap:
a song playing during the stream travels into the clip, and TikTok and YouTube
match audio far more aggressively than a live stream ever does.</p>

${faqBlock(twitchFaq)}

${sources([
  `Competitor plans: our comparison data, checked ${longDate(COMPETITORS.klap.checked)}. See <a href="/alternatives/klap">OpenShorts vs Klap</a> and <a href="/alternatives/opus-clip">OpenShorts vs Opus Clip</a>.`,
  `Silent-footage thresholds (8 words, 5 words per minute), the vision fallback and its stretches: <code>main.py</code>, <code>clip_selection.vision_segments</code>; inset detection: <code>camera_inset.py</code>; burned-subtitle trimming: <code>burned_subtitles.py</code>. Source: <a href="${SITE.repo}" rel="noopener">github.com/mutonby/openshorts</a>.`,
  `OpenShorts Cloud plan minutes: <a href="/pricing">pricing</a>, checked ${longDate(PRICE_MODELS.openshorts.checked)}.`,
])}
${trademarkNote('Twitch', 'Twitch Interactive')}
`,
  faq: twitchFaq,
  extraNodes: [
    howToNode('/platforms/twitch', 'How to turn a Twitch VOD into vertical clips with AI',
      'Paste a Twitch past broadcast and get 9:16 clips with the facecam kept, captions and hooks, ready for TikTok, Shorts and Reels.', twitchSteps),
    appNode('/platforms/twitch', 'OpenShorts Twitch clip maker',
      'AI stream clipper for Twitch: turns VODs into vertical clips with the facecam kept and captions burned in.'),
  ],
})

/* ------------------------------------------------------------------------ */
/* /platforms/kick                                                          */
/* ------------------------------------------------------------------------ */

// Both speeds are quoted from the edition summaries in data.js, so they cannot
// drift from the rest of the site.
const CPU_SPEED = EDITIONS.selfHosted.summary.match(/on a typical CPU an 8-minute video takes [^.]+/)?.[0]
const GPU_SPEED = EDITIONS.cloud.summary.match(/An NVIDIA GPU clips that same 8-minute video in about \d+ seconds/)?.[0]
if (!CPU_SPEED || !GPU_SPEED) throw new Error('[clip-tools] processing speeds not found in EDITIONS summaries')
const SPEED_LINE = `${CPU_SPEED} on your own machine, while on OpenShorts Cloud ${GPU_SPEED.replace('An NVIDIA GPU clips that same', 'an NVIDIA GPU clips the same')}`

const kickSteps = [
  ['Copy the Kick VOD or clip link', 'On the streamer\'s Kick page open Videos and copy the link of the past stream, or copy the link of a Kick clip that runs 45 seconds or more. Paste it in the box. A bare channel link is the live channel, so the box asks for the recording.'],
  ['AI picks the moments worth posting', 'The stream is transcribed word by word, Gemini ranks every stretch of it for how well it plays with no context, and the strongest are cut on sentence boundaries into clips of 15 to 60 seconds. A stream with almost no speech is watched by the model instead.'],
  ['Vertical clips, ready to post', 'Each clip is reframed to 9:16 (facecam stacked under gameplay, IRL footage tracked and steadied), captioned and given a hook. Post to TikTok, YouTube Shorts and Instagram Reels, or download.'],
]

const kickFaq = [
  { q: 'Is this Kick clip maker free?', a: `The hosted free plan clips your first video whole up to ${FIRST_VIDEO_MINUTES} minutes, then ${CLOUD.freeMinutes} minutes a month, with a watermark and no credit card. Paid plans start at $${CLOUD.lowPrice}/month with no watermark. Self-hosted OpenShorts is free under MIT, with no watermark and no meter on how many hours you clip.` },
  { q: 'Which Kick links work?', a: 'Links to a past stream (from the Videos tab of a channel) and links to a Kick clip of at least 45 seconds. A channel link on its own points at the live stream, so it is refused with a note to paste the recording instead.' },
  { q: 'What if a Kick link does not work?', a: 'Kick changes its site more often than the older platforms, and a link that fails to download is not the end of it: upload the recording instead (for your own streams, the local recording your streaming software made is the best copy). Uploads work on every plan.' },
  { q: 'Can it clip a Kick stream while it is live?', a: 'No. It works from the recording after the stream has ended.' },
  { q: 'Does it handle IRL streams?', a: 'Yes. A face in the frame is tracked with a stabiliser that holds the crop still inside a safe zone, so a walking, shaky camera does not turn into a shaky clip. Scenes where nobody is in frame keep the full width over a blurred backdrop. A stream shot vertical on a phone is left as it is.' },
  { q: 'Will the facecam be kept on gaming streams?', a: 'With the vertical layout on auto, a webcam box in a corner is detected and the clip renders with the gameplay across the top and the facecam enlarged underneath.' },
  { q: 'How many clips does one Kick stream give?', a: 'Each job returns its best 6 to 12 moments for anything longer than about 12 minutes. For a stream of several hours, clip it in parts with "clip only part of it" to get more candidates.' },
  { q: 'Can I clip a Kick streamer\'s VOD if I am not that streamer?', a: 'With their permission: as their editor, or as part of a clipping programme they run. Clipping someone\'s stream without it and posting the result is a reupload the platforms will take down when the streamer reports it.' },
  { q: 'Can I post Kick clips to TikTok and YouTube?', a: 'Yes, from the dashboard. YouTube Shorts and Instagram Reels are published directly; TikTok receives each clip as a draft in your TikTok inbox, which you finish in the app.' },
  { q: 'How does it compare with Opus Clip or Klap for Kick?', a: `Opus Clip bills a credit per imported minute, from ${COMPETITORS['opus-clip'].entryPrice}; Klap bills clips, from ${COMPETITORS.klap.entryPrice}, and leans on speech detection (both checked ${longDate(COMPETITORS.klap.checked)}). OpenShorts is open source and self-hostable with no meter, and hosted from $${CLOUD.lowPrice}/month.` },
]

const kickClipMaker = () => common('/platforms/kick', {
  title: 'Kick Clip Maker: Turn Kick Streams Into Viral Clips',
  description: `Turn Kick streams and VODs into vertical clips with AI: moments picked, facecam kept, captions added. First video free up to ${FIRST_VIDEO_MINUTES} min, then watermarked.`,
  h1: 'Kick clip maker: turn Kick streams into clips',
  breadcrumb: [{ name: 'Platforms', path: '/platforms' }, { name: 'Kick clip maker' }],
  tool: { entry: 'tool-paste-link', html: widget('https://kick.com/<channel>/videos/...', 'Kick VOD or clip link') },
  cta: {
    label: 'Clip a Kick stream',
    title: 'Paste the VOD, get the clips',
    body: `Free plan: ${PRICE_LINE}`,
  },
  related: ['/platforms/twitch', '/platforms/tiktok', '/platforms/youtube-shorts', '/alternatives/opus-clip'],
  tldr: [
    'Kick clips are how a Kick stream travels: the twenty seconds of a long broadcast that make sense to someone scrolling TikTok who has never heard of the streamer. Paste the link of a Kick VOD and OpenShorts listens to the whole stream, ranks every stretch for how well it stands alone, and cuts the best into vertical clips with captions and a hook, facecam kept.',
    'It works the same whether you are the streamer, their editor or part of a clipping programme they run. What it will not do is decide for you whether you have permission: that part is yours.',
  ],
  body: `
${stepsBlock('How to make Kick clips with AI', kickSteps)}

<h2>Which Kick links to paste</h2>
<p>Three kinds of Kick address look alike and only two of them are recordings:</p>
<ul>
<li><strong>A past stream</strong>, from the Videos tab of a channel. Its address contains <code>/videos/</code>. This is the one to paste for clipping a whole broadcast.</li>
<li><strong>A Kick clip</strong>, made by the streamer or a viewer. Its address contains <code>clip</code>. It works if it runs at least 45 seconds, which is the minimum any source needs, and comes back reframed and captioned.</li>
<li><strong>The channel itself</strong> (<code>kick.com/name</code>). That is the live stream or an offline page, not a recording, so the box asks for the VOD instead.</li>
</ul>
<p>Kick changes its site more often than YouTube or Twitch do. If a link that
looks right fails to download, the fallback is the file: if it is your stream,
the local recording your streaming software made works best, and an upload
works on every plan, the free one included.</p>

<h2>IRL, just chatting, gaming: three kinds of stream, three crops</h2>
<p>A lot of Kick is not a person at a desk. Each clip is laid out by what is
actually in the frame, decided scene by scene:</p>
<h3>IRL and outdoor streams</h3>
<p>A camera on a gimbal or a selfie stick moves all the time, and a crop that
follows every movement makes a clip nobody can watch. The tracker follows the
face with a stabiliser that only moves the frame when the subject leaves a
safe zone in the middle, and holds still otherwise. Shots with nobody in them
(the street, the view, the car) keep their full width over a blurred backdrop
instead of being cropped to a random strip. A stream shot vertical on a phone
is passed through untouched; it already fits.</p>
<h3>Just chatting</h3>
<p>One face filling a 16:9 frame is the easy case: a steady face-tracked crop.
A guest in the same shot gets stacked, one person per half, with the captions
on the seam between them where they cover nobody.</p>
<h3>Gaming with a facecam</h3>
<p>The webcam box in the corner is found by its geometry (small, off centre,
still between samples) and the clip renders gameplay full width on top and
the facecam enlarged below. The <a href="/platforms/twitch">Twitch clip
maker</a> page covers that layout in more detail, along with what happens on
gameplay with no commentary.</p>

<h2>Volume: clipping a streamer who goes live every day</h2>
<p>A daily streamer produces more hours than anyone can watch back, which is
the whole reason clipping is a job. Two things matter at that volume.</p>
<p>First, the shape of a job. One job returns the best 6 to 12 moments of
whatever it is given, however long that is. A five-hour stream clipped in one
job is ranked against itself and loses good moments to better ones. Clipped in
parts with <strong>clip only part of it</strong> (each hour, or the segment
someone flagged), it yields more candidates, and only the stretch you pick is
downloaded and billed.</p>
<p>Second, the meter. OpenShorts Cloud bills minutes of source, like the
credit-based tools below; at several hours a day that adds up on any hosted
plan. The self-hosted edition has no meter at all: it runs on your own machine
with Docker and your own API keys. Speed is the trade-off: ${esc(SPEED_LINE)}.</p>

<h2>Posting Kick clips</h2>
<p>Kick clips are meant to leave Kick. Connect your accounts in the dashboard
and each clip can go to YouTube Shorts and Instagram Reels directly, and to
TikTok, where it arrives as a draft in your inbox to finish with a sound and a
caption. <strong>Schedule a week</strong> spreads a batch over the coming days
at the hour you choose. Each clip comes with a title and descriptions written
for the platforms, and a hook burned over the first seconds.</p>

${freeTable(['opus-clip', 'klap', 'submagic', '2short'])}
<p>Submagic and Klap have no ongoing free plan, and 2short.ai takes YouTube
links only on its free plan, which leaves Kick VODs out unless you upload them
somewhere first. Per-minute meters (Opus Clip, OpenShorts Cloud) count every
minute of the stream you import, which is why clipping a stretch rather than a
whole VOD matters.</p>

<h2>What it does not do</h2>
<ul>
<li><strong>It does not clip live.</strong> It works on the VOD or a clip after the fact.</li>
<li><strong>It does not read Kick chat</strong> or follow alerts; it ranks what is said, and for silent footage what is shown.</li>
<li><strong>It does not guarantee a Kick link downloads.</strong> When one fails, upload the file.</li>
<li><strong>It does not take clips under 45 seconds</strong> as a source.</li>
<li><strong>It does not post to Kick.</strong> Clips go to TikTok, YouTube and Instagram, or to your disk.</li>
<li><strong>Free clips carry a watermark.</strong> A paid plan removes it, from earlier clips too.</li>
</ul>

<h2>Permission, plainly</h2>
<p>A Kick VOD belongs to the streamer. Clipping it is fine when the streamer
said so: you are the streamer, you edit for them, or you are part of a
clipping programme they run and its rules. Clipping it without that and
posting the result is a reupload, and TikTok, YouTube and Instagram remove those
when the owner reports them. Music playing on stream travels into the clip and
gets matched on those platforms far more strictly than on a live stream. Every
OpenShorts job asks you to confirm the rights before it starts.</p>

${faqBlock(kickFaq)}

${sources([
  `Competitor plans: our comparison data, checked ${longDate(COMPETITORS['opus-clip'].checked)}. See <a href="/alternatives">all alternatives</a>.`,
  `Processing speeds and the pipeline: <a href="/how-openshorts-works">how OpenShorts works</a>. Layout code (tracking, inset, stacked speakers): <a href="${SITE.repo}" rel="noopener">github.com/mutonby/openshorts</a>.`,
])}
`,
  faq: kickFaq,
  extraNodes: [
    howToNode('/platforms/kick', 'How to turn a Kick stream into vertical clips with AI',
      'Paste a Kick VOD or clip link and get 9:16 clips with captions and hooks for TikTok, YouTube Shorts and Instagram Reels.', kickSteps),
    appNode('/platforms/kick', 'OpenShorts Kick clip maker',
      'AI clip maker for Kick streams: turns Kick VODs into vertical clips with captions, hooks and the facecam kept.'),
  ],
})

/* ------------------------------------------------------------------------ */
/* /auto-reframe                                                             */
/* ------------------------------------------------------------------------ */

const reframeSteps = [
  ['Paste a link or upload the video', 'Any horizontal recording works: an interview, a podcast with two cameras, a screen tutorial, a stream with a facecam. Paste a public link or upload the file; it needs at least 45 seconds of material.'],
  ['AI cuts the clips and decides the frame', 'The best moments are picked from the transcript, then each clip is reframed scene by scene: faces are tracked, two people in one shot are stacked, screens are kept readable and a corner webcam is enlarged.'],
  ['Check, adjust, export', 'Every clip comes back 9:16 with captions. Where the AI framed a scene wrong, draw the crop yourself for that scene in the reframe editor, then download or post to TikTok, YouTube Shorts and Instagram.'],
]

const reframeFaq = [
  { q: 'What does auto reframe mean?', a: 'Converting a horizontal video to a vertical (or other) aspect ratio by moving the crop to follow what matters in the shot, usually a face, instead of cutting a fixed slice out of the middle. Here it is one stage of turning a long video into vertical clips.' },
  { q: 'Is the auto reframe free?', a: `On the hosted free plan your first video is clipped whole up to ${FIRST_VIDEO_MINUTES} minutes, then ${CLOUD.freeMinutes} minutes a month, with a watermark and no credit card; paid plans start at $${CLOUD.lowPrice}/month with no watermark. The self-hosted edition is free under MIT. For a quick static crop of a short file, the free video to 9:16 converter runs in your browser with no account and no watermark.` },
  { q: 'How is it different from the free 9:16 converter?', a: 'The converter applies one fixed treatment to the whole file (crop at a position you choose, blurred background or black bars) and never looks at what is in the frame. Auto reframe tracks faces, decides a layout per scene and cuts the video into clips. Use the converter for a short clip you have already cut; use auto reframe for a long recording with people in it.' },
  { q: 'Can it reframe a whole video into one vertical video?', a: 'No. It reframes the clips it cuts, each up to 180 seconds when you raise the maximum. To convert a whole short video in one piece with a fixed crop, use the free 9:16 converter, which takes up to 5 minutes in the browser.' },
  { q: 'What happens with two people in the shot?', a: 'When both faces are visible in the same frame for at least half of the sampled frames, the scene is stacked: one person in each half of the vertical frame, with the captions on the seam between them. Shot and reverse shot (one person at a time) stays a single tracked crop, so nobody appears twice.' },
  { q: 'Does it work on screen recordings?', a: 'Yes. With the layout on auto or screencast, each shot is classified as screen, screen beside a presenter, or camera. A screen keeps its full width, cropped to the area being read; a presenter beside the screen is stacked under the content; a webcam floating over the screen is enlarged below it.' },
  { q: 'What if my video is already vertical?', a: 'It is passed through untouched. Every layout exists to win back width a crop throws away, and a vertical source has none to throw away.' },
  { q: 'Can I correct the framing?', a: 'Yes. The reframe editor plays each scene with the uncropped picture and the crop rectangle over it, so you can move the rectangle or split it into two stacked regions. Scenes you do not touch stay automatic.' },
  { q: 'What resolution is the output?', a: 'At least 1080 pixels wide: a 1080p source gives 1080x1920. A smaller crop is scaled up to that floor; a larger source is never scaled down.' },
  { q: 'How is it different from auto reframe in Premiere Pro or CapCut?', a: 'Those reframe footage you have already edited, inside the editor, with you choosing the moments and adding the captions. Here reframing is one automatic stage of turning a long recording into clips: the moments are picked, cut, reframed per scene (including stacked two-speaker and screen layouts) and captioned without an editor. For a finished horizontal edit that only needs to be vertical, the editor feature is the shorter path.' },
  { q: 'Can it convert landscape to portrait video?', a: 'Yes, as clips: each clip cut from a landscape (16:9) recording comes back as portrait 9:16, reframed scene by scene. To convert a short landscape video to portrait in one piece with a fixed crop, use the free 9:16 converter.' },
  { q: 'How does it compare with the auto reframe in Opus Clip or 2short.ai?', a: `Both face-track to 9:16. 2short.ai keeps the speaker centred with one tracked crop and has an unwatermarked free plan of 30 minutes, YouTube links only (checked ${longDate(COMPETITORS['2short'].checked)}). OpenShorts adds stacked two-speaker, screencast and webcam-inset layouts, a per-scene manual override, and an open source, self-hostable edition.` },
]

const autoReframe = () => common('/auto-reframe', {
  title: 'Auto Reframe Video to 9:16 With AI Speaker Tracking',
  description: `Auto reframe horizontal video to 9:16: AI tracks the speaker, stacks two people and keeps screens readable. First video free up to ${FIRST_VIDEO_MINUTES} min, watermarked.`,
  h1: 'Auto reframe: horizontal to vertical video with AI speaker tracking',
  breadcrumb: [{ name: 'Use cases', path: '/use-cases' }, { name: 'Auto reframe' }],
  tool: { entry: 'tool-paste-link', html: widget('Paste a link to a horizontal video', 'Link to the video to reframe') },
  cta: {
    label: 'Reframe a video',
    title: 'Vertical clips that keep the person in the frame',
    body: `Free plan: ${PRICE_LINE}`,
  },
  related: ['/platforms/youtube', '/use-cases/sermons', '/platforms/twitch', '/platforms/instagram-reels', '/alternatives/2short'],
  tldr: [
    'A 16:9 frame is 1.78 times wider than it is tall; a 9:16 one is 0.56. Converting landscape to portrait throws away about two thirds of the width, and the only question is which two thirds. A static crop answers once for the whole video. Auto reframe answers scene by scene, and here it answers with more than one shape: a tracked crop for one speaker, two stacked halves for two, a full-width screen with the presenter under it.',
    'Paste a long horizontal video and you get vertical clips back with the framing already decided, plus an editor to overrule it on the scenes where it chose wrong.',
  ],
  body: `
${stepsBlock('How auto reframe works here', reframeSteps)}

<h2>Auto reframe versus a static crop</h2>
<p>There are two ways to make a horizontal video vertical. The first is a
fixed treatment: keep a vertical slice at one position, or shrink the whole
picture into the middle over a blurred copy or black bars. That is what the
free <a href="/video-aspect-ratio-converter">video to 9:16 converter</a> does,
in your browser, with no upload. It is the right tool for a short clip where
the subject stays in one place, a slide deck, or a wide shot you want to keep
whole. Most tools sold as a vertical video converter are this kind.</p>
<p>The second is to look at the picture. A person who walks across the stage,
two people taking turns, a presenter who shares a screen halfway through:
each of those needs the frame to change with the content, and no single crop
position can do it. Auto reframe does the looking. The difference shows up
most on exactly the footage people most want to clip, interviews and
podcasts, where a fixed centre crop either cuts the guest out or shows an
empty chair. Church services are the other common case: a preacher who walks
the stage through a whole sermon, which the <a href="/use-cases/sermons">sermon
clip maker</a> handles with the same tracking.</p>
${demoBlock('split')}

<h2>The layouts, and when each one is used</h2>
<p>The layout is decided per scene, not once per video, because one video
mixes shots: an edited podcast cuts between close-ups and a wide two-shot, a
tutorial cuts between the camera and the screen.</p>
<h3>Tracked crop (one speaker)</h3>
<p>A single subject is followed with MediaPipe face detection, with a YOLOv8
person detector as the fallback when the face is lost. The camera is damped by
a stabiliser that holds still while the subject stays inside a safe zone and
only moves when they leave it, so the clip looks operated rather than jittery.
A detection that jumps further than the safe zone in one frame has to repeat
before the camera follows, which stops a passing face from yanking the frame.</p>
<h3>Stacked split (two people in one shot)</h3>
<p>When two faces are visible in the same frame for at least half of the
sampled frames, the scene is stacked, one person per half. That condition is
what separates a real two-shot from shot and reverse shot, where stacking would
show the same person twice. The captions on a stacked stretch sit on the seam
between the halves, where they cover nobody.</p>
<h3>Screen and presenter</h3>
<p>For tutorials and demos each shot is classified from one still: a screen, a
screen beside the presenter, or a camera shot. A full screen keeps its width,
cropped to the area actually being read; a presenter beside their content is
stacked under the content columns; a webcam window floating over the screen
is boxed and enlarged below it.</p>
<h3>Wide content</h3>
<p>When what matters spans more of the frame than any crop can keep (a group
on a stage, a map, a chase), the scene is zoomed about 1.3 times around where
the motion is, keeping roughly three quarters of the width between the hook and
the captions, instead of a thin full-width strip or a random vertical column.</p>
<h3>Facecam inset</h3>
<p>A stream or recording with a webcam composited into a corner is detected by
geometry: a subject that is small, off centre and still. The clip renders the
main picture across the top and the webcam enlarged below.</p>
<h3>Speaker cuts and punch-ins</h3>
<p>Two more options are available on the API (<code>layouts</code>:
<code>speaker_cut</code>, <code>punch_in</code>): hard cuts to whoever is
talking, decided from mouth activity normalised per speaker, and a small push
in on the beats of the audio.</p>
<p>The layout choice itself comes from twelve stills of the video, sent to
Gemini at 1024 pixels with a closed question, rather than the whole video: on
a hand-labelled set of 48 clips that gets the layout right 92 to 96% of the
time, and costs the same whether the source is ten minutes or three hours.</p>

<h2>When the AI frames a scene wrong</h2>
<p>Automatic framing is right most of the time and occasionally wrong in a way
nobody would accept: the crop on the interviewer while the guest talks, a face
in the audience instead of on stage. The reframe editor fixes that scene by
scene. Each scene plays with the uncropped picture and its crop rectangle over
it, so you can judge it with motion and sound; drag the rectangle, or split it
into two stacked regions, and only the scenes you touched are re-rendered. The
rest stay automatic.</p>

<h2>Sources that are already vertical, and output size</h2>
<p>A source shot vertical is passed through untouched: every layout above
exists to win back width that a crop throws away, and a vertical video has
none to give. Horizontal sources are delivered at least 1080 pixels wide, so a
1080p recording becomes 1080x1920. A smaller crop is scaled up to that floor;
a larger source is never scaled down.</p>

<h2>Auto Reframe in Premiere Pro or CapCut vs OpenShorts</h2>
<p>Premiere Pro and CapCut both have an auto reframe feature, and it is good at
what it does: inside your edit, it reframes a clip or a sequence you have
already cut to a new aspect ratio, following the motion in the shot. It is a
manual workflow with an automatic step in it. You still choose the moments,
cut them, set up the sequence, check the framing and add the captions
yourself, one video at a time.</p>
<p>OpenShorts puts reframing somewhere else: it is one stage of turning a long
recording into clips, after the moments are picked and before the captions are
burned, with nobody in an editor. And it can change the shape of the frame
(stacked halves for two people, screen over presenter, an enlarged webcam), not
only the position of one crop.</p>
<ul>
<li><strong>Use the editor's feature</strong> when you already have a finished horizontal edit and only need it vertical, or when you want frame-by-frame control in an editor you know.</li>
<li><strong>Use OpenShorts</strong> when you start from a long recording and want the clips found, cut, reframed and captioned for you.</li>
<li><strong>Use the free converter</strong> for a short file where one fixed crop is enough.</li>
</ul>

${freeTable(['2short', 'klap', 'opus-clip', 'submagic'])}
<p>Every tool here reframes with face tracking. The differences are the layouts
beyond a single crop: Klap lists split-screen, screencast and gaming layouts,
2short.ai keeps one centred speaker, Submagic is a caption editor first.</p>

<h2>What it does not do</h2>
<ul>
<li><strong>It does not reframe a whole long video into one vertical file.</strong> It reframes the clips it cuts, each up to 180 seconds.</li>
<li><strong>It does not output square or 4:5.</strong> The target is 9:16.</li>
<li><strong>It does not take sources under 45 seconds.</strong> For a short file, use the free converter.</li>
<li><strong>It does not track objects or animals</strong> as subjects: tracking is for faces and people.</li>
<li><strong>It cannot invent picture.</strong> When the content is wider than any crop, it keeps about three quarters of the width, centred on the motion, over a blurred backdrop rather than guessing.</li>
<li><strong>Free clips carry a watermark.</strong> A paid plan removes it, also from clips already made.</li>
</ul>

${faqBlock(reframeFaq)}

${sources([
  `Layout picker accuracy (12 stills at 1024 px, 92 to 96% on a hand-labelled 48-clip set), split and screencast rules, and the vertical pass-through: <code>layout_picker.py</code>, <code>split_layout.py</code>, <code>screencast_layout.py</code>, <code>reframe_v2.py</code> at <a href="${SITE.repo}" rel="noopener">github.com/mutonby/openshorts</a>.`,
  `Competitor plans: our comparison data, checked ${longDate(COMPETITORS['2short'].checked)}. See <a href="/alternatives/2short">2short AI</a>.`,
])}
`,
  faq: reframeFaq,
  extraNodes: [
    howToNode('/auto-reframe', 'How to auto reframe a horizontal video to vertical 9:16 with AI',
      'Turn a horizontal recording into vertical 9:16 clips with face tracking, stacked speakers and screen layouts, then correct any scene by hand.', reframeSteps),
    appNode('/auto-reframe', 'OpenShorts auto reframe',
      'AI auto reframe from 16:9 to 9:16 with speaker tracking, two-speaker split, screencast and webcam-inset layouts.'),
  ],
})

/* ------------------------------------------------------------------------ */
/* /use-cases/sermons                                                        */
/* ------------------------------------------------------------------------ */

const sermonSteps = [
  ['Paste the service replay', 'Copy the link of the YouTube or Facebook Live replay (or Vimeo, or upload the recording from your stream computer). Use "clip only part of it" to mark where the sermon starts and ends, so the songs and announcements stay out.'],
  ['The AI finds the moments in the message', 'The sermon is transcribed word by word, and Gemini ranks every stretch of it for how well it would land with someone who was not in the room: a story, a single clear point, a line people would repeat. The strongest are cut on sentence boundaries.'],
  ['Captioned vertical clips for the week', 'Each clip comes back 9:16, following the preacher across the platform, with captions and a hook. Fix the spelling of names in the caption editor, then schedule them across the week to YouTube Shorts, Instagram Reels and TikTok, or download them.'],
]

const sermonFaq = [
  { q: 'Is this sermon clip maker free?', a: `The hosted free plan clips your first video whole up to ${FIRST_VIDEO_MINUTES} minutes, then ${CLOUD.freeMinutes} minutes a month, with a watermark and no credit card. Marking just the sermon keeps a whole service inside that first video. Paid plans start at $${CLOUD.lowPrice}/month with no watermark; the self-hosted edition is free under MIT.` },
  { q: 'Do the free sermon clips have a watermark?', a: 'Yes, on the hosted free plan. Moving to a paid plan removes it from new clips and from the ones already made. Self-hosted OpenShorts never adds one.' },
  { q: 'How many clips does one sermon give?', a: 'A sermon longer than about 12 minutes is asked for 6 to 12 clips, enough for one a day through the week, and fewer come back when fewer passages stand alone. You can set an exact number from 1 to 15.' },
  { q: 'Can I clip only the sermon and skip the worship set?', a: 'Yes. "Clip only part of it" lets you set a start and an end. Only that stretch is downloaded, analysed and counted against your minutes.' },
  { q: 'Can we post clips of the worship songs?', a: 'Be careful. The songs belong to their writers and publishers, and a licence that covers streaming your service does not necessarily cover cutting a song into a clip and posting it on social media. Check the terms of your licence. Clips of the preaching avoid the question.' },
  { q: 'Does it follow a pastor who walks the stage?', a: 'Yes. The preacher\'s face is tracked, and a stabiliser holds the frame still while they stay inside a safe zone and only moves when they leave it, so the clip looks like a camera operator, not a chase.' },
  { q: 'What about a guest interview or a sermon with an interpreter?', a: 'When two people are visible in the same shot for most of a scene, the clip is stacked: one person in each half of the vertical frame, captions on the seam between them.' },
  { q: 'Will it spell names and Bible books correctly?', a: 'Speech recognition spells names the way they sound, so check unusual names and book names before posting. The caption text is editable in the subtitle editor and the timing is kept.' },
  { q: 'Can clips be made in Spanish or another language?', a: 'Captions follow the language that was spoken. With your own ElevenLabs API key, dubbing can also translate a clip into more than 30 languages in the preacher\'s voice and re-caption it.' },
  { q: 'Can new services be clipped automatically every week?', a: 'Yes, with Autopilot on a paid plan, for services streamed to a connected YouTube channel. It clips the first 10 to 90 minutes of each new video (your choice), so set it long enough to reach the sermon. With publishing on, it schedules the best clips one a day.' },
  { q: 'Can it make church clips from more than sermons?', a: 'Yes, from anything in the service or the week where someone is speaking: testimonies, guest speakers, youth talks, Bible studies, conference sessions. Songs score far below speech and carry their own rights questions, so leave them out with "clip only part of it".' },
  { q: 'How does it compare with Opus Clip?', a: `Opus Clip also cuts sermons into clips. Its free plan is 60 credits a month with a watermark and it bills one credit per imported minute, from ${COMPETITORS['opus-clip'].entryPrice} (checked ${longDate(COMPETITORS['opus-clip'].checked)}), so a 90-minute service uses 90 credits unless you trim it first. OpenShorts lets you clip only the sermon, and is open source and self-hostable.` },
]

const sermonClipMaker = () => common('/use-cases/sermons', {
  title: 'Sermon Clip Maker: Turn Church Services Into Shorts',
  description: `Turn Sunday's sermon into captioned vertical clips for Shorts, Reels and TikTok that follow the preacher. First video free up to ${FIRST_VIDEO_MINUTES} min, watermarked.`,
  h1: 'Sermon clip maker: turn church services into Shorts and Reels',
  breadcrumb: [{ name: 'Use cases', path: '/use-cases' }, { name: 'Sermon clip maker' }],
  updated: ANSWER_REVISED,
  lede: `OpenShorts is an AI tool that turns sermons into short clips: paste the replay link of a service, mark where the sermon starts and ends, and it cuts the 6 to 12 passages that hold up on their own into vertical clips with captions, keeping the preacher in frame. ${ANSWER_PRICE} Opus Clip and the other options are priced side by side further down.`,
  tool: { entry: 'tool-paste-link', html: widget('Paste the YouTube or Facebook Live replay link', 'Service replay link') },
  cta: {
    label: 'Clip this Sunday',
    title: 'One sermon, a week of clips',
    body: `Free plan: ${PRICE_LINE}`,
  },
  related: ['/platforms/youtube', '/platforms/instagram-reels', '/platforms/youtube-shorts', '/alternatives/vizard'],
  tldr: [
    'A Sunday service streamed to YouTube or Facebook is often 90 minutes: worship, announcements, a 40-minute sermon, prayer. Mark where the sermon starts and ends, paste the replay link, and OpenShorts reads the message word by word and cuts the 6 to 12 passages that hold up on their own into vertical clips, with captions and the preacher kept in frame as they move.',
    'That is a clip a day until next Sunday, made in the time it takes the replay to process, by whoever runs the church\'s social accounts, without an editor.',
  ],
  body: `
${stepsBlock('How to turn a sermon into clips', sermonSteps)}

<h2>Start from the sermon, not the whole service</h2>
<p>Most of a recorded service is not clip material for social media: the
countdown, the worship set, the welcome, the notices, the offering. The sermon
is. So before pasting the link, note roughly where the message begins and
ends in the replay and use <strong>clip only part of it</strong> to set those
two points. Only that stretch is downloaded and analysed, which does three
things at once: the AI ranks passages of the message against each other
instead of against the songs, the clips carry no music you may not have the
rights to post, and on OpenShorts Cloud you are billed for the 40 minutes you
picked instead of the 90 the replay runs.</p>
<p>If you would rather not trim, it still works: the model reads the whole
transcript and the spoken parts score far higher than lyrics. Trimming simply
removes the guesswork.</p>

<h2>What makes sermon clips work on a phone</h2>
<p>A clip on someone's phone has to make sense with no context: they did not
hear the reading, the series title or the first point. The scoring is built
for exactly that. The transcript is split into windows of about 90 seconds
and Gemini scores every one on hook strength, emotional payload and how well it
stands alone; the best go to a second pass that cuts each clip on sentence
boundaries, 15 to 60 seconds long by default. In practice the passages that win
are a story told in full, one clear point stated plainly, a question put to the
listener, or the line people would repeat on the way out. Long arguments that
depend on the previous ten minutes tend not to, which is the right call for
the feed.</p>
<p>Each clip also gets a hook line over its opening seconds, written from what
is said in that clip, and a title and description for each platform. Read
them before posting: they are written by a model, and your church has its own
voice.</p>

<h2>Filming a church: one preacher, a stage, sometimes two people</h2>
<h3>A preacher who moves</h3>
<p>Few preachers stay behind a pulpit. The tracker follows the face with a
stabiliser that holds the frame still while the preacher stays inside a safe
zone and only moves when they step out of it, so the vertical clip looks like
a steady camera operator rather than a crop chasing every gesture. When your
production cuts between a close camera and a wide one, each shot is framed on
its own.</p>
<h3>Interviews, panels and interpreters</h3>
<p>When two people share the shot for most of a scene (a guest interview, a
Q&amp;A, a sermon preached with an interpreter beside the speaker), the clip is
stacked: one person in each half of the vertical frame, and the captions sit on
the seam between them, where they cover neither face.</p>
<h3>Scripture and slides</h3>
<p>If your stream shows the verse or the sermon slides full screen, those shots
are kept readable rather than cropped to a narrow strip: a screen keeps its
width, and a presenter beside it is stacked under the slide. Burned-in
subtitles already in the stream are noticed and trimmed off the bottom before
OpenShorts adds its own, so you do not get two layers of text.</p>
${demoBlock('split', 'A two-person conversation from a church media ministry: both speakers stacked in half-frames, captions on the seam where they cover nobody.')}

<h2>Captions, names and other languages</h2>
<p>Most people watch these clips with the sound off, so captions are burned in
from a transcript with a timestamp on every word. Speech recognition spells
what it hears, which means unusual names, places and some book names can come
out wrong: open the subtitle editor, correct the text, and the timing is kept.
Styles can be changed in the same place.</p>
<p>For a church that serves more than one language, captions follow the
language that was spoken, and with your own ElevenLabs API key the dubbing
option translates a clip into more than 30 languages, keeping the preacher's
voice and re-captioning it in the new language.</p>

<h2>A week of clips, scheduled once</h2>
<p>Connect the church's YouTube, Instagram and TikTok accounts in the
dashboard, select the clips and use <strong>schedule a week</strong> to spread
them one a day at the hour you pick, so the sermon Shorts and Reels go out
while the message is still fresh. YouTube Shorts and Instagram Reels publish
directly. TikTok receives each clip as a draft in its inbox, to be posted from
the app, which is TikTok's rule for videos sent from other tools.</p>
<p>If the service is streamed to a YouTube channel every week,
<a href="/auto-clip">Autopilot</a> can do it without anyone opening the
dashboard: on a paid plan it clips each new video on the channel and, with
publishing on, schedules the best clips one a day. One setting matters for
churches: Autopilot clips the first 10 to 90 minutes of each video (30 by
default), so if the sermon starts 35 minutes in, set it to 60 or 90 or it will
clip the worship set.</p>

<h2>Church video clips beyond the sermon</h2>
<p>The same clipper works on any part of church life that is mostly someone
talking: a testimony, a guest speaker, a youth talk, a Bible study, a
conference session, a Q&amp;A with the pastor. Paste the replay or upload the
recording and you get church video clips cut the same way, on sentence
boundaries, captioned and vertical. What it is not built for is the music: the
lyrics of a worship set score far below speech, and the song rights question
above applies to any clip of it.</p>

<h2>Where the replay can come from</h2>
<p>YouTube and Facebook Live replays are the usual sources and both are pasted
as links once the stream has ended; Vimeo works the same way. If the service
was recorded locally (by the stream computer or a recorder on the sound desk),
upload the file instead: a local recording is often better quality than the
stream, and uploads work on every plan. Private and unlisted-by-sign-in videos
cannot be fetched from a link.</p>

${freeTable(['opus-clip', 'vizard', 'vidyo-ai', 'submagic'])}
<p>Per-minute meters (Opus Clip, Vizard, Quso and OpenShorts Cloud alike) count
the whole replay if you import the whole replay. Marking only the sermon is
the difference between 40 and 90 minutes on any of them, and on OpenShorts
Cloud you can do that marking before the job starts.</p>

<h2>What it does not do</h2>
<ul>
<li><strong>It does not clip the service while it is live.</strong> It works on the replay.</li>
<li><strong>It does not check song licences.</strong> If a clip contains worship music, whether you may post it is between you and your licence.</li>
<li><strong>It does not add scripture references or graphics.</strong> Add a verse reference to the caption or description yourself.</li>
<li><strong>It does not post to Facebook.</strong> Download the clip and post it there yourself.</li>
<li><strong>It does not guarantee spelling of names.</strong> Proofread the captions before posting.</li>
<li><strong>Free clips carry a watermark.</strong> A paid plan removes it, from earlier clips as well.</li>
</ul>

<h2>Whose sermons can you clip</h2>
<p>Your own church's, and those you have permission for: a guest speaker's
message with their agreement, a network or denomination that has given its
churches the right to share. A sermon belongs to whoever preached and recorded
it, and platforms remove reuploads when the owner reports them. Each job asks
you to confirm the rights before it runs.</p>

${faqBlock(sermonFaq)}

${sources([
  `Competitor plans: our comparison data, checked ${longDate(COMPETITORS.vizard.checked)}. See <a href="/alternatives/vizard">OpenShorts vs Vizard</a>.`,
  `Autopilot window (first 10 to 90 minutes, 30 by default) and scheduling: <code>cloud/autopilot.py</code>; split layout and caption seam: <code>split_layout.py</code>, <code>subtitles.py</code> at <a href="${SITE.repo}" rel="noopener">github.com/mutonby/openshorts</a>.`,
])}
`,
  faq: sermonFaq,
  extraNodes: [
    howToNode('/use-cases/sermons', 'How to turn a sermon into short vertical clips with AI',
      'Cut the sermon from a church service replay into captioned 9:16 clips for YouTube Shorts, Instagram Reels and TikTok.', sermonSteps),
    appNode('/use-cases/sermons', 'OpenShorts sermon clip maker',
      'AI sermon clip maker: turns church service replays into captioned vertical clips that follow the preacher.'),
  ],
})

export function clipToolPages() {
  return [
    youtubeClipMaker(),
    youtubeShortsMaker(),
    tiktokClipMaker(),
    aiReelGenerator(),
    twitchClipMaker(),
    kickClipMaker(),
    autoReframe(),
    sermonClipMaker(),
  ]
}
