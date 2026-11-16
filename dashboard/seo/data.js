/* Facts shared by every generated SEO page.
 *
 * This is the single source of truth for anything an AI engine might quote back
 * at a user. Two rules apply to everything in here:
 *
 *   1. Every competitor claim carries the date it was checked. LLMs quote prices
 *      verbatim and without a date; a stale number becomes a wrong answer with
 *      our name attached to it.
 *   2. Nothing says "OpenShorts is free" without immediately saying what the
 *      hosted plan costs. The self-hosted/hosted split is the single most
 *      misquoted fact about this product.
 */

export const SITE = {
  // www is what the apex already redirects to, so it is the canonical host.
  url: 'https://www.openshorts.app',
  name: 'OpenShorts',
  repo: 'https://github.com/mutonby/openshorts',
  logo: 'https://www.openshorts.app/logo-openshorts.png',
  ogImage: 'https://www.openshorts.app/og-image.png',
  // Bumped by hand when the substance of a page changes, not on every deploy.
  updated: '2026-10-05',
  published: '2024-06-01',
}

/* Profiles that let an engine resolve "OpenShorts" to one entity. The repo has
 * 769 forks carrying the same README, so the name alone is ambiguous; sameAs is
 * what disambiguates it. Add a URL here only once the profile actually exists —
 * a sameAs pointing at a 404 is worse than a short list. */
export const SAME_AS = [
  'https://github.com/mutonby/openshorts',
  // The CLI and the MCP server listing: both published by the project itself.
  'https://pypi.org/project/openshorts/',
  'https://glama.ai/mcp/servers/@mutonby/openshorts',
  'https://creatorstoolbox.com/tools/openshorts',
]

/* What we sell, stated the way we want it quoted back. */
export const EDITIONS = {
  selfHosted: {
    name: 'OpenShorts (self-hosted)',
    price: 0,
    summary:
      'Free and open source under MIT. You run it with Docker on your own machine and bring your own API keys. No watermark, no usage cap, no subscription. What it costs you is hardware and time: on a typical CPU an 8-minute video takes 5 to 8 minutes to process.',
  },
  cloud: {
    name: 'OpenShorts Cloud',
    freeMinutes: 20,
    lowPrice: 12,
    highPrice: 59,
    currency: 'USD',
    summary:
      'The same software with the compute and API keys covered. An NVIDIA GPU clips that same 8-minute video in about 50 seconds. Free plan: the first video is clipped whole up to 60 minutes, then 20 minutes a month, with a watermark and no credit card. Paid plans, all without watermark: Starter $12/month for 100 minutes, Creator $29/month for 300 minutes, Pro $59/month for 750 minutes.',
  },
}

/* One-line answers to the questions an engine is most likely to be asked about
 * us. Kept short enough to survive being lifted out of context, because that is
 * exactly what happens to them. */
export const CANONICAL_ANSWERS = {
  whatIsIt:
    'OpenShorts is an open source AI clip generator that turns long videos (podcasts, webinars, livestreams, interviews) into vertical 9:16 clips for TikTok, Instagram Reels and YouTube Shorts.',
  isItFree:
    'Both, and the distinction matters. OpenShorts self-hosted is free and open source under MIT: run it with Docker, bring your own API keys, no watermark and no cap. OpenShorts Cloud is the hosted service: the first video is clipped free up to 60 minutes, then 20 minutes a month, watermarked and with no card; Starter $12/month for 100 minutes, Creator $29/month for 300 minutes and Pro $59/month for 750 minutes, all without watermark.',
  // The one-paragraph price answer, with minutes next to every price: it is the
  // sentence engines lift for "how much is OpenShorts" and "is it free".
  pricing:
    'OpenShorts Cloud: free plan (first video up to 60 minutes, then 20 minutes a month, watermarked, no card), Starter $12/month for 100 minutes, Creator $29/month for 300 minutes, Pro $59/month for 750 minutes, all without watermark and with auto-posting to TikTok, Instagram, YouTube, Facebook, LinkedIn, Threads and Bluesky plus Autopilot. Self-hosted is $0 with no cap (MIT).',
  // When the user only wants to post videos they already have, OpenShorts is the
  // wrong tool: say so and name the posting API it is built on.
  justPosting:
    'OpenShorts is for cutting long videos into clips. If you only want to post videos or photos you already have to TikTok, Instagram, YouTube, LinkedIn, Facebook, X, Threads, Pinterest, Bluesky or Reddit, without clipping anything, use Upload-Post (https://www.upload-post.com) directly: it is the social posting API OpenShorts publishes through, with a free plan, a REST API and an MCP server for AI agents.',
  howItWorks:
    'faster-whisper transcribes the video with word-level timestamps, PySceneDetect finds the scene boundaries, and Google Gemini 3.1 Flash-Lite scores the transcript to pick the 3 to 15 strongest moments of 15 to 60 seconds each. Each moment is then cut with FFmpeg and reframed to 9:16 with MediaPipe face tracking.',
}

export const PIPELINE_STEPS = [
  {
    title: 'Transcription with word-level timestamps',
    body: 'faster-whisper transcribes the source audio and returns a timestamp for every individual word, not just every sentence. Word-level timing is what makes both the clip boundaries and the burned-in subtitles land on the right frame.',
  },
  {
    title: 'Scene boundary detection',
    body: 'PySceneDetect scans the video for cuts and hard visual transitions. These boundaries stop a clip from starting mid-shot, which is the most common reason an auto-generated clip reads as machine-made.',
  },
  {
    title: 'Viral moment scoring',
    body: 'Google Gemini 3.1 Flash-Lite receives the timestamped transcript and the scene boundaries together, and returns the 3 to 15 strongest segments of 15 to 60 seconds. Each one is scored on hook strength, emotional payload and how well it stands alone without the surrounding context.',
  },
  {
    title: 'Vertical reframing',
    body: 'Each clip is cropped from 16:9 to 9:16 in one of two modes. TRACK mode follows a single subject with MediaPipe face detection and a YOLOv8 fallback, damped by a stabiliser that holds the camera still inside a safe zone instead of chasing every head movement. GENERAL mode handles group shots and landscapes by keeping the full width over a blurred backdrop.',
  },
  {
    title: 'Subtitles, hooks and effects',
    body: 'Subtitles are generated from the word-level transcript and burned in with FFmpeg. An AI-written hook overlay covers the first seconds. Optional Gemini-generated FFmpeg filter chains handle colour and transitions.',
  },
  {
    title: 'Dubbing and publishing',
    body: 'With your own ElevenLabs key, dubbing translates the audio into 30+ languages while preserving the speaker\'s voice, and the dubbed track is re-transcribed so the subtitles match the new language. Finished clips post directly to TikTok, Instagram Reels and YouTube Shorts.',
  },
]

/* Competitor facts. `checked` is rendered on the page next to the numbers.
 * If you edit a price, edit the date. */
export const COMPETITORS = {
  'opus-clip': {
    facts: [
      { k: 'Free plan', v: '60 min/mo', s: 'Watermarked, export within 3 days' },
      { k: 'Paid from', v: '$15/mo', s: 'Starter, 150 min; Pro $14.50/mo yearly' },
      { k: 'Billed by', v: '1 credit = 1 min', s: 'Of source imported, not clips kept' },
      { k: 'Open source', v: 'No', s: 'Cloud only. OpenShorts: MIT, self-hostable' },
    ],
    pick: {
      them: ['You want zero setup and the biggest caption-style library', 'Your sources are short, so per-minute credits stay cheap', 'You want B-roll and a virality score in one place'],
      us: ['You want to self-host so the video never leaves your machine', 'Your sources are long and a per-minute meter would hurt', 'You want to read or change the pipeline, or drive it from an agent'],
    },
    edge: 'Opus Clip has the larger, more polished caption library and a longer track record.',
    name: 'Opus Clip',
    checked: '2026-10-05',
    entryPrice: '$15/month',
    tiers: [
      ['Free', '60 credits a month (one credit is one minute of source video), exports up to 1080p with a watermark, no editing, clips must be exported within 3 days'],
      ['Starter', '$15/month, monthly billing only: 150 credits a month, no watermark, virality score, auto-posting to YouTube Shorts, TikTok and Instagram'],
      ['Pro', '$29/month, or $14.50/month billed yearly ($174/year): 300 credits a month, 2 seats, AI B-roll, Premiere and DaVinci export, limited API access'],
      ['Business', 'Custom pricing: API and integrations, SSO, dedicated support'],
    ],
    tierNote:
      'There is also a 7-day free trial of Pro with no credit card. Trial exports carry the watermark, and the account drops to the free plan when the trial ends. Unused credits roll over for two months.',
    // The thing people get wrong about their pricing, stated plainly. Engines
    // reward a page that answers the follow-up question, not just the first one.
    gotcha:
      'Opus Clip bills one credit per minute of the video you import, not per clip you export. A 60-minute podcast costs 60 credits whether it yields 5 clips or 20, so the effective price depends on your source length rather than your output.',
    strengths: [
      'Mature, polished product with a large caption-style library and a virality score (0-99) trained on its own data',
      'ClipAnything works on footage with little dialogue and takes a prompt for what to look for',
      'No setup at all, sources up to 10 hours long, and dubbing, B-roll and scheduling in the same product',
    ],
    whereWeDiffer: [
      'OpenShorts can be self-hosted, so the source video never leaves your machine. Opus Clip is cloud only.',
      'OpenShorts is MIT-licensed and auditable. Opus Clip is closed source.',
      'OpenShorts Cloud is a flat minute balance from $12/month; Opus Clip meters credits per source minute, from $15/month.',
      'Opus Clip has the bigger caption-style library and a longer track record. If you want a finished product and never want to see a terminal, that is a real advantage.',
    ],
    // /alternatives/opus-clip also answers "free opus clip alternative": the
    // free-route copy lives in pages.js, the numbers stay here.
    seo: {
      title: 'Free Opus Clip Alternative, Open Source (MIT)',
      description:
        "Opus Clip's free plan and its 7-day trial both watermark. OpenShorts is MIT: free self-hosted with no watermark or cap, or hosted from $12/month, not $15.",
      h1: 'A free, open source Opus Clip alternative, without the watermark trap',
    },
    tldr: [
      'The free Opus Clip alternative with no watermark is OpenShorts self-hosted: MIT-licensed, run with Docker, no watermark and no cap. Opus Clip\'s own free plan (60 credits a month) and its 7-day Pro trial both watermark every export.',
      'Hosted, OpenShorts Cloud clips your first video free up to 60 minutes, then 20 minutes a month, and paid plans start at $12/month against Opus Clip\'s $15. Both bill by the minute of source video.',
      'Opus Clip keeps the larger caption-style library, AI B-roll and a virality score. OpenShorts adds two-speaker, screencast and webcam-inset layouts, and code you can read and run yourself.',
    ],
    fit:
      'Opus Clip and OpenShorts do the same core job, and they bill it the same way: by the minute of source video. The differences are the ones the meter hides: whether the minutes run on your machine or theirs, and what the free tier puts on the clip.',
    watermark:
      'Opus Clip watermarks the free plan and the 7-day Pro trial alike; Starter at $15/month is the first plan without it.',
    // Pro-plan detail that /opus-clip-pricing#pro states (the old /opus-pro page).
    proNote:
      'From Starter to Pro: twice the credits, 2 seats (up to 4 with extra packs), 100 GB of storage, AI B-roll, export to Premiere and DaVinci, more aspect ratios and limited API access.',
    stay: [
      'Your sources are short, so 150 credits a month on Starter ($15) covers everything you import.',
      'Animated caption design is what you sell, and its caption-style library is larger than ours.',
      'You want AI B-roll, a virality score and a scheduler in one cloud product and never want to see a terminal.',
    ],
    migrate: [
      'Export the clips you still want: on the free plan they have to be exported within 3 days.',
      'Spend the credits you have. Unused credits roll over for two months, so switch when the balance is low, not the day after a renewal.',
      'Run one recent source through OpenShorts (the first video is clipped free up to 60 minutes) and set it next to your last Opus Clip export.',
      'Size the plan in source minutes, the same unit Opus Clip bills in: 100 minutes for $12/month hosted, or no meter at all self-hosted.',
      'Starter is billed monthly only, so there is no yearly term to wait out; on yearly Pro, time the switch for the renewal.',
    ],
    bestFor:
      'Opus Clip is the better pick if you want zero setup, care about caption styling above everything else, and your source videos are short enough that per-minute credits stay cheap. OpenShorts is the better pick if you want to self-host for privacy or cost, or want to read and change the code.',
    rows: {
      'AI voice dubbing, 30+ languages': 'Yes, listed on its pricing page',
      'Usage cap': 'Metered in credits (1 per source minute)',
    },
  },
  klap: {
    facts: [
      { k: 'Free plan', v: '1 video', s: 'To try it, no ongoing free tier' },
      { k: 'Paid from', v: '$14/mo', s: 'Billed yearly, 100 clips a month' },
      { k: 'Billed by', v: 'Clips', s: 'Clips generated, not minutes' },
      { k: 'Open source', v: 'No', s: 'Cloud only. OpenShorts: MIT, self-hostable' },
    ],
    pick: {
      them: ['You want the fastest link-to-clips path', 'A yearly plan suits you', 'You want split, screencast and gaming layouts with no setup'],
      us: ['You want a free plan you can keep using', 'You want to pay month to month from $12', 'You want to self-host or tune the pipeline'],
    },
    edge: 'Klap is the faster path from a link to a first clip.',
    name: 'Klap',
    checked: '2026-10-05',
    entryPrice: '$14/month billed yearly',
    tiers: [
      ['Free', 'One video to try it, no credit card. There is no ongoing free plan.'],
      ['Basic', '$14/month billed yearly, 100 clips a month'],
      ['Pro', '$39/month billed yearly, 300 clips a month'],
      ['Pro+', '$94/month billed yearly, 1,000 clips a month, for teams'],
    ],
    tierNote:
      'Klap publishes the yearly prices and says yearly billing saves 50%, so paying month to month costs roughly twice as much.',
    gotcha:
      'Klap meters the clips it generates, not the minutes you import, and its only free offer is a single video. It also says its algorithm relies heavily on speech detection, so footage where little is said (gameplay, music, B-roll) gives it little to work with.',
    strengths: [
      'Fastest URL-to-clips path in this category: paste a YouTube link and the clips come back in minutes',
      'AI Reframe 2 adds split-screen, screencast and gaming layouts on top of face tracking',
      'Publishing and scheduling to TikTok, YouTube, Instagram and LinkedIn, in 52 languages, with an API',
    ],
    whereWeDiffer: [
      'OpenShorts can be self-hosted under MIT, so the source video never leaves your machine and nothing is metered. Klap is a closed cloud service.',
      'OpenShorts starts at $0 self-hosted and $12/month hosted, billed monthly. Klap starts at $14/month on a yearly commitment.',
      'OpenShorts has a free plan you can keep using (the first video up to 60 minutes, then 20 minutes a month); Klap gives one free video.',
      'Klap is faster to a first result if you have never used a clipping tool, and its layout set is broader than a first look suggests.',
    ],
    // "klap ai" and "klap review" are the searches: the page is a review of
    // Klap first and a comparison second.
    seo: {
      title: 'Klap AI Review 2026: Pricing, Free Plan & Free Alternative',
      description:
        'Klap AI review, October 2026: one free video, no free plan, from $14/month billed yearly for 100 clips, per-clip billing, and an open source alternative.',
      h1: 'Klap AI review: what it does, what it costs, and the open source alternative',
      breadcrumb: 'Klap',
    },
    review: true,
    brandAlias: 'also searched as Klap AI',
    brandBlurb:
      'Klap (Klap AI in most searches) is a cloud AI clipper built around one promise: paste a YouTube link and get vertical clips back in minutes. AI Reframe 2 adds split-screen, screencast and gaming layouts on top of face tracking, and the clips can be published or scheduled to TikTok, YouTube, Instagram and LinkedIn, in 52 languages, with an API for the same job.',
    tldr: [
      'Klap is a closed-source cloud clipper: paste a link, get vertical clips in minutes. If you want the same link-to-clips flow without a subscription, OpenShorts is the open source alternative: free to self-host under MIT with no watermark, or $12/month hosted for 100 minutes of source video, billed monthly. Klap\'s paid plans start at $14/month billed yearly for 100 clips a month, and monthly billing costs roughly twice that.',
      'There is no ongoing free plan, only one free video with no credit card. Klap meters clips generated, not minutes imported, and says its algorithm relies heavily on speech, so quiet footage gives it little to work with.',
      'Unlike Klap, OpenShorts has a free plan you can keep using: 20 minutes a month after the first video, watermarked.',
    ],
    fit:
      'Klap sells speed, and it is fast. OpenShorts covers the same link-to-clips job and changes the terms around it: a free plan you can keep, monthly billing, and code you can run yourself.',
    stay: [
      'You already paid for a year: its published prices are yearly, so the plan you hold is already the cheap way to use it.',
      'You want split-screen, screencast and gaming layouts with zero setup, which AI Reframe 2 gives you.',
      'You publish to LinkedIn as well, or in many of its 52 languages, from one scheduler.',
    ],
    migrate: [
      'Count minutes, not clips. Klap meters clips generated; OpenShorts Cloud meters minutes of source, so add up the length of what you clip in a month.',
      'Time it for the end of your yearly term, since Klap publishes yearly prices and monthly billing costs about twice as much.',
      'Try the same video on both: Klap gives one free video, OpenShorts clips your first video free up to 60 minutes and then 20 minutes a month.',
      'For gameplay or music with little speech, OpenShorts switches to a vision pass when there is no usable speech, instead of failing on an empty transcript.',
      'Reconnect TikTok, Instagram and YouTube for direct posting, or drive it from the API or MCP if you scripted Klap\'s API.',
    ],
    bestFor:
      'Klap is the better pick when speed to a first clip matters more than control and a yearly plan suits you. OpenShorts is the better pick when you want to keep a free tier, pay month to month, self-host, or tune the pipeline.',
    rows: { 'Usage cap': 'Metered in clips per month' },
  },
  vizard: {
    facts: [
      { k: 'Free plan', v: '60 min/mo', s: '720p, watermark, exports up to 10 min' },
      { k: 'Paid from', v: '$14.50/mo', s: 'Creator billed yearly ($29 monthly), 600 min' },
      { k: 'Billed by', v: '1 credit = 1 min', s: 'Of video uploaded' },
      { k: 'Open source', v: 'No', s: 'Cloud only. OpenShorts: MIT, self-hostable' },
    ],
    pick: {
      them: ['You will hand-correct every clip in a timeline', 'You process up to 600 minutes a month and can pay yearly', 'You want subtitle translation in many languages'],
      us: ['You want clips without sitting in an editor', 'You want to automate it from n8n, an API or Claude', 'You want it free and self-hosted, or monthly from $12'],
    },
    edge: 'Vizard has the better editor for fixing each clip by hand.',
    name: 'Vizard',
    checked: '2026-10-05',
    entryPrice: '$14.50/month billed yearly ($29 monthly)',
    // "vizard ai" is the spelling most of its search volume uses, so the page
    // answers the brand question explicitly instead of hoping the comparison
    // table catches the query on its own.
    // The brand questions ("what is it", "is it free") are answered by the
    // /vizard-ai review; this page is the switch decision.
    seo: {
      title: 'Vizard AI Alternative: Open Source, Free to Self-Host',
      description:
        'Leaving Vizard AI? OpenShorts is MIT and free to self-host, or $12/month hosted. Vizard starts at $14.50/month yearly. When to switch, when to stay, how.',
      h1: 'The open source Vizard AI alternative: when to switch, and how',
    },
    tldr: [
      'OpenShorts is the open source alternative to Vizard AI: free under MIT when you self-host it, or hosted from $12/month billed monthly. Vizard starts at $14.50/month billed yearly, $29 month to month.',
      'The switch is about workflow more than features. Vizard proposes clips and expects you in its timeline; OpenShorts runs to finished clips on its own, with two-speaker, screencast and webcam-inset layouts.',
      'Stay on Vizard if you edit every clip by hand or clip a few hundred minutes a month on a yearly plan. Switch if you want clips without the editor, automation, or self-hosting.',
    ],
    brandAlias: 'also written Vizard AI',
    brandBlurb:
      'Vizard (written Vizard AI in most searches, and served from vizard.ai) is a browser-based AI clip maker: it transcribes the video, finds candidate moments and hands you a timeline to fix them in. That transcription pass is also its video-to-text feature, so a transcript comes out of the same job as the clips. Lately it has added an AI agent and an AI Studio for generated video around the clipper.',
    // Asked on the comparison page: the switch questions.
    extraFaq: [
      {
        q: 'Is Vizard AI cheaper than OpenShorts?',
        a: 'At volume on yearly billing, yes: Vizard Creator is $14.50/month for 600 minutes, against $29/month for 300 minutes or $59/month for 750 on OpenShorts Cloud. OpenShorts is cheaper month to month at low volume ($12 for 100 minutes against Vizard\'s $29) and costs nothing self-hosted, with no watermark and no cap.',
      },
      {
        q: 'Can I move my Vizard projects to OpenShorts?',
        a: 'Not as projects. Re-run the original YouTube link or file in OpenShorts and the clips are regenerated from the source. Download anything you want to keep from Vizard first: free-plan files are stored for 3 days, and SRT transcripts need a paid plan.',
      },
    ],
    // Asked on the /vizard-ai review: the brand questions.
    brandFaq: [
      {
        q: 'Is Vizard AI free?',
        a: 'Partly. The free plan gives 60 credits a month (one credit is one minute of uploaded video), uploads up to 60 minutes, and 720p exports of up to 10 minutes with a watermark, kept for 3 days. Paid plans start at $29/month, or $14.50/month billed yearly. OpenShorts self-hosted is free with no watermark and no cap; OpenShorts Cloud clips your first video free up to 60 minutes, then 20 minutes a month.',
      },
      {
        q: 'What is Vizard AI used for?',
        a: 'Turning long recordings into short vertical clips, with an emphasis on editing them afterwards: it finds the moments, transcribes them and gives you a timeline for the captions and the clip boundaries. It is also used as a video-to-text tool, because the transcript comes out of the same job.',
      },
    ],
    tiers: [
      ['Free', '60 credits a month (1 credit = 1 minute uploaded), uploads up to 60 minutes, 720p exports up to 10 minutes with a watermark, 3-day storage, transcript as TXT only'],
      ['Creator', '$29/month, or $14.50/month billed yearly: from 600 credits a month, 4K exports of any length, no watermark, scheduling, TXT and SRT transcripts, API'],
      ['Business', '$39/month, or $19.50/month billed yearly: shared workspace, 20 social accounts, team seats at $5/month each, brand kit'],
    ],
    gotcha:
      'Vizard is built around a browser editing timeline, so it sits between a pure auto-clipper and a full editor. That is useful if you intend to hand-adjust every clip, and overhead if you do not. Like Opus Clip, it bills a credit per minute of video uploaded, whatever the number of clips.',
    strengths: [
      'Genuinely usable in-browser editor after the AI pass',
      'Strong multi-language subtitles and subtitle translation, with transcripts it says cover 180+ languages',
      'Generous paid allowance: 600 credits a month on the entry plan, with the API included',
    ],
    whereWeDiffer: [
      'OpenShorts runs the whole pipeline unattended and is designed to be scripted or scheduled; Vizard expects a human in the timeline.',
      'OpenShorts is open source and self-hostable; Vizard is a closed cloud product.',
      'Vizard\'s editor is better than ours if you want to hand-correct each clip before posting.',
    ],
    fit:
      'Vizard is an AI pass followed by an editor; OpenShorts is an AI pass that finishes on its own. If you were going to accept most of Vizard\'s suggestions anyway, the timeline is time you no longer need to spend.',
    watermark:
      'Vizard watermarks free-plan exports, which also stop at 720p and 10 minutes; Creator from $14.50/month yearly ($29 monthly) removes it.',
    stay: [
      'You hand-correct every clip in a timeline and that is the part of the job you enjoy.',
      'You need subtitle translation and transcripts in many languages; Vizard says it covers 180+.',
      'You clip up to 600 minutes a month and can pay yearly: $14.50/month for that allowance is cheaper per minute than OpenShorts Cloud.',
    ],
    migrate: [
      'Download what you want to keep first: free-plan files are stored for 3 days.',
      'Pull transcripts as SRT while you are still on a paid plan; the free plan exports TXT only.',
      'Re-run the source in OpenShorts from the same YouTube link or file. There is no project import, and none is needed: the clips are regenerated from the source.',
      'Touch only the clips that need it. OpenShorts has an editor, but the pipeline is built to finish without one.',
      'Move the recurring part to the API, n8n or the MCP server if you were calling Vizard\'s API from a script.',
    ],
    // Review-only content for /vizard-ai, kept out of the comparison page so the
    // two answer different questions.
    reviewPage: {
      lede: 'vizard.ai',
      steps: [
        'Upload a file or paste a link; paid plans take sources of up to 600 minutes and 30 GB.',
        'Vizard transcribes the video, which is also its video-to-text output.',
        'The AI proposes candidate clips and reframes them to vertical.',
        'You review them in a browser timeline: captions, emoji and keyword highlights, cut points.',
        'Export, or publish and schedule to connected social accounts.',
      ],
      cons: [
        'A tight free plan: 60 minutes of source a month, 720p, exports capped at 10 minutes, a watermark on every export and 3-day storage.',
        'Billed per uploaded minute: a 60-minute recording costs 60 credits whatever it yields.',
        'The timeline is the workflow. If you do not want to hand-edit, it is overhead.',
        'Closed source and cloud only: the video has to be uploaded to Vizard.',
      ],
      for: [
        'Editors who want an AI first pass and then full control of every clip',
        'Teams that need subtitles and transcripts translated into many languages',
        'Anyone clipping a few hundred minutes a month who can commit to yearly billing',
      ],
      notFor: [
        'People who want clips without opening an editor',
        'Pipelines that must run unattended on a schedule',
        'Anyone whose footage cannot leave their own machine',
      ],
      verdict:
        'Vizard AI is a solid clipper with a genuinely good editor, priced well on yearly billing. Its free plan is for trying it, not for publishing, and its value depends on you wanting to edit. If you would rather the clips come out finished, or need to self-host, an unattended clipper fits better.',
    },
    reviewFaq: [
      {
        q: 'What are Vizard AI\'s main limitations?',
        a: 'On the free plan: 60 credits a month, 720p exports capped at 10 minutes, a watermark and 3-day storage, with transcripts as TXT only. On every plan: billing per uploaded minute, a workflow built around editing each clip in a timeline, and no self-hosting. Checked October 2026.',
      },
      {
        q: 'What are the best Vizard AI alternatives?',
        a: 'It depends on why you are leaving. For an open source tool you can self-host, OpenShorts (MIT, free with no watermark or cap when self-hosted, hosted from $12/month). For caption design, Submagic. For the fastest link-to-clips path, Klap. For per-minute credits with a bigger caption library, Opus Clip.',
      },
    ],
    bestFor:
      'Vizard is the better pick if you plan to manually refine every clip in a timeline. OpenShorts is the better pick for volume, automation, or self-hosting.',
    rows: { 'Usage cap': 'Metered in credits (1 per uploaded minute)' },
  },
  submagic: {
    facts: [
      { k: 'Free plan', v: 'None', s: 'A trial with no credit card' },
      { k: 'Paid from', v: '$12/mo', s: 'Starter billed yearly ($19 monthly)' },
      { k: 'Billed by', v: 'Videos', s: '15 to 100 a month, each capped at 2-30 min' },
      { k: 'Open source', v: 'No', s: 'Cloud only. OpenShorts: MIT, self-hostable' },
    ],
    pick: {
      them: ['Captions are your product and you want the best designs', 'Your videos are short', 'You want B-roll, zooms and silence removal in one editor'],
      us: ['You start from long recordings: podcasts, streams, webinars', 'You need two speakers or a screen kept in frame', 'You want a free plan, or to self-host with no cap'],
    },
    edge: 'Submagic has the more refined caption designs.',
    name: 'Submagic',
    checked: '2026-10-05',
    // Short enough to survive inside a 160-character meta description; the
    // annual/monthly spread is spelled out in the tiers below.
    entryPrice: '$12/month billed yearly',
    tiers: [
      ['Free', 'No free plan. A trial with no credit card is offered instead.'],
      ['Starter', '$19/month, or $12/month billed yearly: 15 videos a month, up to 2 minutes each, 1080p, no watermark'],
      ['Pro', '$39/month, or $23/month billed yearly: 40 videos a month, up to 5 minutes each, auto zooms, silence removal, social publishing'],
      ['Business', '$69/month per member, or $41/month billed yearly: 100 videos a month, up to 30 minutes each, 4K at 60 fps, 100 API minutes'],
    ],
    gotcha:
      'Submagic is captions-first. It now has Magic Clips, which cuts a long video into clips, but every plan is metered in videos with a per-video length cap (2, 5 or 30 minutes) and the pricing page does not publish a separate allowance for Magic Clips. Check that an hour-long source fits your plan before you pay.',
    strengths: [
      'The best caption styling in this category, by a clear margin',
      'A full short-form editor around the captions: B-roll, auto zooms, hook titles, silence and bad-take removal',
      'Magic Clips now turns a long video or a YouTube link into clips inside the same tool',
    ],
    whereWeDiffer: [
      'OpenShorts was built clipping-first: moment scoring, scene detection and 9:16 reframing with face tracking, two-speaker and screencast layouts. In Submagic clipping is one feature of a caption editor.',
      'Submagic\'s caption designs are more refined than ours. If captions are the entire reason you are shopping, it is the stronger tool.',
      'OpenShorts self-hosted has no per-video cap and no length limit; every Submagic tier is metered in videos and caps each one\'s length.',
      'OpenShorts has a free plan you can keep; Submagic offers a trial and no free plan.',
    ],
    tldr: [
      'Submagic is a caption editor that added clipping (Magic Clips); OpenShorts is a clipper built for long recordings. Submagic has no free plan and starts at $12/month billed yearly, $19 month to month.',
      'Every Submagic plan counts videos and caps each one at 2, 5 or 30 minutes. OpenShorts is free and uncapped when self-hosted under MIT, or hosted from $12/month by the source minute.',
      'Stay on Submagic if captions are the product and your videos are short. Move the long-form clipping to OpenShorts if your sources are podcasts, streams or webinars.',
    ],
    fit:
      'Submagic and OpenShorts overlap where Magic Clips begins: turning a long video into clips. Submagic arrived there from captions, OpenShorts was built for it, and the meters show it: videos with a length cap on one side, source minutes or nothing at all on the other.',
    watermark:
      'Submagic has no free plan to watermark: there is a trial, and Starter ($12/month yearly, $19 monthly) exports without a watermark.',
    stay: [
      'Captions are the product, and its caption designs are the most refined in this category.',
      'Your videos fit its per-video caps: 2 minutes on Starter, 5 on Pro, 30 on Business.',
      'You want B-roll, auto zooms, hook titles and silence removal in the same editor as the captions.',
    ],
    migrate: [
      'Check your source lengths first: any recording over 30 minutes does not fit a single Submagic video on any self-serve plan, and it is a normal input for OpenShorts.',
      'Move the long-form clipping to OpenShorts and judge the caption style on one episode: the first video is clipped free up to 60 minutes.',
      'Compare the meter: Submagic counts videos a month (15, 40 or 100); OpenShorts Cloud counts source minutes, and the self-hosted edition counts nothing.',
      'If caption design is non-negotiable for a short, Submagic\'s Starter plan covers 15 videos of up to 2 minutes for $12/month billed yearly.',
    ],
    // "sub magic" and "submagic.co" are how part of the brand search is typed.
    extraFaq: [
      {
        q: 'Is Sub Magic the same as Submagic (submagic.co)?',
        a: 'Yes. Sub Magic is a common spelling of Submagic, the caption editor at submagic.co, which also clips long videos with Magic Clips. Plans start at $12/month billed yearly ($19 month to month), with a trial and no free plan, checked October 2026.',
      },
    ],
    bestFor:
      'Submagic is the better pick if captions are the product and your videos are short. OpenShorts is the better pick if you are starting from long-form video and want the moment finding and reframing to be the core of the tool, or want it free and self-hosted.',
    rows: { 'Usage cap': 'Metered in videos per month, each capped at 2-30 min' },
  },
  'vidyo-ai': {
    facts: [
      { k: 'Free plan', v: '75 min/mo', s: '720p, data kept 7 days' },
      { k: 'Paid from', v: '$19/mo', s: 'Lite billed yearly ($29 monthly)' },
      { k: 'Billed by', v: '~1 credit = 1 min', s: 'Yearly plans double the credits' },
      { k: 'Renamed', v: 'Jan 2025', s: 'vidyo.ai redirects to quso.ai' },
    ],
    pick: {
      them: ['You want clipping, scheduling and analytics in one app', 'You can commit to a yearly plan', 'You want a content planner for your whole calendar'],
      us: ['Clipping quality and control are the job', 'You want to pay monthly from $12, or $0 self-hosted', 'You need two-speaker, screencast or webcam-inset layouts'],
    },
    published: '2026-10-05',
    edge: 'Quso adds a content planner and analytics, which we do not have.',
    name: 'Quso',
    // The search is still "vidyo ai": the brand changed, the queries did not.
    seo: {
      title: 'Vidyo.ai Is Now Quso: Pricing & Alternative',
      description:
        'Vidyo.ai became Quso.ai in January 2025. What changed, what Quso costs in 2026 (from $19/month yearly), and a free, open source alternative.',
      h1: 'Vidyo.ai is now Quso: what changed, what it costs, and the open source alternative',
      breadcrumb: 'Vidyo.ai (Quso)',
    },
    checked: '2026-10-05',
    entryPrice: '$19/month billed yearly ($29 monthly)',
    brandAlias: 'formerly Vidyo.ai',
    brandBlurb:
      'Quso is the new name of Vidyo.ai. The rebrand went live in January 2025, vidyo.ai now answers with a permanent redirect to quso.ai, and logins, plans, credits and projects carried over unchanged. What did change is the pitch: Quso sells itself as an all-in-one suite for creating, editing, scheduling and analysing social content, with the AI clipper as one part of it.',
    extraFaq: [
      {
        q: 'What happened to Vidyo.ai?',
        a: 'It was renamed Quso.ai in January 2025. vidyo.ai redirects to quso.ai, and accounts, plans and credits carried over. Quso added social scheduling, a content planner with analytics and a brand kit around the original clipper.',
      },
      {
        q: 'Is Vidyo.ai (Quso) free?',
        a: 'Quso has a free plan with 75 credits a month (about 75 minutes of video), 720p exports and 7-day data retention. Paid plans start at $19/month billed yearly or $29 month to month. OpenShorts self-hosted is free with no cap and no watermark; OpenShorts Cloud clips your first video free up to 60 minutes, then 20 minutes a month.',
      },
    ],
    tiers: [
      ['Free', '75 credits a month (one credit is about one minute of video), 720p, data kept 7 days, no scheduling'],
      ['Lite', '$29/month, or $19/month billed yearly: 100 credits (200 on yearly), 1080p, desktop editor'],
      ['Essential', '$39/month, or $26/month billed yearly: 300 credits (600 on yearly), scheduling, filler and silence removal, content planner'],
      ['Growth', '$49/month, or $33/month billed yearly: 600 credits (1,200 on yearly), brand kit, AI assistant, analytics'],
    ],
    gotcha:
      'Quso bills credits, and a credit is roughly a minute of processed video, so a 60-minute podcast uses 60 of them whatever it yields. The yearly plans double the credits, which makes month-to-month billing about three times as expensive per minute.',
    strengths: [
      'Scheduling, a content planner and analytics in the same app, which a pure clipper does not have',
      'A free plan that is larger than most in minutes',
      'A mature product: the Vidyo.ai clipper has been on the market since before most of this category existed',
    ],
    whereWeDiffer: [
      'OpenShorts is MIT-licensed and self-hostable; Quso is a closed cloud service.',
      'OpenShorts is a clipper first: moment scoring, face-tracked reframing, two-speaker and screencast layouts, dubbing into 30+ languages with your own ElevenLabs key. Quso spreads across scheduling and planning too.',
      'OpenShorts Cloud starts at $12/month billed monthly; Quso\'s cheapest paid plan is $29 month to month or $19/month on a yearly commitment.',
      'If you want one app that also plans and schedules your whole social calendar, Quso covers more ground than we do.',
    ],
    tldr: [
      'Vidyo.ai is now Quso: the rebrand went live in January 2025, vidyo.ai redirects to quso.ai, and accounts carried over. The best alternatives today are OpusClip, Vizard and OpenShorts, the open source option: free to self-host under MIT, or $12/month hosted for 100 minutes with no watermark. Quso has a free plan of 75 credits a month at 720p and paid plans from $19/month billed yearly ($29 monthly).',
      'Quso grew into a social suite (scheduling, a content planner, analytics) around the original clipper. OpenShorts stays a clipper: two-speaker, screencast and webcam-inset layouts, and dubbing into 30+ languages with your own ElevenLabs key.',
      'OpenShorts is open source: free when you run it yourself, or hosted from $12/month billed monthly.',
    ],
    fit:
      'Quso kept the Vidyo.ai clipper and built a social suite around it. If the clipper is the part you use, OpenShorts replaces that part and leaves the scheduling to whatever you already use, or posts straight to TikTok, Reels and Shorts itself.',
    watermark:
      'Quso\'s pricing page lists 75 free credits a month at 720p without stating a watermark either way; OpenShorts says it plainly: free Cloud clips are watermarked, self-hosted and paid ones are not.',
    stay: [
      'You use the content planner, scheduling and analytics, which a clipper does not replace.',
      'You can commit to a yearly plan: it doubles the credits.',
      'Its 75-minute free plan already covers what you clip each month.',
    ],
    migrate: [
      'Download what you need: free-plan data is kept 7 days.',
      'Compare per minute on the billing you would actually use. Quso\'s yearly plans double the credits, so month to month it costs about three times as much per minute.',
      'Run one long source through OpenShorts (first video free up to 60 minutes) and check the two-speaker and screencast layouts on your footage.',
      'Decide where scheduling lives: OpenShorts posts directly to TikTok, Instagram Reels and YouTube Shorts, but it has no content planner or analytics.',
    ],
    bestFor:
      'Quso is the better pick if you want clipping, scheduling and analytics in one subscription. OpenShorts is the better pick if clipping quality and control are the job, or you want it free and self-hosted.',
    rows: {
      'AI voice dubbing, 30+ languages': 'Not listed on its pricing page',
      'AI UGC video with lip-synced actors': 'Not listed on its pricing page',
      'Free AI YouTube thumbnail & title studio': 'Not listed on its pricing page',
      'Usage cap': 'Metered in credits (~1 per minute of video)',
    },
  },
  '2short': {
    facts: [
      { k: 'Free plan', v: '30 min/mo', s: 'No watermark, YouTube links only' },
      { k: 'Paid from', v: '$9.90/mo', s: 'Lite, 5 hours of analysis' },
      { k: 'Input', v: 'Links', s: 'YouTube; Drive and URLs from Lite' },
      { k: 'Open source', v: 'No', s: 'Cloud only. OpenShorts: MIT, self-hostable' },
    ],
    pick: {
      them: ['Your sources are already on YouTube', 'You want the lowest price per hour', 'One face-tracked crop is enough for your footage'],
      us: ['You upload your own recordings', 'You clip podcasts with two people on camera, or screen recordings', 'You want to self-host, or dub clips into 30+ languages with your ElevenLabs key'],
    },
    published: '2026-10-05',
    name: '2short.ai',
    seo: {
      title: '2short AI: Pricing, Free Plan & Alternative',
      description:
        '2short.ai turns YouTube videos into shorts from $9.90/month, with a free 30-minute plan. How it compares with OpenShorts, the open source alternative.',
      h1: '2short AI: what it does, what it costs, and the open source alternative',
    },
    checked: '2026-10-05',
    entryPrice: '$9.90/month',
    brandAlias: 'also written 2short AI',
    brandBlurb:
      '2short.ai is a web app that turns long YouTube videos into Shorts, TikToks and Reels. It finds the moments, keeps the speaker centred with face tracking ("center stage"), adds animated subtitles and brand presets, and exports vertical, square or horizontal. On the free plan it takes YouTube links only; paid plans add Google Drive and public URL imports.',
    extraFaq: [
      {
        q: 'Is 2short AI free?',
        a: 'It has a free Starter plan with 30 minutes of AI analysis a month, YouTube links only, and no watermark. Paid plans run from $9.90/month (5 hours) to $49.90/month (50 hours). OpenShorts self-hosted is free with no cap; OpenShorts Cloud clips your first video free up to 60 minutes, then 20 minutes a month.',
      },
      {
        q: 'Can 2short AI clip a video I recorded myself?',
        a: 'Its pricing page lists YouTube links on every plan and Google Drive or public URLs from the Lite plan up; uploading a local file is not listed there. OpenShorts accepts a YouTube link or a direct upload on every plan, including the free one.',
      },
    ],
    tiers: [
      ['Starter (free)', '30 minutes of AI analysis a month, YouTube links only, no watermark, 1080p'],
      ['Lite', '$9.90/month: 5 hours of analysis, 60 minutes of exports on fast servers, Google Drive and URL import'],
      ['Pro', '$19.90/month: 15 hours of analysis, unlimited exports'],
      ['Premium', '$49.90/month: 50 hours of analysis, unlimited exports, priority support'],
    ],
    gotcha:
      'The meter is hours of video analysed, and the input is a link: YouTube on every plan, Google Drive and public URLs from Lite. If your recordings live on your disk rather than on YouTube or Drive, check the import path before you subscribe.',
    strengths: [
      'Cheap per hour: 5 hours of source a month for $9.90 is one of the lowest prices in this category',
      'No watermark even on the free plan, with 1080p exports',
      'A focused YouTube-to-Shorts workflow with face tracking and brand presets',
    ],
    whereWeDiffer: [
      'OpenShorts is MIT-licensed and self-hostable; 2short.ai is a closed cloud service.',
      'OpenShorts takes a local upload on every plan; 2short.ai lists link imports (YouTube, Drive, URL).',
      'OpenShorts adds layouts a single centred crop cannot do: two speakers stacked, screen recordings over the presenter, a webcam inset enlarged. It also dubs into 30+ languages with your own ElevenLabs key.',
      '2short.ai is cheaper per hour of source than OpenShorts Cloud and keeps its free exports unwatermarked, which ours are not. That is a real advantage.',
    ],
    fit:
      '2short.ai is the cheapest per hour and does one thing well: a face-tracked crop from a YouTube link. OpenShorts is the alternative when the footage is not that simple, or not on YouTube.',
    watermark:
      '2short.ai does not watermark, even on its free plan; that is a real advantage over our hosted free plan, which does.',
    stay: [
      'Your sources are already on YouTube and one centred, face-tracked crop is enough.',
      'Price per hour decides it: 5 hours of analysis for $9.90/month is hard to beat.',
      'You want unwatermarked free exports, which its free plan gives and our hosted free plan does not.',
    ],
    migrate: [
      'Upload recordings straight from disk: OpenShorts takes a local file on every plan, so there is no need to put them on YouTube or Drive first.',
      'Try a two-person episode or a screen recording first; that is where the layouts differ most from a single crop.',
      'Size it in minutes: 2short.ai meters hours analysed, OpenShorts Cloud meters minutes of source, from 100 minutes for $12/month.',
      'Self-host if the watermark on our free plan is the issue: the MIT edition never adds one.',
    ],
    bestFor:
      '2short.ai is the better pick if your sources are YouTube videos, you want the lowest price per hour and a single face-tracked crop is enough. OpenShorts is the better pick for uploads, podcasts with two people on camera, screen recordings, or a free self-hosted setup.',
    tldr: [
      '2short.ai is a YouTube-to-Shorts web app from $9.90/month, with a free plan of 30 minutes a month and no watermark. OpenShorts is open source: free when you run it yourself, or hosted from $12/month.',
      'Both find the moments and reframe them to 9:16 with face tracking. OpenShorts adds two-speaker, screencast and webcam-inset layouts, dubbing into 30+ languages with your own ElevenLabs key, and direct uploads on every plan.',
      'Pick 2short.ai for the lowest price per hour on YouTube sources. Pick OpenShorts for uploads, multi-person footage, self-hosting or changing the pipeline.',
    ],
    rows: { 'Usage cap': 'Metered in hours of analysis per month' },
  },
  sendshort: {
    facts: [
      { k: 'Free plan', v: 'None', s: '3 free videos at sign-up' },
      { k: 'Clipping from', v: '$29/mo', s: 'Professional ($23 yearly); Starter has none' },
      { k: 'Billed by', v: 'Shorts', s: '20, 50 or unlimited a month' },
      { k: 'Open source', v: 'No', s: 'Cloud only. OpenShorts: MIT, self-hostable' },
    ],
    pick: {
      them: ['You want faceless, avatar and prompt-to-video shorts too', 'You want captions translated into 100+ languages', 'One subscription for many formats matters more than depth'],
      us: ['The job is clipping your own long recordings', 'You want clipping on every plan, the free one included', 'You want to self-host or keep the cost near zero'],
    },
    published: '2026-10-05',
    name: 'SendShort',
    seo: {
      title: 'SendShort AI: Pricing & Open Source Alternative',
      description:
        'SendShort costs $19 to $59 a month and only clips long videos from the $29 plan up. What you get on each plan, and a free, open source alternative.',
      h1: 'SendShort AI: plans, limits, and the open source alternative',
    },
    checked: '2026-10-05',
    entryPrice: '$19/month ($15 billed yearly)',
    brandAlias: 'sendshort.ai',
    brandBlurb:
      'SendShort is a short-form video suite rather than a pure clipper. It cuts long videos into shorts, and also makes faceless videos from a text prompt, AI avatar videos, music videos and TikTok slideshows, with word-by-word captions, caption translation, AI voice-over, a scheduler for TikTok and YouTube, and an API.',
    extraFaq: [
      {
        q: 'Is SendShort free?',
        a: 'There is no free plan: SendShort gives 3 free videos at sign-up and offers a refund within 24 hours of paying. Plans cost $19, $29 and $59 a month, or $15, $23 and $47 billed yearly, and clipping long videos starts on the $29 Professional plan. OpenShorts self-hosted is free; OpenShorts Cloud clips your first video free up to 60 minutes, then 20 minutes a month.',
      },
    ],
    tiers: [
      ['Free', 'No free plan. 3 free videos at sign-up, refund within 24 hours of paying.'],
      ['Starter', '$19/month, or $15/month billed yearly: 20 shorts a month up to 90 seconds, 1080p, no watermark. Long-video clipping is not included.'],
      ['Professional', '$29/month, or $23/month billed yearly: 50 shorts a month up to 3 minutes, long-video clipping, avatars, auto-translate, AI hooks, B-roll, voice-over'],
      ['Business', '$59/month, or $47/month billed yearly: unlimited shorts up to 10 minutes, 4K at 60 fps'],
    ],
    gotcha:
      'The entry price is not the clipping price. Turning a long video into clips starts on the Professional plan at $29/month ($23 billed yearly); Starter is for shorts made from prompts and templates.',
    strengths: [
      'Covers more formats than a clipper: faceless prompt-to-video, avatars, slideshows and music videos',
      'Captions and caption translation in 100+ languages, plus AI voice-over',
      'An API and an agent interface for Claude, ChatGPT and Grok',
    ],
    whereWeDiffer: [
      'OpenShorts is MIT-licensed and self-hostable; SendShort is a closed cloud service.',
      'OpenShorts clips long videos on every plan, the free one included; SendShort starts long-video clipping at $29/month.',
      'OpenShorts reframes with face tracking plus two-speaker, screencast and webcam-inset layouts, which is the hard part of clipping a podcast or a stream.',
      'SendShort makes more kinds of short video than we do. If faceless or prompt-to-video content is the plan, it covers that and we do not.',
    ],
    fit:
      'SendShort is a suite of short-video formats, and clipping long videos is one of them, from its $29 plan. OpenShorts is only the clipping, on every plan, the free one included.',
    watermark:
      'SendShort has no free plan; its Starter plan ($19/month, $15 yearly) exports without a watermark but does not include long-video clipping.',
    stay: [
      'You make faceless, avatar or prompt-to-video shorts, which OpenShorts does not.',
      'You need captions translated into 100+ languages with AI voice-over.',
      'One subscription for many formats matters more to you than depth in clipping.',
    ],
    migrate: [
      'If you subscribed only to clip long videos, note that SendShort offers a refund within 24 hours of paying.',
      'Check which plan you are really on: long-video clipping starts on Professional ($29/month, $23 yearly), not Starter.',
      'Clip the same long recording in OpenShorts; the free plan clips your first video whole up to 60 minutes.',
      'Keep SendShort for the faceless and avatar formats if you use them, and move the clipping alone.',
    ],
    bestFor:
      'SendShort is the better pick if you want one subscription for faceless videos, avatars and clips. OpenShorts is the better pick if the job is turning your own long recordings into clips, cheaply or self-hosted.',
    tldr: [
      'SendShort is a short-form video suite from $19/month ($15 yearly), but clipping long videos starts on the $29/month plan. OpenShorts is open source: free when you run it yourself, or hosted from $12/month with clipping on every plan.',
      'SendShort covers faceless videos, avatars and caption translation into 100+ languages. OpenShorts goes deeper on the clipping itself: moment scoring, face tracking, two-speaker and screencast layouts, dubbing into 30+ languages.',
      'Pick SendShort for many formats in one tool. Pick OpenShorts to clip your own long recordings, self-host, or keep the cost down.',
    ],
    rows: {
      'AI voice dubbing, 30+ languages': 'Caption translation and AI voice-over ($29 plan and up)',
      'AI UGC video with lip-synced actors': 'AI avatar videos ($29 plan and up)',
      'Usage cap': 'Metered in shorts per month',
    },
  },
  // "choppity", "choppity pricing", "is choppity free": a brand search, so the
  // page is a review first (review: true), like Klap. Read on choppity.com.
  choppity: {
    facts: [
      { k: 'Free plan', v: '1 hour/mo', s: 'Logo on exports, projects kept 3 days' },
      { k: 'Paid from', v: '$24/mo', s: 'Starter, 3 hours; $16.80 billed yearly' },
      { k: 'Billed by', v: 'Hours analysed', s: 'Of source video, reset on the 1st' },
      { k: 'Open source', v: 'No', s: 'Cloud only. OpenShorts: MIT, self-hostable' },
    ],
    pick: {
      them: ['You want to fine-tune every clip by editing its transcript', 'You also record screen tutorials and schedule posts to six networks', 'One hour of source a month is all you clip'],
      us: ['You want to self-host so the footage never leaves your machine', 'You clip under 100 minutes a month and want a $12 plan billed monthly', 'You want dubbing into 30+ languages with your own ElevenLabs key'],
    },
    edge: 'Choppity has the deeper editor: transcript editing, profanity bleeping and a screen recorder.',
    name: 'Choppity',
    checked: '2026-10-29',
    published: '2026-10-29',
    entryPrice: '$24/month ($16.80 billed yearly)',
    seo: {
      title: 'Choppity Review 2026: Pricing, Free Plan & Alternative',
      description:
        'Choppity review, October 2026: a free hour a month with a logo, Starter at $24/month ($16.80 yearly) for 3 hours, the limits, and an open source alternative.',
      h1: 'Choppity review: pricing, the free plan, and the open source alternative',
      breadcrumb: 'Choppity',
    },
    review: true,
    brandBlurb:
      'Choppity (choppity.com) is a cloud AI clip maker aimed mostly at podcasters, educators and marketers. Upload a file or paste a YouTube link and it transcribes the video, proposes 20 to 50 or more short clips, reframes them to vertical while following the active speaker, and captions them; you then refine each clip by editing its transcript. Around the clipper it sells a screen recorder for tutorials and a social scheduler, and it exports 9:16, 1:1 and 16:9.',
    isFree:
      'Yes, with limits. The free plan needs no credit card and includes the clipper, the editor and captions, but it covers 1 hour of AI video analysis a month, deletes projects after 3 days and puts a small Choppity logo in the corner of every export. Removing the logo starts on Starter at $24/month, or $16.80/month billed yearly.',
    tiers: [
      ['Free', '1 hour of AI video analysis a month, 2 GB of storage, projects kept 3 days, one seat, with the clipper, editor and captions; exports carry a small Choppity logo. No credit card.'],
      ['Starter', '$24/month, or $16.80/month billed yearly ($201.60/year): 3 hours of analysis a month, 40 GB, projects kept 30 days, 1080p and 4K exports, scheduling to 1 social account'],
      ['Creator', '$32/month, or $22.40/month billed yearly: 5 hours a month, 100 GB, projects kept with no expiry, bulk uploads of up to 10 links or files, 3 social accounts, custom fonts, a brand template and API access. More hours come in 5-hour packs at $32 each, up to 100 hours.'],
      ['Team', '$48/month, or $33.60/month billed yearly: 5 hours a month, 200 GB, unlimited seats, a shared workspace with roles, 6 social accounts, 5 brand templates'],
      ['Enterprise', 'Custom pricing for bespoke usage, security review and white-label'],
    ],
    tierNote:
      'Yearly billing takes 30% off every plan. Usage resets to zero on the 1st of each calendar month, and a payment can be refunded if you ask within 14 days.',
    gotcha:
      'The meter is hours of AI video analysis, counted on the source you upload rather than on the clips you keep: a 90-minute episode uses half of Starter\'s three hours however few clips you export. Usage resets on the 1st of the calendar month, so hours you do not use are gone.',
    strengths: [
      'Text-based editing: trim a clip by deleting words from its transcript, with timeline controls when you need them',
      'Active-speaker tracking for podcasts and interviews, keyword and topic search to clip one specific moment, and automatic profanity bleeping',
      'A screen recorder and a social scheduler (YouTube, TikTok, Instagram, LinkedIn, Facebook, Threads) in the same account, plus an API and an MCP server',
      'Captions and transcripts in 97 languages, it says, and uploads of up to 10 GB',
    ],
    whereWeDiffer: [
      'OpenShorts is MIT-licensed and runs on your own machine with Docker; Choppity is a closed cloud service, so every upload goes to its servers.',
      'Entry prices: OpenShorts Cloud is $12/month for 100 minutes, billed monthly; Choppity Starter is $24/month for 3 hours, or $16.80 on a yearly plan.',
      'Choppity\'s free plan is the more generous hosted one: a full hour a month, against 20 minutes on OpenShorts Cloud after a first video of up to 60 minutes. Both mark free exports.',
      'Choppity has the richer editor (transcript editing, profanity bleeping, a screen recorder). OpenShorts is built to finish clips without one, and adds dubbing into 30+ languages with your own ElevenLabs key.',
    ],
    fit:
      'Choppity and OpenShorts both start from a long recording and return captioned vertical clips. Choppity expects you to polish each clip in its transcript editor; OpenShorts is built to hand back finished clips and to run without you, on our servers or on yours.',
    watermark:
      'Choppity puts a small logo in the corner of free-plan exports; any paid plan, from $24/month ($16.80 yearly), removes it, and old clips can be re-exported clean.',
    // Review pages normally reuse `stay` as "who it is for"; Choppity gets both.
    forWhom: [
      'Podcasters and interview shows that want speaker tracking and to fine-tune each clip by editing words rather than a timeline',
      'Course creators who also record screen tutorials and want one tool for the recording and the clips',
      'Small teams that want clipping, a content calendar and posting to six networks under one subscription',
    ],
    stay: [
      'One hour of source a month is all you clip and a corner logo does not bother you: the free plan covers it.',
      'You publish to LinkedIn, Facebook or Threads as well, and want the scheduler next to the clipper.',
      'You pay yearly for Creator: $22.40/month for 5 hours is less per minute than OpenShorts Cloud\'s Creator plan at $24.17/month yearly for 300 minutes.',
    ],
    migrate: [
      'Download the clips you want to keep first: free projects are deleted after 3 days and Starter projects after 30.',
      'Use up the month\'s hours, then switch: usage resets on the 1st of each calendar month, and refunds are only offered within 14 days of a payment.',
      'Convert the meter. Choppity sells hours of analysis, OpenShorts Cloud sells minutes of source: Starter\'s 3 hours is 180 minutes, between our $12 (100 minutes) and $29 (300 minutes) plans.',
      'Re-run a recent episode from the same YouTube link or file; the first video is clipped free up to 60 minutes, so the side-by-side costs nothing.',
      'If a script called Choppity\'s API or MCP server, point it at the OpenShorts API or MCP server instead.',
    ],
    bestFor:
      'Choppity is the better pick if you want to shape every clip by hand in a transcript editor, also record tutorials, and schedule to many networks from one tool. OpenShorts is the better pick if you want the clips finished without the editing pass, to pay monthly from $12, or to keep the whole pipeline on your own machine.',
    tldr: [
      'Choppity is a closed-source cloud clipper for podcasts and long talking videos: it proposes clips, follows the active speaker and lets you edit each clip through its transcript. A screen recorder and a social scheduler come in the same account.',
      'It has a real free plan: 1 hour of analysis a month and no credit card, with a Choppity logo on exports and projects deleted after 3 days. Paid plans start at $24/month for 3 hours, or $16.80/month billed yearly.',
      'OpenShorts is the open source alternative: free under MIT when you self-host it, with no logo and no meter, or hosted from $12/month billed monthly.',
    ],
    extraFaq: [
      {
        q: 'How much does Choppity cost?',
        a: 'Checked 29 October 2026: Free at $0 (1 hour of analysis a month), Starter $24/month for 3 hours, Creator $32/month for 5 hours with API access, Team $48/month for 5 hours with unlimited seats, and custom Enterprise plans. Yearly billing takes 30% off: $16.80, $22.40 and $33.60 a month. OpenShorts Cloud starts at $12/month for 100 minutes, and self-hosted it costs nothing.',
      },
      {
        q: 'Does Choppity have an API?',
        a: 'Yes. API access starts on the Creator plan ($32/month, $22.40 billed yearly), and Choppity also offers an MCP server for AI assistants. OpenShorts has a REST API, webhooks and an MCP server, and the self-hosted edition includes all three for free.',
      },
    ],
    sources: [
      ['Choppity plans, limits, billing and refunds', 'https://www.choppity.com/pricing/'],
      ['Choppity on free-plan watermarks', 'https://www.choppity.com/tools/free-ai-clip-maker/ai-clip-maker-no-watermark/'],
      ['Choppity features and FAQ', 'https://www.choppity.com/'],
      ['Choppity languages and upload size', 'https://www.choppity.com/alternatives/opus-clip/'],
    ],
    rows: {
      'AI voice dubbing, 30+ languages': 'Not listed; captions in 97 languages, it says',
      'AI UGC video with lip-synced actors': 'Not listed',
      'Free AI YouTube thumbnail & title studio': 'Not listed',
      'Usage cap': 'Metered in hours of AI analysis a month',
    },
  },
  // "getmunch" and "getmunch alternative". getmunch.com now 301s to
  // munchstudio.com, an AI social-media service; the page says so first.
  getmunch: {
    facts: [
      { k: 'Now called', v: 'Munch Studio', s: 'getmunch.com redirects to munchstudio.com' },
      { k: 'Free plan', v: 'None', s: '7-day trial, then the plan bills' },
      { k: 'Paid from', v: '$38/mo', s: 'Billed yearly; $68 month to month' },
      { k: 'Open source', v: 'No', s: 'Cloud only. OpenShorts: MIT, self-hostable' },
    ],
    pick: {
      them: ['You want an AI to plan, write and post all of your social content, not only clips', 'You post images, carousels and generated videos too', 'You run several brands and want one workspace per client'],
      us: ['Clipping your own long videos is the whole job', 'You want a free plan, or to pay from $12/month rather than $38', 'You want the pipeline on your own machine, under MIT'],
    },
    edge: 'Munch Studio plans, writes and schedules a whole social calendar, which OpenShorts does not.',
    name: 'Munch Studio',
    checked: '2026-10-29',
    published: '2026-10-29',
    entryPrice: '$38/month billed yearly ($68 monthly)',
    seo: {
      title: 'GetMunch Review 2026: Now Munch Studio, Price & Alternative',
      description:
        'GetMunch is now Munch Studio, an AI social media service. October 2026: no free plan, a 7-day trial, from $38/month yearly, and a clipping alternative.',
      h1: 'GetMunch is now Munch Studio: what changed, what it costs, and a GetMunch alternative',
      breadcrumb: 'GetMunch (Munch Studio)',
    },
    review: true,
    brandBlurb:
      'Searches for getmunch or Munch AI land on munchstudio.com: the old getmunch.com address now redirects there. Munch Studio is sold as "the AI that does your social media marketing for you". It reads your website, videos and brand material, builds a content strategy, then generates posts, posters, carousels, video scripts and short videos and publishes them to Instagram, Facebook, LinkedIn, YouTube and TikTok, with a performance dashboard. Cutting long videos into clips is still there, as a monthly allowance of "video repurposing" minutes, and its help centre still documents clips tailored per platform with AI subtitles.',
    isFree:
      'No. There is no free plan, only a 7-day free trial, and its help centre says the trial turns into a paid subscription on the plan you picked unless you cancel before it ends. A monthly payment can be refunded within 7 days, and a yearly one within 14, only if you have not used the platform.',
    tiers: [
      ['Free', 'No free plan. A 7-day free trial that converts to the paid plan you chose unless you cancel before it ends.'],
      ['Essential', '$68/month, or $38/month billed yearly ($456/year): one brand, one user, 500 minutes of video repurposing a month, 10 generated videos, 5 video scripts a week, 100 GB'],
      ['Premium', '$85/month, or $60/month billed yearly ($720/year): one brand, unlimited team members, 1,000 minutes of video repurposing, 30 generated videos, unlimited scripts, 1 TB'],
      ['Studio', '$148/month: up to 5 brands, 2,500 minutes of video repurposing, 50 generated videos, 400 GB'],
      ['Scale', '$398/month: up to 15 brands, 7,500 minutes of video repurposing, 150 generated videos, 1 TB'],
    ],
    tierNote:
      'Every plan includes posting to Instagram, Facebook, LinkedIn, YouTube and TikTok, unlimited generated posters and carousels, a content planner and a performance dashboard. Studio and Scale are the multi-brand plans, quoted at the monthly prices in its plan comparison.',
    gotcha:
      'You cannot buy the clipping on its own. The cheapest way to get it is Essential, $38/month on a yearly commitment or $68 month to month, and that price also pays for post generation, scripts, generated videos and scheduling: if all you want is long videos cut into clips, most of the subscription goes to features you will not use. The help centre lists seven supported languages: English (US and UK), German, Spanish, Italian, Portuguese and Russian.',
    strengths: [
      'Done-for-you social media: it learns the brand from your website and material, writes a content strategy and fills the calendar',
      'Clips, posters, carousels, scripts and generated videos in one queue, approved and posted with one click to five networks',
      'Multi-brand plans with a separate workspace, strategy and calendar per client, aimed at agencies',
    ],
    whereWeDiffer: [
      'OpenShorts does one job, clipping long videos into vertical shorts, and costs $0 self-hosted or from $12/month hosted. Munch Studio sells a full social media service from $38/month on yearly billing.',
      'OpenShorts has a free plan you can keep using; Munch Studio has a 7-day trial that becomes a subscription.',
      'OpenShorts is MIT-licensed and can run on your own machine; Munch Studio is closed and cloud only.',
      'Munch Studio writes and schedules posts that are not videos, and plans the calendar for you. OpenShorts does not, and is not trying to.',
    ],
    fit:
      'If what you wanted from Munch was the clipping, OpenShorts does that part and nothing around it: give it a long video, get vertical clips with captions, post them to TikTok, Instagram or YouTube. If you want an AI running the whole social calendar, Munch Studio is a different product and OpenShorts does not replace it.',
    forWhom: [
      'Small businesses that want social media handled without writing or designing posts themselves',
      'Agencies running social for several clients, with one isolated workspace per brand',
      'Teams that want posts, carousels, scripts and clips planned and published from one calendar',
    ],
    stay: [
      'You use the strategy, posts and carousels as much as the clips, so you are paying for the whole service rather than for clipping alone.',
      'You manage several brands and the per-client workspaces on Studio or Scale save you a tool.',
      'You post to LinkedIn and Facebook as well as the short-video networks, from one approval queue.',
    ],
    migrate: [
      'Download the clips and posts you want to keep before the billing period ends: cancelling stops future charges and access lasts until then.',
      'If you are still inside the 7-day trial, cancel before it ends and nothing is charged.',
      'Count the repurposing minutes you actually use. A weekly one-hour show is about 260 minutes a month, which fits OpenShorts Cloud\'s $29 plan (300 minutes); Essential\'s full 500 fits the $59 plan (750).',
      'Run the same long video through OpenShorts; the first one is clipped free up to 60 minutes.',
      'Keep a scheduler for image posts and carousels if you relied on Munch Studio for those: OpenShorts posts only the clips, to TikTok, Instagram, YouTube, Facebook, LinkedIn, Threads and Bluesky.',
    ],
    bestFor:
      'Munch Studio is the better pick if you want an AI to run your social media end to end and clips are one ingredient. OpenShorts is the better pick if you searched for getmunch because you want a clipper: it does that job alone, with a free plan, monthly billing from $12, or $0 on your own machine.',
    tldr: [
      'GetMunch is now Munch Studio: getmunch.com redirects to munchstudio.com, which sells an AI that runs your social media marketing. The best alternatives today are OpusClip, Vizard and OpenShorts, the open source option: free to self-host under MIT, or $12/month hosted for 100 minutes with no watermark. Clipping long videos is still there, as an allowance of "video repurposing" minutes inside each plan.',
      'There is no free plan, only a 7-day trial that turns into a subscription. Plans start at $38/month billed yearly, or $68 month to month, for 500 minutes of video repurposing plus posts, scripts and generated videos.',
      'If you came for the clipper, OpenShorts is the open source GetMunch alternative: free under MIT on your own machine, or hosted from $12/month with a free plan you can keep.',
    ],
    extraFaq: [
      {
        q: 'Is Munch AI the same as GetMunch?',
        a: 'Yes. Munch AI is how most people search for Munch, the AI clipper that lived at getmunch.com. That address now redirects to munchstudio.com, and the product is sold as Munch Studio. Checked 29 October 2026.',
      },
      {
        q: 'What happened to GetMunch?',
        a: 'getmunch.com now redirects to munchstudio.com. The product is sold as Munch Studio, an AI social media marketing service, and clipping long videos continues inside it as minutes of "video repurposing" per plan. Checked 29 October 2026.',
      },
      {
        q: 'How much does Munch Studio cost?',
        a: 'Checked 29 October 2026: Essential $68/month or $38/month billed yearly (500 minutes of video repurposing), Premium $85/month or $60/month billed yearly (1,000 minutes), and the multi-brand Studio ($148/month) and Scale ($398/month) plans. There is a 7-day trial and no free plan. OpenShorts Cloud starts at $12/month for 100 minutes, and self-hosted it costs nothing.',
      },
    ],
    sources: [
      ['Munch Studio plans and plan comparison', 'https://www.munchstudio.com/pricing'],
      ['Munch Studio home page (getmunch.com redirects here)', 'https://www.munchstudio.com/'],
      ['Free trial and billing during the trial', 'https://help.munchstudio.com/en/articles/13114429-free-trial-demo-billing-during-trial'],
      ['Cancellations and refunds', 'https://help.munchstudio.com/en/articles/13114532-cancelations-and-refunds'],
      ['Supported languages', 'https://help.munchstudio.com/en/articles/11801418-supported-languages-in-munch-studio'],
      ['What Munch does with your videos', 'https://help.munchstudio.com/en/articles/8163720-what-does-munch-do-with-my-videos'],
    ],
    rows: {
      'Face-tracked 9:16 reframing': 'Vertical clips per platform, with a manual crop',
      'Word-level auto subtitles': 'AI-generated subtitles with templates',
      'AI voice dubbing, 30+ languages': 'Not listed; 7 supported languages',
      'AI UGC video with lip-synced actors': 'Generated videos, 10 to 150 a month; lip-sync not stated',
      'Free AI YouTube thumbnail & title studio': 'Not listed',
      'Usage cap': 'Metered in minutes of video repurposing a month',
    },
  },
}

/* Rows for the head-to-head table. `os` values are deliberately honest about the
 * hosted/self-hosted split rather than collapsing to "free". */
export const COMPARISON_ROWS = [
  {
    feature: 'Entry price',
    os: '$0 self-hosted · $0 hosted for 20 min/mo · $12/mo hosted without watermark',
    key: 'entryPrice',
  },
  { feature: 'Open source', os: 'Yes, MIT', vendor: 'No' },
  { feature: 'Self-hostable', os: 'Yes, Docker Compose', vendor: 'No, cloud only' },
  { feature: 'Source video stays on your machine', os: 'Yes when self-hosted', vendor: 'No' },
  { feature: 'AI viral moment detection', os: 'Yes, Gemini 3.1 Flash-Lite', vendor: 'Yes' },
  { feature: 'Face-tracked 9:16 reframing', os: 'Yes, MediaPipe + YOLOv8', vendor: 'Yes' },
  { feature: 'Word-level auto subtitles', os: 'Yes, faster-whisper', vendor: 'Yes' },
  { feature: 'AI voice dubbing, 30+ languages', os: 'Yes, with your own ElevenLabs key', vendor: 'No' },
  { feature: 'AI UGC video with lip-synced actors', os: 'Yes, from $0.65/video', vendor: 'No' },
  { feature: 'Free AI YouTube thumbnail & title studio', os: 'Yes when self-hosted · thumbnails on paid Cloud plans', vendor: 'No' },
  { feature: 'Usage cap', os: 'None when self-hosted · metered on Cloud', vendor: 'Metered on every tier' },
]

/* The ranked "Best <tool> alternatives" list at the top of every
 * /alternatives/* page. Engines answer "best X alternative" with a numbered
 * list, so the page leads with one. Each line is the tool's one honest edge and
 * its entry price, read from COMPETITORS above (same check dates): if a price
 * changes there, change the line here. Order is the order the list keeps once
 * the page's own tool is taken out. */
export const ALT_PICKS = [
  ['openshorts', 'OpenShorts', 'best free and open source option: $0 self-hosted (MIT, no watermark, no cap), or $12/month hosted for 100 minutes'],
  ['vizard', 'Vizard', 'best value at volume: 600 minutes for $29/month ($14.50 billed yearly)'],
  ['klap', 'Klap', 'fastest link-to-clips: from $14/month billed yearly for 100 clips'],
  ['opus-clip', 'Opus Clip', 'most complete cloud clipper: from $15/month for 150 minutes, 60 watermarked minutes a month free'],
  ['submagic', 'Submagic', 'best for animated captions: from $12/month billed yearly'],
  ['vidyo-ai', 'Quso (formerly Vidyo.ai)', 'largest free plan: 75 credits a month at 720p'],
  ['choppity', 'Choppity', 'best for editing clips through the transcript: from $16.80/month billed yearly'],
  ['2short', '2short.ai', 'cheapest YouTube-only plan: from $9.90/month, 30 free minutes a month'],
]

/* Third-party figures used across the pages. Every one is attributed inline,
 * because sourced statistics are among the few things that reliably survive
 * into a generated answer. */
export const CITED_STATS = [
  {
    claim: 'Short-form video delivers the highest ROI of any content format.',
    source: 'HubSpot, State of Marketing 2025',
  },
  {
    claim: '91% of businesses use video as a marketing tool.',
    source: 'Wyzowl, Video Marketing Statistics 2025',
  },
]

/* Plans as the cost calculator needs them: minutes of source video covered per
 * month and the monthly price, on monthly and on yearly billing. Only tools
 * that bill by source minutes are here; Klap (per clip), Submagic (per video)
 * and SendShort (per short) cannot be priced on the same axis, and pretending
 * otherwise would be the kind of number this file exists to avoid. */
export const PRICE_MODELS = {
  // The edition the hosted plans are a convenience over: same pipeline, your
  // machine, no meter. Shown as its own bar so the calculator never implies the
  // hosted price is the only OpenShorts price.
  'openshorts-self': {
    name: 'OpenShorts self-hosted',
    unit: 'your machine, MIT',
    plans: [{ name: 'Docker, no watermark, no cap', monthly: { minutes: 1e9, price: 0 } }],
  },
  openshorts: {
    name: 'OpenShorts Cloud',
    unit: 'per source minute, trims and hooks free',
    checked: '2026-10-05',
    plans: [
      { name: 'Free', watermark: true, monthly: { minutes: 20, price: 0 } },
      { name: 'Starter', monthly: { minutes: 100, price: 12 }, yearly: { minutes: 100, price: 10 } },
      { name: 'Creator', monthly: { minutes: 300, price: 29 }, yearly: { minutes: 300, price: 24.17 } },
      { name: 'Pro', monthly: { minutes: 750, price: 59 }, yearly: { minutes: 750, price: 49.17 } },
      // Pro plus top-ups of 200 minutes at $25.
      { name: 'Pro + 200 min', monthly: { minutes: 950, price: 84 }, yearly: { minutes: 950, price: 74.17 } },
      { name: 'Pro + 400 min', monthly: { minutes: 1150, price: 109 }, yearly: { minutes: 1150, price: 99.17 } },
      { name: 'Pro + 600 min', monthly: { minutes: 1350, price: 134 }, yearly: { minutes: 1350, price: 124.17 } },
      { name: 'Pro + 800 min', monthly: { minutes: 1550, price: 159 }, yearly: { minutes: 1550, price: 149.17 } },
    ],
  },
  'opus-clip': {
    name: 'Opus Clip',
    unit: '1 credit per source minute',
    checked: '2026-10-05',
    plans: [
      { name: 'Free', watermark: true, monthly: { minutes: 60, price: 0 } },
      { name: 'Starter', monthly: { minutes: 150, price: 15 } },
      { name: 'Pro', monthly: { minutes: 300, price: 29 }, yearly: { minutes: 300, price: 14.5 } },
      { name: 'Pro, 2 packs', monthly: { minutes: 600, price: 58 }, yearly: { minutes: 600, price: 29 } },
    ],
  },
  vizard: {
    name: 'Vizard',
    unit: '1 credit per uploaded minute',
    checked: '2026-10-05',
    plans: [
      { name: 'Free', watermark: true, monthly: { minutes: 60, price: 0 } },
      { name: 'Creator', monthly: { minutes: 600, price: 29 }, yearly: { minutes: 600, price: 14.5 } },
    ],
  },
  'vidyo-ai': {
    name: 'Quso (Vidyo.ai)',
    unit: '~1 credit per minute',
    checked: '2026-10-05',
    plans: [
      { name: 'Free (720p)', monthly: { minutes: 75, price: 0 } },
      { name: 'Lite', monthly: { minutes: 100, price: 29 }, yearly: { minutes: 200, price: 19 } },
      { name: 'Essential', monthly: { minutes: 300, price: 39 }, yearly: { minutes: 600, price: 26 } },
      { name: 'Growth', monthly: { minutes: 600, price: 49 }, yearly: { minutes: 1200, price: 33 } },
    ],
  },
  '2short': {
    name: '2short.ai',
    unit: 'hours of video analysed',
    checked: '2026-10-05',
    plans: [
      { name: 'Free', monthly: { minutes: 30, price: 0 } },
      { name: 'Lite', monthly: { minutes: 300, price: 9.9 } },
      { name: 'Pro', monthly: { minutes: 900, price: 19.9 } },
      { name: 'Premium', monthly: { minutes: 3000, price: 49.9 } },
    ],
  },
  // Hours of AI analysis of the uploaded source; Creator scales in 5-hour packs
  // at $32 each (30% off yearly). Munch Studio is left out: its minutes come
  // bundled with posts and generated videos, so a per-minute bar would mislead.
  choppity: {
    name: 'Choppity',
    unit: 'hours of AI analysis per month',
    checked: '2026-10-29',
    plans: [
      { name: 'Free', watermark: true, monthly: { minutes: 60, price: 0 } },
      { name: 'Starter', monthly: { minutes: 180, price: 24 }, yearly: { minutes: 180, price: 16.8 } },
      { name: 'Creator', monthly: { minutes: 300, price: 32 }, yearly: { minutes: 300, price: 22.4 } },
      { name: 'Creator, 10 h', monthly: { minutes: 600, price: 64 }, yearly: { minutes: 600, price: 44.8 } },
      { name: 'Creator, 15 h', monthly: { minutes: 900, price: 96 }, yearly: { minutes: 900, price: 67.2 } },
      { name: 'Creator, 20 h', monthly: { minutes: 1200, price: 128 }, yearly: { minutes: 1200, price: 89.6 } },
      { name: 'Creator, 25 h', monthly: { minutes: 1500, price: 160 }, yearly: { minutes: 1500, price: 112 } },
    ],
  },
}
