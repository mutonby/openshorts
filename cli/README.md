# OpenShorts CLI

Clip long videos into vertical 9:16 shorts from the terminal. Zero
dependencies; talks to the same API the dashboard, the MCP server and the
webhooks use.

```bash
pip install openshorts        # or: uvx openshorts / pipx run openshorts

export OPENSHORTS_API_KEY=osk_...   # from your account page at openshorts.app

openshorts process "https://youtube.com/watch?v=..." --wait
openshorts clips <job_id>
openshorts publish <job_id> 0 --platforms tiktok,youtube
openshorts quota
```

Self-hosted instance? Point it at your own machine and skip the key:

```bash
export OPENSHORTS_API_URL=http://localhost:8000
openshorts process "https://youtube.com/watch?v=..." --wait
```

For pipelines, prefer the webhook to `--wait`: pass `--webhook` and
`--webhook-secret` and OpenShorts POSTs once (HMAC-signed,
`X-OpenShorts-Signature: sha256=<hex>`) when the job ends, with clip titles
and durable download links.

The hosted free tier is 20 minutes/month with a watermark; paid plans from
$12/month. The self-hosted edition is MIT and has no meter. Agent-native
version of the same surface: [openshorts.app/mcp](https://www.openshorts.app/mcp).

## Trend-driven clips

From the repository checkout, install the current CLI with `pip install ./cli`.

```bash
openshorts trends --region ZA
openshorts --json trends --region ZA --topic "candy making" --days 7 --limit 5
```

The first command uses Google Trending Now's regional RSS feed. Its counts
measure Google searches. The second needs `YOUTUBE_DATA_API_KEY` and uses YouTube
Data API search and video metadata to find recent clips up to 180 seconds long,
ranked by public views. YouTube search uses the region for video availability.
Check the player to confirm the format, since YouTube also uses aspect ratio
to determine Shorts eligibility. [Google Trends RSS](https://support.google.com/trends/answer/3076011),
[YouTube search](https://developers.google.com/youtube/data/reference/rest/v3/search/list),
[Shorts eligibility](https://support.google.com/youtube/answer/15424877)

Create a JSON file with video URLs you own or have permission to clip. List
specific words or phrases that identify the subject. For long videos with
several subjects, set `source_start` and `source_end` in seconds to clip a
specific section.

```json
[
  {
    "url": "https://example.com/my-candy-tutorial.mp4",
    "topics": ["candy making", "hand-pulled sugar"],
    "profile": "process",
    "source_start": 60,
    "source_end": 240
  }
]
```

```bash
openshorts trend-generate sources.json --region ZA
openshorts trend-generate sources.json --region ZA --topic "candy making"
openshorts --json trend-generate sources.json --region ZA --submit --acknowledge
```

The first two commands preview matches. `--topic` accepts reviewed terms from
other sources and can be repeated. The third queues vertical clips with the
regular OpenShorts pipeline and automatic motion- and detail-based framing.
`--limit` sets the job cap per run (default one), and `--clips` sets the target
clip count per job (default three).

| Profile | Target length | Opening overlay |
| --- | --- | --- |
| `explainer` (default) | 20–45s | High-contrast pill |
| `process` | 15–40s | Footage starts unobstructed |
| `story` | 35–65s | Outlined hook |

Run the submit command from your scheduler to check for new matches. Use the
same `--state` SQLite path for every run against the same library and account,
or keep the default file beside the manifest. A URL and its selected time
range are queued once. If a response is lost, the claim stays in the history
and the command reports it as unconfirmed on the next matching run. Check
OpenShorts for the job before manually resubmitting. Pass `--webhook` to receive
completion events, and set `OPENSHORTS_WEBHOOK_SECRET` to sign them. Generated
clips appear in OpenShorts for review.

For editorial references, YouTube highlighted serialized microdramas from
GOKKO CLUB and Shorts Break in October 2026, and featured candy-making process
Shorts the same month. TikTok's 2026 report highlights behind-the-scenes
stories and demonstrations that show how a result was achieved. Use these as
format references and check the regional TikTok Creative Center for current
videos, songs, and hashtags. [YouTube microdramas](https://blog.youtube/culture-and-trends/the-evolution-of-web-series-on-youtube-microdramas/),
[Candy-making examples](https://blog.youtube/culture-and-trends/youtube-candy-makers-halloween-yoodle/),
[TikTok 2026 report](https://ads.tiktok.com/business/en-US/next),
[TikTok Creative Center](https://ads.tiktok.com/creative/creativeCenter/trends)
