# OpenShorts Integration Guide: Multi-Provider LLM Fallback & YouTube Auto-Publish

This guide covers deploying two major features added in this release:

1. **Multi-Provider LLM Fallback** (Part A)
2. **YouTube Shorts Auto-Publisher** (Part B)

---

## Part A: Multi-Provider LLM Fallback

### Overview

Instead of failing when Gemini is unavailable, the system now tries a configurable chain of providers:

**Default order:** Gemini → Claude → DeepSeek → OpenAI → Grok → Amazon Bedrock Nova

Each provider is independent; only set API keys for ones you want to use.

### Setup

#### 1. Update Environment Variables

Add to `.env` (or pass via Docker):

```bash
# Pick which providers you want to enable:

# Google Gemini (free, default first)
GEMINI_API_KEY=your_key_here

# Anthropic Claude (cost-effective fallback)
ANTHROPIC_API_KEY=your_key_here

# OpenAI (gpt-4o-mini is cheapest for reasoning)
OPENAI_API_KEY=your_key_here

# DeepSeek (extremely cheap ~$0.14/$0.28 per 1M tokens)
DEEPSEEK_API_KEY=your_key_here

# Grok / xAI (optional)
XAI_API_KEY=your_key_here

# Amazon Bedrock (uses AWS credentials from IAM role)
AWS_REGION=us-east-1

# Custom fallback order (optional, default: gemini,claude,deepseek,openai,grok,bedrock-nova)
LLM_FALLBACK_ORDER=gemini,claude,deepseek,openai
```

#### 2. Update Python Dependencies

```bash
pip install -r requirements.txt
```

New packages added:
- `anthropic==0.28.1` (Claude support)
- `openai==1.52.0` (OpenAI, DeepSeek, Grok support via OpenAI-compatible API)

#### 3. Docker Image Rebuild

If using Docker:

```bash
cd /opt/openshorts
docker compose down
docker compose up --build
```

### Pricing Reference (as of Sept 2026)

| Provider | Model | Input | Output | Notes |
|----------|-------|-------|--------|-------|
| **Gemini** | flash-lite | $0.25 | $1.50 | Free, highly recommended |
| Claude | 3.5 Haiku | $0.80 | $4.00 | Cost-effective fallback |
| OpenAI | gpt-4o-mini | $0.15 | $0.60 | Cheap reasoning model |
| **DeepSeek** | chat | $0.14 | $0.28 | **Cheapest option** |
| Grok | 2-1212 | $2.00 | $10.00 | Expensive, last resort |
| Bedrock | Nova Lite | $0.40 | $1.60 | AWS-native, include in bill |

**Recommendation:** Keep Gemini (free) first, add DeepSeek as fallback (~$0.004 per clip detection).

### Monitoring

Logs now show which provider handled each job:

```bash
# Watch logs for provider usage
docker logs -f openshorts-backend | grep "Provider used:"
```

Example output:
```
2026-10-01 14:23:45 Trying gemini for score...
2026-10-01 14:23:50 ⚠️ gemini transient error (attempt 1/3), retrying in 5s: 503 Service Unavailable
2026-10-01 14:24:00 Trying claude for score...
2026-10-01 14:24:05 ✅ claude succeeded for score
2026-10-01 14:24:05 Provider used: claude
```

### Troubleshooting

**Problem:** All providers fail
```
❌ All providers exhausted for score. Errors: gemini: 401 API key invalid; claude: ANTHROPIC_API_KEY not set
```

**Solution:** Verify env vars are set and valid. Check which providers are available:
```python
import llm_providers
for name, provider in llm_providers.PROVIDERS.items():
    print(f"{name}: {'available' if provider.available() else 'not available'}")
```

---

## Part B: YouTube Shorts Auto-Publisher

### Overview

Automatically upload completed OpenShorts clips to YouTube with:
- Scheduled publishing (respects timezone & post times)
- Resumable uploads for large files
- Quota awareness
- Retry logic
- Queue management (local state.json)

### Setup

#### 1. Get YouTube OAuth Credentials

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable **YouTube Data API v3**
4. Create OAuth2 credentials (type: Desktop application)
5. Download JSON credentials
6. Save to `secrets/youtube_client_secrets.json`

Then run the auth setup:

```bash
cd /opt/openshorts
python youtube_auth.py --output secrets/youtube_token.json
```

This opens a browser for you to authorize; the token is saved for future use.

**⚠️ Important:** Keep `youtube_token.json` secure! It allows uploading to your YouTube channel.

#### 2. Create Configuration Files

**`config.json`:**
```json
{
  "openshorts_url": "http://localhost:8000",
  "state_file": "state.json",
  "youtube_token_path": "secrets/youtube_token.json",
  "posts_per_day": 2,
  "post_times": ["09:00", "18:00"],
  "timezone": "America/Bogota"
}
```

**`state.json`** (auto-created):
```json
{
  "queue": [],
  "published": []
}
```

#### 3. Update Crontab

Add cron jobs to run the publisher periodically:

```bash
# Every 5 minutes: try to publish if scheduled
*/5 * * * * cd /opt/openshorts && /usr/bin/python3 publish_to_youtube.py --publish >> publish.log 2>&1

# Every 10 minutes: check status
*/10 * * * * cd /opt/openshorts && /usr/bin/python3 publish_to_youtube.py --status >> publish.log 2>&1
```

### Integration with OpenShorts Job Pipeline

#### Option 1: Manual Integration in sync_and_retry.py

When a job is detected as completed, call:

```python
from publish_to_youtube import YouTubePublisher

def handle_completed_job(job_id, output_dir, base_name):
    """Called when sync_and_retry.py detects a completed job."""
    publisher = YouTubePublisher(
        config_path="config.json",
        state_path="state.json",
        token_path="secrets/youtube_token.json"
    )
    
    # Extract clips from job and add to queue
    clips_added = publisher.extract_clips_from_job(
        job_id=job_id,
        output_dir=output_dir,
        base_name=base_name
    )
    
    print(f"Added {clips_added} clips to publishing queue")
```

#### Option 2: HTTP Webhook Integration

Modify OpenShorts to call your webhook when a job completes:

```bash
POST /api/process

{
  ...job params...,
  "webhook_url": "http://your-server/webhook/job-complete",
  "webhook_secret": "your-secret-key"
}
```

Your webhook handler:

```python
@app.post("/webhook/job-complete")
async def handle_job_complete(request: Request):
    """Called by OpenShorts when a job completes."""
    payload = await request.json()
    job_id = payload["job_id"]
    
    # Verify signature (if webhook_secret was set)
    # ...
    
    # Extract and queue clips
    publisher = YouTubePublisher("config.json", "state.json", "secrets/youtube_token.json")
    publisher.extract_clips_from_job(
        job_id=job_id,
        output_dir=f"output/{job_id}",
        base_name=f"{job_id}_source"
    )
    
    return {"status": "queued"}
```

### Usage

#### View Queue Status

```bash
python publish_to_youtube.py --status
```

Output:
```
📊 Publishing Status:
   Queue: 3 clip(s)
   Published: 12 clip(s)

   Next clip: xyz123/0
   Scheduled: 2026-10-01T18:00:00+00:00
```

#### Manually Add a Clip

```bash
python publish_to_youtube.py --add-clip '{
  "job_id": "xyz123",
  "clip_index": 0,
  "video_path": "/opt/openshorts/output/xyz123/clip_00.mp4",
  "title": "Amazing Moment from Anime",
  "description": "Check out this amazing scene!",
  "tags": ["anime", "viral", "shorts"]
}'
```

#### Publish Next Clip Now

```bash
python publish_to_youtube.py --publish
```

#### Extract Clips from a Completed Job

```bash
python publish_to_youtube.py --extract xyz123 --extract-dir output
```

### YouTube Publishing Timeline

1. **Clip added to queue** → state.json updated
2. **Cron runs every 5 min** → Checks if it's time to publish
3. **If time to publish:**
   - Upload starts (resumable, chunks)
   - Video set to Private initially
   - publishAt scheduled for next configured time
   - Moved to "published" in state.json
4. **At publishAt time:**
   - YouTube automatically publishes the video as Public

### Monitoring

Watch the publishing log:

```bash
tail -f publish_to_youtube.log
```

Example log output:

```
2026-10-01 09:00:15 [INFO] Publishing clip xyz123/0
2026-10-01 09:00:15 [INFO] Resumable session started: https://www.googleapis.com/upload/youtube/v3/...
2026-10-01 09:00:45 [INFO] Upload progress: 50% (50.2 MB)
2026-10-01 09:01:15 [INFO] Upload progress: 100% (100.5 MB)
2026-10-01 09:01:20 [INFO] ✅ Upload complete: https://youtu.be/abc123xyz
2026-10-01 09:01:21 [INFO] Moved to published: xyz123/0 → abc123xyz
```

### Troubleshooting

#### Problem: "YOUTUBE_API_KEY not set"
- You need to authenticate first: `python youtube_auth.py`
- This creates `secrets/youtube_token.json`

#### Problem: "Quota exceeded"
- YouTube has a 10,000 unit/day limit (~6 uploads)
- Each upload costs ~1,600 units
- Wait until next UTC day for quota reset
- Or get higher quota from Google Cloud

#### Problem: Video stuck "Private" after publishAt time
- YouTube's scheduled publishing sometimes has delays
- Check video status: `python -c "from youtube_utils import YouTubeUploader; u = YouTubeUploader('secrets/youtube_token.json'); print(u.get_video_status('VIDEO_ID'))"`
- Manually publish via YouTube Studio if needed

---

## Deployment Checklist

### Pre-Deployment

- [ ] **Part A:**
  - [ ] Set GEMINI_API_KEY (or at least one API key)
  - [ ] Optionally set ANTHROPIC_API_KEY, OPENAI_API_KEY, DEEPSEEK_API_KEY
  - [ ] Test: `python -c "import llm_providers; llm_providers.PROVIDERS['gemini'].score('test', 'gemini-3.1-flash-lite')"`

- [ ] **Part B:**
  - [ ] Generate YouTube OAuth token: `python youtube_auth.py`
  - [ ] Create `config.json` with your settings
  - [ ] Test CLI: `python publish_to_youtube.py --status`
  - [ ] Add cron jobs for periodic publishing

### On Production Server

```bash
cd /opt/openshorts

# 1. Pull latest code (or merge your branches)
git fetch origin
git checkout feature/multi-provider-llm-fallback
git merge feature/youtube-auto-publish

# 2. Update dependencies
pip install -r requirements.txt

# 3. Configure environment
# Edit .env and add LLM_FALLBACK_ORDER, API keys, YouTube token path

# 4. Set up YouTube (if using Part B)
mkdir -p secrets
python youtube_auth.py --output secrets/youtube_token.json
cp config.example.json config.json
# Edit config.json to customize

# 5. Restart backend
docker compose down openshorts-backend
docker compose up -d openshorts-backend

# 6. Add cron jobs (for YouTube publishing)
crontab -e
# Add:
# */5 * * * * cd /opt/openshorts && /usr/bin/python3 publish_to_youtube.py --publish >> publish.log 2>&1

# 7. Verify
docker logs openshorts-backend | grep "Provider used:"
python publish_to_youtube.py --status
```

### Rollback

If something breaks:

```bash
cd /opt/openshorts
git checkout main
docker compose down openshorts-backend
docker compose up -d openshorts-backend
```

---

## Cost Estimation

### Part A: LLM Fallback

**Per 15-second clip:**
- Gemini Flash-Lite: $0.0001 (free)
- Claude Haiku: $0.0001
- DeepSeek: $0.000003 (cheapest)

**Daily cost (assuming 10 clips):**
- Gemini only: $0 (free)
- Gemini + DeepSeek fallback: ~$0.0001
- Claude: ~$0.001

### Part B: YouTube Publishing

**Per clip upload:**
- YouTube Data API: ~1,600 units per upload (10k/day limit = ~6 free uploads)
- Bandwidth: ~100 MB × your ISP rate (~$0.01-0.05)

**Monthly cost:** Free (within quota)

---

## Support & Troubleshooting

**For LLM fallback issues:**
- Check logs: `docker logs openshorts-backend | tail -100`
- Verify API keys: `python -c "import os; print(os.getenv('GEMINI_API_KEY'))"`
- Test provider directly: `python llm_providers.py` (implement quick test)

**For YouTube issues:**
- Check logs: `tail -f publish_to_youtube.log`
- Verify OAuth: `python youtube_auth.py --refresh`
- Check quota: Log in to [Google Cloud Console](https://console.cloud.google.com/) → YouTube API → Quotas

**Contact:**
- For OpenShorts: https://github.com/mutonby/openshorts
- For integration help: Check CLAUDE.md in your fork

---

## Next Steps

1. Merge both feature branches to `main`
2. Test on staging before production
3. Monitor logs for 1-2 days to confirm stability
4. Adjust `LLM_FALLBACK_ORDER` or `posts_per_day` based on actual usage
5. Consider adding webhooks or database integration for advanced queue management
