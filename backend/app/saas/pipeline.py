"""
SaaSShorts: AI-powered UGC video generator for SaaS products.

Generates viral TikTok/Instagram Reels content from a SaaS URL.
Pipeline:
  1. Scrape & analyze SaaS website (Gemini)
  2. Generate video scripts (hook → problem → solution → CTA)
  3. Generate AI actor portrait (Flux Pro via fal.ai)
  4. Generate voiceover (ElevenLabs TTS)
  5. Generate talking head video (Kling Avatar v2 via fal.ai)
  6. Generate b-roll clips (Kling v2.6 via fal.ai)
  7. Composite final video with subtitles (FFmpeg)
"""

import os
import re
import json
import time
import subprocess
import httpx
from urllib.parse import urljoin
from typing import Optional, List, Dict, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed


ELEVENLABS_API_BASE = "https://api.elevenlabs.io/v1"
FAL_QUEUE_BASE = "https://queue.fal.run"

# Default ElevenLabs voices (name → voice_id)
DEFAULT_VOICES = {
    "Rachel (Female, calm)": "21m00Tcm4TlvDq8ikWAM",
    "Drew (Male, confident)": "29vD33N1CtxCmqQRPOHJ",
    "Bella (Female, soft)": "EXAVITQu4vr4xnSDxMaL",
    "Antoni (Male, warm)": "ErXwobaYiN019PkySvjV",
    "Josh (Male, deep)": "TxGEqnHWrfWFTfGW9XjX",
    "Sam (Male, raspy)": "yoZ06aMxZJJ28mfd3POQ",
}


GEMINI_MODEL = "gemini-3-flash-preview"


# ═══════════════════════════════════════════════════════════════════════
# Phase 1: Website Scraping, Web Research & Analysis
# ═══════════════════════════════════════════════════════════════════════

def research_saas_online(url: str, gemini_key: str) -> dict:
    """
    Use Gemini with Google Search grounding to deeply research a SaaS product
    across the internet: reviews, Reddit threads, Twitter, competitor comparisons,
    pricing complaints, user testimonials, etc.
    """
    from google import genai
    from google.genai import types

    print(f"[SaaSShorts] 🔍 Researching {url} across the web (Google Search grounding)...")

    client = genai.Client(api_key=gemini_key)

    # Extract domain name for search queries
    domain = url.replace("https://", "").replace("http://", "").split("/")[0]

    prompt = f"""You are a world-class SaaS market researcher. Research this product thoroughly using Google Search.

Product URL: {url}
Domain: {domain}

SEARCH AND INVESTIGATE:
1. What does this SaaS product do? (search their website, Product Hunt, G2, Capterra)
2. What are REAL user reviews saying? (G2, Capterra, TrustPilot, Reddit, Twitter/X)
3. What are the most common complaints and pain points users mention?
4. Who are their main competitors and how do they compare?
5. What is their pricing and do users think it's worth it?
6. What is their target market and ideal customer profile?
7. Are there any viral posts, memes, or discussions about this product?
8. What content creators or influencers have talked about them?

Return a comprehensive JSON research report:
{{
    "product_name": "...",
    "website_url": "{url}",
    "what_it_does": "Detailed description of the product based on web research",
    "target_market": "Who this product is for",
    "pricing_info": "Pricing details found online (plans, costs, free tier)",
    "user_sentiment": "overall positive/mixed/negative",
    "real_reviews": [
        {{"source": "G2/Reddit/Twitter/etc", "quote": "actual user quote or paraphrase", "sentiment": "positive/negative/neutral"}},
        ...
    ],
    "common_complaints": ["complaint 1 from real users", "complaint 2", ...],
    "common_praise": ["what users love 1", "what users love 2", ...],
    "competitors": [
        {{"name": "competitor", "comparison": "how they compare"}}
    ],
    "viral_potential": ["angle 1 based on real discussions", "angle 2", ...],
    "key_differentiators": ["what makes them unique based on research"],
    "content_angles_from_web": ["angles found from existing content about this product"],
    "sources_found": ["list of URLs where information was found"]
}}

Be thorough. Use REAL data from your search results, not made-up information."""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[prompt],
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
        ),
    )

    # Extract grounding sources
    sources = []
    try:
        metadata = response.candidates[0].grounding_metadata
        if metadata and metadata.grounding_chunks:
            for chunk in metadata.grounding_chunks:
                if chunk.web:
                    sources.append({"title": chunk.web.title, "url": chunk.web.uri})
        if metadata and metadata.web_search_queries:
            print(f"[SaaSShorts]   Searches performed: {metadata.web_search_queries}")
    except Exception:
        pass

    # Parse response text as JSON
    raw = response.text
    if not raw:
        print("[SaaSShorts] ⚠️ Gemini returned empty response for web research")
        return {"raw_research": "", "product_name": domain, "grounding_sources": sources}

    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text)

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start : end + 1]

    try:
        research = json.loads(text)
    except json.JSONDecodeError:
        research = {"raw_research": text, "product_name": domain}

    research["grounding_sources"] = sources
    print(f"[SaaSShorts] ✅ Web research complete: {len(sources)} sources found")
    return research


def scrape_website(url: str) -> dict:
    """Scrape a SaaS website to extract key content for analysis."""
    from bs4 import BeautifulSoup

    print(f"[SaaSShorts] 🌐 Scraping {url}...")

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        response = client.get(url, headers=headers)
        response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove non-content elements
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "svg", "iframe"]):
        tag.decompose()

    # Extract metadata
    meta_desc = ""
    meta_tag = soup.find("meta", attrs={"name": "description"})
    if meta_tag:
        meta_desc = meta_tag.get("content", "")

    og_desc = ""
    og_tag = soup.find("meta", attrs={"property": "og:description"})
    if og_tag:
        og_desc = og_tag.get("content", "")

    title = soup.title.string.strip() if soup.title and soup.title.string else ""

    # Extract headings
    headings = []
    for h in soup.find_all(["h1", "h2", "h3"]):
        text = h.get_text(strip=True)
        if text and len(text) < 200:
            headings.append(text)

    # Main text content
    text = soup.get_text(separator="\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text[:10000]

    # Find subpages to scrape
    base_host = httpx.URL(url).host
    subpages = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].lower()
        if any(kw in href for kw in ["pricing", "features", "about", "product", "why", "how-it-works", "use-case"]):
            try:
                full_url = urljoin(url, a["href"])
                full_host = httpx.URL(full_url).host
                if base_host and full_host and base_host == full_host:
                    subpages.add(full_url)
            except Exception:
                pass

    # Scrape subpages (max 3)
    additional = ""
    for sub_url in list(subpages)[:3]:
        try:
            print(f"[SaaSShorts]   → Subpage: {sub_url}")
            with httpx.Client(timeout=20.0, follow_redirects=True) as client:
                resp = client.get(sub_url, headers=headers)
                if resp.status_code == 200:
                    sub_soup = BeautifulSoup(resp.text, "html.parser")
                    for tag in sub_soup(["script", "style", "nav", "footer", "header", "noscript"]):
                        tag.decompose()
                    sub_text = sub_soup.get_text(separator="\n", strip=True)[:5000]
                    additional += f"\n\n--- {sub_url} ---\n{sub_text}"
        except Exception as e:
            print(f"[SaaSShorts]   ⚠️ Failed: {e}")

    result = {
        "url": url,
        "title": title,
        "meta_description": meta_desc or og_desc,
        "headings": headings[:20],
        "main_content": text,
        "additional_pages": additional[:15000],
        "pages_scraped": 1 + min(len(subpages), 3),
    }

    print(f"[SaaSShorts] ✅ Scraped {result['pages_scraped']} pages, {len(text)} chars")
    return result


def analyze_saas(scraped_data: dict, gemini_key: str, web_research: dict = None) -> dict:
    """
    Deep analysis of a SaaS product combining website scraping + web research.
    Uses Gemini 3 Flash for synthesis.
    """
    from google import genai
    from google.genai import types

    print(f"[SaaSShorts] 🧠 Analyzing {scraped_data['url']} (with web research)...")

    client = genai.Client(api_key=gemini_key)

    # Build web research context
    research_context = ""
    if web_research:
        research_context = f"""
=== WEB RESEARCH (from Google Search) ===
Product: {web_research.get('product_name', 'Unknown')}
What it does: {web_research.get('what_it_does', 'N/A')}
Target market: {web_research.get('target_market', 'N/A')}
Pricing: {web_research.get('pricing_info', 'N/A')}
User sentiment: {web_research.get('user_sentiment', 'N/A')}

Real user reviews:
{json.dumps(web_research.get('real_reviews', [])[:8], indent=2)}

Common complaints from real users:
{json.dumps(web_research.get('common_complaints', []), indent=2)}

What users love:
{json.dumps(web_research.get('common_praise', []), indent=2)}

Competitors:
{json.dumps(web_research.get('competitors', []), indent=2)}

Viral angles from existing content:
{json.dumps(web_research.get('viral_potential', []), indent=2)}

Key differentiators:
{json.dumps(web_research.get('key_differentiators', []), indent=2)}

Content angles found online:
{json.dumps(web_research.get('content_angles_from_web', []), indent=2)}
"""

    prompt = f"""You are an expert SaaS marketing analyst and UGC content strategist. Analyze this SaaS product for creating viral UGC-style marketing videos.

You have TWO sources of information:
1. The product's OWN WEBSITE (scraped content)
2. EXTERNAL WEB RESEARCH (real reviews, Reddit, competitor analysis, user sentiment from Google Search)

Combine BOTH to create the most accurate and compelling analysis possible. Prioritize REAL user pain points and sentiments from the web research.

Website: {scraped_data['url']}
Title: {scraped_data['title']}
Meta: {scraped_data['meta_description']}
Headings: {json.dumps(scraped_data['headings'][:15])}

=== WEBSITE CONTENT ===
{scraped_data['main_content'][:6000]}

=== ADDITIONAL PAGES ===
{scraped_data['additional_pages'][:8000]}
{research_context}

Return a JSON object:
{{
    "product_name": "Name of the SaaS",
    "one_liner": "One-sentence description",
    "target_audience": ["audience 1", "audience 2", "audience 3"],
    "pain_points": [
        {{"pain": "specific pain point (from real user feedback if available)", "intensity": "high/medium/low", "emotional_trigger": "frustration/fear/time-waste/money-loss/overwhelm", "source": "website/user-reviews/reddit/general"}}
    ],
    "key_features": ["feature 1", "feature 2", "feature 3"],
    "unique_selling_points": ["usp 1", "usp 2"],
    "competitors": [
        {{"name": "competitor", "comparison": "how they compare"}}
    ],
    "pricing_model": "freemium/subscription/one-time/usage-based",
    "pricing_details": "specific pricing info if found",
    "industry": "category",
    "user_sentiment_summary": "what real users think overall",
    "emotional_hooks": [
        "Stop wasting X hours on...",
        "Your competitors are already using...",
        "I wish I knew about this sooner..."
    ],
    "transformation_story": "Before (with real pain) → After (with product) narrative",
    "viral_angles": [
        {{"angle": "description", "platform": "tiktok/instagram/both", "style": "ugc/educational/shock/story", "why_viral": "reason this angle works"}}
    ]
}}

IMPORTANT: Use REAL pain points from user reviews when available. Real frustrations make the best UGC content.
Include 5-8 pain points, 4-6 emotional hooks, and 4+ viral angles."""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[prompt],
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )

    raw = response.text
    if not raw:
        raise Exception("Gemini returned empty response for SaaS analysis")

    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text)

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start : end + 1]

    try:
        analysis = json.loads(text)
    except json.JSONDecodeError as e:
        raise Exception(f"Failed to parse analysis JSON: {e}\nRaw: {text[:500]}")

    # Attach web research sources for reference
    if web_research and web_research.get("grounding_sources"):
        analysis["_web_sources"] = web_research["grounding_sources"]

    print(f"[SaaSShorts] ✅ Analysis: {analysis.get('product_name', '?')} ({len(analysis.get('pain_points', []))} pain points)")
    return analysis


def generate_scripts(
    analysis: dict,
    gemini_key: str,
    num_scripts: int = 3,
    style: str = "ugc",
    language: str = "en",
    actor_gender: str = "female",
) -> list:
    """Generate video scripts based on SaaS analysis."""
    from google import genai
    from google.genai import types

    lang_name = "Spanish" if language == "es" else "English"
    print(f"[SaaSShorts] 📝 Generating {num_scripts} scripts ({style}, {lang_name})...")

    client = genai.Client(api_key=gemini_key)

    style_guide = {
        "ugc": "Natural, authentic UGC style. Person talking to camera like sharing a discovery with a friend. Casual, genuine.",
        "educational": "Educational style. Clear explanations.",
        "shock": "Shock/discovery style. Surprising opener.",
        "story": "Storytelling style. Mini narrative.",
        "comparison": "Before/after comparison.",
    }

    lang_instructions = ""
    if language == "es":
        lang_instructions = """
LANGUAGE: ALL narrations, subtitles, captions, and hashtags MUST be in SPANISH (Spain/Latin America).
Use natural casual Spanish like a real person would speak on TikTok. Contractions, slang OK.
Examples of Spanish UGC hooks: "Tío, no me puedo creer que nadie me haya dicho esto antes...", "Si usas Excel para esto, necesitas ver esto YA", "Os voy a enseñar algo que me ha cambiado la vida..."
"""
    else:
        lang_instructions = """
LANGUAGE: ALL narrations, subtitles, captions, and hashtags MUST be in ENGLISH.
Use natural casual American English like a real person on TikTok. Contractions, slang OK.
Examples of English UGC hooks: "Okay so I just found this tool and...", "Stop doing this manually, there's a better way", "I can't believe nobody told me about this sooner..."
"""

    prompt = f"""You are a viral short-form video scriptwriter for TikTok/Instagram Reels.
Generate {num_scripts} video scripts to promote this product/business.
{lang_instructions}
PRODUCT ANALYSIS:
{json.dumps(analysis, indent=2)}

STYLE: {style_guide.get(style, style_guide['ugc'])}

Each script MUST be 20-25 seconds total. NEVER longer than 25 seconds.

YOU MUST USE EXACTLY THIS 5-SEGMENT STRUCTURE. NO EXCEPTIONS:
1. HOOK (0-5s): type="hook", visual="actor_talking", broll_prompt=null — Avatar says a punchy hook.
2. B-ROLL 1 (5-9s): type="problem", visual="broll", broll_prompt="..." (REQUIRED) — Visual of the problem.
3. BODY (9-16s): type="solution", visual="actor_talking", broll_prompt=null — Avatar presents the solution.
4. B-ROLL 2 (16-21s): type="demo", visual="broll", broll_prompt="..." (REQUIRED) — Visual of the product.
5. CTA (21-25s): type="cta", visual="actor_talking", broll_prompt=null — Avatar says CTA with link in bio.

CRITICAL — READ CAREFULLY:
- EXACTLY 5 segments. Not 3, not 4, not 6. FIVE.
- Segments 2 and 4 MUST have visual="broll" and a non-null broll_prompt string.
- Segments 1, 3, 5 MUST have visual="actor_talking" and broll_prompt=null.
- duration_seconds MUST be between 20 and 25.
- full_narration = all narration text joined together.

Return a JSON array:
[
    {{
        "title": "Short internal title",
        "style": "{style}",
        "duration_seconds": 23,
        "target_platform": "tiktok",
        "hook_text": "Hook overlay text (2-5 words max)",
        "segments": [
            {{
                "type": "hook",
                "start": 0,
                "end": 5,
                "narration": "Punchy hook the actor says",
                "visual": "actor_talking",
                "broll_prompt": null,
                "emotion": "excited",
                "subtitle_text": "Hook phrase"
            }},
            {{
                "type": "problem",
                "start": 5,
                "end": 9,
                "narration": "Voiceover describing the pain point",
                "visual": "broll",
                "broll_prompt": "REQUIRED: visual of the problem, e.g. person frustrated at laptop, cluttered spreadsheet on screen",
                "emotion": "frustrated",
                "subtitle_text": "Pain phrase"
            }},
            {{
                "type": "solution",
                "start": 9,
                "end": 16,
                "narration": "Actor introduces the product naturally",
                "visual": "actor_talking",
                "broll_prompt": null,
                "emotion": "confident",
                "subtitle_text": "Solution phrase"
            }},
            {{
                "type": "demo",
                "start": 16,
                "end": 21,
                "narration": "Voiceover showing the product in action",
                "visual": "broll",
                "broll_prompt": "REQUIRED: visual of the product/result, e.g. clean dashboard with metrics, modern app interface",
                "emotion": "excited",
                "subtitle_text": "Result phrase"
            }},
            {{
                "type": "cta",
                "start": 21,
                "end": 23,
                "narration": "Short CTA mentioning link in bio",
                "visual": "actor_talking",
                "broll_prompt": null,
                "emotion": "confident",
                "subtitle_text": "Link in bio"
            }}
        ],
        "full_narration": "All narration text joined (only actor_talking segments)",
        "actor_description": "Specific person description: age, gender, ethnicity, hair style, clothing. Casual everyday look.",
        "hashtags": ["#saas", "#productivity", "#techtools"],
        "caption": "Suggested Instagram/TikTok caption"
    }}
]

RULES:
- EXACTLY 5 segments in order: actor, broll, actor, broll, actor
- EXACTLY 2 broll segments with detailed broll_prompt (NOT null)
- full_narration = ALL narration text (both actor and broll voiceover segments joined)
- Total duration MUST be 18-22 seconds, never more
- Keep narrations punchy, conversational, with contractions
- Actor descriptions: casual, real-person look (NOT model/influencer)
- B-roll prompts: cinematic, specific, detailed visual descriptions
- Each script should use a different pain point / angle
- Vary actor demographics across scripts
- CTA MUST always mention "link in bio" / "enlace en la bio". Examples: "Link in bio, go try it", "Check the link in my bio", "El enlace está en la bio, probadlo"
- Write ALL text in {lang_name}
- Actor gender: {actor_gender}. ALL actor_description fields MUST describe a {actor_gender} person. Use diverse ages/ethnicities across scripts.
- IMPORTANT: actor_description MUST ALWAYS be in ENGLISH regardless of script language. Only describe physical appearance: age, gender, ethnicity, hair, clothing. NO actions, NO background, NO scene description.
- Actors must look European, attractive but natural, slightly nerdy/tech vibe. Vary across: blonde, brunette, redhead. Ages 22-35.
- If female: casual summer look (tank top, camisole, simple tee). If male: casual tee or hoodie.
- Example female: "a 26 year old attractive european woman, light brown wavy hair, wearing a white tank top, natural minimal makeup, friendly face"
- Example male: "a 29 year old european man, short dark hair, light stubble, wearing a navy t-shirt, smart casual look" """

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            max_output_tokens=8192,
        ),
    )

    raw = response.text
    if not raw:
        raise Exception("Gemini returned empty response for script generation")

    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text)

    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1:
        text = text[start : end + 1]

    try:
        scripts = json.loads(text)
    except json.JSONDecodeError as e:
        raise Exception(f"Failed to parse scripts JSON: {e}\nRaw: {text[:500]}")

    print(f"[SaaSShorts] ✅ Generated {len(scripts)} scripts")
    return scripts


# ═══════════════════════════════════════════════════════════════════════
# Phase 2: Asset Generation
# ═══════════════════════════════════════════════════════════════════════

def _fal_run(model_id: str, input_data: dict, fal_key: str, timeout: int = 600) -> dict:
    """
    Submit a job to fal.ai queue, poll for completion, return result.
    Uses the URLs returned by the submit response (as per fal.ai docs).
    """
    headers = {
        "Authorization": f"Key {fal_key}",
        "Content-Type": "application/json",
    }

    # ── Step 1: Submit to queue ──
    submit_url = f"{FAL_QUEUE_BASE}/{model_id}"
    print(f"[fal.ai] Submitting to {submit_url}...")

    with httpx.Client(timeout=120.0) as client:
        resp = client.post(submit_url, headers=headers, json=input_data)

    if resp.status_code >= 400:
        print(f"[fal.ai] Submit error: {resp.text[:500]}")
        raise Exception(f"fal.ai error ({resp.status_code}): {resp.text[:300]}")

    try:
        submit_data = resp.json()
    except json.JSONDecodeError:
        raise Exception(f"fal.ai invalid JSON: {resp.text[:300]}")

    request_id = submit_data.get("request_id")
    if not request_id:
        # Synchronous result (no queue)
        return submit_data

    # Use the URLs from the submit response (guaranteed correct per docs)
    status_url = submit_data.get("status_url", f"{FAL_QUEUE_BASE}/{model_id}/requests/{request_id}/status")
    response_url = submit_data.get("response_url", f"{FAL_QUEUE_BASE}/{model_id}/requests/{request_id}")

    print(f"[fal.ai] Queued: {request_id}")
    print(f"[fal.ai] Status URL: {status_url}")

    # ── Step 2: Poll for completion ──
    poll_headers = {"Authorization": f"Key {fal_key}"}
    start = time.time()

    while time.time() - start < timeout:
        elapsed = int(time.time() - start)
        try:
            with httpx.Client(timeout=30.0) as client:
                poll_resp = client.get(f"{status_url}?logs=1", headers=poll_headers)
            status_data = poll_resp.json()
        except Exception as e:
            print(f"[fal.ai] Poll error (retrying): {e}")
            time.sleep(5)
            continue

        status = status_data.get("status", "UNKNOWN")

        if status == "COMPLETED":
            print(f"[fal.ai] ✅ Completed in {elapsed}s! Fetching result...")
            with httpx.Client(timeout=120.0) as client:
                result_resp = client.get(response_url, headers=poll_headers)
                return result_resp.json()

        elif status in ("FAILED", "CANCELLED"):
            error = status_data.get("error", "unknown error")
            raise Exception(f"fal.ai job {status}: {error}")

        # Log progress
        queue_pos = status_data.get("queue_position", "")
        pos_info = f" (pos: {queue_pos})" if queue_pos != "" else ""
        print(f"[fal.ai] {model_id}: {status}{pos_info} ({elapsed}s)")
        time.sleep(5)

    raise Exception(f"fal.ai job timed out after {timeout}s for {model_id}")


def _fal_upload_file(file_path: str, fal_key: str) -> str:
    """Upload a local file to fal.ai CDN storage and return public URL."""
    headers = {"Authorization": f"Key {fal_key}"}

    filename = os.path.basename(file_path)
    ext = os.path.splitext(filename)[1].lower()
    content_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".mp4": "video/mp4",
        ".webp": "image/webp",
    }
    content_type = content_types.get(ext, "application/octet-stream")

    # Initiate upload
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(
            "https://rest.alpha.fal.ai/storage/upload/initiate",
            headers={**headers, "Content-Type": "application/json"},
            json={"file_name": filename, "content_type": content_type},
        )
        resp.raise_for_status()
        upload_info = resp.json()

    upload_url = upload_info["upload_url"]
    file_url = upload_info["file_url"]

    # Upload file content
    with open(file_path, "rb") as f:
        file_bytes = f.read()

    with httpx.Client(timeout=120.0) as client:
        resp = client.put(
            upload_url,
            content=file_bytes,
            headers={"Content-Type": content_type},
        )
        resp.raise_for_status()

    print(f"[fal.ai] Uploaded {filename} → {file_url}")
    return file_url


def generate_actor_images(
    description: str, fal_key: str, output_dir: str, title_slug: str, num_options: int = 3,
    product_description: str = None,
) -> List[str]:
    """Generate multiple hyper-realistic actor portrait options using Flux 2 Pro."""
    print(f"[SaaSShorts] 🎨 Generating {num_options} actor image options (Flux 2 Pro)...")

    # Clean description: strip scene/actions, keep only physical appearance
    clean_desc = description
    for remove in ["hablando", "talking", "sentad", "sitting", "desde", "from", "con una", "with a", "detrás", "behind"]:
        if remove in clean_desc.lower():
            idx = clean_desc.lower().find(remove)
            if idx > 10:
                clean_desc = clean_desc[:idx].rstrip(" ,.")

    import random
    img_num = random.randint(1000, 9999)

    if product_description:
        prompt = f"""IMG_{img_num}.jpg Raw candid selfie of {clean_desc}, casually holding {product_description}, showing it to the camera with a natural smile. Product clearly visible in hand. Casual and real, not an ad. Low quality front camera, soft room lighting. Reddit selfie."""
    else:
        prompt = f"""IMG_{img_num}.jpg Raw candid selfie of {clean_desc}, sitting at their desk at home, looking at camera with a relaxed natural smile. Headphones around neck, monitor glow behind them. Not posed, casual and real. Low quality front camera, soft room lighting. Reddit selfie."""

    print(f"[SaaSShorts]   Prompt: {prompt[:120]}...{' (with product)' if product_description else ''}")

    paths = []
    # Flux 2 Pro — #1 for photorealistic faces
    def _gen_one(i):
        result = _fal_run(
            "fal-ai/flux-2-pro",
            {
                "prompt": prompt,
                "image_size": "portrait_4_3",
                "safety_tolerance": 5,
                "seed": random.randint(0, 999999),
            },
            fal_key,
            timeout=300,
        )
        images = result.get("images") or result.get("output", [])
        if not images:
            raise Exception(f"No images in actor result: {list(result.keys())}")
        img_url = images[0]["url"] if isinstance(images[0], dict) else images[0]
        img_path = os.path.join(output_dir, f"{title_slug}_actor_option_{i}.png")
        with httpx.Client(timeout=60.0) as client:
            img_resp = client.get(img_url)
            with open(img_path, "wb") as f:
                f.write(img_resp.content)
        print(f"[SaaSShorts] ✅ Actor option {i+1}: {img_path}")
        return img_path

    with ThreadPoolExecutor(max_workers=num_options) as executor:
        futures = [executor.submit(_gen_one, i) for i in range(num_options)]
        for future in as_completed(futures):
            paths.append(future.result())

    return sorted(paths)

    paths = []
    for i, img in enumerate(result.get("images", [])):
        img_path = os.path.join(output_dir, f"{title_slug}_actor_option_{i}.png")
        with httpx.Client(timeout=60.0) as client:
            img_resp = client.get(img["url"])
            with open(img_path, "wb") as f:
                f.write(img_resp.content)
        paths.append(img_path)
        print(f"[SaaSShorts] ✅ Actor option {i+1}: {img_path}")

    return paths


def generate_actor_image(
    description: str, fal_key: str, output_path: str
) -> str:
    """Generate a single actor image using Recraft V4."""
    output_dir = os.path.dirname(output_path)
    title_slug = os.path.basename(output_path).replace("_actor.png", "")
    paths = generate_actor_images(description, fal_key, output_dir, title_slug, num_options=1)
    if paths:
        import shutil
        shutil.move(paths[0], output_path)
    return output_path


def generate_voiceover(
    text: str,
    elevenlabs_key: str,
    output_path: str,
    voice_id: str = "21m00Tcm4TlvDq8ikWAM",
) -> str:
    """Generate voiceover audio using ElevenLabs TTS."""
    print(f"[SaaSShorts] 🎙️ Generating voiceover ({len(text)} chars)...")

    url = f"{ELEVENLABS_API_BASE}/text-to-speech/{voice_id}"

    headers = {
        "xi-api-key": elevenlabs_key,
        "Content-Type": "application/json",
    }

    body = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75,
            "style": 0.4,
            "use_speaker_boost": True,
        },
    }

    with httpx.Client(timeout=120.0) as client:
        resp = client.post(url, headers=headers, json=body)
        if resp.status_code != 200:
            raise Exception(f"ElevenLabs TTS error ({resp.status_code}): {resp.text}")

        with open(output_path, "wb") as f:
            f.write(resp.content)

    print(f"[SaaSShorts] ✅ Voiceover: {output_path}")
    return output_path


def get_elevenlabs_voices(elevenlabs_key: str) -> list:
    """Fetch available voices from ElevenLabs."""
    url = f"{ELEVENLABS_API_BASE}/voices"
    headers = {"xi-api-key": elevenlabs_key}

    with httpx.Client(timeout=15.0) as client:
        resp = client.get(url, headers=headers)
        if resp.status_code != 200:
            return []
        data = resp.json()

    voices = []
    for v in data.get("voices", []):
        voices.append({
            "voice_id": v["voice_id"],
            "name": v["name"],
            "category": v.get("category", ""),
            "labels": v.get("labels", {}),
            "preview_url": v.get("preview_url", ""),
        })

    return voices


# ═══════════════════════════════════════════════════════════════════════
# Phase 3: Video Generation
# ═══════════════════════════════════════════════════════════════════════

def generate_talking_head(
    image_path: str,
    audio_path: str,
    fal_key: str,
    output_path: str,
) -> str:
    """Generate talking head video using Kling Avatar v2 Standard on fal.ai."""
    print(f"[SaaSShorts] 🗣️ Generating talking head (Kling Avatar v2)...")

    # Upload image and audio to fal.ai CDN
    image_url = _fal_upload_file(image_path, fal_key)
    audio_url = _fal_upload_file(audio_path, fal_key)

    result = _fal_run(
        "fal-ai/kling-video/ai-avatar/v2/standard",
        {
            "image_url": image_url,
            "audio_url": audio_url,
            "prompt": (
                "Natural UGC creator talking to camera. Expressive and energetic. "
                "Subtle hand gestures to emphasize points. Slight head movements and nods. "
                "Occasional leaning forward for emphasis. Relaxed shoulders, casual vibe. "
                "Maintain eye contact with camera. Natural blinking and micro-expressions."
            ),
        },
        fal_key,
        timeout=600,
    )

    video_url = result["video"]["url"]

    # Download video
    with httpx.Client(timeout=180.0) as client:
        vid_resp = client.get(video_url)
        with open(output_path, "wb") as f:
            f.write(vid_resp.content)

    print(f"[SaaSShorts] ✅ Talking head: {output_path}")
    return output_path


def generate_talking_head_lowcost(
    image_path: str,
    audio_path: str,
    fal_key: str,
    output_path: str,
) -> str:
    """
    Low-cost talking head: Hailuo 2.3 Fast img2video → VEED Lipsync.
    ~$0.39 vs ~$1.69 for Kling Avatar v2.
    """
    print(f"[SaaSShorts] 🗣️ Generating talking head (Low Cost: Hailuo + VEED Lipsync)...")

    # Step 1: Generate 6s video from image using MiniMax Hailuo 2.3 Fast ($0.19)
    # Cache the Hailuo clip so retries don't re-generate it
    hailuo_cache_path = output_path.replace(".mp4", "_hailuo_cache.mp4")

    if os.path.exists(hailuo_cache_path) and os.path.getsize(hailuo_cache_path) > 0:
        print(f"[SaaSShorts]   Hailuo clip cached, skipping generation.")
        hailuo_video_url = _fal_upload_file(hailuo_cache_path, fal_key)
    else:
        image_url = _fal_upload_file(image_path, fal_key)

        hailuo_result = _fal_run(
            "fal-ai/minimax/hailuo-2.3-fast/standard/image-to-video",
            {
                "image_url": image_url,
                "prompt": (
                    "Person talking to camera, subtle head nods and natural micro-expressions. "
                    "Gentle head movement, slight shoulder sway. Eye contact with camera. "
                    "Natural blinking. Soft ambient lighting. Smooth cinematic motion."
                ),
            },
            fal_key,
            timeout=300,
        )

        print(f"[SaaSShorts]   Hailuo response keys: {list(hailuo_result.keys())}")
        if "video" in hailuo_result:
            hailuo_video_url = hailuo_result["video"]["url"] if isinstance(hailuo_result["video"], dict) else hailuo_result["video"]
        elif "video_url" in hailuo_result:
            hailuo_video_url = hailuo_result["video_url"]
        elif "output" in hailuo_result:
            hailuo_video_url = hailuo_result["output"]["url"] if isinstance(hailuo_result["output"], dict) else hailuo_result["output"]
        else:
            raise Exception(f"No video in Hailuo result: {hailuo_result}")

        # Save Hailuo clip locally for retry cache
        with httpx.Client(timeout=180.0) as client:
            vid_resp = client.get(hailuo_video_url)
            with open(hailuo_cache_path, "wb") as f:
                f.write(vid_resp.content)

        print(f"[SaaSShorts]   Hailuo 2.3 Fast 6s clip ready (cached for retry).")

    # Step 2: Upload audio for lip-sync
    audio_url = _fal_upload_file(audio_path, fal_key)

    # Step 3: VEED Lipsync — high quality lip-sync with loop ($0.20 for 30s)
    lipsync_result = _fal_run(
        "veed/lipsync",
        {
            "video_url": hailuo_video_url,
            "audio_url": audio_url,
        },
        fal_key,
        timeout=900,
    )

    print(f"[SaaSShorts]   VEED Lipsync response keys: {list(lipsync_result.keys())}")
    if "video" in lipsync_result:
        lipsync_video_url = lipsync_result["video"]["url"] if isinstance(lipsync_result["video"], dict) else lipsync_result["video"]
    else:
        raise Exception(f"No video in VEED Lipsync result: {lipsync_result}")

    with httpx.Client(timeout=180.0) as client:
        vid_resp = client.get(lipsync_video_url)
        with open(output_path, "wb") as f:
            f.write(vid_resp.content)

    print(f"[SaaSShorts] ✅ Talking head (low cost): {output_path}")
    return output_path


def generate_broll(
    prompt: str, fal_key: str, output_path: str, duration: str = "5"
) -> str:
    """
    Generate b-roll: Recraft V4 image + Ken Burns zoom effect via FFmpeg.
    """
    print(f"[SaaSShorts] 🎬 Generating b-roll image + Ken Burns effect...")

    dur_secs = int(duration)
    img_path = output_path.replace(".mp4", "_img.png")

    # Step 1: Generate a high-quality still image with Flux 2 Pro
    result = _fal_run(
        "fal-ai/flux-2-pro",
        {
            "prompt": f"{prompt}. Cinematic, shallow depth of field, professional photography.",
            "image_size": "portrait_4_3",
            "safety_tolerance": 5,
        },
        fal_key,
        timeout=300,
    )

    # Flux 2 Pro returns images in "images" or "output" key
    images = result.get("images") or result.get("output", [])
    if not images:
        raise Exception(f"No images in b-roll result: {list(result.keys())}")
    img_url = images[0]["url"] if isinstance(images[0], dict) else images[0]

    with httpx.Client(timeout=60.0) as client:
        img_resp = client.get(img_url)
        with open(img_path, "wb") as f:
            f.write(img_resp.content)

    # Step 2: Ken Burns effect — slow zoom in with slight pan
    fps = 30
    total_frames = dur_secs * fps
    # Zoom from 1.0x to 1.15x over duration (subtle, cinematic)
    zoompan_filter = (
        f"scale=2160:3840,"
        f"zoompan=z='1+0.15*on/{total_frames}':"
        f"x='iw/2-(iw/zoom/2)+10*on/{total_frames}':"
        f"y='ih/2-(ih/zoom/2)':"
        f"d={total_frames}:s=1080x1920:fps={fps},"
        f"setsar=1"
    )
    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-i", img_path,          # Input 0: image
        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",  # Input 1: silent audio
        "-vf", zoompan_filter,
        "-t", str(dur_secs),
        "-map", "0:v", "-map", "1:a",
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        "-shortest",
        output_path,
    ]

    subprocess.run(cmd, check=True, capture_output=True)

    # Cleanup temp image
    if os.path.exists(img_path):
        os.remove(img_path)

    print(f"[SaaSShorts] ✅ B-roll (Ken Burns): {output_path}")
    return output_path


# ═══════════════════════════════════════════════════════════════════════
# Phase 4: Compositing (FFmpeg)
# ═══════════════════════════════════════════════════════════════════════

def _get_media_duration(path: str) -> float:
    """Get duration of a media file using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        output = result.stdout.strip()
        if output:
            return float(output)
    except Exception as e:
        print(f"[SaaSShorts] ⚠️ ffprobe failed for {path}: {e}")
    return 30.0  # Fallback to 30s estimate


def _format_ass_time(seconds: float) -> str:
    """Format time for ASS subtitle format: H:MM:SS.cc"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds - int(seconds)) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def transcribe_audio_for_subs(audio_path: str) -> list:
    """
    Transcribe audio with word-level timestamps using faster-whisper.
    Returns list of {"word": str, "start": float, "end": float}.
    """
    from faster_whisper import WhisperModel

    print(f"[SaaSShorts] 🎙️ Transcribing audio for subtitles...")
    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, info = model.transcribe(audio_path, word_timestamps=True)

    words = []
    for segment in segments:
        if segment.words:
            for w in segment.words:
                words.append({
                    "word": w.word.strip(),
                    "start": w.start,
                    "end": w.end,
                })

    print(f"[SaaSShorts] ✅ Transcribed {len(words)} words")
    return words


def generate_tiktok_subs(audio_path: str, output_path: str, max_words: int = 3) -> str:
    """
    Generate TikTok-style ASS subtitles from audio using Whisper transcription.

    Style: Big bold centered text, 1-3 words at a time, white with black outline.
    Matches actual spoken words with precise timestamps.
    """
    words = transcribe_audio_for_subs(audio_path)
    if not words:
        # Fallback: empty subtitle file
        with open(output_path, "w") as f:
            f.write("")
        return output_path

    # Group words into chunks of max_words
    chunks = []
    for i in range(0, len(words), max_words):
        group = words[i : i + max_words]
        text = " ".join(w["word"] for w in group).upper()
        start = group[0]["start"]
        end = group[-1]["end"]
        chunks.append({"text": text, "start": start, "end": end})

    # Build ASS file with TikTok style
    ass_content = """[Script Info]
Title: TikTok Style Subs
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: TikTok,Arial Black,90,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,4,0,2,40,40,120,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    for chunk in chunks:
        start = _format_ass_time(chunk["start"])
        end = _format_ass_time(chunk["end"])
        text = chunk["text"].replace("\n", "\\N")
        ass_content += f"Dialogue: 0,{start},{end},TikTok,,0,0,0,,{text}\n"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(ass_content)

    print(f"[SaaSShorts] ✅ TikTok subs: {len(chunks)} captions from {len(words)} words")
    return output_path


def generate_srt_from_script(segments: list, output_path: str) -> str:
    """Fallback: generate basic SRT from script segments (used if no audio available)."""
    srt_content = ""
    index = 1
    for seg in segments:
        text = seg.get("subtitle_text") or seg.get("narration", "")
        if not text:
            continue
        words = text.split()
        chunk_size = 3
        start_time = seg["start"]
        end_time = seg["end"]
        duration = end_time - start_time
        chunks = [words[i : i + chunk_size] for i in range(0, len(words), chunk_size)]
        chunk_dur = duration / max(len(chunks), 1)
        for i, chunk in enumerate(chunks):
            cs = start_time + i * chunk_dur
            ce = min(start_time + (i + 1) * chunk_dur, end_time)
            h, m, s, ms = int(cs//3600), int((cs%3600)//60), int(cs%60), int((cs-int(cs))*1000)
            h2, m2, s2, ms2 = int(ce//3600), int((ce%3600)//60), int(ce%60), int((ce-int(ce))*1000)
            srt_content += f"{index}\n{h:02d}:{m:02d}:{s:02d},{ms:03d} --> {h2:02d}:{m2:02d}:{s2:02d},{ms2:03d}\n{' '.join(chunk).upper()}\n\n"
            index += 1
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(srt_content)
    return output_path


def composite_video(
    talking_head_path: str,
    broll_clips: List[Dict],
    srt_path: str,
    hook_text: str,
    output_path: str,
) -> str:
    """
    Composite talking head + b-roll inserts + subtitles into final video.

    broll_clips: [{"path": "/path/to/clip.mp4", "start": 12, "end": 17}]
    """
    print(f"[SaaSShorts] 🎞️ Compositing final video...")

    # Determine subtitle filter based on file type
    safe_sub = srt_path.replace("\\", "/").replace(":", "\\:")
    if srt_path.endswith(".ass"):
        # ASS has styles embedded — use ass filter directly
        sub_filter = f"ass='{safe_sub}'"
    else:
        # SRT fallback with TikTok-ish styling
        sub_style = (
            "Alignment=2,Fontname=Arial Black,Fontsize=24,PrimaryColour=&H00FFFFFF,"
            "OutlineColour=&H00000000,BorderStyle=1,Outline=4,Shadow=0,MarginV=120,Bold=-1"
        )
        sub_filter = f"subtitles='{safe_sub}':force_style='{sub_style}'"

    if not broll_clips:
        # Simple: talking head + subtitles only
        cmd = [
            "ffmpeg", "-y",
            "-i", talking_head_path,
            "-vf", sub_filter,
            "-c:v", "libx264", "-preset", "fast", "-crf", "22",
            "-c:a", "aac", "-b:a", "128k",
            output_path,
        ]
        subprocess.run(cmd, check=True)
        print(f"[SaaSShorts] ✅ Final video (simple): {output_path}")
        return output_path

    # Complex: talking head with b-roll inserts
    th_duration = _get_media_duration(talking_head_path)
    sorted_broll = sorted(broll_clips, key=lambda x: x["start"])

    # Get actual b-roll durations and limit segment lengths
    broll_durations = {}
    for i, clip in enumerate(sorted_broll):
        broll_durations[i] = _get_media_duration(clip["path"])
        print(f"[SaaSShorts] B-roll {i} actual duration: {broll_durations[i]:.1f}s")

    # Build segment list — limit b-roll segments to actual clip duration
    segments = []
    prev_end = 0.0

    for i, clip in enumerate(sorted_broll):
        bstart = clip["start"]
        actual_dur = broll_durations[i]
        # B-roll segment can't be longer than the actual clip
        bend = min(clip["end"], bstart + actual_dur)

        if prev_end < bstart:
            segments.append({"type": "th", "start": prev_end, "end": bstart})

        segments.append({
            "type": "broll",
            "index": i,
            "start": bstart,
            "end": bend,
            "duration": bend - bstart,
        })
        prev_end = bend

    if prev_end < th_duration:
        segments.append({"type": "th", "start": prev_end, "end": th_duration})

    # Build FFmpeg filter_complex
    inputs = ["-i", talking_head_path]
    for clip in sorted_broll:
        inputs.extend(["-i", clip["path"]])

    filter_parts = []
    concat_parts = []

    # Normalize all segments to same resolution and fps for concat
    norm = "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,fps=30,setsar=1"

    for j, seg in enumerate(segments):
        if seg["type"] == "th":
            filter_parts.append(
                f"[0:v]trim=start={seg['start']:.3f}:end={seg['end']:.3f},setpts=PTS-STARTPTS,{norm}[tv{j}]"
            )
            filter_parts.append(
                f"[0:a]atrim=start={seg['start']:.3f}:end={seg['end']:.3f},asetpts=PTS-STARTPTS[ta{j}]"
            )
            concat_parts.append(f"[tv{j}][ta{j}]")
        else:
            idx = seg["index"] + 1
            dur = seg["duration"]
            filter_parts.append(
                f"[{idx}:v]trim=start=0:end={dur:.3f},setpts=PTS-STARTPTS,{norm}[bv{j}]"
            )
            filter_parts.append(
                f"[0:a]atrim=start={seg['start']:.3f}:end={seg['end']:.3f},asetpts=PTS-STARTPTS[ba{j}]"
            )
            concat_parts.append(f"[bv{j}][ba{j}]")

    n = len(segments)
    filter_parts.append(
        f"{''.join(concat_parts)}concat=n={n}:v=1:a=1[outv][outa]"
    )
    filter_parts.append(
        f"[outv]{sub_filter}[finalv]"
    )

    filter_str = ";".join(filter_parts)

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_str,
        "-map", "[finalv]",
        "-map", "[outa]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-c:a", "aac", "-b:a", "128k",
        output_path,
    ]

    subprocess.run(cmd, check=True)
    print(f"[SaaSShorts] ✅ Final video (composite): {output_path}")
    return output_path


# ═══════════════════════════════════════════════════════════════════════
# Orchestrator: Full Pipeline
# ═══════════════════════════════════════════════════════════════════════

def generate_full_video(
    script: dict,
    config: dict,
    output_dir: str,
    log: Callable[[str], None] = print,
) -> dict:
    """
    Full SaaSShorts video generation pipeline.

    Args:
        script: A single script object from generate_scripts()
        config: {
            "fal_key": str,
            "elevenlabs_key": str,
            "voice_id": str (optional),
            "actor_description": str (optional, overrides script),
        }
        output_dir: Directory to write output files
        log: Callback for progress logging

    Returns:
        {"video_path": str, "srt_path": str, "actor_image": str, "cost_estimate": dict}
    """
    os.makedirs(output_dir, exist_ok=True)

    fal_key = config["fal_key"]
    elevenlabs_key = config["elevenlabs_key"]
    voice_id = config.get("voice_id", "21m00Tcm4TlvDq8ikWAM")
    actor_desc = config.get("actor_description") or script.get("actor_description", "a young professional in their late 20s, wearing a casual modern outfit, clean background")

    title_slug = re.sub(r"[^a-z0-9]+", "_", script.get("title", "video").lower())[:30]

    # Paths
    actor_img = os.path.join(output_dir, f"{title_slug}_actor.png")
    audio_path = os.path.join(output_dir, f"{title_slug}_voice.mp3")
    talking_head = os.path.join(output_dir, f"{title_slug}_head.mp4")
    srt_path = os.path.join(output_dir, f"{title_slug}_subs.ass")
    final_path = os.path.join(output_dir, f"{title_slug}_final.mp4")

    full_narration = script.get("full_narration", "")
    if not full_narration:
        full_narration = " ".join(
            seg.get("narration", "") for seg in script.get("segments", [])
        )

    def _exists(path):
        return os.path.exists(path) and os.path.getsize(path) > 0

    # ── Step 1 & 2: Generate actor image + voiceover in parallel ──
    # If user pre-selected an actor image, copy it
    selected_actor = config.get("selected_actor_path")
    if selected_actor and os.path.exists(selected_actor) and not _exists(actor_img):
        import shutil
        shutil.copy2(selected_actor, actor_img)
        log("[1/6] Using pre-selected actor image.")

    need_img = not _exists(actor_img)
    need_voice = not _exists(audio_path)

    if need_img or need_voice:
        tasks = []
        if need_img:
            tasks.append("actor image")
        if need_voice:
            tasks.append("voiceover")
        log(f"[1/6] Generating {' + '.join(tasks)} (parallel)...")

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_img = executor.submit(generate_actor_image, actor_desc, fal_key, actor_img) if need_img else None
            future_voice = executor.submit(
                generate_voiceover, full_narration, elevenlabs_key, audio_path, voice_id
            ) if need_voice else None

            if future_img:
                actor_img = future_img.result()
            if future_voice:
                audio_path = future_voice.result()

        log("[2/6] Actor image and voiceover ready.")
    else:
        log("[1/6] Actor image and voiceover cached, skipping.")
        log("[2/6] ✅ Using cached assets.")

    # ── Step 3: Generate talking head ──
    video_mode = config.get("video_mode", "premium")
    if not _exists(talking_head):
        if video_mode == "lowcost":
            log("[3/6] Generating talking head (Low Cost: Hailuo + VEED Lipsync)... This takes 2-5 minutes.")
            talking_head = generate_talking_head_lowcost(actor_img, audio_path, fal_key, talking_head)
        else:
            log("[3/6] Generating talking head video (Kling Avatar v2)... This takes 2-5 minutes.")
            talking_head = generate_talking_head(actor_img, audio_path, fal_key, talking_head)
        log("[3/6] Talking head ready.")
    else:
        log("[3/6] ✅ Talking head cached, skipping.")

    # ── Step 4: Generate b-roll clips ──
    broll_segments = [
        seg for seg in script.get("segments", [])
        if seg.get("broll_prompt") and seg.get("visual") == "broll"
    ]

    broll_clips = []
    if broll_segments:
        # Check which b-roll clips need generating
        broll_to_generate = []
        for i, seg in enumerate(broll_segments):
            broll_path = os.path.join(output_dir, f"{title_slug}_broll_{i}.mp4")
            if _exists(broll_path):
                broll_clips.append({
                    "path": broll_path,
                    "start": seg["start"],
                    "end": seg["end"],
                })
                log(f"  ✅ B-roll {i} cached, skipping.")
            else:
                broll_to_generate.append((i, seg, broll_path))

        if broll_to_generate:
            log(f"[4/6] Generating {len(broll_to_generate)} b-roll clips...")
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {}
                for i, seg, broll_path in broll_to_generate:
                    future = executor.submit(
                        generate_broll, seg["broll_prompt"], fal_key, broll_path
                    )
                    futures[future] = {"seg": seg, "path": broll_path}

                for future in as_completed(futures):
                    info = futures[future]
                    try:
                        path = future.result()
                        broll_clips.append({
                            "path": path,
                            "start": info["seg"]["start"],
                            "end": info["seg"]["end"],
                        })
                        log(f"  ✅ B-roll clip ready: {os.path.basename(path)}")
                    except Exception as e:
                        log(f"  ⚠️ B-roll failed (skipping): {e}")
        else:
            log("[4/6] ✅ All b-roll cached, skipping.")
    else:
        log("[4/6] No b-roll segments in script, skipping.")

    # ── Step 5: Generate subtitles (from actual audio, not script text) ──
    log("[5/6] Transcribing audio and generating TikTok-style subtitles...")
    generate_tiktok_subs(audio_path, srt_path, max_words=2)

    # ── Step 6: Composite final video ──
    log("[6/6] Compositing final video with FFmpeg...")
    hook_text = script.get("hook_text", "")
    composite_video(talking_head, broll_clips, srt_path, hook_text, final_path)

    log("🎉 Video generation complete!")

    # Cost estimate
    audio_duration = _get_media_duration(audio_path)
    if video_mode == "lowcost":
        cost = {
            "actor_image_flux": 0.05,
            "voiceover_elevenlabs": round(len(full_narration) * 0.00003, 3),
            "hailuo_img2video": 0.19,
            "veed_lipsync": 0.20,
            "broll_flux": round(len(broll_clips) * 0.05, 2),
            "ffmpeg_compositing": 0.00,
        }
    else:
        cost = {
            "actor_image_flux": 0.05,
            "voiceover_elevenlabs": round(len(full_narration) * 0.00003, 3),
            "talking_head_kling": round(audio_duration * 0.056, 2),
            "broll_kling": round(len(broll_clips) * 5 * 0.07, 2),
            "ffmpeg_compositing": 0.00,
        }
    cost["total"] = round(sum(cost.values()), 2)

    return {
        "video_path": final_path,
        "video_filename": os.path.basename(final_path),
        "srt_path": srt_path,
        "actor_image": actor_img,
        "duration": audio_duration,
        "cost_estimate": cost,
    }
