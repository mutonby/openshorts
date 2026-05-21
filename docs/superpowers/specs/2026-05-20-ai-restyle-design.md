# Design — AI Restyle (new sidebar product)

**Status:** Approved by user 2026-05-20. Implementation plan pending.
**Working name:** AI Restyle (sidebar label "AI Restyle"). Internal product ID: `ai-restyle`.

---

## 1. Goal & non-goals

### Goal

Let users upload a video they've already produced and get back the **same video with a transformed background and lighting**. Same person, same words, same motion, same cuts, same duration — restyled to look like it was shot in the user-chosen location with the user-chosen lighting setup.

Concretely: a hand-held phone clip filmed in a dim living room becomes a clip that looks like it was filmed in a Bahamas beach at golden hour, or in a clean white studio with softbox lighting.

### Non-goals (v1)

This product **does not**:
- Run viral extraction or pick "best moments" from a long source (that's Short-form's job).
- Apply subtitles, AI edits (zoompan/cuts), color grade, or silence removal. These are deliberately out of scope. Users export the restyled output and feed it into Short-form if they want polish.
- Generate net-new video content (no Kling/Sora image-to-video). The output content is the input content; only style changes.
- Handle videos >30s. v1 caps at 30 seconds (single video-to-video call). Longer durations come in a separate milestone (see §10 Future Work).
- Replace the user's audio (original audio is preserved bit-for-bit).
- Generate net-new audio (TTS, music, SFX).

---

## 2. User flow

```
Sidebar: AI Restyle  (between Long-form and Short-form)
        │
        ▼
┌─────────────┐    ┌─────────────┐    ┌──────────────┐
│  1. Upload  │ →  │ 2. Configure│ →  │  3. Review   │
│  (≤30s MP4) │    │ (presets +  │    │ (before/after│
│             │    │  prompt)    │    │  + Send to   │
│             │    │             │    │  Short-form) │
└─────────────┘    └─────────────┘    └──────────────┘
                                              │
                                              ▼
                              Optional: Send-to-Short-form
                              hands the file to /short-form's
                              Upload step, pre-loaded.
```

### Step 1 — Upload

- Single drop zone. MP4/MOV. ≤2GB. ≤30s **(client-side `<video>.duration` probe BEFORE upload; server re-verifies via ffprobe and rejects with HTTP 413)**.
- Rejection message for too-long videos: *"AI Restyle v1 caps at 30s. Trim your video first or use Short-form."*
- Reuses `Upload.jsx`'s MIME + ftyp validation logic.
- CTA: `Continue →` disabled until file passes validation.

### Step 2 — Configure

- Two dropdowns side by side: **Background** + **Lighting**. Each lists user's presets (read from `localStorage`).
- Selected preset's prompt text is rendered in an **editable textarea** beneath the dropdowns. Editing this box overrides the preset prompt *for this job only* — the saved preset is unchanged.
- The "effective prompt" sent to Nano Banana = `background_text + " • " + lighting_text + " • " + safety_constraints` (safety constraints are hard-coded server-side).
- CTA: `Start restyle →`.

### Step 3 — Review

- Phone-frame preview of the restyled clip.
- **Before/After** toggle (Original vs Restyled) — same UX as Short-form `Review.jsx`.
- **Download** link to the restyled MP4.
- **Send to Short-form** primary CTA. Stashes the restyled file into a routing payload that `/short-form` reads on mount and pre-loads into its Upload step. Closes the loop with the existing pipeline.
- No editing controls (no stage selector, no LUT picker — those belong to Short-form).

---

## 3. Backend architecture

### Route module

**New file:** `backend/app/routes/ai_restyle.py` — first populated file in `backend/app/routes/`. Establishes the pattern future router-split work will follow.

```python
POST /api/restyle
  Form fields:
    file: UploadFile             # MP4/MOV, ≤30s, ≤2GB
    background_prompt: str        # Effective background text (≤500 chars)
    lighting_prompt: str          # Effective lighting text (≤500 chars)
  Headers:
    X-Gemini-Key: str            # required (Nano Banana)
    X-Fal-Key: str               # required (video-to-video)
  Returns:
    { "job_id": "<uuid>" }

GET /api/restyle/{job_id}
  Returns:
    { "status": "processing" | "completed" | "failed",
      "logs": [str, ...],
      "progress_pct": 0..100,    # coarse: 0/10/40/85/100
      "result": { "video_url": "/videos/{job_id}/restyled_{filename}.mp4",
                  "original_url": "/videos/{job_id}/{filename}",
                  "duration_sec": float } | null }
```

`POST /api/restyle` and `GET /api/restyle/{job_id}` are wired into `backend/app/main.py`'s router list in the same place the existing routes are registered.

Job state lives in the same `jobs` dict that `main.py` already manages, with `status` semantics identical to Short-form (`processing → completed | failed`). The frontend reuses `useJobPolling.js` unchanged.

### Pipeline orchestrator

**New file:** `backend/app/saas/restyle_pipeline.py`. (Lives under `saas/` because it's the same architectural neighbor as the existing SaaSShorts pipeline — both are multi-stage ML orchestrators that compose Gemini + fal.ai + FFmpeg. Long-term it may move to its own package; out of scope for v1.)

Single public function:

```python
async def run_restyle_job(
    job_id: str,
    input_path: str,
    background_prompt: str,
    lighting_prompt: str,
    gemini_key: str,
    fal_key: str,
) -> None:
    """Orchestrate the 7-step restyle pipeline. Updates jobs[job_id] in place."""
```

### Pipeline steps

| # | Step | Module | Failure mode |
|---|---|---|---|
| 1 | Validate upload (MIME + ftyp + duration ≤30s) | `routes/ai_restyle.py` | HTTP 415 / 413 |
| 2 | Probe duration via ffprobe | `video/ffmpeg.py:probe_duration` (exists) | 500 |
| 3 | Extract first frame to PNG | `ml/frame_extract.py` (NEW) | 500, log + fail job |
| 4 | Nano Banana relight | `ml/frame_relight.py` (NEW) | 500, log + fail job |
| 5 | Video-to-video restyle | `ml/video_restyle.py` (NEW) | 500, log + fail job |
| 6 | Mux original audio | `video/ffmpeg.py:mux_video_audio` (exists) | 500, log + fail job |
| 7 | Write metadata.json + serve | `routes/ai_restyle.py` | 500 |

All FFmpeg operations route through `backend/app/video/ffmpeg.py` (Convention #1).

### New ML modules

#### `backend/app/ml/frame_extract.py`

```python
def extract_first_frame(video_path: str, out_path: str) -> str:
    """Extract frame at t=0 to PNG. Returns out_path on success."""
```

One-line FFmpeg call (`-ss 0 -frames:v 1 -y`). ~30 lines including docstring + error handling.

#### `backend/app/ml/frame_relight.py`

```python
def relight_frame(
    api_key: str,
    frame_path: str,
    background_prompt: str,
    lighting_prompt: str,
    out_path: str,
) -> str:
    """Call gemini-2.5-flash-image-preview with the frame + relight prompts.
    Returns out_path on success."""
```

Mirrors the existing pattern in `backend/app/thumbnails/images.py:generate_thumbnail` (already calls the same model). The effective prompt template:

```
Relight this image with the following style. Keep the person, pose,
clothing, and composition EXACTLY as in the source. Only change the
background and lighting.

Background: {background_prompt}
Lighting: {lighting_prompt}

Do not add or remove any people or objects. Do not change facial features
or body proportions. Preserve the framing and camera angle.
```

Safety constraints (the "keep person/pose/clothing" and "no add/remove people" clauses) are hard-coded in this module, not user-controllable.

#### `backend/app/ml/video_restyle.py`

```python
def restyle_video(
    api_key: str,
    video_path: str,
    reference_frame_path: str,
    out_path: str,
) -> str:
    """Call fal.ai video-to-video with the source video + reference frame.
    Returns out_path on success."""
```

**Model selection deferred to implementation Phase 0** — needs a quick research spike against fal.ai's current catalog (as of 2026-05) to pick between Wan v2.5 video-to-video, Luma Ray2 reference-conditioned, Runway Gen-3 Alpha Turbo v2v (if available on fal.ai), or an alternative. Acceptance criteria for the model:
- Accepts a source video (≤30s) AND a reference image.
- Restyles video to match reference's lighting + background while preserving motion + content.
- Cost: ≤$2 per 30s gen.
- Latency: ≤5min per 30s gen.

If no fal.ai-hosted model meets the bar, fallback is direct Runway API integration (Runway has a v2v product). This is the highest-risk decision in the entire plan and the implementation plan's Phase 0 is dedicated to validating it.

---

## 4. Frontend architecture

### Pages directory

**New folder:** `frontend/src/pages/AIRestyle/` (sibling to `LongForm/` and `ShortForm/`). Per Convention #7, this folder does **not** cross-import from `LongForm/` or `ShortForm/`. Shared code goes through `hooks/`, `components/ui/`, `state/`, or `lib/`.

Files:

```
frontend/src/pages/AIRestyle/
├── index.jsx        # Routes + Wizard mount + History tab
├── Wizard.jsx       # 3-step wizard wrapper (mirrors ShortForm/Wizard.jsx)
├── History.jsx      # Past restyle jobs (read from localStorage, last 20)
└── steps/
    ├── Upload.jsx
    ├── Configure.jsx
    └── Review.jsx
```

### State

Wizard state shape (persisted to `localStorage` via `useWizard`):

```js
{
  step: 0,
  data: {
    file: { id, name, size, durationSec, file: File } | null,
    selection: {
      backgroundPresetId: string,    // FK into presetsStore
      lightingPresetId: string,
      backgroundPromptOverride: string | null,  // if user edited the textarea
      lightingPromptOverride: string | null,
    },
    job: {
      jobId: string,
      status: 'idle' | 'processing' | 'completed' | 'failed',
      result: { video_url, original_url, duration_sec } | null,
      progressPct: number,
      logs: string[],
    } | null,
  },
}
```

`File` handle is lost on reload — same constraint as Short-form. The `resetOnRehydrate` guard mirrors `ShortForm/Wizard.jsx`'s shortFormNeedsFreshUpload pattern.

### Sidebar entry

**Modified file:** `frontend/src/layouts/Sidebar.jsx`. Add one entry to `NAV` between Long-form and Short-form:

```js
{ to: '/ai-restyle', label: 'AI Restyle', icon: Wand2 },
```

Icon: `Wand2` from `lucide-react`.

### Settings tab

**Modified file:** `frontend/src/pages/Settings/` — adds a new tab "AI Restyle" alongside existing tabs.

Tab layout: two sections (Backgrounds, Lightings), each showing the user's preset list with row-level actions:

```
┌──────────────────────────────────────────────────────────────┐
│ ★ Studio white                          [Edit] [Delete*]    │
│   clean white seamless backdrop, minimalist photo studio     │
├──────────────────────────────────────────────────────────────┤
│   Sunlit office                         [Edit] [Delete] [★]  │
│   sunlit modern office with floor-to-ceiling windows…        │
├──────────────────────────────────────────────────────────────┤
│ + Add background preset                                      │
└──────────────────────────────────────────────────────────────┘
                  *Disabled for starred default
```

- **★** marks one preset per dimension as "Recommended/Default" (pre-selected in the wizard's Configure step). Clicking [★] on another preset moves the star atomically.
- **[Edit]** opens a modal with two fields: `name` (≤40 chars) + `prompt` (≤500 chars). Cancel discards.
- **[Delete]** is disabled for the starred default (prevents the dropdown from going empty).
- **+ Add preset** opens the same modal as Edit, with empty fields.

### Preset storage

**New file:** `frontend/src/state/aiRestylePresets.js`. Mirrors the pattern in `keysStore.js` and `lib/brandKit.js`:

- LocalStorage key: `openshorts.aiRestyle.presets`
- Custom event for cross-component reactivity: `openshorts:ai-restyle-presets-changed`
- Hook: `useAIRestylePresets()` returns `{ backgrounds: [], lightings: [], setBackgrounds, setLightings, setDefault }`.

### Seed presets (first-load defaults)

| Backgrounds | Prompt fragment |
|---|---|
| ★ Studio white | clean white seamless backdrop, minimalist photo studio, no clutter, perfect color separation |
| Sunlit office | bright modern office interior with floor-to-ceiling windows, soft natural light, plants, wooden desk |
| Bahamas beach | tropical beach with palm trees, turquoise ocean water in the distance, soft white sand |
| Cyberpunk neon | nighttime city street with vivid neon signs, pink-and-cyan color palette, light fog |
| Cinematic forest | deep forest with dappled sunlight through tall pine trees, mossy ground, atmospheric haze |

| Lightings | Prompt fragment |
|---|---|
| ★ Studio softbox | soft diffused studio softbox lighting from camera-left, gentle fill on the right, no harsh shadows |
| Sunlit office | bright daylight pouring through large windows, soft fill on subject's face |
| Golden hour | warm golden-hour sun low and to the side, long shadows, amber and rose tones |
| Cinematic moody | low-key cinematic lighting with strong directional key, deep shadows, single soft fill |
| Neon nighttime | colored neon spill lighting (pink and cyan accents), low ambient, subject lit from multiple sides |

These 10 presets ship with the build; first-load seeding writes them to `localStorage` only if no preset list exists yet. Existing users with custom presets are not overwritten.

---

## 5. Cross-cutting concerns

### Security baseline (per global CLAUDE.md `securing-http-and-llm-endpoints` skill)

`POST /api/restyle` classifies as **STATE-MUTATING + LLM-CALL** (calls Gemini for Nano Banana + fal.ai for v2v). Required controls per the skill's tier matrix, with status:

| Control | Status | Notes |
|---|---|---|
| C1 Auth (BYO API key via header) | ✓ Inherited | `X-Gemini-Key`, `X-Fal-Key` headers |
| C2 Rate limit | DEFER | Same opt-out as HANDOFF.md §5; lands in `/gsd-secure-phase` sweep |
| C3 Input validation | ✓ Required at impl time | Pydantic + duration cap + prompt length cap |
| C4 Timeout/retry/breaker | DEFER | Same opt-out; fal.ai client already has internal retry |
| C5 Output rate limit | N/A | Single-file response |
| C6 PII redaction | N/A | No structured PII in logs |
| C7 Idempotency | DEFER | Same opt-out |
| C8 Concurrency lock | N/A | No shared mutable resource |
| C9 Audit logging | ✓ Required at impl time | Job logs already capture; add cost line per Gemini/fal call |
| C10 Cost / abuse cap | DEFER | Same opt-out |

Implementation plan must add **C3 (Pydantic + duration cap + prompt length cap)** and **C9 (per-call cost logging)** as in-scope tasks. C2/C4/C7/C10 wait for the cross-router sweep.

### Cost telemetry

Each Nano Banana + fal.ai call appends a line to the job logs:

```
💰 Nano Banana relight: $0.039 (1 image)
💰 fal.ai v2v: $1.20 (30s × $0.04/s — model: <name>)
💰 Total: $1.24
```

Total cost surfaces in the Review step as a small footer beneath the Download button.

### Failure handling

- **Nano Banana fails** (content policy / network): job fails with `status='failed'`, logs include the API error message. No partial output.
- **Video-to-video fails** (timeout, content policy, model overload): same.
- **Audio mux fails** (rare): retry once with `-c:a aac` fallback before failing the job.
- **Job times out** (>15min): wizard surfaces a "Job stuck — refresh to retry" message. Backend marks `status='failed'` after 15min.

The frontend never crashes on a failed job — Review step renders the failure logs and offers `Try again` (returns to Configure with the same selection) or `Start over` (back to Upload).

### Tests

Backend (added to `backend/tests/`):
- `unit/test_frame_extract.py` — fixture MP4 → first frame → asserts PNG file exists + dimensions ≥320x320.
- `unit/test_frame_relight.py` — mocks Gemini client, asserts prompt-template formatting + retry-on-content-policy fallback.
- `unit/test_video_restyle.py` — mocks fal.ai client, asserts payload shape + cost telemetry log line.
- `unit/test_restyle_pipeline.py` — runs the orchestrator end-to-end with all three ML modules mocked; asserts job logs cycle through all 7 steps + status flips to `completed`.
- `api/test_openapi_contract.py` — snapshot picks up `/api/restyle` + `/api/restyle/{job_id}`.

Frontend: `npm run build` 0 warnings + browser smoke test per HANDOFF.md §6 rule 6.

---

## 6. Files added / modified

### Added (backend)

- `backend/app/routes/__init__.py` — empty (creates the package)
- `backend/app/routes/ai_restyle.py` — FastAPI router (≈150 lines)
- `backend/app/saas/restyle_pipeline.py` — orchestrator (≈120 lines)
- `backend/app/ml/frame_extract.py` — ≈30 lines
- `backend/app/ml/frame_relight.py` — ≈80 lines
- `backend/app/ml/video_restyle.py` — ≈100 lines (model TBD)

### Added (frontend)

- `frontend/src/pages/AIRestyle/index.jsx`
- `frontend/src/pages/AIRestyle/Wizard.jsx`
- `frontend/src/pages/AIRestyle/History.jsx`
- `frontend/src/pages/AIRestyle/steps/Upload.jsx`
- `frontend/src/pages/AIRestyle/steps/Configure.jsx`
- `frontend/src/pages/AIRestyle/steps/Review.jsx`
- `frontend/src/state/aiRestylePresets.js`

### Added (tests)

- `backend/tests/unit/test_frame_extract.py`
- `backend/tests/unit/test_frame_relight.py`
- `backend/tests/unit/test_video_restyle.py`
- `backend/tests/unit/test_restyle_pipeline.py`

### Modified

- `backend/app/main.py` — register the new router (one-line `app.include_router(ai_restyle.router)`)
- `backend/tests/snapshots/baseline.openapi.json` — regenerate after route is wired
- `frontend/src/App.jsx` — add `<Route path="ai-restyle/*" element={<AIRestyle />} />`
- `frontend/src/layouts/Sidebar.jsx` — add the NAV entry
- `frontend/src/pages/Settings/sections/AIRestylePresetsSection.jsx` (new) — adds the "AI Restyle" tab to the existing sections pattern (`ApiKeysSection`, `BrandKitSection`, etc.)
- `frontend/src/pages/Settings/index.jsx` — register the new section in the tab list
- `ROADMAP.md` — promote "Planned product: AI Restyle" from later to shipped/in-progress
- `~/.claude/CLAUDE.md` — `## OpenShorts (project-specific)` repo-map / module-map auto-managed sections regenerate after backend modules land

---

## 7. Implementation milestones (preview — full plan via `writing-plans`)

| Phase | Scope | Estimate |
|---|---|---|
| **0** | Research spike: validate the video-to-video model on fal.ai. Run 3-5 test gens; verify quality + cost + latency hit the bar. Pick the model. | 0.5–1 day |
| **1** | Backend `ml/frame_extract.py` + `ml/frame_relight.py` + 2 unit tests. Verify Nano Banana relight quality against 3 hand-picked source frames. | 1 day |
| **2** | Backend `ml/video_restyle.py` + `saas/restyle_pipeline.py` + 2 unit tests + manual end-to-end with a single fixture clip. | 1.5 days |
| **3** | Backend `routes/ai_restyle.py` + OpenAPI snapshot regen + pytest gate green. | 0.5 day |
| **4** | Frontend `pages/AIRestyle/` (3-step wizard) + sidebar entry + routing. | 1.5 days |
| **5** | Frontend Settings tab + `aiRestylePresets.js` + seed defaults. | 1 day |
| **6** | Browser smoke test (per HANDOFF.md §6 rule 6) + Codex adversarial review per global CLAUDE.md + commit + ship. | 0.5 day |

**Total estimate:** 6–7 working days, gated on Phase 0 validating the model. If Phase 0 fails (no fal.ai v2v model meets the bar), pivot to Runway direct integration adds ~1 day.

---

## 8. Roadmap entry

Promotes the entry already saved as project memory (`project_ai_short_form.md`) into `ROADMAP.md` under a new section:

```
### AI Restyle (new product, v1 in progress)

**Stubbed in v1**
- Restyle a video's lighting + background while preserving content, motion, and audio.
- Sidebar entry between Long-form and Short-form.
- 3-step wizard: Upload → Configure → Review.
- Settings tab "AI Restyle" with CRUD for Background + Lighting preset prompts.
- Cap: 30s per video.

**Later**
- Lift the 30s cap via chunked v2v with shared reference frame (Approach B from design).
- Bridge from Short-form Review's stage selector ("+ AI Restyle" stage).
- Auto-suggest preset based on the source frame (Gemini-driven).
```

---

## 9. Decisions

| ID | Decision | Rationale |
|---|---|---|
| **D1** | v1 caps at 30s. | Single video-to-video call. Predictable cost (~$1-2). No stitch artifacts. Validates the model choice cheaply before committing to chunking. |
| **D2** | No editing chain in this product. | User's explicit ask. Restyled output can be fed into Short-form for polish. Keeps this product's value proposition crisp. |
| **D3** | Presets in `localStorage`, not backend. | Same pattern as `keysStore` + `brandKit`. No backend persistence needed. User-editable in Settings without an API round-trip. |
| **D4** | Two preset dimensions (background + lighting), not one combined preset. | Users want to mix-and-match: "Bahamas beach" × "Cinematic moody" is a different mood than "Bahamas beach" × "Golden hour". Two dropdowns is 5×5=25 combos with 10 stored fragments — much higher leverage than 25 stored combos. |
| **D5** | Per-job prompt override via textarea. | Lets power users iterate without polluting their saved presets. Editing the textarea overrides for THIS job only. |
| **D6** | Model choice deferred to Phase 0. | High-uncertainty — fal.ai's v2v catalog evolves quickly. A 0.5-day spike early de-risks the whole plan. If no fal.ai option meets the bar, fallback is Runway direct API (~1 extra day). |
| **D7** | Reuse `backend/app/saas/` neighbor for pipeline orchestration. | Architectural fit: SaaSShorts is the existing multi-stage Gemini+fal.ai+FFmpeg orchestrator. Restyle is structurally identical. Moving to its own package can come later. |
| **D8** | Audio is original audio, untouched. | Subtitles in Short-form rely on audio-timestamp alignment with the transcript. Re-encoding or replacing audio would break that downstream contract. |
| **D9** | First file populated under `backend/app/routes/`. | The router-split refactor (HANDOFF.md §5) has been deferred indefinitely. AI Restyle landing here as the first inhabitant sets the precedent. Future router-split work moves existing routes to neighbors. |
| **D10** | Defer C2 / C4 / C7 / C10 security controls. | Inherited from existing per-route opt-outs (Phase 1 D3, Phase 2 D4 in the polish plan). Cross-router sweep lands in `/gsd-secure-phase`. |

---

## 10. Future work (explicitly out of scope for v1)

- **Long-form support (>30s up to 3min)** — requires chunk-and-stitch logic with shared reference frame. Plan separately as "AI Restyle: long-form".
- **Voiceover / TTS** — for AI ads use case. Probably belongs in a different product (e.g. SaaSShorts) than AI Restyle.
- **Multi-reference frames** — pick a reference frame per scene boundary instead of just the first frame. Improves quality for source videos with hard cuts.
- **Preset marketplace / sharing** — currently presets are per-browser localStorage. Backend-stored presets with team sharing is a v3 feature.
- **In-line preset preview** — show a Nano-Banana-relit thumbnail of the source's first frame in the Configure step before committing to the full v2v call. Saves cost when users iterate.

---

*End of design.*
