---
title: 'Indonesian Podcast Comedy Multi-Pass Selection'
type: 'feature'
created: '2026-06-13'
status: 'done'
baseline_commit: '3f58ccb086bc123a67ab136d3f570303e45ff59a'
context:
  - '{project-root}/docs/superpowers/specs/2026-06-13-indonesian-podcast-comedy-multipass-design.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** `podcast_comedy` sends duplicated transcript content and asks one AI pass to discover and rank nuanced Indonesian podcast moments, producing inconsistent selections for slang-heavy comedy, hot takes, and personal stories.

**Approach:** Send one timestamped segment transcript to a scout pass, compare consolidated candidates in an Indonesian judge pass, then apply deterministic validation, balanced ranking, word-boundary snapping, and a scout fallback when judging fails.

## Boundaries & Constraints

**Always:** Preserve local word timestamps for final snapping; use the existing OpenAI-compatible client and environment configuration; balance `COMEDY`, `HOT_TAKE`, and `PERSONAL_STORY` by quality; target 20-40 seconds for comedy and 35-60 seconds for other lanes; keep existing downstream metadata fields.

**Ask First:** Adding a provider-specific SDK, changing API-key handling, altering extraction/cropping/subtitle behavior, or expanding changes beyond clip analysis and its focused tests.

**Never:** Send both full transcript text and word-level transcript content to the AI; hard-code individual creator names as selection rules; allow judge output to introduce unknown candidates; fail the full job solely because the judge pass fails.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|---------------|----------------------------|----------------|
| Normal multi-pass | Indonesian timestamped segments and valid scout/judge JSON | Globally ranked, validated, mixed-lane clips | Invalid individual candidates are dropped |
| Long transcript | Multiple bounded segment windows | Only overlapping segments appear in each scout prompt | Duplicate candidates are consolidated |
| Judge failure | Timeout, exception, or invalid JSON | Validated scout candidates are ranked and returned | Log fallback and continue |
| Invalid judge selection | Unknown candidate ID or out-of-context timestamps | Selection is rejected | Other valid selections continue |

</frozen-after-approval>

## Code Map

- `main.py` -- transcript shaping, scout/judge prompts, candidate validation, ranking, orchestration, and corrected window filtering.
- `tests/test_clip_quality.py` -- dependency-light unit and orchestration regression tests.
- `docs/superpowers/specs/2026-06-13-indonesian-podcast-comedy-multipass-design.md` -- approved design and acceptance boundary.

## Tasks & Acceptance

**Execution:**
- [x] `tests/test_clip_quality.py` -- add failing tests for canonical transcripts, bounded windows, candidate consolidation, judge validation, lane-aware duration/ranking, and judge fallback.
- [x] `main.py` -- add focused helpers for timestamped segments, scout prompts, consolidation, candidate contexts, judge prompts, final validation, and fallback ranking.
- [x] `main.py` -- route `podcast_comedy` through two AI passes while retaining the existing path for other categories.
- [x] `tests/test_clip_quality.py` -- retain existing metadata and boundary regression coverage.

**Acceptance Criteria:**
- Given a `podcast_comedy` transcript, when the scout prompt is built, then transcript language appears once as timestamped segments and word arrays are absent.
- Given long-video windows, when a window prompt is built, then segments outside both window bounds are excluded.
- Given overlapping or malformed scout candidates, when consolidation runs, then only valid grounded candidates and the strongest duplicates remain.
- Given judge selections, when validation runs, then unknown IDs, invalid boundaries, and lane-inappropriate durations are rejected.
- Given similarly strong candidates from multiple lanes, when ranking runs, then lane diversity is preserved without promoting candidates substantially weaker than the best options.
- Given a failed judge call, when analysis completes, then validated Pass 1 candidates populate compatible clip metadata.

## Design Notes

The judge receives compact candidate records plus at most 8 seconds of lead-in and 5 seconds of follow-up context. Default hard duration limits are 15-70 seconds, with lane targets enforced using a 5-second tolerance. Pass 1 is the availability fallback; deterministic code validates contracts but does not attempt to infer sarcasm or humor.

## Verification

**Commands:**
- `python3 -m pytest tests/test_clip_quality.py -q` -- expected: all focused tests pass.
- `python3 -m py_compile main.py subtitles.py hooks.py` -- expected: exit code 0.

## Suggested Review Order

**Orchestration**

- Routes only `podcast_comedy` through the two-pass path and preserves generic categories.
  [`main.py:2666`](../../main.py#L2666)

- Coordinates bounded scout windows, global judging, validation, and fallback.
  [`main.py:2575`](../../main.py#L2575)

**Validation**

- Shapes one monotonic timestamped transcript and splits unusually long segments.
  [`main.py:253`](../../main.py#L253)

- Rejects malformed, ungrounded, duplicate, and out-of-range scout candidates.
  [`main.py:380`](../../main.py#L380)

- Enforces candidate identity, payoff inclusion, durations, scores, and grounded evidence.
  [`main.py:570`](../../main.py#L570)

**Downstream Compatibility**

- Normalizes hook metadata while preserving existing consumer fields.
  [`main.py:2244`](../../main.py#L2244)

- Expands cuts around speech instead of trimming through boundary words.
  [`main.py:2362`](../../main.py#L2362)

**Tests**

- Exercises the real OpenAI-compatible two-call route without network access.
  [`test_clip_quality.py:417`](../../tests/test_clip_quality.py#L417)

- Covers judge failures, malformed configuration, and scout fallback.
  [`test_clip_quality.py:349`](../../tests/test_clip_quality.py#L349)

- Covers invalid judge IDs, boundaries, payoff cuts, duration, and score gates.
  [`test_clip_quality.py:250`](../../tests/test_clip_quality.py#L250)
