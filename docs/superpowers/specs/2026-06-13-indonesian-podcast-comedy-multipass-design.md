# Indonesian Podcast Comedy Multi-Pass Clip Selection

## Summary

Upgrade the existing global `podcast_comedy` behavior in `main.py` for Indonesian
podcast content similar to Grind Boys and Praz Teguh. The pipeline will use an
OpenAI-compatible DeepSeek V4 Flash endpoint in two semantic passes, followed by
deterministic validation and the existing word-level boundary snapping.

The target content mix is balanced across:

- Comedy, roasting, and conversational banter
- Hot takes and controversial opinions
- Personal stories and relatable moments

Clip duration remains dynamic:

- Comedy: 20-40 seconds
- Hot takes and personal stories: 35-60 seconds

The system prioritizes quality over quantity and may return fewer clips when the
source contains few strong moments.

## Goals

- Improve understanding of Indonesian slang, sarcasm, callbacks, and group banter.
- Select complete moments with enough setup, a clear payoff, and a clean ending.
- Compare candidates globally instead of trusting scores produced independently.
- Reduce prompt duplication by sending one canonical timestamped transcript.
- Keep final timestamps precise without sending word-level data to the AI.
- Preserve usable results when the second AI pass fails.

## Non-Goals

- Training or fine-tuning a custom model.
- Speaker diarization as part of this change.
- Audio-based laughter, volume, or emotion detection.
- Changing video cropping, subtitles, hook rendering, or extraction codecs.
- Creating channel-specific hard-coded rules for individual creators.

## Current Problems

The current implementation sends both full transcript text and `WORDS_JSON`.
These contain substantially duplicated language content and increase prompt size.

The specialized comedy prompt asks one model response to discover, score, rank,
and write metadata at once. It cannot reliably compare candidates produced in
different long-video windows.

`post_process_clips()` mainly trusts the model score. It validates duration,
confidence, overlap, and pattern diversity, but does not validate semantic
evidence such as setup, payoff, reaction, or standalone completeness.

The current `_build_window_prompt()` selects transcript segments whose end is
after the window start without also requiring their start to be before the
window end. A window can therefore include transcript content far beyond its
word range.

## Architecture

```text
Whisper result
    |
    +-- timestamped utterance transcript --> Pass 1: Candidate Scout
    |                                          |
    |                                          v
    |                                  candidate collection
    |                                          |
    |                         local context extraction per candidate
    |                                          |
    |                                          v
    |                              Pass 2: Indonesian Judge
    |                                          |
    +-- word timestamps -----------------------+--> deterministic validator
                                                       |
                                                       v
                                                boundary snapping
                                                       |
                                                       v
                                                   extraction
```

The AI provider remains behind the existing OpenAI-compatible client. Model and
base URL stay environment-configurable so the pipeline does not depend on a
provider-specific SDK.

## Canonical Transcript

AI prompts receive only one transcript representation:

```json
[
  {
    "id": 42,
    "start": 120.4,
    "end": 126.8,
    "text": "Gue waktu itu sebenarnya takut banget...",
    "confidence": 0.94
  }
]
```

The transcript builder will:

- Preserve complete utterances or sentence-like segments.
- Split unusually long segments at strong punctuation or natural pauses.
- Keep monotonically increasing absolute timestamps.
- Calculate a segment confidence from its word probabilities when available.
- Exclude word arrays from AI prompts.

The original word timestamps remain local and continue to feed
`snap_clip_to_boundaries()`.

## Pass 1: Candidate Scout

Pass 1 receives the canonical timestamped transcript once. For long videos, it
receives bounded overlapping windows containing only segments that overlap the
window.

Each window requests more candidates than the final output requires. Across the
whole video, the default limit is 24 raw candidates before deduplication. The
model may return fewer candidates when the source lacks strong moments.

Each candidate must include:

```json
{
  "candidate_id": "w2-c4",
  "content_lane": "COMEDY",
  "start": 120.4,
  "end": 154.2,
  "setup_start": 120.4,
  "payoff_start": 145.1,
  "payoff_end": 151.8,
  "reaction_end": 154.2,
  "hook_evidence": "quoted or directly referenced transcript evidence",
  "payoff_evidence": "quoted or directly referenced transcript evidence",
  "standalone_summary": "what a new viewer understands",
  "suggested_hook": "MAX 6 WORDS",
  "scout_score": 0.88
}
```

`content_lane` must be one of:

- `COMEDY`
- `HOT_TAKE`
- `PERSONAL_STORY`

Selection guidance:

- Comedy must contain a recognizable setup/build/payoff cycle. Reaction is
  preferred when it adds value, but dead air must not be retained.
- Hot takes must contain a clear claim and enough reasoning or consequence to
  avoid becoming empty rage bait.
- Personal stories must contain a meaningful turn, reveal, lesson, or emotional
  payoff rather than only background exposition.
- Indonesian filler, repeated acknowledgements, greetings, sponsor reads, and
  context-dependent inside jokes receive explicit penalties.
- Hooks and metadata must stay grounded in spoken content.

Pass 1 does not make the final selection.

## Candidate Consolidation

Python merges candidates from all windows before Pass 2:

- Normalize numeric fields and content lane names.
- Reject malformed or out-of-video timestamps.
- Reject candidates with no hook or payoff evidence.
- Remove near-duplicates using timestamp overlap.
- Keep the stronger candidate when duplicate windows identify the same moment.
- Extract a limited transcript context around each surviving candidate.

By default, context covers the candidate plus up to 8 seconds before it and
5 seconds after it, clamped to the video bounds. It must not include the full
transcript.

## Pass 2: Indonesian Judge

Pass 2 receives:

- Compact candidate records.
- Limited timestamped transcript context for each candidate.
- Final clip count and lane-balance requirements.
- Dynamic duration targets.

The judge compares all candidates globally and returns the final ranking. It
evaluates:

- Hook strength in the first 1-3 seconds.
- Setup efficiency.
- Punchline, reveal, claim, or emotional payoff strength.
- Standalone clarity for a viewer with no prior context.
- Natural Indonesian phrasing, slang, sarcasm, and cultural context.
- Reaction value for comedy.
- Replay, comment, and share potential.
- Ending completeness.
- Redundancy with stronger candidates.

The judge returns:

```json
{
  "shorts": [
    {
      "candidate_id": "w2-c4",
      "start": 120.4,
      "end": 154.2,
      "content_lane": "COMEDY",
      "judge_score": 0.92,
      "hook_score": 0.91,
      "payoff_score": 0.95,
      "standalone_score": 0.88,
      "ending_score": 0.93,
      "reasoning": "grounded Indonesian explanation",
      "viral_pattern_type": "PATTERN_INTERRUPT",
      "viral_hook_text": "DIA MALAH NGAKU",
      "social_caption": "grounded Indonesian caption and hashtags"
    }
  ]
}
```

The judge should select a balanced mix when quality allows. Balance is a
secondary objective: a weak lane must not displace a substantially stronger
candidate.

## Deterministic Validation

The final Python validator does not attempt to replace semantic judgment. It
enforces contracts that should not depend on model compliance:

- Required fields and valid numeric timestamps.
- Candidate IDs must exist in the Pass 1 candidate set.
- Selected boundaries must stay within the candidate context.
- Comedy duration target: 20-40 seconds.
- Hot-take and personal-story target: 35-60 seconds.
- Up to 5 seconds of configurable duration tolerance may preserve a complete
  thought.
- Hard duration limits of 15-70 seconds prevent unusably short or long clips.
- Evidence fields cannot be empty.
- Hooks are whitespace-normalized, uppercased, and limited to six words.
- Captions and hooks cannot be replaced with unrelated fallback claims.
- Excessive overlap and duplicate moments are removed.
- Final ordering uses judge score with deterministic penalties and lane balance.

After validation, `snap_clip_to_boundaries()` expands cuts around local word
boundaries and nearby scene boundaries without trimming through speech.

## Fallback Behavior

If Pass 2 times out, raises an API error, or returns invalid JSON:

1. Log the judge failure.
2. Rank validated Pass 1 candidates using scout score and deterministic checks.
3. Apply overlap removal, duration handling, and soft lane balance.
4. Generate final legacy-compatible metadata from grounded Pass 1 fields.
5. Continue extraction rather than failing the whole job.

If Pass 1 produces no valid candidates, retain the existing whole-video
fallback behavior.

## Configuration

The following values should be configurable through constants or environment
variables with conservative defaults:

- Model name and OpenAI-compatible base URL
- Enable or disable Pass 2
- Scout and judge temperatures
- Maximum raw candidates
- Final clip count
- Candidate context padding, defaulting to 8 seconds before and 5 seconds after
- Per-lane duration targets
- Duration tolerance, defaulting to 5 seconds
- Hard duration limits, defaulting to 15-70 seconds
- Judge score threshold

No provider keys are added to source files.

## Implementation Boundaries

Keep changes focused on the analysis surface:

- `main.py`: transcript shaping, prompts, two-pass orchestration, validation,
  fallback, and corrected window filtering.
- `tests/test_clip_quality.py`: focused unit tests for prompt payloads,
  validation, ranking, fallback, and window bounds.

Small helper functions should separate:

- Transcript construction
- Scout prompt construction
- Candidate normalization and consolidation
- Judge context construction
- Judge prompt construction
- Final validation and ranking

This avoids expanding `get_viral_clips()` into one larger control-flow block.

## Testing

### Unit Tests

- AI prompt contains timestamped segments and does not contain a second
  word-level transcript.
- Window prompts contain only transcript segments overlapping that window.
- Scout candidates without evidence or valid timestamps are rejected.
- Duplicate window candidates retain the stronger version.
- Judge output cannot select an unknown candidate ID.
- Dynamic durations are applied by lane.
- Weak lane balance does not displace a clearly stronger candidate.
- Hooks are normalized and capped at six words.
- Judge failure falls back to Pass 1 candidates.
- Existing boundary snapping behavior remains intact.

### Fixture Evaluation

Create anonymized Indonesian transcript fixtures representing:

- Fast multi-speaker roasting.
- Sarcasm whose literal text is misleading.
- A joke requiring short setup and reaction.
- A controversial claim with supporting explanation.
- A personal story with a late reveal.
- Generic banter and filler that should be rejected.

For each fixture, specify expected selected moment ranges, rejected distractors,
content lanes, and acceptable boundary tolerance.

### Regression Verification

Run:

```bash
python3 -m pytest tests/test_clip_quality.py
python3 -m py_compile main.py subtitles.py hooks.py
```

When API credentials and representative source material are available, run an
offline comparison:

- Current single-pass output.
- New Pass 1 fallback output.
- New full multi-pass output.

Review selection relevance, standalone clarity, ending completeness, duplicate
rate, and boundary quality. API-dependent evaluation is informative and must
not make the unit test suite depend on network access.

## Acceptance Criteria

- The AI receives transcript language content in one canonical representation.
- Long-video windows do not include transcript content outside their bounds.
- `podcast_comedy` uses the two-pass pipeline by default when enabled.
- Pass 2 compares consolidated candidates from the whole video.
- Final results can include a quality-weighted balance of all three lanes.
- Comedy clips target 20-40 seconds; hot takes and stories target 35-60 seconds.
- Invalid judge output cannot bypass deterministic validation.
- Pass 2 failure still produces usable clips from Pass 1.
- Existing downstream metadata fields remain populated.
- Focused tests and Python compilation pass.

## Risks

- A second model call increases latency and token cost.
- A judge may overvalue provocative wording unless grounding remains strict.
- Sarcasm and callbacks may still require speaker identity or audio cues.
- Excessive context per candidate can erase prompt-size savings.
- Very long videos may produce too many candidates for one judge call.

Mitigations include compact candidate schemas, bounded context, strict candidate
ID validation, quality-first lane balance, configurable candidate limits, and
the Pass 1 fallback.
