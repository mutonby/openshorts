# Creator workflow regression audit

This audit covers the creator, publishing, stream-discovery, and persistence changes in PR #216, including their framing and highlight-review foundation. It combines code review, fault injection, a disposable PostgreSQL database, synthetic FFmpeg exports, and browser interaction checks. Passing these checks establishes the tested behavior; it does not establish Ssemble quality parity or prove every external service works.

## Corrected failures

| Area | Failure and resulting correction |
| --- | --- |
| Pinterest | Posting offered Pinterest without its required board. Board discovery and membership validation now use the resolved owner profile. Manual posts, weekly scheduling, Autopilot, CLI and MCP carry the selected board. |
| Publishing errors | Validation errors inside the upload path lost their HTTP status. Preserve the status and refund a claimed free post on failure. |
| Source credits | Delayed settings reads could erase a typed draft, and reopening a dialog could reuse stale credit. Separate saved values from explicit overrides and inherit the current server setting when untouched. |
| X attribution | Link removal could leave a dangling source label. Send the credit URL on its own line and explain the provider account requirement in the posting controls. |
| Filtered clip IDs | Failed renders make array positions differ from source clip IDs. Resolve IDs in dashboard edits, weekly publishing, CLI/MCP summaries, backend edits/dubbing/effects, upgrades, downloads, and restored results. |
| Completed-session cache | Reloading a completed browser session could keep a stale result URL. Refresh it after auth settles, preserving an intervening local edit or newly opened job. |
| Repaired results | Repairing another clip could restore a generation filename over a saved edit. Prefer the persisted edit URL when rebuilding delivered results. |
| Gameplay timeline | A teaser-first canonical file is already edited. Preserve its immutable source EDL and map subsequent cuts onto its actual file clock; recover valid older maps from recorded metadata. |
| Short generated cuts | Generated and legacy short kept segments must remain editable without relaxing the minimum for new cuts. See the regression tests for the preserved-segment exception. |
| Music duration | A short original audio stream stopped the music bed early. Pad the voice stream to the video duration before mixing and ducking. |
| Overlay sync | Resetting main-video timestamps could move B-roll and bottom-panel video away from the original audio clock. Preserve main timestamps and align overlay clocks. |
| Cue sync | Sound mixing could pull delayed source audio into its lead-in. Preserve the original audio offset while mixing cues. |
| Caption position | Fast recuts and Pro edits dropped stored seam and caption-height anchors. Carry them through sidecar remapping and writes. |
| Caption toggle | Reopening an uncaptioned clip could enable captions on its next save. Initialize and track the toggle from the rendered caption state. |
| Transcript access | Hiding or losing the source monitor also hid retained words and disabled word-linked effects. Keep the transcript available and seek the corresponding clip frame. |
| Cold upgrade restore | Frozen metadata could resurrect a watermark after the durable served pointer had advanced to the clean twin. Reconcile authoritative project pointers into restored metadata. |
| Warm upgrade restore | Existing working files could retain the free-plan delivery guard after upgrade. Clear the marker when the authoritative project state is clean. |
| Metadata revisions | A newer snapshot could be ignored when its basename and video filename stayed the same. Track the restored metadata key and content hash. Preserve locally modified metadata while its upload is pending. |
| Overlapping saves | AI editing and dubbing did not share the other editor operations’ job lock. Serialize their complete metadata read/modify/write sequence. |
| Metadata-only saves | Source-credit updates uploaded an unversioned object instead of advancing the saved project snapshot. Use the same captured snapshot archive as video edits. |
| Asset deletion | Deleting a current asset could break older snapshots; retaining it could resurrect it on restore. Keep immutable remote media and record current deletion tombstones. |
| Partial restore | One failed download could release the restore lock while other download threads still wrote into its directory. Drain all downloads before cleanup, including cancellation. |
| Upgrade/upload race | An older marked snapshot finishing after payment could repoint history to a watermark. Reconcile marked edit archives against the owner’s current paid plan. |
| Upgrade/deletion race | Background clean-twin rendering/uploading lacked a project file lease. Hold it until both finish. |
| Highlight count and lengths | Hype review could ignore the requested clip count and silently clamp supported duration bands. Respect the configured bounds. |
| Candidate playback | Missing previews were advertised as playable. Publish only existing browser previews; an optional analysis-copy failure can still leave a playable manual-review candidate. |
| Judge trim bounds | A suggested trim could leave its staged preview range. Clamp it to the exact interval that was reviewed. |
| Channel watcher | A transport error could stop the watcher, while a lost accepted response could cause duplicate submission. Retry definite connection refusals and retain an unconfirmed claim for ambiguous responses. |
| Chat offsets | A later selected Twitch range attached the start of the original chat. Select the matching VOD-relative interval and preserve its timestamps. |
| Media coverage | Long audio could conceal a shorter video stream. Require the requested range to fit both streams, accounting for their start clocks. |
| Format resume | A newly available higher format could invalidate a still-available saved choice. Match pinned video/audio IDs against all usable refreshed formats. |

## Verification and limits

The final backend run passed **3,145 tests with 32 skips**, including the full synthetic viral render and the PostgreSQL tests. Dashboard lint and the production build also passed. The skipped tests require optional environment capabilities. Validation details are also recorded in the PR description. Database tests exercise concurrent project creation, monotonic revision publication, preservation of browser state, owner isolation, and retryable deletion while retaining Autopilot deduplication. Storage tests also inject missing files and download/upload failures; no production object store is mutated.

Synthetic media checks inspect stream timing and decoded pixels/audio, including offset tracks, short voice tracks, free-plan clean recipes, canonical teaser ordering, and a single appended end card. Browser checks cover manual and weekly publishing payloads, delayed settings responses, filtered IDs, and caption-linked cues with the source monitor hidden. Completed-session checks also delay authentication and server responses, verify end-card layer normalization, and preserve a new edit or reset against an older response. Provider responses are mocked; no social posts are sent.

Live stock imports still require configured provider credentials. Live publishing, calendars, and automated delivery require authorized provider test profiles. GPU equivalence and full model-quality comparisons require suitable hardware and matched footage. The existing feature comparison in `ssemble-reference.md` records broader product gaps that this regression audit does not represent as complete.
