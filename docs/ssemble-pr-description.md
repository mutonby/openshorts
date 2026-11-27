# Expand creator editing, exact-range exports, and project persistence

## 1. Creator workflows and feature checklist

Creators need to finish a selected source range, correct its captions, add visual and audio layers, reopen the project, and publish the same finished result. This change extends those paths across the dashboard, API, MCP, CLI, local rendering, and managed project storage. It implements behavior observed in Ssemble's documented workflows using OpenShorts' own components and rendering code. It includes no Ssemble source code, account credentials, stock files, or proprietary artwork.

The branch includes the existing unmerged framing and highlight-review foundation on which these editor changes depend. Review that foundation separately from the new creator controls. The earlier creator-workflow PR is [#212](https://github.com/mutonby/openshorts/pull/212); this proposal collects the broader implementation and its subsequent fixes. The comparison matrix in `docs/ssemble-reference.md` records implemented behavior, provider dependencies, and remaining gaps. It does not claim measured Ssemble quality parity.

- Export one exact source range or a complete short file, preserving its timing without automatic highlight selection or editorial time cuts. The dashboard, REST API, MCP tool, and CLI expose the same choice.
- Accept fractional source start and end times. Exact-range exports can process short input without a Gemini key when automatic captions and selection are unnecessary.
- Preserve explicit generation choices in resume manifests and remote worker options so a restart retains the requested output.
- Discover available YouTube audio languages and select a source track. Carry that selection through source preparation, downloads, source caching, REST, MCP, and CLI requests.
- Add sports framing with local ball tracking when suitable weights are installed. Use motion or full-picture fallbacks when tracking evidence is weak. Athlete coverage remains a separate evaluation task.
- Preserve source facecams and show the streamer once, with adaptive stack geometry, gameplay crops, per-shot fallbacks, and composition checks.
- Add staged highlight discovery, source proxies, candidate approval, selected-range retrieval, and restartable analysis checkpoints.
- Record and align Twitch chat with VODs; combine chat and audio hype signals with a clip judge and discard poll-only peaks.
- Support stream cold opens, punch-ins, dead-air cuts, emoji cues, and built-in sound effects with independent controls.
- Carry viewer highlights and feedback into candidate ranking, expose local review/watch workflows, and queue new completed channel recordings for review.
- Keep long recordings bounded through chunked transcription and host slots, with quota-aware analysis, model fallback, and recoverable source fetching.
- Expose agent job listing, CLI generation controls and trend workflows, translated captions without redubbing, and post engagement metrics. Preserve owner isolation, source-rights acknowledgement, and existing metering boundaries.
- Save named caption styles within the current browser and account, then apply them to one clip or a batch.
- Correct caption words and their start/end times. Preserve those corrections when captions are restyled.
- Limit words per caption, alongside existing character and duration controls. Merge continuation fragments before supplying words to the browser preview.
- Download original-language SRT or copy caption text from the saved clip timeline. Exports follow reordered cuts and use the saved corrections.
- Configure opening hooks and closing text CTAs with compatible timing. Keep the opening text clear of the closing overlay.
- Customize hook shape, font, text/background colors, outline, corners, and letter case. Use the same artwork renderer for the design preview and exported overlays.
- Save named hook designs within browser/account storage. Preserve their settings when the clip is edited again.
- Append a separate one-, two-, or three-second end card with a message, channel handle, and optional chime.
- Enable a text end card during initial generation through dashboard, REST, MCP, or CLI. Carry its configuration through restart and remote execution.
- Upload a still image or bounded animated GIF/WebP for an editor end card. Normalize supported artwork into project-owned assets.
- Keep captions, opening hooks, and the closing CTA on the editable body. Append the card once after those layers. Display the delivery duration including its tail.

## 2. Media layers, publishing, and persistence

The editor's saved state now includes picture layers, audio layers, timed effects, and the end card. Undo/redo and dirty-state detection cover these controls together with the source segments. A rerender builds the body first and then adds the delivery tail. Subsequent caption, hook, reframe, mode, and dubbing operations use the body beneath that tail.

- Upload MP4, MOV, or WebM B-roll. Place it at the playhead or attach it to a selected caption word.
- Adjust a B-roll asset's source in-point, start, and duration. Remap source-word anchors across cuts and reordered or repeated source segments.
- Retain the original clip audio while replacing its picture with B-roll. Reapply saved inserts during rerenders and reframing.
- Search optional Pexels stock videos and import a provider ID into the same B-roll library. Keep creator, source, provider, and license links with the asset.
- Require a server-side Pexels key. Keep that key off media requests, disable ambient proxy use, validate every redirect host, and bound download size and supported media duration.
- Select original, vivid, warm, cool, cinematic, or monochrome picture looks. Grade the main picture before later overlays are applied.
- Add a muted looping bottom video from an uploaded or imported asset. Preserve the main image and its audio above that panel, and position captions relative to the resulting layout.
- Upload background music, select its in-point and level, loop it to the clip duration, and optionally duck it against the original audio.
- Add built-in hit, whoosh, and boom sound cues at the playhead or a selected caption word, with a bounded duration and volume.
- Add timed dip-to-black, color-pop, black-and-white, and vignette effects. Preview a cue on its projected frame and map repeated source occurrences onto the edited timeline.
- Persist media asset references in managed projects and prevent deletion of assets still used by saved layers.
- Configure an Autopilot minimum score from 0 to 100. Leave unscored or lower-scoring clips for review when the threshold is active.
- Save an explicit source-credit URL per project and optionally on Autopilot. Append it to immediate and scheduled post descriptions.
- Add X to the dashboard's supported publishing choices using the existing connected-profile provider. Explain the account requirement for source links.
- Discover and validate Pinterest boards for the resolved profile, and carry the selected board through manual posts, weekly scheduling, Autopilot, CLI, and MCP.
- Preserve canonical clip IDs across filtered results, edits, batch publishing, downloads, and agent summaries so a failed neighboring render cannot redirect an action.
- View upcoming scheduled posts in a content calendar and reschedule or cancel them through owner-checked provider calls.
- Delete a project from results or managed history. Restrict deletion to its owner and reject active/busy projects.
- Use shared filesystem leases, deletion markers, and a managed tombstone so deletion can recover after a partial purge and does not race active file requests or archives.
- Archive a stable metadata snapshot together with every referenced clip, editable ancestor, and media asset before advancing durable project/history pointers.
- Serialize durable pointer updates under the project row lock and reject older metadata revisions. A slow previous save cannot replace a newer completed revision.
- Keep earlier snapshot media and complete lazy clean-picture recipes until normal project retention or deletion. An in-flight restore can finish using its recorded metadata version. This increases retained storage compared with deleting each superseded export immediately.
- Restore only the metadata object recorded on the project, even when older versions have the same basename. Treat its committed video URL as authoritative over abandoned render artifacts.
- Defer Pro-mode metadata changes until the complete delivery exists, leaving a failed end-card operation retryable. Preserve generated-voice labeling on the delivered dub and its clean twin.

## 3. Validation, installation, and remaining work

The final full audit run passed **3,145 tests with 32 skips**, including the long synthetic render and the disposable PostgreSQL checks. `docs/creator-workflow-audit.md` records the reproduced failures and fixes. The skipped checks require optional environment capabilities. It covered end-card and media renders, exact ranges, captions and exports, custom hook designs, source languages, sports decisions, cue remapping, rerender contracts, project deletion, snapshot archives, Autopilot, publishing ownership, remote settings, MCP transport, and CLI flags. Tests use synthetic media and mocked external providers where appropriate. They do not establish live delivery to a social network or comparative AI quality.

- Real FFmpeg fixtures verify body/delivery durations, animated card artwork, audio/video offsets, and silent body audio when an input audio stream starts after the visible video ends.
- End-card endpoint checks apply a card, replace captions and hooks, remove captions, and clear the card while keeping exactly one tail.
- Archive fault tests verify that missing editable bodies or storage failures cannot publish pointers, that a captured snapshot remains stable after the shared JSON path changes, and that a stale revision cannot repoint newer history.
- Restore checks cover same-basename revisions, clean served pointers after upgrade, pending local edits, asset deletion tombstones, drained partial downloads, and filtered clip IDs. A completed-session browser refresh fetches the latest server result and guards against overwriting an intervening local edit.
- Disposable PostgreSQL checks exercise concurrent first archives, monotonic revision publication, browser-state preservation, owner-scoped deletion, retryability, and retained Autopilot deduplication.
- The audit corrects media-clock drift, early music cutoff, teaser-first canonical recuts, saved short-range compatibility, missing preview URLs, watcher retry/deduplication, selected-range chat alignment, pinned-format resume, and overlapping metadata saves. The audit document lists the reproduced failures and fixes.
- Project deletion checks cover sibling isolation, ownership, active work, held leases, and retryable partial cleanup.
- Provider tests verify attribution, ID-based imports, redirect restrictions, download limits, source-credit handling, and profile ownership for scheduling.
- Dashboard lint and the production build pass. The build reports the existing bundle-size and optional analytics-environment notices.
- Saved end-card preview plays through the delivery tail while pending edits remain unsaved. A browser check reached the five-second endpoint with the editable body readout held at three seconds.
- The local backend and dashboard were restarted with the update. The desktop dashboard bundle and CLI were refreshed, with the prior dashboard bundle retained.
- A live synthetic source export selected seconds 1.25–4.25, produced a three-second body plus a two-second chime card, and completed without composition warnings. A second live rerender combined B-roll, a looping bottom panel, ducked music, timed effects, and an animated card. FFprobe and sampled decoding verified a 5.000 second 1080×1920 delivery with audio. CamoFox checks inspected the result card, loaded editor controls, effect catalog, playback metadata, and rendered artwork.

Optional stock search still needs a Pexels key. Live publishing and calendar delivery need configured test profiles. Managed storage behavior has fault-injection and disposable PostgreSQL coverage; a production storage exercise remains separate. These features should report unavailable configuration clearly and should not silently substitute a provider.

Remaining product work includes applying saved custom hook designs through generation/automation presets, image cards in channel automation, per-caption visibility and free placement, stock music and meme-intro catalogs, user-uploaded cue sounds, additional transitions, public-channel URL scheduling, and generated-template automation. A funded rewards marketplace also needs campaign policy, authorized platform metrics, moderation, accounting, payment-provider onboarding, and reconciliation. No payout flow is represented as complete.

Reproduce the local release checks with `pytest tests/ -q -k 'not test_render_end_to_end_on_synthetic_footage'`, then `pytest tests/test_viral_edit.py::test_render_end_to_end_on_synthetic_footage -q`. Dashboard validation uses `npm run lint` and `npm run build`. The default CI command runs the whole suite together.

Quality comparisons still require matched, authorized footage and paired exports. Measure selection relevance, word accuracy/timing, speaker and sports coverage, duplicate presenters, layout stability, output sync, and completion latency separately. The installed CamoFox MCP browser supports authenticated feature inspection; its browser profiles, credentials, screenshots, and downloaded runtime remain local and untracked. Product behavior can be reproduced from accessible workflows, while Ssemble's private ranking and tracking internals remain unverified.
