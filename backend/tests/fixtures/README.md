# Test fixtures

Put small reusable test inputs here.

## `smoke.mp4`

The e2e pipeline smoke test (`tests/e2e/test_pipeline_smoke.py`) looks for a
file at `tests/fixtures/smoke.mp4`.

If the file is missing, the e2e test is **skipped** (not failed) so the
non-e2e suite stays green on machines without a fixture committed.

Requirements for the fixture:

- Roughly 5 seconds long
- Landscape (16:9) source so the vertical reframing actually has to crop
- Contains at least one detectable face for the speaker tracker
- Has an audio track
- Small file (<= 2 MB so it can be committed)
- Creative-commons or self-recorded so it can be redistributed

To generate a quick synthetic one with ffmpeg:

```bash
ffmpeg -y -f lavfi -i testsrc2=size=1280x720:rate=30:duration=5 \
       -f lavfi -i sine=frequency=440:duration=5 \
       -c:v libx264 -preset fast -crf 28 -c:a aac \
       tests/fixtures/smoke.mp4
```

That file has no face though, so the speaker tracker will hit the YOLO
fallback. For a more representative fixture, trim a 5-second clip from any
talking-head video you own.
