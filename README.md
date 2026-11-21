# plic.ai

**AI-powered video clipping for creators.** Turn long videos and livestreams into vertical clips with facecam-aware framing, subtitles, hooks and platform-ready exports.

## What plic.ai does

- Finds strong moments in long videos using Gemini or an OpenAI-compatible model.
- Produces vertical 9:16 clips with facecam above gameplay when a webcam inset is detected.
- Burns word-timed subtitles and optional hooks into the final video.
- Keeps the original clip and derived versions so clips can be re-edited.
- Supports YouTube links, uploaded videos, clip editing and social publishing integrations already present in the project.

## Faster long-video workflow

Plic AI uses a transcript-first route for YouTube sources:

1. Download the lightweight audio track and inspect the source duration.
2. Try the original-language YouTube automatic transcript first.
3. Use the timestamped transcript to find highlight moments without transcribing the entire stream.
4. Download only the selected video ranges, then concatenate those ranges.
5. Run local ASR on the selected footage for accurate final subtitle timings.
6. Render clips in parallel using FFmpeg and the existing reframe pipeline.

If YouTube does not provide a sufficiently complete transcript, the existing local audio-transcription path remains the fallback. This avoids making transcript availability a hard dependency.

## Run locally

Requirements: Docker Desktop, Docker Compose and a working NVIDIA Container Toolkit setup if you want GPU acceleration.

```powershell
docker compose up -d --build
```

Open the dashboard at [http://localhost:5175](http://localhost:5175). The API is available at [http://localhost:8000](http://localhost:8000).

A clean source ZIP is built by the GitHub Actions workflow **Build plic.ai ZIP**. Open the repository's [Actions page](https://github.com/Gurmehtab1/openshorts/actions/workflows/build-plic-ai-zip.yml), open the latest successful run, and download the `plic-ai-source` artifact. The archive excludes the separately licensed `cloud/` directory, local data, credentials and generated media.

For a non-Docker Python setup, install the packages in `requirements.txt` and start the FastAPI application using the project's existing startup instructions.

## Useful settings

- `PLIC_GAMEPLAY_LAYOUT=1` enables automatic facecam-over-gameplay framing. Set it to `0` to disable.
- `WHISPER_DEVICE=cuda` and `WHISPER_COMPUTE=float16` select CUDA transcription when the installed backend supports it.
- `AUTO_CAPTIONS=1` keeps automatic subtitles enabled.
- `PLIC_PUBLIC_URL` and `PLIC_API_URL` set public URLs when running behind a reverse proxy. Local defaults are used otherwise.

## Licence and upstream notices

The core application is distributed under the MIT licence. The original copyright and permission notice in [LICENSE](LICENSE) is intentionally retained as required by that licence.

The upstream repository's `cloud/` directory has a separate commercial licence and is intentionally excluded from the plic.ai ZIP. It is not covered by the root MIT licence. Do not redistribute, white-label, or offer that commercial code as a hosted service without permission from its copyright holder. The self-hosted core runs with billing disabled and does not require it.

Plic AI's transcript-first optimization is an independent implementation inspired by the publicly documented workflow of PeakCut. The PeakCut source code is not bundled into this project.
