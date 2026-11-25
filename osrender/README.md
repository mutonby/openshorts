# osrender

Single-pass GPU renderer for reframed clips. `gpu_render.py` hands it the
clip, a JSON plan and an output path:

    osrender <input.mp4> <plan.json> <output.mp4>
    osrender burn <spec.json>

`burn` (`src/burn.rs`, called by `gpu_burn.py`) is the hook + captions (+ the
free-plan mark) pass of a finished clip: NVDEC decodes, `src/kernels.cu`
repeats the 8-bit blend arithmetic of ffmpeg's `overlay` and `ass` filters
(libass is linked and set up as `vf_subtitles` does, `src/ass.rs`), and NVENC
encodes each requested stage (`hook`, `ass`, `mark`) from device memory; the
audio is copied. A `raw` output writes planar yuv420p instead, which is how
the equivalence test compares it with ffmpeg byte for byte.

NVDEC decodes into CUDA frames, the kernels in `src/kernels.cu` compose each
output frame from the plan's layers (a source rectangle scaled into an output
rectangle, optionally blurred) plus an optional image overlay, and NVENC
encodes straight from device memory. The frames never leave the card, and
the whole clip is one process instead of one ffmpeg per scene plus a concat.
The output is video only; the caller muxes the clip's audio back in.

The plan is not a second copy of the layouts: `gpu_render.layers_from_graph`
reads the filtergraph each scene would hand ffmpeg. A graph it cannot read,
a source NVDEC cannot decode, a missing binary or any error renders through
ffmpeg exactly as before. `GPU_RENDER=0` turns it off.

## Building

The GPU image builds it (`Dockerfile`, `osrender` stage, `GPU=1`) against the
same Debian FFmpeg libraries the final image installs. By hand:

    apt-get install pkg-config clang libclang-dev libavcodec-dev libavformat-dev \
      libavutil-dev libavfilter-dev libavdevice-dev libswscale-dev libswresample-dev libass-dev
    cargo build --release --locked

`libcuda.so.1` is loaded at run time, so the binary builds without the CUDA
toolkit. `src/kernels.ptx` is generated from `src/kernels.cu` and committed;
after editing the kernels, regenerate it with

    nvcc -ptx -O3 -arch=compute_75 src/kernels.cu -o src/kernels.ptx
    nvcc -ptx -O3 -arch=compute_75 src/burn_kernels.cu -o src/burn_kernels.ptx

(`compute_75` PTX is compiled by the driver for any newer card).

## Tests

`tests/test_gpu_render.py` covers the plan side in CI. On a machine with the
binary and NVENC, `tests/test_gpu_render_equivalence.py` renders every layout
through both paths and requires the same frame count and SSIM >= 0.97
(`OSRENDER_BIN=... OSRENDER_TEST_VIDEO=clip.mp4 pytest tests/test_gpu_render_equivalence.py`).
