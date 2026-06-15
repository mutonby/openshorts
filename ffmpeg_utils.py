import torch

DEFAULT_NVENC_CQ = 19
DEFAULT_CPU_CRF = 18


def has_cuda() -> bool:
    """Check if CUDA is available for NVENC encoding."""
    try:
        return torch.cuda.is_available()
    except (ImportError, Exception):
        return False


def get_nvenc_opts(cq: int = DEFAULT_NVENC_CQ) -> list:
    """Returns NVENC encoder options for best quality."""
    return [
        "-c:v", "h264_nvenc",
        "-preset", "p7",
        "-tune", "hq",
        "-rc", "vbr_hq",
        "-cq", str(cq),
        "-b:v", "0",
        "-profile:v", "high",
        "-bf", "3",
    ]


def get_cpu_opts(crf: int = DEFAULT_CPU_CRF) -> list:
    """Returns CPU libx264 encoder options."""
    return [
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", str(crf),
    ]


def get_video_encoder_opts(use_cuda: bool = True, cq: int = DEFAULT_NVENC_CQ, crf: int = DEFAULT_CPU_CRF) -> list:
    """Returns appropriate encoder options based on CUDA availability."""
    if use_cuda and has_cuda():
        return get_nvenc_opts(cq)
    return get_cpu_opts(crf)


def get_audio_encoder_opts() -> list:
    """Returns audio encoder options (AAC, no GPU benefit)."""
    return ["-c:a", "aac", "-b:a", "192k"]


def build_ffmpeg_cmd(input_path: str, output_path: str, video_opts: list, audio_opts: list = None, extra_args: list = None) -> list:
    """Builds a complete ffmpeg command with common options."""
    cmd = ["ffmpeg", "-y"]
    if extra_args:
        cmd.extend(extra_args)
    cmd.extend(["-i", input_path])
    cmd.extend(video_opts)
    if audio_opts:
        cmd.extend(audio_opts)
    cmd.append(output_path)
    return cmd