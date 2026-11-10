"""The GPU image must end up with the CUDA build of onnxruntime.

requirements-gpu.lock pins both ``onnxruntime`` (CPU, pulled in by
faster-whisper) and ``onnxruntime-gpu``. Both wheels write the same
``onnxruntime`` package directory, so a single ``pip install -r`` of the lock
can leave the CPU build on top: Parakeet then transcribes on the CPU without
any error and every job slows down by minutes. The Dockerfile has to
reinstall the GPU wheel after the lock and fail the build when the CUDA
provider is missing.
"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(name):
    with open(os.path.join(ROOT, name)) as f:
        return f.read()


def _pin(lock, package):
    m = re.search(rf"^{re.escape(package)}==(\S+)$", lock, re.MULTILINE)
    return m.group(1) if m else None


def test_gpu_lock_pins_matching_onnxruntime_builds():
    lock = _read("requirements-gpu.lock")
    gpu = _pin(lock, "onnxruntime-gpu")
    assert gpu, "requirements-gpu.lock must pin onnxruntime-gpu (Parakeet on CUDA)"
    cpu = _pin(lock, "onnxruntime")
    if cpu:
        assert cpu == gpu, "onnxruntime and onnxruntime-gpu must share a version"


def test_dockerfile_puts_the_gpu_build_back_after_the_lock():
    dockerfile = _read("Dockerfile")
    lock_install = dockerfile.find('pip install --no-cache-dir --no-deps -r "$LOCK"')
    assert lock_install != -1, "lock install step not found"
    reinstall = re.search(
        r"pip install[^\n]*--force-reinstall[^\n]*\$ORT_GPU", dockerfile)
    assert reinstall, "onnxruntime-gpu must be force-reinstalled after the lock"
    assert reinstall.start() > lock_install, "the reinstall must come after the lock"
    assert "grep -E '^onnxruntime-gpu==' requirements-gpu.lock" in dockerfile


def test_dockerfile_fails_the_build_without_the_cuda_provider():
    dockerfile = _read("Dockerfile")
    check = dockerfile.find("'CUDAExecutionProvider' not in p")
    assert check != -1, "the build must assert CUDAExecutionProvider is available"
    assert check > dockerfile.find("--force-reinstall"), \
        "the provider check must run after the GPU wheel is reinstalled"
