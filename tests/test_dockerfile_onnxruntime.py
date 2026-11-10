"""The GPU image must run the onnxruntime-gpu build, not the CPU one.

onnxruntime and onnxruntime-gpu are two distributions of the same import
package. The GPU lock pins both (faster-whisper requires the name
"onnxruntime", so pip check needs it), and when one pip install unpacks both,
the CPU wheel can land last: Parakeet then asks for CUDAExecutionProvider,
gets a UserWarning and transcribes on the CPU. The Dockerfile rewrites the GPU
wheel on top and fails the build if the CUDA provider is missing.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _lock_pins(name):
    text = (ROOT / "requirements-gpu.lock").read_text()
    return re.findall(rf"^{re.escape(name)}==(\S+)$", text, flags=re.M)


def _lock_install_step():
    text = (ROOT / "Dockerfile").read_text()
    # Join continuation lines so each RUN is one logical line.
    logical = re.sub(r"\\\n", " ", text)
    steps = [l for l in logical.splitlines()
             if l.startswith("RUN ") and '-r "$LOCK"' in l]
    assert len(steps) == 1, "expected exactly one RUN that installs the lock"
    return steps[0]


def test_gpu_lock_pins_both_onnxruntime_wheels_at_one_version():
    cpu, gpu = _lock_pins("onnxruntime"), _lock_pins("onnxruntime-gpu")
    assert len(cpu) == 1 and len(gpu) == 1
    # Different versions would mix the two builds' files in one package.
    assert cpu == gpu


def test_gpu_wheel_is_written_last_and_checked():
    step = _lock_install_step()
    lock_at = step.index('-r "$LOCK"')
    reinstall_at = step.index("--force-reinstall")
    assert reinstall_at > lock_at
    assert "^onnxruntime-gpu==" in step[reinstall_at:]
    check_at = step.index("CUDAExecutionProvider")
    assert check_at > reinstall_at
    # pip check still runs, after the reinstall.
    assert step.index("pip check") > check_at


def test_cpu_build_is_left_alone():
    # The CPU image has no onnxruntime-gpu: the reinstall is GPU-only.
    step = _lock_install_step()
    between = step[step.index('-r "$LOCK"'):step.index("--force-reinstall")]
    assert 'if [ "$GPU" = "1" ]' in between
