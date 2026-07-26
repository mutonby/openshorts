from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JSX = (ROOT / "dashboard" / "src" / "App.jsx").read_text(encoding="utf-8")
APP_PY = (ROOT / "app.py").read_text(encoding="utf-8")
MAIN_PY = (ROOT / "main.py").read_text(encoding="utf-8")


def test_clip_generator_requires_only_gemini_key():
    assert "const keysMissing = !billingEnabled && !apiKey;" in APP_JSX
    assert "TrueLife Clipper needs both a" not in APP_JSX


def test_self_hosted_clips_are_kept_local_by_default():
    assert 'JOB_RETENTION_SECONDS = int(os.environ.get("JOB_RETENTION_SECONDS", "0"))' in APP_PY
    assert 'OUTPUT_MAX_GB = int(os.environ.get("OUTPUT_MAX_GB", "0"))' in APP_PY
    assert "run_in_executor(None, upload_job_artifacts" not in APP_PY
    assert "if JOB_RETENTION_SECONDS > 0 and now - os.path.getmtime(file_path) > JOB_RETENTION_SECONDS:" in APP_PY


def test_processing_job_uses_the_server_python_environment():
    run_job_source = APP_PY[APP_PY.index("async def run_job"):]
    assert 'cmd = [sys.executable, "-u", "main.py"]' in APP_PY
    assert 'env.pop("PYTHONPATH", None)' in run_job_source


def test_clip_generator_defaults_to_gemini_3_flash_preview():
    assert MAIN_PY.count("or 'gemini-3-flash-preview'") == 2
