"""`openshorts process` forwards the generation controls to /api/process.

The CLI advertises parity with the REST/MCP API; these flags (clip count,
length band, captions, hook, quality) are part of that contract.
"""
import importlib.util
import os

_CLI_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "cli", "openshorts_cli.py")


def _load_cli():
    spec = importlib.util.spec_from_file_location("openshorts_cli", _CLI_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_process_forwards_the_generation_controls(monkeypatch):
    cli = _load_cli()
    sent = {}

    def _fake_request(method, path, body=None):
        sent.update(method=method, path=path, body=body)
        return 200, {"job_id": "job-1"}

    monkeypatch.setattr(cli, "_request", _fake_request)
    cli.main(["process", "https://youtu.be/x",
              "--target-clips", "6", "--clip-min", "20", "--clip-max", "45",
              "--no-captions", "--auto-hook", "--hook-style", "bold",
              "--force-low-quality", "--format", "vertical"])
    body = sent["body"]
    assert sent["method"] == "POST" and sent["path"] == "/api/process"
    assert body["url"] == "https://youtu.be/x" and body["acknowledged"] is True
    assert body["target_clips"] == 6
    assert body["clip_min_seconds"] == 20.0 and body["clip_max_seconds"] == 45.0
    assert body["captions"] is False
    assert body["auto_hook"] is True and body["auto_hook_style"] == "bold"
    assert body["force_low_quality"] is True
    assert body["output_format"] == "vertical"


def test_process_without_controls_sends_only_the_basics(monkeypatch):
    cli = _load_cli()
    sent = {}
    monkeypatch.setattr(cli, "_request",
                        lambda m, p, body=None: (sent.update(body=body), (200, {"job_id": "j"}))[1])
    cli.main(["process", "https://youtu.be/x"])
    assert set(sent["body"]) == {"url", "acknowledged"}
