import pytest

from cli import openshorts_cli as cli


def test_process_forwards_framing_and_explicit_quality_confirmation(monkeypatch):
    calls = []

    def request(method, path, body):
        calls.append((method, path, body))
        return 200, {"job_id": "job-1"}

    monkeypatch.setattr(cli, "_request", request)
    cli.main(["process", "https://youtu.be/x", "--format", "vertical", "--framing", "fit",
              "--force-low-quality"])
    assert calls == [("POST", "/api/process", {
        "url": "https://youtu.be/x", "acknowledged": True, "output_format": "vertical",
        "framing_method": "fit", "force_low_quality": True,
    })]


def test_process_reports_quality_confirmation_before_polling(monkeypatch, capsys):
    calls = []

    def request(method, path, body):
        calls.append((method, path, body))
        return 200, {"needs_confirmation": True, "quality_check": {"max_height": 480}}

    monkeypatch.setattr(cli, "_request", request)
    with pytest.raises(SystemExit) as error:
        cli.main(["process", "https://youtu.be/x", "--wait"])
    assert error.value.code == 1
    assert calls == [("POST", "/api/process", {"url": "https://youtu.be/x", "acknowledged": True})]
    assert "480p" in (stderr := capsys.readouterr().err)
    assert "--force-low-quality" in stderr
