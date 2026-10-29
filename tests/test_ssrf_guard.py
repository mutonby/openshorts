"""SSRF guard: nothing but globally routable addresses may be fetched.

The old check listed ranges (is_private, is_loopback, ...) and missed
100.64.0.0/10, the carrier-grade NAT block Tailscale assigns: a job URL
pointing at another server's tailnet address (SSH, the Coolify API) went
straight through. The quality probe also ran yt-dlp on the URL with no check
at all, before the balance and rate checks.
"""
import json
import socket
import sys
import types

import pytest

import security_utils
from security_utils import UnsafeURLError, assert_public_url


@pytest.mark.parametrize("ip", [
    "100.101.1.2",          # Tailscale node
    "100.64.0.1",            # CGNAT, first address
    "100.127.255.254",       # CGNAT, last address
    "10.0.0.1", "172.16.0.1", "192.168.1.1",
    "127.0.0.1", "0.0.0.0", "169.254.169.254",
    "192.0.0.1", "198.18.0.1", "224.0.0.1", "255.255.255.255",
    "::1", "fe80::1", "fd7a:115c:a1e0::1",   # Tailscale's IPv6 ULA
    "::ffff:100.101.1.2",   # IPv4-mapped tailnet address
    "::ffff:127.0.0.1",
    "2002:6465:102::1",      # 6to4 wrapping 100.101.1.2
    "64:ff9b::6465:102",     # NAT64 wrapping 100.101.1.2
])
def test_non_global_addresses_are_refused(ip):
    assert security_utils._ip_is_public(ip) is False


@pytest.mark.parametrize("ip", ["8.8.8.8", "142.250.184.14", "2606:4700:4700::1111"])
def test_global_addresses_pass(ip):
    assert security_utils._ip_is_public(ip) is True


@pytest.mark.parametrize("url", [
    "http://100.101.1.2/",
    "http://100.90.3.4:8080/admin",
    "https://[::ffff:100.101.1.2]/",
    "http://[fd7a:115c:a1e0::1]/",
])
def test_literal_tailnet_urls_are_refused(url):
    with pytest.raises(UnsafeURLError):
        assert_public_url(url)


def test_hostname_resolving_to_tailnet_is_refused(monkeypatch):
    """MagicDNS names (``db-host``) or a public name pointed at 100.x."""
    def fake_getaddrinfo(host, port, *a, **k):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("100.101.1.2", 0))]
    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    with pytest.raises(UnsafeURLError):
        assert_public_url("https://videos.example.com/watch.mp4")


def test_hostname_with_one_bad_record_is_refused(monkeypatch):
    def fake_getaddrinfo(host, port, *a, **k):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0)),
                (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::ffff:100.64.1.1", 0, 0, 0))]
    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    with pytest.raises(UnsafeURLError):
        assert_public_url("https://videos.example.com/watch.mp4")


@pytest.mark.parametrize("url", [
    "https://" + "a" * 64 + ".example.com/watch.mp4",   # label over 63 chars
    "https://videos..example.com/watch.mp4",            # empty label
])
def test_hostname_the_idna_codec_refuses_is_unsafe_not_a_crash(url):
    """getaddrinfo raises UnicodeError for these before any DNS lookup; it
    used to escape the guard and turn POST /api/process into a 500."""
    with pytest.raises(UnsafeURLError):
        assert_public_url(url)


def test_public_hostname_passes(monkeypatch):
    def fake_getaddrinfo(host, port, *a, **k):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("142.250.184.14", 0))]
    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    url = "https://www.youtube.com/watch?v=abc"
    assert assert_public_url(url) == url


# --------------------------------------------------------------------------- #
# quality_probe.py: same gates, noplaylist, no disabled TLS checks
# --------------------------------------------------------------------------- #
class _FakeYDL:
    seen = []

    def __init__(self, opts):
        self.opts = opts
        _FakeYDL.seen.append(opts)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def extract_info(self, url, download=False):
        return {"formats": [{"height": 1080, "vcodec": "avc1"}], "duration": 600}


def _run_probe(monkeypatch, capsys, url):
    import quality_probe
    _FakeYDL.seen = []
    monkeypatch.setitem(sys.modules, "yt_dlp", types.SimpleNamespace(YoutubeDL=_FakeYDL))
    monkeypatch.setattr(sys, "argv", ["quality_probe.py", "--url", url])
    assert quality_probe.main() == 0
    return json.loads(capsys.readouterr().out.strip())


def test_probe_refuses_tailnet_url_without_running_ytdlp(monkeypatch, capsys):
    out = _run_probe(monkeypatch, capsys, "http://100.101.1.2:22/")
    assert out["max_height"] == 0 and out["duration"] == 0
    assert _FakeYDL.seen == []


def test_probe_refuses_non_video_youtube_page(monkeypatch, capsys):
    monkeypatch.setattr(security_utils, "assert_public_url", lambda u: u)
    out = _run_probe(monkeypatch, capsys, "https://www.youtube.com/results?search_query=x")
    assert out["max_height"] == 0
    assert _FakeYDL.seen == []


def test_probe_runs_single_video_with_noplaylist(monkeypatch, capsys):
    monkeypatch.setattr(security_utils, "assert_public_url", lambda u: u)
    out = _run_probe(monkeypatch, capsys, "https://www.youtube.com/watch?v=abc&list=PL1")
    assert out["max_height"] == 1080 and out["duration"] == 600
    assert _FakeYDL.seen
    for opts in _FakeYDL.seen:
        assert opts["noplaylist"] is True
        assert not opts.get("nocheckcertificate")
