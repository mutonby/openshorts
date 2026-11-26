import httpx
import pytest

import broll
import stock_media


def video(url="https://videos.pexels.com/a.mp4"):
    return {"id": 123, "duration": 12, "url": "https://www.pexels.com/video/123/",
            "image": "https://images.pexels.com/123.jpg", "user": {"name": "Fixture Creator"},
            "video_files": [{"id": 1, "width": 640, "height": 360, "file_type": "video/mp4", "link": url}]}


def client(monkeypatch, handler):
    constructor = httpx.Client
    monkeypatch.setenv("PEXELS_API_KEY", "fixture-key")
    monkeypatch.setattr(stock_media.httpx, "Client", lambda **kwargs: constructor(transport=httpx.MockTransport(handler), **kwargs))


def test_search_filters_unsupported_assets_and_retains_creator_credit(monkeypatch):
    def respond(request):
        assert str(request.url).startswith(stock_media.API + "/search?")
        assert request.headers["authorization"] == "fixture-key"
        return httpx.Response(200, json={"videos": [video(), video("http://127.0.0.1/a.mp4")], "total_results": 13})

    client(monkeypatch, respond)
    result = stock_media.search("nature")
    assert len(result["videos"]) == 1
    assert result["videos"][0]["creator"] == "Fixture Creator"
    assert result["has_more"]


def test_import_resolves_provider_id_and_keeps_api_key_off_media_downloads(monkeypatch, tmp_path):
    def respond(request):
        if request.url.host == "api.pexels.com":
            assert request.url.path.endswith("/videos/123")
            return httpx.Response(200, json=video())
        assert request.url.host == "videos.pexels.com"
        assert "authorization" not in request.headers
        return httpx.Response(200, content=b"video-bytes")

    client(monkeypatch, respond)
    target = tmp_path / "asset.mp4"
    result = stock_media.download(123, target)
    assert target.read_bytes() == b"video-bytes"
    assert result["source_url"] == "https://www.pexels.com/video/123/"


def test_import_rejects_large_files(monkeypatch, tmp_path):
    def respond(request):
        if request.url.host == "api.pexels.com":
            return httpx.Response(200, json=video())
        return httpx.Response(200, content=b"x" * 11)

    client(monkeypatch, respond)
    monkeypatch.setattr(broll, "MAX_UPLOAD_BYTES", 10)
    with pytest.raises(stock_media.StockMediaError, match="40 MB"):
        stock_media.download(123, tmp_path / "asset.mp4")


def test_missing_key_and_bad_provider_urls_are_explicit(monkeypatch):
    monkeypatch.delenv("PEXELS_API_KEY", raising=False)
    with pytest.raises(stock_media.StockMediaError, match="API key"):
        stock_media.search("nature")
    assert stock_media._view(video("https://videos.pexels.com.evil.example/file.mp4")) is None
    assert stock_media._view(video("https://user:password@videos.pexels.com/file.mp4")) is None


@pytest.mark.parametrize("destination", ["http://127.0.0.1/secret", "https://example.com/file.mp4", "https://videos.pexels.com:8443/file.mp4"])
def test_redirects_cannot_leave_provider_hosts(monkeypatch, tmp_path, destination):
    visited = []
    def respond(request):
        visited.append(str(request.url))
        if request.url.host == "api.pexels.com":
            return httpx.Response(200, json=video())
        return httpx.Response(302, headers={"location": destination})
    client(monkeypatch, respond)
    target = tmp_path / "asset.mp4"
    with pytest.raises(stock_media.StockMediaError, match="download link"):
        stock_media.download(123, target)
    assert len(visited) == 2
    assert not target.exists()


def test_provider_redirect_is_followed_without_ambient_proxy(monkeypatch, tmp_path):
    constructor = httpx.Client
    def respond(request):
        if request.url.host == "api.pexels.com":
            return httpx.Response(200, json=video())
        if request.url.host == "videos.pexels.com":
            return httpx.Response(302, headers={"location": "https://player.vimeo.com/final.mp4"})
        assert "authorization" not in request.headers
        return httpx.Response(200, content=b"video")
    def create(**kwargs):
        assert kwargs["trust_env"] is False
        return constructor(transport=httpx.MockTransport(respond), **kwargs)
    monkeypatch.setenv("PEXELS_API_KEY", "fixture-key")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    monkeypatch.setattr(stock_media.httpx, "Client", create)
    target = tmp_path / "asset.mp4"
    stock_media.download(123, target)
    assert target.read_bytes() == b"video"
