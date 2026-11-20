import pytest

from news_shorts import discover, publish
from news_shorts.licensing import is_reusable
from news_shorts.queue import Queue


class Resp:
    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self.data


class FakeHttp:
    def get(self, url, params=None):
        if url.endswith("/search"):
            assert params["videoLicense"] == "creativeCommon"
            return Resp({"items": [{"id": {"videoId": "a"}}, {"id": {"videoId": "b"}}]})
        mk = lambda i, lic: {"id": i, "snippet": {"title": "t", "channelTitle": "Chan"},
                             "status": {"license": lic}}
        return Resp({"items": [mk("a", "creativeCommon"), mk("b", "youtube")]})


def test_only_cc_survives_recheck():
    out = discover.discover_youtube_cc("x", api_key="k", client=FakeHttp())
    assert [c["id"] for c in out] == ["a"]


def test_unlicensed_or_unattributed_rejected():
    assert not is_reusable({"license": "youtube", "source_url": "u", "attribution": "a"})
    assert not is_reusable({"license": "owned", "source_url": "u", "attribution": ""})
    with pytest.raises(ValueError):
        discover.manual_candidate("https://x/y", "bogus", "AN")


def test_publish_requires_approval_and_dry_run_default(tmp_path, monkeypatch):
    q = Queue(str(tmp_path / "q.json"))
    q.add({"clip_path": "c.mp4", "title": "t", "license": "owned",
           "source_url": "u", "attribution": "me"})
    assert publish.publish_approved(q) == []
    assert q.approve("c.mp4")
    res = publish.publish_approved(q)
    assert all(r["dry_run"] for _, r in res) and len(res) == 2
    assert q.load()[0]["status"] == "approved"


def test_real_publish_needs_env(monkeypatch):
    monkeypatch.delenv("TIKTOK_ACCESS_TOKEN", raising=False)
    with pytest.raises(RuntimeError):
        publish.publish_tiktok({"clip_path": "c.mp4"}, dry_run=False)


def test_hook_card_text_fits_9_16_frame(tmp_path):
    from PIL import Image, ImageDraw
    from news_shorts import template

    draw = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    lines = template._lines(draw, "PLAN SECRET POUR STOPPER MÉLENCHON EN 2027", template._font(92), template.W - 200)
    assert len(lines) > 1
    assert all(draw.textlength(l, font=template._font(92)) <= template.W - 200 for l in lines)
    out = tmp_path / "h.png"
    template.hook_card("Plan secret", str(out))
    assert Image.open(out).size == (1080, 1920)
