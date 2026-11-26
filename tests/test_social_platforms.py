"""The networks a clip can be posted to and the Upload-Post fields each one gets."""
import pytest

import social_platforms as sp


def test_clean_keeps_known_networks_in_a_stable_order():
    assert sp.clean(["Bluesky", "tiktok", "snapchat", "LinkedIn"]) == ["tiktok", "linkedin", "bluesky"]
    assert sp.clean(None) == []


def test_unknown_names_what_is_not_offered():
    assert sp.unknown(["tiktok", "snapchat", "myspace"]) == ["snapchat", "myspace"]


def test_x_and_pinterest_are_supported_networks():
    assert sp.clean(["X", "Pinterest"]) == ["x", "pinterest"]
    data = sp.upload_fields(["x", "pinterest"], title="T", description="D")
    assert data["x_title"] == "D"                       # X post text = caption
    assert data["pinterest_title"] == "T"
    assert data["pinterest_description"] == "D"


def test_each_network_gets_its_own_text_fields():
    data = sp.upload_fields(sp.PLATFORMS, title="T", description="D", yt_title="YT")
    assert data["tiktok_title"] == "D" and data["instagram_title"] == "D"
    assert data["youtube_title"] == "YT" and data["youtube_description"] == "D"
    assert data["facebook_title"] == "T" and data["facebook_description"] == "D"
    assert data["facebook_media_type"] == "REELS"
    assert data["linkedin_title"] == "T" and data["linkedin_description"] == "D"
    # Threads posts its *_title as the text: it must be the caption, not the title.
    assert data["threads_title"] == "D"
    assert data["bluesky_description"] == "D"


def test_only_the_chosen_networks_get_fields():
    data = sp.upload_fields(["linkedin"], title="T", description="D")
    assert set(data) == {"linkedin_title", "linkedin_description"}


def test_no_description_falls_back_to_the_title():
    assert sp.upload_fields(["threads"], title="T", description="")["threads_title"] == "T"


def test_post_endpoints_refuse_an_unsupported_network():
    import app as app_module
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as e:
        app_module._checked_platforms(["tiktok", "snapchat"])
    assert e.value.status_code == 400 and "snapchat" in e.value.detail
    with pytest.raises(HTTPException):
        app_module._checked_platforms([])
    assert app_module._checked_platforms(["Threads", "tiktok"]) == ["tiktok", "threads"]


def test_upload_post_client_retries_only_the_connect():
    import asyncio
    client = sp.upload_post_client(20)
    assert client.timeout.connect == sp.UPLOAD_POST_CONNECT_TIMEOUT
    assert client.timeout.read == 20
    asyncio.run(client.aclose())
