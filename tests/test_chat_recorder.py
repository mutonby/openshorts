"""The live chat recorder: Twitch IRC lines in, VOD-relative replay out."""
import csv
import json
from datetime import datetime, timezone

import pytest

import chat_hype
import chat_recorder as cr
import livestream


LINE = ("@badge-info=;color=#FF0000;display-name=Viewer;tmi-sent-ts=1795306029510;user-id=227223884 "
        ":viewer!viewer@viewer.tmi.twitch.tv PRIVMSG #kaicenat :BRO HUH")


def test_parse_a_chat_message_uses_twitchs_own_time():
    channel, record = cr.parse(LINE)
    assert channel == "kaicenat"
    assert record == {"ts": 1795306029.51, "author_id": "227223884", "text": "BRO HUH"}


@pytest.mark.parametrize("line,expected", [
    ("PING :tmi.twitch.tv", ("PING", "tmi.twitch.tv")),
    (":tmi.twitch.tv RECONNECT", ("RECONNECT", None)),
    (":tmi.twitch.tv 001 justinfan12345 :Welcome, GLHF!", None),
    ("@tmi-sent-ts=1;user-id=1 :nightbot!nightbot@nightbot.tmi.twitch.tv PRIVMSG #x :!discord", None),
    ("garbage", None),
])
def test_parse_control_lines_and_bots(line, expected):
    assert cr.parse(line) == expected


@pytest.mark.parametrize("value,expected", [
    ("KaiCenat", "kaicenat"), ("https://www.twitch.tv/xqc", "xqc"), ("#summit1g", "summit1g"),
    ("bad name!", None), ("ab", None),
])
def test_channel_names(value, expected):
    assert cr.channel_name(value) == expected


def test_writer_rolls_over_at_utc_midnight(tmp_path):
    writer = cr._Writer(tmp_path)
    before = datetime(2026, 11, 21, 23, 59, 50, tzinfo=timezone.utc).timestamp()
    writer.write("chan", {"ts": before, "author_id": "1", "text": "a"})
    writer.write("chan", {"ts": before + 20, "author_id": "2", "text": "b"})
    writer.close()
    assert (tmp_path / "chan" / "2026-11-21.jsonl").read_text().count("\n") == 1
    assert (tmp_path / "chan" / "2026-11-22.jsonl").read_text().count("\n") == 1


def test_attach_writes_a_vod_relative_replay_the_hype_engine_reads(tmp_path):
    vod_start = datetime(2026, 11, 21, 23, 0, tzinfo=timezone.utc).timestamp()
    writer = cr._Writer(tmp_path / "rec")
    for offset, author, text in ((-30, "1", "before"), (10.5, "2", "KEKW"), (3605, "3", "next day"),
                                 (99999, "4", "after the VOD")):
        writer.write("chan", {"ts": vod_start + offset, "author_id": author, "text": text})
    writer.close()
    dest = tmp_path / "job" / "imported-chat.csv"
    assert cr.attach("chan", vod_start, 3700, dest, tmp_path / "rec") == 2
    rows = list(csv.DictReader(open(dest)))
    assert [(r["time_seconds"], r["text"]) for r in rows] == [("10.5", "KEKW"), ("3605.0", "next day")]
    messages = list(chat_hype.read_messages(dest))
    assert [m.seconds for m in messages] == [10.5, 3605.0]


def test_attach_without_recordings_writes_nothing(tmp_path):
    assert cr.attach("chan", 1795000000.0, 600, tmp_path / "x.csv", tmp_path) == 0
    assert not (tmp_path / "x.csv").exists()


def test_discovery_attaches_recorded_chat_for_a_twitch_vod(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAT_RECORDINGS_DIR", str(tmp_path / "rec"))
    writer = cr._Writer(tmp_path / "rec")
    writer.write("somestreamer", {"ts": 1795223830 + 42.0, "author_id": "9", "text": "Pog"})
    writer.close()
    info = {"uploader_id": "somestreamer", "timestamp": 1795223830}
    assert livestream.attach_recorded_chat(info, tmp_path / "job", 27395) == 1
    assert livestream.imported_chat(tmp_path / "job").name == "imported-chat.csv"
    # An uploaded replay always wins: nothing is attached over it.
    assert livestream.attach_recorded_chat(info, tmp_path / "job", 27395) == 0


def test_run_refuses_no_channels():
    with pytest.raises(ValueError):
        cr.run(["not a channel!"], stop=lambda: True)
