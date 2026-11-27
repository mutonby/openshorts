import json

import pytest

from chat_hype import Message, read_messages, spikes
from highlight_signals import additional_windows, chat_evidence, related


def ordinary(duration):
    for offset in range(2, duration, 5):
        yield Message(offset, "regular", f"ordinary message {offset}")


@pytest.mark.parametrize("minutes", (10, 30, 120, 720))
def test_real_burst_on_sources_of_different_lengths(minutes):
    end = minutes * 60
    peak = end - 35
    messages = list(ordinary(end))
    messages += [Message(peak + i / 100, f"viewer{i}", "OMG") for i in range(35)]
    result, accepted = spikes(messages, end, reaction_delay=7)
    assert accepted > 30
    assert len(result) == 1
    burst = result[0]
    assert burst["start"] < peak < burst["end"]
    assert burst["participants"] >= 35
    assert burst["ratio"] >= 2
    assert "surprise" in burst["reactions"]
    assert "viewer" not in json.dumps(burst)
    assert "viewers" not in burst
    assert related(peak - 5, peak + 5, {"chat": result})["chat"] == burst


def test_repeat_author_and_giveaway_cannot_create_a_spike():
    messages = list(ordinary(600))
    messages.extend(Message(500 + i / 200, "single", "LOL!") for i in range(200))
    messages.extend(Message(500 + i / 200, f"give{i}", "giveaway type YES to enter") for i in range(200))
    assert spikes(messages, 600)[0] == []


def test_twitch_export_id_and_explicit_alignment(tmp_path):
    replay = tmp_path / "replay.json"
    replay.write_text(json.dumps({"video_id": "456", "comments": [
        {"content_offset_seconds": 50.5, "commenter": {"_id": "private"},
         "message": {"body": "Oh wow"}},
    ]}))
    assert list(read_messages(replay, source_id="456")) == [Message(50.5, "private", "Oh wow")]
    assert chat_evidence(replay, 600, source_id="other")["chat_status"] == "invalid"
    assert chat_evidence(replay, 600, source_id="456", offset=-60)["chat_status"] == "unavailable"
    assert chat_evidence(None, 600)["chat_status"] == "unavailable"


def test_yt_dlp_replay_line_preserves_offset_but_skips_membership_events(tmp_path):
    replay = tmp_path / "replay.json"
    row = {"replayChatItemAction": {"videoOffsetTimeMsec": "50700", "actions": [
        {"addChatItemAction": {"item": {"liveChatTextMessageRenderer": {
            "message": {"runs": [{"text": "no way"}, {"emoji": {"shortcuts": [":surprise:"]}}]},
             "authorExternalChannelId": "viewer"}}}},
        {"addChatItemAction": {"item": {"liveChatPaidMessageRenderer": {"authorExternalChannelId": "paid"}}}},
    ]}}
    replay.write_text(json.dumps(row) + "\n" + json.dumps({}) + "\n")
    assert list(read_messages(replay)) == [Message(50.7, "viewer", "no way :surprise:")]


def test_evidence_keeps_transcript_shortlist_intact():
    quiet = {"id": "quiet", "start": 0, "end": 80}
    busy = {"id": "busy", "start": 200, "end": 280}
    evidence = {"chat": [{"start": 230, "end": 250, "ratio": 8}]}
    assert additional_windows([quiet, busy], [quiet], evidence) == [busy]
    assert additional_windows([quiet, busy], [quiet], {"chat": []}) == []


def _replay_line(ms, author, text):
    return json.dumps({"replayChatItemAction": {"videoOffsetTimeMsec": str(ms), "actions": [
        {"addChatItemAction": {"item": {"liveChatTextMessageRenderer": {
            "authorExternalChannelId": author, "message": {"runs": [{"text": text}]}}}}}]}})


def test_truncated_last_replay_line_keeps_the_rest(tmp_path):
    """An interrupted replay fetch leaves half a JSON line at the end: the
    messages before it are still evidence, not an unreadable file."""
    path = tmp_path / "replay.live_chat.json"
    lines = [_replay_line(1000 * i, f"a{i}", "lol") for i in range(1, 6)]
    path.write_text("\n".join(lines) + '\n{"replayChatItemAction": {"videoOff')
    messages = list(read_messages(path))
    assert [m.seconds for m in messages] == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_malformed_actions_are_skipped_not_fatal(tmp_path):
    path = tmp_path / "replay.live_chat.json"
    odd = json.dumps({"replayChatItemAction": {"videoOffsetTimeMsec": "500",
                                               "actions": ["not-a-dict", None, {"addChatItemAction": "x"}]}})
    path.write_text(odd + "\n" + _replay_line(2000, "a", "gg") + "\n")
    assert [(m.seconds, m.text) for m in read_messages(path)] == [(2.0, "gg")]


def test_a_file_that_is_not_json_at_all_is_still_invalid(tmp_path):
    path = tmp_path / "chat.json"
    path.write_text("this is not\na chat replay\n")
    assert chat_evidence(path, 600)["chat_status"] == "invalid"
