"""Stage 1 of highlight discovery: the hype curve from chat and the streamer.

Synthetic inputs with known answers: a chat burst, a shout in the speech
band, a loud low-frequency game sound that must NOT count, and a chat that
trails the streamer by a known number of seconds."""
import shutil
import subprocess

import numpy as np
import pytest

import hype_engine as he
from chat_hype import Message

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")


def _flat_chat(duration, per_second=2, start_author=0):
    out = []
    for t in range(duration):
        for k in range(per_second):
            out.append(Message(t + 0.1 * k, f"u{(t * per_second + k + start_author) % 40}", "just chatting here"))
    return out


def _burst(at, seconds=6, chatters=30, text="KEKW"):
    return [Message(at + s + i / chatters, f"fan{i}", text) for s in range(seconds) for i in range(chatters)]


def test_reaction_tokens_cover_stream_emotes():
    assert he.reaction_of("KEKW KEKW") == "laughter"
    assert he.reaction_of("OMEGALUL") == "laughter"
    assert he.reaction_of("Pog") == "shock"
    assert he.reaction_of("?????") == "shock"
    assert he.reaction_of("W") == "hype"
    assert he.reaction_of("hello everyone") is None


def test_chat_curve_counts_people_not_spam():
    msgs = [Message(5.1, "a", "LUL")] * 20                    # one author spamming
    msgs += [Message(5.2, f"b{i}", "LUL") for i in range(4)]  # four distinct people
    msgs += [Message(5.3, "c", "!giveaway enter now")]        # a command
    msgs += [Message(6.0, "d", "!clip"), Message(6.1, "e", "clip it")]
    curve = he.chat_curve(msgs, 10)
    assert curve["chatters"][5] == 5            # a + b0..b3; the command is not a chatter
    assert curve["reactions"][5] == 5
    assert curve["clip_requests"][6] == 2
    assert curve["messages"][5] == 25


def test_chat_intensity_rises_on_a_burst_not_on_a_busy_chat():
    duration = 900
    msgs = _flat_chat(duration, per_second=6) + _burst(600)
    z = he.chat_intensity(he.chat_curve(msgs, duration))
    assert int(np.argmax(z)) in range(598, 610)
    assert float(np.median(z)) < 0.5 and float(z.max()) > 3


def test_chat_lag_recovers_the_delay():
    rng = np.random.default_rng(1)
    voice = np.zeros(1800, dtype="float32")
    for p in (200, 520, 900, 1300, 1600):
        voice[p:p + 4] = 3.0
    chat = np.roll(voice, 9) * 1.5 + rng.random(1800).astype("float32") * 0.2
    lag, confidence = he.chat_lag(chat, voice)
    assert lag == 9 and confidence > 0.05


def test_chat_lag_falls_back_without_agreement(monkeypatch):
    monkeypatch.setenv("HIGHLIGHT_CHAT_DELAY", "6")
    lag, _ = he.chat_lag(np.zeros(600), np.ones(600))
    assert lag == 6.0


def test_windows_fit_the_band_and_do_not_overlap():
    hype = np.zeros(3600, dtype="float32")
    hype[1000:1006] = 0.9
    hype[1010:1012] = 0.7     # same event: must not become a second window
    hype[2500:2503] = 0.6
    found = he.windows(hype, 3600, min_secs=30, max_secs=60)
    assert len(found) == 2
    assert found[0]["peak"] in range(1000, 1006) and found[1]["peak"] in range(2500, 2503)
    for w in found:
        assert 30 <= w["end"] - w["start"] <= 60
        assert w["start"] <= w["peak"] <= w["end"]
        # The setup comes before the peak: most of the window is lead-in.
        assert w["peak"] - w["start"] >= 12
    assert not any(a["start"] < b["end"] and b["start"] < a["end"]
                   for i, a in enumerate(found) for b in found[i + 1:])


def test_windows_respect_the_recording_edges():
    hype = np.zeros(100, dtype="float32")
    hype[3] = 1.0
    hype[97] = 0.9
    for w in he.windows(hype, 100, min_secs=30, max_secs=60):
        assert 0 <= w["start"] and w["end"] <= 100 and w["end"] - w["start"] >= 30


def test_fuse_rewards_agreement_over_either_alone():
    chat = np.zeros(300, dtype="float32")
    voice = np.zeros(300, dtype="float32")
    chat[107] = chat[257] = 3.0      # chat 7 s after the streamer at 100
    voice[100] = 3.0
    voice[200] = 3.0                 # streamer alone
    hype, _, _ = he.fuse(chat, voice, 7)
    assert hype[100] > hype[200] and hype[100] > hype[250]


def _tone_track(path, events):
    """120 s of quiet speech-band tone with ``events`` [(start, dur, freq, gain)]."""
    filters = ["sine=frequency=700:sample_rate=16000:duration=120,volume=0.05[base]"]
    labels = ["[base]"]
    for i, (start, dur, freq, gain) in enumerate(events):
        filters.append(f"sine=frequency={freq}:sample_rate=16000:duration={dur},volume={gain},"
                       f"adelay={int(start * 1000)},apad=whole_dur=120[e{i}]")
        labels.append(f"[e{i}]")
    graph = ";".join(filters) + f";{''.join(labels)}amix=inputs={len(labels)}:normalize=0[out]"
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-filter_complex", graph,
                    "-map", "[out]", "-ac", "1", str(path)], check=True)


@needs_ffmpeg
def test_a_shout_in_the_speech_band_is_excitement_a_bass_boom_is_not(tmp_path):
    wav = tmp_path / "stream.wav"
    _tone_track(wav, [(60, 2, 2200, 0.9),     # yell: loud, speech band, high
                      (30, 2, 60, 0.9)])      # explosion rumble: loud, below the band
    features = he.voice_features(wav)
    assert features is not None and abs(features["duration"] - 120) < 1
    excite = he.per_second(he.voice_excitement(features), 120)
    assert int(np.argmax(excite)) in range(59, 63)
    assert excite[30:33].max() < 0.5 * excite[59:63].max()


@needs_ffmpeg
def test_voice_features_without_audio_is_none(tmp_path):
    silent = tmp_path / "video.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "lavfi",
                    "-i", "color=black:s=32x32:d=2", str(silent)], check=True)
    assert he.voice_features(silent) is None


@needs_ffmpeg
def test_scan_end_to_end_with_lagged_chat(tmp_path):
    wav = tmp_path / "stream.wav"
    _tone_track(wav, [(80, 3, 2200, 0.9)])
    chat = tmp_path / "chat.csv"
    rows = ["time_seconds,author_id,text"]
    rows += [f"{m.seconds:.2f},{m.author},{m.text}" for m in _flat_chat(120, 2)]
    rows += [f"{m.seconds:.2f},{m.author},{m.text}" for m in _burst(86, seconds=5, chatters=25)]
    chat.write_text("\n".join(rows))
    result = he.scan(wav, 120, chat_path=chat, min_secs=30, max_secs=60)
    assert result["chat_status"] == "available" and result["voice"]
    best = result["candidates"][0]
    assert best["start"] <= 80 <= best["end"]
    assert best["signals"]["chat"]["peak_chatters_per_second"] >= 25
    assert "laughter" in best["signals"]["chat"]["reactions"]
    assert result["curve"]["step"] == 5 and len(result["curve"]["values"]) == 24


def test_past_the_end_of_a_partial_replay_the_streamer_alone_counts():
    chat = np.zeros(1000, dtype="float32")
    voice = np.zeros(1000, dtype="float32")
    chat[100:105] = 3.0
    voice[100] = 3.0
    voice[800] = 3.0                         # chat data stops at 500
    full, _, _ = he.fuse(chat, voice, 0)
    cut, _, _ = he.fuse(chat, voice, 0, chat_until=500)
    assert cut[100] == full[100]
    assert cut[800] > full[800]              # no longer punished for missing chat
    assert cut[800] == pytest.approx(he._unit(np.float32(3.0)))


def test_a_poll_or_copypasta_is_not_a_reaction_but_a_shared_emote_is():
    """Measured on a real stream: 114 viewers typing "1" for a poll ranked as a
    hype moment. Many people typing the same reaction is hype; many typing
    the same non-reaction line is a vote, a copy-paste or a giveaway entry."""
    votes = [Message(10.1 + i / 200, f"v{i}", "1") for i in range(100)]
    votes += [Message(10.5 + i / 100, f"d{i}", "dog") for i in range(14)]
    emotes = [Message(20.1 + i / 200, f"e{i}", "KEKW") for i in range(100)]
    talk = [Message(30.1 + i / 200, f"t{i}", f"different message {i}") for i in range(20)]
    curve = he.chat_curve(votes + emotes + talk, 40)
    assert curve["chatters"][10] == 2            # "1" and "dog": two lines, not 114 people
    assert curve["chatters"][20] == 100          # a shared reaction counts everyone
    assert curve["chatters"][30] == 20           # ordinary chat is unchanged
