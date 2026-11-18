"""Command line: ``python -m livewatch <command>``.

  ui         open the web dashboard: start/stop, live view, clips, settings, music
  run        watch the channels in livewatch.toml, clip peaks, post them
  doctor     check tools, encoder, keys and the config
  chat       print a channel's live chat score for a while (tune threshold)
  clip       turn part of a local video into a Short with the same pipeline
  replay     run the whole pipeline on an ended YouTube stream (test outputs/posting)
  post       post a clip that was held back (``clips/<channel>/<time>``)
  snapshot   save a frame of a live channel with a 10% grid (to set facecam)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import __version__, config, procs, render


def _utf8_console():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def _load(args) -> config.Settings:
    try:
        return config.load(Path(args.config))
    except config.ConfigError as e:
        sys.exit(f"config error: {e}")


def cmd_run(args):
    from .service import Service
    s = _load(args)
    Service(s).run(duration=args.minutes * 60 if args.minutes else None)


def cmd_ui(args):
    from .webui import serve
    path = Path(args.config)
    if not path.exists():
        example = Path(__file__).resolve().parent.parent / "livewatch.example.toml"
        if example.is_file():
            shutil.copyfile(example, path)
            print(f"created {path} from the example; edit it in the Settings tab")
    serve(path, port=args.port, open_browser=not args.no_browser)


def cmd_doctor(args):
    ok = True

    def line(good, what, hint=""):
        nonlocal ok
        ok &= bool(good) or hint.startswith("optional")
        print(f"  [{'ok' if good else '--'}] {what}" + (f"  ({hint})" if hint and not good else ""))

    print(f"livewatch {__version__}")
    line(shutil.which(procs.ffmpeg()) or Path(procs.ffmpeg()).exists(), "ffmpeg", "install ffmpeg")
    line(shutil.which(procs.ffprobe()) or Path(procs.ffprobe()).exists(), "ffprobe", "install ffmpeg")
    try:
        import streamlink  # noqa: F401
        line(True, "streamlink")
    except ImportError:
        line(False, "streamlink", "pip install streamlink")
    line(shutil.which("yt-dlp"), "yt-dlp (YouTube chat)", "optional: pip install yt-dlp")
    from . import transcribe
    line(transcribe.available(), f"whisper ({transcribe.available() or 'none'})",
         "optional: pip install faster-whisper (captions)")
    enc = render.pick_encoder("auto")
    line(True, f"video encoder: {'Intel QuickSync' if enc == 'qsv' else 'libx264 (CPU)'}")
    line(render.FONTS_DIR.is_dir(), "caption fonts", "fonts/ folder missing")
    if Path(args.config).exists():
        try:
            s = config.load(Path(args.config))
            line(True, f"config: {len(s.channels)} channel(s), style {s.caption_style}"
                       f"{' (rotating)' if s.rotate_styles else ''}")
            line(bool(s.gemini_api_key), "GEMINI_API_KEY", "optional: offline hooks are used without it")
            if s.post_enabled:
                line(bool(s.upload_post_api_key), "UPLOADPOST_API_KEY", "needed to post")
                line(bool(s.upload_post_user), "post.upload_post_user", "the Upload-Post profile name")
            else:
                line(False, "posting", "optional: off; clips are saved for review")
            free = shutil.disk_usage(s.data_dir.parent if not s.data_dir.exists() else s.data_dir).free / 1e9
            line(free > 5, f"disk: {free:.1f} GB free", "keep 5 GB+ free")
        except config.ConfigError as e:
            line(False, "config", str(e))
    else:
        line(False, f"config {args.config}", "copy livewatch.example.toml to livewatch.toml")
    print("ready" if ok else "fix the items marked -- first")
    return 0 if ok else 1


def cmd_chat(args):
    from .chat import TwitchChatReader, YouTubeChatReader
    from .config import Channel
    from .detector import PeakDetector
    ch = Channel(platform=args.platform, name=args.name)
    det = PeakDetector(threshold=args.threshold, warmup=args.warmup, cooldown=30)
    if ch.platform == "twitch":
        reader = TwitchChatReader(ch.name, det.add_message)
    elif ch.platform == "youtube":
        reader = YouTubeChatReader(ch.chat_url, Path(".livewatch_chat_tmp"), det.add_message)
    else:
        sys.exit("kick has no chat reader; livewatch uses audio peaks there")
    reader.start()
    t_end = time.time() + args.seconds
    try:
        while time.time() < t_end:
            time.sleep(1)
            peak = det.evaluate()
            sc, cz, az = det.last_score
            print(f"\r{reader.count:6d} msgs  score {sc:5.1f}  chat {cz:5.1f}  "
                  f"{'(warming up)' if time.time() - (det._first or time.time()) < args.warmup else '            '}",
                  end="", flush=True)
            if peak:
                print(f"\nPEAK {peak.score} {peak.reason} :: {' | '.join(peak.messages[:4])}")
    except KeyboardInterrupt:
        pass
    reader.stop()
    print()


def cmd_clip(args):
    from .detector import Peak
    from .service import ClipJob, LiveInfo, Service
    from .config import Channel
    s = _load(args) if Path(args.config).exists() else config.from_dict({})
    if args.style:
        s.caption_style, s.rotate_styles = args.style, False
    if args.no_post:
        s.post_enabled = False
    src = Path(args.file).resolve()
    if not src.is_file():
        sys.exit(f"no such file: {src}")
    info = render.probe(src)
    end = args.end if args.end else min(info["duration"], args.start + 45)
    ch = Channel(platform=args.platform, name=args.streamer, facecam=args.facecam)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job = ClipJob(ch, Peak(ts=time.time(), score=0, chat_z=0, audio_z=0, hype_share=0,
                           messages=args.chat or [], reason="manual"),
                  0, 0, LiveInfo(True, title=args.title or "", category=args.category),
                  s.style_for(ch), s.data_dir / "clips" / "manual" / stamp)
    job.dir.mkdir(parents=True, exist_ok=True)
    job.raw = src
    job.raw_offset, job.duration = args.start, max(1.0, end - args.start)
    s.keep_raw_clips = True  # never delete the user's own file
    svc = Service(s)
    out = svc.process(job)
    print(out)


def cmd_replay(args):
    from .replay import Replay, ReplayError
    from .service import Logger
    s = _load(args) if Path(args.config).exists() else config.from_dict({})
    log = Logger(s.data_dir / "livewatch.log") if s.data_dir.exists() or s.data_dir.parent.exists() \
        else Logger(None)
    post = True if args.post else (False if args.no_post else None)
    try:
        rep = Replay(s, args.url, minutes=args.minutes, start_minute=args.start,
                     max_clips=args.max_clips, post=post, log=log)
        results = rep.run()
    except ReplayError as e:
        sys.exit(f"replay: {e}")
    except KeyboardInterrupt:
        sys.exit("replay: cancelled")
    for r in results:
        print(r.get("file") or f"failed at {r['at']:.0f}s: {r.get('error')}")


def cmd_post(args):
    from . import poster
    from .service import post_now, Logger
    s = _load(args)
    d = Path(args.clip_dir)
    meta_path = d / "meta.json"
    if not meta_path.is_file() or not (d / "short.mp4").is_file():
        sys.exit(f"{d} has no short.mp4 + meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not (s.upload_post_api_key and s.upload_post_user):
        sys.exit("set UPLOADPOST_API_KEY and post.upload_post_user first")
    ledger = poster.PostLedger(s.data_dir / "post_ledger.json")
    when = ledger.slot(datetime.now(timezone.utc), timedelta(minutes=s.min_gap_minutes), s.daily_cap)
    if when is None:
        sys.exit("daily_cap reached; try later")
    posts = meta.get("posts") or {}
    if args.platforms:
        posts = {p: v for p, v in posts.items() if p in args.platforms}
    meta["posted"] = post_now(d / "short.mp4", posts, s, ledger, when, Logger(None))
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def cmd_snapshot(args):
    from .config import Channel
    ch = Channel(platform=args.platform, name=args.name)
    out = Path(args.out).resolve()
    grid = "drawgrid=w=iw/10:h=ih/10:t=2:c=yellow@0.7"
    sl = procs.popen(procs.streamlink_cmd() + ["--stdout", "--loglevel", "error", ch.url, "720p,best"],
                     stdout=__import__("subprocess").PIPE)
    r = procs.run([procs.ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-i", "pipe:0",
                   "-frames:v", "1", "-vf", grid, str(out)],
                  stdin=sl.stdout, timeout=90)
    procs.terminate(sl)
    if r.returncode != 0 or not out.exists():
        sys.exit("could not grab a frame (is the channel live?)")
    print(f"saved {out}: read the facecam box in grid squares, e.g. 3 squares from the left, "
          f"6 down, 3 wide, 3 tall -> facecam = [0.3, 0.6, 0.3, 0.3]")


def main(argv=None):
    _utf8_console()
    p = argparse.ArgumentParser(prog="python -m livewatch", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-c", "--config", default="livewatch.toml")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="watch and clip")
    r.add_argument("--minutes", type=float, default=0, help="stop after this long (0 = forever)")
    r.set_defaults(fn=cmd_run)

    u = sub.add_parser("ui", help="open the web dashboard (start/stop, live view, clips, settings)")
    u.add_argument("--port", type=int, default=8765)
    u.add_argument("--no-browser", action="store_true")
    u.set_defaults(fn=cmd_ui)

    sub.add_parser("doctor", help="check the setup").set_defaults(fn=cmd_doctor)

    c = sub.add_parser("chat", help="watch a chat's peak score")
    c.add_argument("platform", choices=["twitch", "youtube"])
    c.add_argument("name")
    c.add_argument("--seconds", type=float, default=300)
    c.add_argument("--threshold", type=float, default=3.5)
    c.add_argument("--warmup", type=float, default=60)
    c.set_defaults(fn=cmd_chat)

    k = sub.add_parser("clip", help="make a Short from a local video")
    k.add_argument("file")
    k.add_argument("--start", type=float, default=0)
    k.add_argument("--end", type=float, default=0)
    k.add_argument("--streamer", default="streamer")
    k.add_argument("--platform", default="twitch", choices=config.PLATFORMS)
    k.add_argument("--category", default="Grand Theft Auto VI")
    k.add_argument("--title", default="")
    k.add_argument("--style", choices=config.CAPTION_STYLES)
    k.add_argument("--chat", nargs="*", help="chat lines to give the writer")
    k.add_argument("--facecam", type=float, nargs=4, metavar=("X", "Y", "W", "H"))
    k.add_argument("--no-post", action="store_true")
    k.set_defaults(fn=cmd_clip)

    rp = sub.add_parser("replay", help="run the pipeline on an ended YouTube stream")
    rp.add_argument("url")
    rp.add_argument("--minutes", type=float, default=30, help="how much of the stream to analyse")
    rp.add_argument("--start", type=float, default=0, help="start at this minute")
    rp.add_argument("--max-clips", type=int, default=5)
    g = rp.add_mutually_exclusive_group()
    g.add_argument("--post", action="store_true", help="post the clips (overrides the config)")
    g.add_argument("--no-post", action="store_true", help="never post (overrides the config)")
    rp.set_defaults(fn=cmd_replay)

    po = sub.add_parser("post", help="post a held clip")
    po.add_argument("clip_dir")
    po.add_argument("--platforms", nargs="*", choices=config.POST_PLATFORMS)
    po.set_defaults(fn=cmd_post)

    sn = sub.add_parser("snapshot", help="grab a frame with a grid to set facecam")
    sn.add_argument("platform", choices=config.PLATFORMS)
    sn.add_argument("name")
    sn.add_argument("--out", default="snapshot.png")
    sn.set_defaults(fn=cmd_snapshot)

    args = p.parse_args(argv)
    return args.fn(args) or 0


if __name__ == "__main__":
    sys.exit(main())
