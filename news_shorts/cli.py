"""python -m news_shorts.cli {discover,approve,publish,list} ..."""
import argparse
import glob
import os
import subprocess
import sys

from . import discover as D
from .licensing import ALLOWED_LICENSES
from .publish import publish_approved
from .queue import Queue


def cut(cand, out_dir):
    """Reuse the repo's clipper (main.py) on a licensed source."""
    os.makedirs(out_dir, exist_ok=True)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    subprocess.run([sys.executable, os.path.join(root, "main.py"), "-u", cand["source_url"],
                    "-o", out_dir], check=True)
    return sorted(glob.glob(os.path.join(out_dir, "*.mp4")))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="news_shorts")
    ap.add_argument("--queue", default="output/news_queue.json")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("discover", help="find licensed videos, cut clips, queue for review")
    d.add_argument("query")
    d.add_argument("--url", help="manual source (Assemblée nationale / owned)")
    d.add_argument("--license", choices=[k for k in ALLOWED_LICENSES if k != "creativeCommon"])
    d.add_argument("--attribution")
    d.add_argument("--limit", type=int, default=3)
    d.add_argument("--out", default="output/news")
    a = sub.add_parser("approve")
    a.add_argument("clip_path")
    p = sub.add_parser("publish")
    p.add_argument("--platform", action="append", choices=["youtube", "tiktok"])
    p.add_argument("--no-dry-run", action="store_true", help="really publish (default: dry run)")
    sub.add_parser("list")
    args = ap.parse_args(argv)
    q = Queue(args.queue)

    if args.cmd == "discover":
        if args.url:
            cands = [D.manual_candidate(args.url, args.license, args.attribution, args.query)]
        else:
            cands = D.discover_youtube_cc(args.query, limit=args.limit)
        for c in cands:
            print(f"licensed source: {c['source_url']} ({c['license']})")
            try:
                clips = cut(c, os.path.join(args.out, c["id"]))
            except subprocess.CalledProcessError as e:
                print(f"  cutting failed (exit {e.returncode}); run main.py manually to see why")
                continue
            for clip in clips:
                q.add({**c, "clip_path": clip})
        print(f"{len(cands)} licensed source(s) found; review with `list`")
    elif args.cmd == "approve":
        print("approved" if q.approve(args.clip_path) else "not found or not pending")
    elif args.cmd == "list":
        for i in q.load():
            print(i["status"], i["clip_path"], i["license"], i["source_url"])
    else:
        res = publish_approved(q, tuple(args.platform or ("youtube", "tiktok")),
                               dry_run=not args.no_dry_run)
        for path, r in res:
            print(path, r)
        if not res:
            print("nothing approved to publish")


if __name__ == "__main__":
    main()
