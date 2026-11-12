"""OpenShorts CLI: clip long videos into vertical shorts from the terminal.

Zero dependencies by design so `uvx openshorts` and `pipx run openshorts`
start instantly. Talks to the same REST API the dashboard and the MCP server
use; nothing here can drift from what the app actually does.

Auth and target come from the environment:
  OPENSHORTS_API_KEY  osk_... key from the account page (cloud only)
  OPENSHORTS_API_URL  defaults to https://api.openshorts.app; set to
                      http://localhost:8000 for a self-hosted instance,
                      where no key is needed.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

DEFAULT_API = "https://api.openshorts.app"
POLL_SECONDS = 10


def _base():
    return os.environ.get("OPENSHORTS_API_URL", DEFAULT_API).rstrip("/")


def _request(method, path, body=None):
    headers = {"Accept": "application/json"}
    key = os.environ.get("OPENSHORTS_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(_base() + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, json.loads(resp.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        try:
            payload = json.loads(e.read().decode() or "{}")
        except Exception:
            payload = {"detail": str(e.reason)}
        return e.code, payload
    except urllib.error.URLError as e:
        print(f"error: cannot reach {_base()} ({e.reason})", file=sys.stderr)
        sys.exit(2)


def _die(status, payload):
    detail = payload.get("detail", payload) if isinstance(payload, dict) else payload
    if not isinstance(detail, str):
        detail = json.dumps(detail)
    print(f"error ({status}): {detail}", file=sys.stderr)
    sys.exit(1)


def _absolutize(url):
    if url and url.startswith("/"):
        return _base() + url
    return url


def _print_clips(result):
    clips = (result or {}).get("clips") or []
    if not clips:
        print("no clips in result")
        return
    for i, clip in enumerate(clips):
        title = clip.get("title") or clip.get("video_title_for_youtube_short") or f"clip {i}"
        print(f"[{i}] {title}")
        # Cloud: download_url is signed and opens without the API key.
        print(f"    {clip.get('download_url') or _absolutize(clip.get('video_url') or '')}")


def cmd_process(args):
    body = {"url": args.url, "acknowledged": True}
    if args.layouts:
        body["layouts"] = args.layouts
    if args.format:
        body["output_format"] = args.format
    if args.webhook:
        body["webhook_url"] = args.webhook
    if args.webhook_secret:
        body["webhook_secret"] = args.webhook_secret
    if args.blur_sensitive:
        body["blur_sensitive"] = True
    status, payload = _request("POST", "/api/process", body)
    if status >= 400:
        _die(status, payload)
    job_id = payload.get("job_id")
    if args.json and not args.wait:
        print(json.dumps(payload))
        return
    print(f"job queued: {job_id}")
    if args.wait:
        _watch(job_id, as_json=args.json)
    else:
        print(f"follow it with: openshorts status {job_id} --watch")


def _watch(job_id, as_json=False):
    seen_logs = 0
    while True:
        status, payload = _request("GET", f"/api/status/{job_id}")
        if status >= 400:
            _die(status, payload)
        logs = payload.get("logs") or []
        for line in logs[seen_logs:]:
            print(f"  {line}")
        seen_logs = len(logs)
        state = payload.get("status")
        if state == "completed":
            if as_json:
                print(json.dumps(payload.get("result") or {}))
            else:
                print("completed:")
                _print_clips(payload.get("result"))
            return
        if state == "failed":
            print("job failed; last log lines above", file=sys.stderr)
            sys.exit(1)
        time.sleep(POLL_SECONDS)


def cmd_status(args):
    if args.watch:
        _watch(args.job_id, as_json=args.json)
        return
    status, payload = _request("GET", f"/api/status/{args.job_id}")
    if status >= 400:
        _die(status, payload)
    if args.json:
        print(json.dumps(payload))
        return
    print(f"status: {payload.get('status')}")
    logs = payload.get("logs") or []
    if logs:
        print(f"last log: {logs[-1]}")


def cmd_clips(args):
    status, payload = _request("GET", f"/api/status/{args.job_id}")
    if status >= 400:
        _die(status, payload)
    if payload.get("status") != "completed":
        print(f"job is {payload.get('status')}, no clips yet", file=sys.stderr)
        sys.exit(1)
    if args.json:
        print(json.dumps((payload.get("result") or {}).get("clips") or []))
        return
    _print_clips(payload.get("result"))


def cmd_quota(args):
    status, payload = _request("GET", "/api/me")
    # 401 anonymous, 404 self-host (the cloud router is not mounted): neither
    # is an error, there is simply no minute quota to report.
    if status in (401, 404):
        print("no cloud account in play: self-hosted or anonymous, no minute quota")
        return
    if status >= 400:
        _die(status, payload)
    if args.json:
        print(json.dumps(payload))
        return
    print(f"plan: {payload.get('plan')}")
    print(f"minutes: {payload.get('minutes')}")
    print(f"entitled: {payload.get('entitled')}")


def cmd_publish(args):
    body = {
        "job_id": args.job_id,
        "clip_index": args.clip_index,
        "platforms": [p.strip() for p in args.platforms.split(",") if p.strip()],
    }
    if args.title:
        body["title"] = args.title
    if args.schedule:
        body["scheduled_date"] = args.schedule
    if args.timezone:
        body["timezone"] = args.timezone
    status, payload = _request("POST", "/api/social/post", body)
    if status >= 400:
        _die(status, payload)
    print(json.dumps(payload) if args.json else f"publishing: {payload}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="openshorts",
        description="Clip long videos into vertical shorts via the OpenShorts API.",
    )
    parser.add_argument("--json", action="store_true", help="raw JSON output")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("process", help="submit a video URL for clipping")
    p.add_argument("url", help="YouTube or direct video URL")
    p.add_argument("--layouts", help="comma list: auto,split,screencast,speaker_cut,punch_in")
    p.add_argument("--format", help="output format, e.g. 1080p")
    p.add_argument("--webhook", help="webhook URL fired once when the job ends")
    p.add_argument("--webhook-secret", help="HMAC secret for X-OpenShorts-Signature")
    p.add_argument("--wait", action="store_true", help="stream logs until the job ends")
    p.add_argument("--blur-sensitive", action="store_true",
                   help="blur faces, license plates and on-screen personal details (off unless passed)")
    p.set_defaults(func=cmd_process)

    p = sub.add_parser("status", help="job status and last log line")
    p.add_argument("job_id")
    p.add_argument("--watch", action="store_true", help="stream logs until the job ends")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("clips", help="list finished clips with links")
    p.add_argument("job_id")
    p.set_defaults(func=cmd_clips)

    p = sub.add_parser("quota", help="plan and remaining minutes")
    p.set_defaults(func=cmd_quota)

    p = sub.add_parser("publish", help="post or schedule one clip to social platforms")
    p.add_argument("job_id")
    p.add_argument("clip_index", type=int)
    p.add_argument("--platforms", required=True, help="comma list: tiktok,instagram,youtube")
    p.add_argument("--title")
    p.add_argument("--schedule", help="ISO datetime for scheduled posting")
    p.add_argument("--timezone", help="IANA timezone for --schedule")
    p.set_defaults(func=cmd_publish)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
