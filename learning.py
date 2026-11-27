"""Learn what YOUR clips need from what you pick and what gets views.

The hype engine and the judge rank candidates with fixed weights. What makes
a clip work differs per channel, and two things this install already sees
say what it is:

- approvals: in every review you render some candidates and skip others;
- views: clips posted through Upload-Post come back with their view counts
  whenever the analytics pass through the API (/api/social/analytics/posts).

Each candidate's measured features (hype, chat against its normal, clip
requests, viewer clips, streamer excitement and laughter, the judge's score
and ratings, length) are stored with those outcomes in a local SQLite file
(LEARNING_DIR, default output/.learning). After every new outcome a small
model is refitted with numpy:

- with MIN_VIEWS posted clips that have views: ridge regression of
  log(1 + views), the signal that matters;
- else with MIN_APPROVALS decisions of both kinds: logistic regression of
  "you rendered it";
- else nothing: the fixed ranking stands.

``score`` turns a candidate into 0..1 under the current model; hype_judge.rank
blends it in. Self-host only: in the cloud this would be per-user data.
"""
import json
import math
import os
import sqlite3
import threading
import time
from pathlib import Path

MIN_VIEWS = 15
MIN_APPROVALS = 20
LEVELS = {"weak": 0.0, "clear": 1.0, "strong": 2.0}
FEATURES = ("hype", "chat_excitement", "chat_ratio", "clip_requests", "viewer_clips",
            "voice_excitement", "laughing", "virality", "hook", "visual_action", "reaction",
            "clarity", "payoff", "duration", "alert")
_lock = threading.Lock()
_model_cache = {"mtime": None, "model": None}


def enabled():
    return (os.environ.get("BILLING_ENABLED", "").strip().lower() not in ("1", "true", "yes")
            and os.environ.get("LEARNING", "1").strip() != "0")


def directory():
    configured = os.environ.get("LEARNING_DIR", "").strip()
    return Path(configured) if configured else Path(os.environ.get("OUTPUT_DIR", "output")) / ".learning"


def features(clip):
    """The feature vector (dict) of one candidate or rendered clip."""
    hype = clip.get("hype") if isinstance(clip.get("hype"), dict) else {}
    signals = hype.get("signals") if isinstance(hype.get("signals"), dict) else {}
    chat = signals.get("chat") if isinstance(signals.get("chat"), dict) else {}
    streamer = signals.get("streamer") if isinstance(signals.get("streamer"), dict) else {}
    a = clip.get("assessment") if isinstance(clip.get("assessment"), dict) else {}
    ratings = a.get("ratings") if isinstance(a.get("ratings"), dict) else {}

    def num(value, default=0.0):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return default
        return value if math.isfinite(value) else default

    normal = num(chat.get("normal_chatters_per_second"))
    peak = num(chat.get("peak_chatters_per_second"))
    virality = a.get("virality")
    return {
        "hype": num(hype.get("score")),
        "chat_excitement": num(chat.get("excitement")),
        "chat_ratio": math.log1p(peak / normal) if normal > 0 else math.log1p(peak / 5.0),
        "clip_requests": math.log1p(num(chat.get("clip_requests"))),
        "viewer_clips": math.log1p(num(signals.get("viewer_clips"))),
        "voice_excitement": num(streamer.get("excitement")),
        "laughing": 1.0 if streamer.get("laughing") else 0.0,
        "virality": num(virality, 50.0) / 100.0 if virality is not None else 0.5,
        **{name: LEVELS.get(ratings.get(name), 1.0) for name in ("hook", "visual_action", "reaction",
                                                                 "clarity", "payoff")},
        "duration": num(clip.get("end")) - num(clip.get("start")),
        "alert": num(hype.get("alert")),
    }


from contextlib import contextmanager


@contextmanager
def _db():
    """One connection under the module lock, committed and closed."""
    with _lock:
        db = _connect()
        try:
            with db:
                yield db
        finally:
            db.close()


def _connect():
    folder = directory()
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    db = sqlite3.connect(folder / "learning.sqlite3", timeout=10)
    db.execute("""CREATE TABLE IF NOT EXISTS examples (
        key TEXT PRIMARY KEY, job_id TEXT NOT NULL, features TEXT NOT NULL,
        approved INTEGER, views REAL, request_id TEXT, post_url TEXT, updated REAL)""")
    db.execute("CREATE INDEX IF NOT EXISTS examples_request ON examples(request_id)")
    return db


def _key(job_id, clip):
    return f"{job_id}:{float(clip.get('start', 0)):.1f}-{float(clip.get('end', 0)):.1f}"


def record_review(job_id, shorts):
    """Every candidate of a saved review, outcome unknown yet."""
    if not enabled():
        return
    with _db() as db:
        for clip in shorts or []:
            db.execute("INSERT OR IGNORE INTO examples(key, job_id, features, updated) VALUES (?,?,?,?)",
                       (_key(job_id, clip), job_id, json.dumps(features(clip)), time.time()))


def record_approval(job_id, shorts, selected_ids):
    """Which of a review's candidates (1-based ids) were rendered."""
    if not enabled():
        return
    chosen = set(selected_ids or [])
    with _db() as db:
        for i, clip in enumerate(shorts or [], 1):
            db.execute("""INSERT INTO examples(key, job_id, features, approved, updated) VALUES (?,?,?,?,?)
                          ON CONFLICT(key) DO UPDATE SET approved=excluded.approved, updated=excluded.updated""",
                       (_key(job_id, clip), job_id, json.dumps(features(clip)), int(i in chosen), time.time()))
    refit()


def record_post(job_id, clip, request_id):
    """A clip posted through Upload-Post: its request id links the views."""
    if not enabled() or not request_id:
        return
    with _db() as db:
        db.execute("""INSERT INTO examples(key, job_id, features, approved, request_id, updated) VALUES (?,?,?,1,?,?)
                      ON CONFLICT(key) DO UPDATE SET request_id=excluded.request_id, updated=excluded.updated""",
                   (_key(job_id, clip), job_id, json.dumps(features(clip)), str(request_id), time.time()))


def sync_views(rows):
    """Views from Upload-Post analytics rows (any shape social_metrics reads),
    matched to posted clips by request id. Returns how many clips changed."""
    if not enabled():
        return 0
    import social_metrics

    totals = {}
    for row in rows or []:
        if not isinstance(row, dict) or not row.get("request_id"):
            continue
        key = (str(row["request_id"]), row.get("platform"), row.get("post_id") or row.get("post_url"))
        totals[key] = max(totals.get(key, 0.0), social_metrics.row_views(row))
    per_request = {}
    for (request_id, _platform, _post), views in totals.items():
        per_request[request_id] = per_request.get(request_id, 0.0) + views
    changed = 0
    with _db() as db:
        for request_id, views in per_request.items():
            changed += db.execute("UPDATE examples SET views=?, updated=? WHERE request_id=? AND "
                                  "(views IS NULL OR views < ?)",
                                  (views, time.time(), request_id, views)).rowcount
    if changed:
        refit()
    return changed


def _matrix(rows):
    import numpy as np
    return np.array([[float(json.loads(f).get(name, 0.0)) for name in FEATURES] for f in rows], dtype="float64")


def _fit_ridge(x, y, alpha=1.0):
    import numpy as np
    a = np.hstack([x, np.ones((len(x), 1))])
    reg = alpha * np.eye(a.shape[1])
    reg[-1, -1] = 0.0
    return np.linalg.solve(a.T @ a + reg, a.T @ y)


def _fit_logistic(x, y, alpha=1.0, steps=800, rate=0.3):
    import numpy as np
    a = np.hstack([x, np.ones((len(x), 1))])
    w = np.zeros(a.shape[1])
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-(a @ w)))
        grad = a.T @ (p - y) / len(y) + alpha * np.r_[w[:-1], 0.0] / len(y)
        w -= rate * grad
    return w


def refit():
    """Refit and save the model from every outcome so far. Returns its kind
    ("views", "approvals") or None when there is not enough to learn from."""
    if not enabled():
        return None
    import numpy as np

    with _db() as db:
        viewed = db.execute("SELECT features, views FROM examples WHERE views IS NOT NULL").fetchall()
        judged = db.execute("SELECT features, approved FROM examples WHERE approved IS NOT NULL").fetchall()
    kind, rows, target = None, None, None
    if len(viewed) >= MIN_VIEWS:
        kind, rows = "views", viewed
        target = np.log1p(np.array([float(v) for _, v in viewed]))
    elif len(judged) >= MIN_APPROVALS and 0 < sum(a for _, a in judged) < len(judged):
        kind, rows = "approvals", judged
        target = np.array([float(a) for _, a in judged])
    path = directory() / "model.json"
    if kind is None:
        return None
    x = _matrix([f for f, _ in rows])
    mean, std = x.mean(axis=0), x.std(axis=0)
    std[std < 1e-9] = 1.0
    z = (x - mean) / std
    if kind == "views":
        weights = _fit_ridge(z, target)
        fitted = np.hstack([z, np.ones((len(z), 1))]) @ weights
        scale = {"low": float(np.percentile(fitted, 5)), "high": float(np.percentile(fitted, 95))}
    else:
        weights = _fit_logistic(z, target)
        scale = {}
    model = {"kind": kind, "features": list(FEATURES), "mean": mean.tolist(), "std": std.tolist(),
             "weights": weights.tolist(), "examples": len(rows), "scale": scale, "fitted_at": time.time()}
    tmp = path.with_suffix(".partial")
    tmp.write_text(json.dumps(model))
    tmp.replace(path)
    _model_cache["mtime"] = None
    return kind


def model():
    """The saved model (cached by file time), or None."""
    path = directory() / "model.json"
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    if _model_cache["mtime"] != mtime:
        try:
            data = json.loads(path.read_text())
            if data.get("features") != list(FEATURES):
                data = None
        except (OSError, ValueError):
            data = None
        _model_cache.update(mtime=mtime, model=data)
    return _model_cache["model"]


def score(clip):
    """0..1: how much this candidate looks like what worked for you, or None
    without a model."""
    m = model() if enabled() else None
    if not m:
        return None
    f = features(clip)
    z = [(f[name] - mu) / sd for name, mu, sd in zip(m["features"], m["mean"], m["std"])]
    raw = sum(w * v for w, v in zip(m["weights"], z)) + m["weights"][-1]
    if m["kind"] == "approvals":
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, raw))))
    low, high = m["scale"].get("low", 0.0), m["scale"].get("high", 1.0)
    return max(0.0, min(1.0, (raw - low) / ((high - low) or 1.0)))


def status():
    """Counts and the current model's strongest features, for the API."""
    if not enabled():
        return {"enabled": False}
    with _db() as db:
        total, approved, viewed = db.execute(
            "SELECT COUNT(*), SUM(approved IS NOT NULL), SUM(views IS NOT NULL) FROM examples").fetchone()
    m = model()
    top = []
    if m:
        pairs = sorted(zip(m["features"], m["weights"][:-1]), key=lambda p: -abs(p[1]))
        top = [{"feature": name, "weight": round(w, 3)} for name, w in pairs[:5]]
    return {"enabled": True, "candidates": total or 0, "decisions": approved or 0, "posts_with_views": viewed or 0,
            "model": m and {"kind": m["kind"], "examples": m["examples"], "top_features": top},
            "needs": {"approvals": MIN_APPROVALS, "views": MIN_VIEWS}}
