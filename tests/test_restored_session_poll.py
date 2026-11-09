"""A restored dashboard session whose job the server no longer has.

Cloud drops finished jobs after JOB_RETENTION_SECONDS (1 h) and keeps their
clips in History. A browser reopened later restored the saved "processing"
session and polled /api/status forever: every 404 was only logged, so the
screen sat on "waiting for clips..." with empty logs. The poll must give up on
a gone job and send the user to History.
"""
import json
import os
import shutil
import subprocess

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULE = os.path.join(REPO, "dashboard", "src", "lib", "jobPoll.js")
APP = os.path.join(REPO, "dashboard", "src", "App.jsx")


def _decide(cases):
    """Run pollFailureAction over [(status|None, failures), ...] in node."""
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    script = (
        "const m = await import(process.argv[1]);"
        "const cases = JSON.parse(process.argv[2]);"
        "console.log(JSON.stringify(cases.map(([s, n]) => m.pollFailureAction("
        "  s === null ? new Error('network') : new m.JobPollError(s), n))));"
    )
    out = subprocess.run(
        [node, "--input-type=module", "-e", script, "file://" + MODULE, json.dumps(cases)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout)


def test_purged_or_foreign_job_stops_the_poll_at_once():
    assert _decide([[404, 1], [403, 1], [410, 1]]) == ["gone", "gone", "gone"]


def test_transient_failures_keep_polling_until_the_ceiling():
    assert _decide([[502, 1], [500, 5], [None, 1], [None, 59]]) == ["retry"] * 4
    assert _decide([[502, 60], [None, 60]]) == ["gone", "gone"]


def test_app_ends_the_poll_and_drops_the_saved_session():
    src = open(APP).read()
    poll = src[src.index("const pollJob"):src.index("function App()")]
    assert "throw new JobPollError(res.status)" in poll
    effect = src[src.index("pollFailureAction(e, failures)") - 400:
                 src.index("pollFailureAction(e, failures)") + 600]
    assert "failures += 1" in effect
    assert "clearInterval(interval)" in effect
    # handleReset sets status to idle, which removes the saved session.
    assert "handleResetRef.current()" in effect
    assert "setActiveTab('history')" in effect
    assert "setJobGone(true)" in effect
