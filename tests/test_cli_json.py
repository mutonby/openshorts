"""CLI JSON output must remain parseable while jobs stream progress."""

import contextlib
import io
import json
import unittest
from unittest.mock import patch

from cli import openshorts_cli as cli


class CliJsonTests(unittest.TestCase):
    def run_cli(self, argv, responses):
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(cli, "_request", side_effect=responses), \
                patch.object(cli.time, "sleep"), \
                contextlib.redirect_stdout(stdout), \
                contextlib.redirect_stderr(stderr):
            cli.main(argv)
        return stdout.getvalue(), stderr.getvalue()

    def test_process_wait_json_keeps_progress_off_stdout(self):
        result = {"clips": [{"title": "A clip"}]}
        stdout, stderr = self.run_cli(
            ["--json", "process", "https://example.com/video.mp4", "--wait"],
            [(200, {"job_id": "job1"}),
             (200, {"status": "processing", "logs": ["Starting"]}),
             (200, {"status": "completed", "logs": ["Starting", "Done"],
                    "result": result})],
        )
        self.assertEqual(json.loads(stdout), result)
        self.assertIn("job queued: job1", stderr)
        self.assertEqual(stderr.count("Starting"), 1)
        self.assertIn("Done", stderr)

    def test_status_watch_json_keeps_logs_off_stdout(self):
        stdout, stderr = self.run_cli(
            ["--json", "status", "job1", "--watch"],
            [(200, {"status": "completed", "logs": ["Done"], "result": {}})],
        )
        self.assertEqual(json.loads(stdout), {})
        self.assertIn("Done", stderr)

    def test_process_json_without_wait_still_returns_submission(self):
        payload = {"job_id": "job1"}
        stdout, stderr = self.run_cli(
            ["--json", "process", "https://example.com/video.mp4"],
            [(200, payload)],
        )
        self.assertEqual(json.loads(stdout), payload)
        self.assertEqual(stderr, "")

    def test_human_output_still_includes_progress(self):
        stdout, stderr = self.run_cli(
            ["process", "https://example.com/video.mp4", "--wait"],
            [(200, {"job_id": "job1"}),
             (200, {"status": "completed", "logs": ["Done"], "result": {}})],
        )
        self.assertIn("job queued: job1", stdout)
        self.assertIn("Done", stdout)
        self.assertIn("completed:", stdout)
        self.assertEqual(stderr, "")

    def test_failed_json_watch_leaves_stdout_empty(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(cli, "_request", return_value=(
                200, {"status": "failed", "logs": ["Render failed"]})), \
                contextlib.redirect_stdout(stdout), \
                contextlib.redirect_stderr(stderr), \
                self.assertRaises(SystemExit) as raised:
            cli.main(["--json", "status", "job1", "--watch"])
        self.assertEqual(raised.exception.code, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("Render failed", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
