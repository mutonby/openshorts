#!/usr/bin/env python3
"""
YouTube Shorts auto-publisher: queue clips from OpenShorts jobs and publish to YouTube.

This script is designed to run periodically via cron (typically every 10-15 minutes).
It reads a state file to manage clip queues, respects scheduling (posts_per_day, times, timezone),
and uploads to YouTube with configurable metadata.

Configuration (config.json):
  {
    "openshorts_url": "http://localhost:8000",
    "state_file": "state.json",
    "youtube_token_path": "secrets/youtube_token.json",
    "posts_per_day": 2,
    "post_times": ["09:00", "18:00"],
    "timezone": "America/Bogota"
  }

State file (state.json):
  {
    "queue": [
      {
        "job_id": "xyz123",
        "clip_index": 0,
        "video_path": "/path/to/clip.mp4",
        "title": "Amazing Moment",
        "description": "...",
        "tags": ["anime", "short"],
        "added_at": 1696000000
      }
    ],
    "published": [
      {
        "job_id": "xyz123",
        "clip_index": 0,
        "video_id": "abc123",
        "published_at": 1696050000
      }
    ]
  }
"""

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone as tz
from pathlib import Path
from typing import Dict, List, Optional

import httpx
import pytz
from youtube_utils import YouTubeUploader, YouTubeUploadError, YouTubeQuotaExceeded

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("publish_to_youtube.log")
    ]
)


class YouTubePublisher:
    """Manage clip queue and YouTube uploads."""

    def __init__(self, config_path: str, state_path: str, token_path: str):
        """
        Initialize publisher.

        Args:
            config_path: Path to config.json
            state_path: Path to state.json
            token_path: Path to YouTube OAuth token
        """
        self.config_path = Path(config_path)
        self.state_path = Path(state_path)
        self.token_path = Path(token_path)

        # Load configuration
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        with open(self.config_path) as f:
            self.config = json.load(f)

        self.openshorts_url = self.config.get("openshorts_url", "http://localhost:8000").rstrip("/")
        self.posts_per_day = self.config.get("posts_per_day", 2)
        self.post_times = self.config.get("post_times", ["09:00", "18:00"])
        self.timezone_name = self.config.get("timezone", "UTC")

        # Initialize YouTube uploader
        self.uploader = YouTubeUploader(str(self.token_path))

        # Load or initialize state
        self.state = self._load_state()

    def _load_state(self) -> Dict:
        """Load state from file, or create empty state."""
        if self.state_path.exists():
            with open(self.state_path) as f:
                return json.load(f)
        return {"queue": [], "published": []}

    def _save_state(self):
        """Save state to file atomically."""
        # Write to temp file first, then move (atomic on most filesystems)
        with tempfile.NamedTemporaryFile(mode='w', delete=False, dir=self.state_path.parent) as tmp:
            json.dump(self.state, tmp, indent=2)
            tmp_path = tmp.name

        os.replace(tmp_path, self.state_path)
        logger.info(f"State saved to {self.state_path}")

    def add_clip_to_queue(
        self,
        job_id: str,
        clip_index: int,
        video_path: str,
        title: str,
        description: str = "",
        tags: List[str] = None,
    ):
        """Add a clip to the publishing queue."""
        clip = {
            "job_id": job_id,
            "clip_index": clip_index,
            "video_path": video_path,
            "title": title,
            "description": description,
            "tags": tags or [],
            "added_at": int(time.time()),
            "retry_count": 0,
            "last_error": None,
        }
        self.state["queue"].append(clip)
        self._save_state()
        logger.info(f"Added clip to queue: {job_id}/{clip_index}")

    def get_next_scheduled_time(self) -> datetime:
        """
        Calculate when the next clip should be published based on schedule.

        Returns next scheduled time in UTC.
        """
        tz_obj = pytz.timezone(self.timezone_name)
        now_local = datetime.now(tz_obj)
        now_utc = now_local.astimezone(pytz.UTC)

        # Parse configured post times
        scheduled_times = []
        for time_str in self.post_times:
            try:
                hour, minute = map(int, time_str.split(":"))
                scheduled_times.append((hour, minute))
            except (ValueError, AttributeError):
                logger.warning(f"Invalid post time format: {time_str}, skipping")
                continue

        if not scheduled_times:
            # Fallback: post every N hours
            return now_utc + timedelta(hours=24 // self.posts_per_day)

        # Find next scheduled time
        scheduled_times.sort()

        # Create datetime objects for today's scheduled times (in local timezone)
        today_scheduled = []
        for hour, minute in scheduled_times:
            dt_local = now_local.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if dt_local > now_local:
                today_scheduled.append(dt_local)

        if today_scheduled:
            # Use earliest unmet time today
            next_time_local = min(today_scheduled)
        else:
            # All today's times have passed; use first time tomorrow
            tomorrow = now_local + timedelta(days=1)
            first_time_local = tomorrow.replace(
                hour=scheduled_times[0][0],
                minute=scheduled_times[0][1],
                second=0,
                microsecond=0
            )
            next_time_local = first_time_local

        next_time_utc = next_time_local.astimezone(pytz.UTC)
        return next_time_utc

    def publish_next(self) -> Optional[Dict]:
        """
        Publish the next clip in the queue if it's time.

        Returns the published clip dict, or None if queue empty or not yet scheduled.
        """
        queue = self.state.get("queue", [])
        if not queue:
            logger.debug("Queue empty")
            return None

        clip = queue[0]

        # Check if it's time to publish
        next_scheduled = self.get_next_scheduled_time()
        now_utc = datetime.now(pytz.UTC)

        if now_utc < next_scheduled:
            seconds_until = (next_scheduled - now_utc).total_seconds()
            logger.info(f"Next publish in {seconds_until:.0f}s ({next_scheduled.isoformat()})")
            return None

        # Time to publish!
        logger.info(f"Publishing clip {clip['job_id']}/{clip['clip_index']}")

        try:
            video_path = clip["video_path"]
            if not os.path.exists(video_path):
                # Try to download from OpenShorts if local path doesn't exist
                logger.info(f"Video path not found: {video_path}, skipping")
                self._move_clip_to_published(clip, None)
                return clip

            # Determine publish time (next scheduled slot)
            scheduled_time = self.get_next_scheduled_time()

            video_id, video_url = self.uploader.upload_video(
                video_path=video_path,
                title=clip["title"],
                description=clip["description"],
                tags=clip.get("tags", []),
                scheduled_time=scheduled_time,
                made_for_kids=False,
            )

            logger.info(f"✅ Published: {video_url}")

            # Move to published
            self._move_clip_to_published(clip, video_id)
            return clip

        except YouTubeQuotaExceeded:
            logger.error("YouTube quota exceeded! Stopping uploads.")
            return None
        except YouTubeUploadError as e:
            logger.error(f"Upload failed: {e}")
            clip["retry_count"] = clip.get("retry_count", 0) + 1
            clip["last_error"] = str(e)

            # Retry up to 3 times
            if clip["retry_count"] >= 3:
                logger.error(f"Max retries reached for {clip['job_id']}/{clip['clip_index']}, removing from queue")
                queue.pop(0)
                self._save_state()
            else:
                self._save_state()  # Save retry count

            return None
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            return None

    def _move_clip_to_published(self, clip: Dict, video_id: Optional[str]):
        """Move a clip from queue to published."""
        queue = self.state.get("queue", [])
        if clip in queue:
            queue.remove(clip)

        published_entry = {
            "job_id": clip["job_id"],
            "clip_index": clip["clip_index"],
            "video_id": video_id,
            "published_at": int(time.time()),
        }
        self.state.setdefault("published", []).append(published_entry)
        self._save_state()
        logger.info(f"Moved to published: {clip['job_id']}/{clip['clip_index']} → {video_id}")

    def extract_clips_from_job(
        self,
        job_id: str,
        output_dir: str,
        base_name: str,
    ) -> int:
        """
        Extract clips from a completed OpenShorts job and add to queue.

        Args:
            job_id: OpenShorts job ID
            output_dir: Path to job output directory
            base_name: Metadata base name (e.g., "abc123_source")

        Returns:
            Number of clips added to queue
        """
        logger.info(f"Extracting clips from job {job_id}")

        # Load metadata
        metadata_path = Path(output_dir) / f"{base_name}_metadata.json"
        if not metadata_path.exists():
            logger.error(f"Metadata not found: {metadata_path}")
            return 0

        with open(metadata_path) as f:
            metadata = json.load(f)

        shorts = metadata.get("shorts", [])
        if not shorts:
            logger.warning(f"No clips in metadata for {job_id}")
            return 0

        added_count = 0
        for i, clip_data in enumerate(shorts):
            # Try to find the video file
            clip_filename = f"{base_name}_clip_{i:02d}.mp4"
            clip_path = Path(output_dir) / clip_filename

            if not clip_path.exists():
                # Try alternate naming
                clip_path = Path(output_dir) / f"clip_{i}.mp4"

            if not clip_path.exists():
                logger.warning(f"Clip file not found for {job_id}/{i}: {clip_filename}")
                continue

            title = clip_data.get("video_title_for_youtube_short", f"Clip {i+1}")
            description = clip_data.get("video_description_for_tiktok", "")
            tags = ["shorts", "viral"]  # Customize as needed

            self.add_clip_to_queue(
                job_id=job_id,
                clip_index=i,
                video_path=str(clip_path),
                title=title,
                description=description,
                tags=tags,
            )
            added_count += 1

        logger.info(f"Extracted {added_count} clip(s) from job {job_id}")
        return added_count

    def status(self):
        """Print queue status."""
        queue_len = len(self.state.get("queue", []))
        published_len = len(self.state.get("published", []))

        print(f"\n📊 Publishing Status:")
        print(f"   Queue: {queue_len} clip(s)")
        print(f"   Published: {published_len} clip(s)")

        if queue_len > 0:
            next_clip = self.state["queue"][0]
            next_time = self.get_next_scheduled_time()
            print(f"\n   Next clip: {next_clip['job_id']}/{next_clip['clip_index']}")
            print(f"   Scheduled: {next_time.isoformat()}")


def main():
    parser = argparse.ArgumentParser(
        description="Publish OpenShorts clips to YouTube"
    )
    parser.add_argument(
        "--config",
        default="config.json",
        help="Config file path"
    )
    parser.add_argument(
        "--state",
        default="state.json",
        help="State file path"
    )
    parser.add_argument(
        "--youtube-token",
        default="secrets/youtube_token.json",
        help="YouTube OAuth token path"
    )
    parser.add_argument(
        "--extract",
        metavar="JOB_ID",
        help="Extract clips from a completed job and add to queue"
    )
    parser.add_argument(
        "--extract-dir",
        default="output",
        help="Output directory to search for job (used with --extract)"
    )
    parser.add_argument(
        "--extract-base",
        help="Metadata base name (e.g., 'abc123_source', defaults to JOB_ID_source)"
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Publish the next queued clip if scheduled"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show queue and publication status"
    )
    parser.add_argument(
        "--add-clip",
        metavar="JSON",
        help="Add a single clip to queue (JSON dict with job_id, clip_index, video_path, title, description, tags)"
    )

    args = parser.parse_args()

    try:
        publisher = YouTubePublisher(args.config, args.state, args.youtube_token)

        if args.extract:
            base_name = args.extract_base or f"{args.extract}_source"
            job_dir = os.path.join(args.extract_dir, args.extract)
            publisher.extract_clips_from_job(args.extract, job_dir, base_name)

        elif args.add_clip:
            clip_data = json.loads(args.add_clip)
            publisher.add_clip_to_queue(**clip_data)

        elif args.publish:
            result = publisher.publish_next()
            if result:
                print(f"\n✅ Published: {result['title']}")
            else:
                print("\n⏳ No clips ready to publish yet")

        elif args.status:
            publisher.status()

        else:
            # Default: try to publish if ready
            publisher.publish_next()
            publisher.status()

    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
