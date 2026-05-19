"""Compat shim: re-exports openshorts.integrations.s3 at the original import path.

This module moved to openshorts/integrations/s3.py as part of the restructure.
New code should import from `openshorts.integrations.s3` directly; this shim
keeps existing imports (e.g. `from s3_uploader import upload_job_artifacts`)
working while the restructure is in flight.
"""
from openshorts.integrations.s3 import *  # noqa: F401,F403
from openshorts.integrations.s3 import (  # noqa: F401
    upload_file_to_s3,
    get_s3_client,
    generate_presigned_url,
    list_all_clips,
    upload_actor_to_s3,
    list_actor_gallery,
    upload_video_to_gallery,
    list_video_gallery,
    upload_job_artifacts,
)
