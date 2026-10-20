"""Attachable poster for finished short clips.

OpenShorts imports :func:`publish_now`, :func:`schedule`, and :func:`run_due`.
The HTTP app, the other platform adapters, and the web UI stay in this tree
and are not part of that call.
"""

from opendispatch.credentials import Credentials, InstagramCreds, YouTubeCreds
from opendispatch.shorts import DispatchError, PlatformResult, ShortPost, publish_now, run_due, schedule

__all__ = [
    "Credentials",
    "DispatchError",
    "InstagramCreds",
    "PlatformResult",
    "ShortPost",
    "YouTubeCreds",
    "publish_now",
    "run_due",
    "schedule",
]
