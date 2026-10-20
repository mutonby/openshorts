"""ContentUnit — the single payload every adapter consumes.

Derived from the content-poster schema in CommandCenter (private), stripped of
CommandCenter-specific fields (source_project, paperclip approval hooks) so the
public Open-Dispatch repo stays generic.
"""

from __future__ import annotations

import json
import re
import uuid
import ipaddress
from urllib.parse import urlparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CAPTION_LIMITS = {
    "instagram": 2200,
    "twitter": 280,
    "telegram": 4096,
    "youtube": 5000,
    "tiktok": 2200,
    "bluesky": 300,
    "linkedin": 3000,
    "threads": 500,
    "facebook": 63206,
}

TARGET_RE = re.compile(r"^(?P<platform>[a-z]+)(?::(?P<account>[a-z0-9._-]+))?$")


class ValidationError(Exception):
    """Raised when a ContentUnit fails validation."""


def _validate_webhook_url(value: str | None) -> str | None:
    if not value:
        return None
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        return "webhook_url must be an HTTPS URL without embedded credentials"
    host = parsed.hostname.lower().rstrip(".")
    if host in {"localhost", "localhost.localdomain"}:
        return "webhook_url host is not allowed"
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address and (address.is_private or address.is_loopback or address.is_link_local or address.is_reserved or address.is_unspecified or address.is_multicast):
        return "webhook_url host is not a public address"
    return None


def parse_target(target: str) -> tuple[str, str | None]:
    m = TARGET_RE.match(target.strip().lower())
    if not m:
        raise ValidationError(f"invalid target: {target!r}")
    return m.group("platform"), m.group("account")


@dataclass
class ContentUnit:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now(tz=timezone.utc).isoformat())
    category: str = "general"
    targets: list[str] = field(default_factory=list)
    scheduled_for: str | None = None
    formats: dict[str, Any] = field(default_factory=dict)
    webhook_url: str | None = None
    profile_id: str | None = None  # which profile's creds to use when dispatching
    # Opaque handle the caller resolves back to credentials. Never a token.
    creds_ref: str | None = None
    timezone_name: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> "ContentUnit":
        return cls(
            id=d.get("id") or str(uuid.uuid4()),
            created_at=d.get("created_at") or datetime.now(tz=timezone.utc).isoformat(),
            category=d.get("category", "general"),
            targets=list(d.get("targets") or []),
            scheduled_for=d.get("scheduled_for"),
            formats=dict(d.get("formats") or {}),
            webhook_url=d.get("webhook_url"),
            profile_id=d.get("profile_id"),
            creds_ref=d.get("creds_ref"),
            timezone_name=d.get("timezone_name"),
        )

    @classmethod
    def load(cls, path: Path) -> "ContentUnit":
        return cls.from_dict(json.loads(Path(path).read_text()))


def _validate_media_paths(value: Any, path: str = "formats") -> list[str]:
    errors: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            errors.extend(_validate_media_paths(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            errors.extend(_validate_media_paths(item, f"{path}[{index}]"))
    elif isinstance(value, str):
        key = path.rsplit(".", 1)[-1].split("[", 1)[0]
        if key.endswith("_path") or key.endswith("_paths"):
            candidate = Path(value)
            if candidate.is_absolute() or ".." in candidate.parts:
                errors.append(f"{path} must be a relative media path without '..'")
    return errors


def validate(unit: ContentUnit) -> list[str]:
    errs: list[str] = []
    if not unit.targets:
        errs.append("targets must be non-empty")
    if len(unit.targets) > 20:
        errs.append("targets must contain at most 20 platforms")
    for t in unit.targets:
        try:
            parse_target(t)
        except ValidationError as e:
            errs.append(str(e))
    if not unit.formats:
        errs.append("formats must be non-empty")
    errs.extend(_validate_media_paths(unit.formats))
    webhook_error = _validate_webhook_url(unit.webhook_url)
    if webhook_error:
        errs.append(webhook_error)
    if unit.scheduled_for:
        try:
            datetime.fromisoformat(unit.scheduled_for.replace("Z", "+00:00"))
        except ValueError:
            errs.append(f"scheduled_for is not ISO-8601: {unit.scheduled_for!r}")
    return errs
