"""Secure Vantsports -> VANTCLIP job contract.

This module only builds and signs requests. Transport, retries and persistence stay
in the API layer so the video workers remain independent from tournament logic.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ClipJobRequest:
    project_id: str
    idempotency_key: str
    source_url: str | None = None
    source_file_path: str | None = None
    tournament_id: str | None = None

    def payload(self) -> dict[str, Any]:
        if not self.source_url and not self.source_file_path:
            raise ValueError("A source_url or source_file_path is required")
        return {
            "project_id": self.project_id,
            "idempotency_key": self.idempotency_key,
            "source_url": self.source_url,
            "source_file_path": self.source_file_path,
            "tournament_id": self.tournament_id,
        }


def sign_payload(payload: Mapping[str, Any], secret: str, timestamp: int | None = None) -> tuple[int, str]:
    """Return a timestamp and hex HMAC signature for an internal webhook."""
    if not secret:
        raise ValueError("VANTCLIP webhook secret is required")
    issued_at = int(time.time()) if timestamp is None else timestamp
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    message = f"{issued_at}.".encode("ascii") + body
    digest = hmac.new(secret.encode("utf-8"), message, hashlib.sha256).hexdigest()
    return issued_at, digest


def build_headers(payload: Mapping[str, Any], secret: str, request_id: str, timestamp: int | None = None) -> dict[str, str]:
    """Build the signed headers expected by the VANTCLIP API."""
    issued_at, signature = sign_payload(payload, secret, timestamp)
    return {
        "Content-Type": "application/json",
        "X-Vantsports-Request-Id": request_id,
        "X-Vantsports-Timestamp": str(issued_at),
        "X-Vantsports-Signature": signature,
        "Idempotency-Key": str(payload["idempotency_key"]),
    }
