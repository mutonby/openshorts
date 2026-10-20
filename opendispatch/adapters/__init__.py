"""Adapter registry. Each adapter exposes `publish(unit, account) -> (ok, post_id, error)`."""

from __future__ import annotations

from typing import Protocol

from opendispatch.api.schema import ContentUnit

from . import bluesky, discord, facebook, instagram, linkedin, telegram, threads, tiktok, twitter, youtube


class Adapter(Protocol):
    def publish(self, unit: ContentUnit, account: str | None = None) -> tuple[bool, str, str]: ...


ADAPTERS: dict[str, Adapter] = {
    "telegram": telegram,
    "twitter": twitter,
    "instagram": instagram,
    "bluesky": bluesky,
    "linkedin": linkedin,
    "threads": threads,
    "youtube": youtube,
    "tiktok": tiktok,
    "facebook": facebook,
    "discord": discord,
}
