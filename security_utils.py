"""Small, dependency-free SSRF guards shared by the download / scrape paths.

The app fetches user-supplied URLs server-side (yt-dlp downloads, SaaS landing
page scraping, actor-image download). Without a guard, a caller can point those
at internal services or the cloud metadata endpoint (169.254.169.254) to read
instance credentials. ``assert_public_url`` rejects non-HTTP(S) schemes and any
host that resolves to anything but a globally routable address (private,
loopback, link-local, reserved, and the 100.64.0.0/10 range Tailscale uses).
"""
from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse


def truncate_ip(ip: str) -> str:
    """Drop the host part of an address so it identifies a network, not a device.

    Used for the rights declaration kept with a project (art. 5.1.c: the record
    has to show the confirmation was made from somewhere at some time, which a
    /24 answers as well as a full address, and it does not hand anyone a
    per-device identifier that survives for years). IPv4 keeps three octets,
    IPv6 keeps the /48 the RIRs actually assign. Anything unparseable becomes
    "unknown" rather than being stored verbatim.
    """
    try:
        addr = ipaddress.ip_address((ip or "").strip())
    except ValueError:
        return "unknown"
    if addr.version == 4:
        return str(ipaddress.ip_network(f"{addr}/24", strict=False).network_address)
    return str(ipaddress.ip_network(f"{addr}/48", strict=False).network_address)


class UnsafeURLError(ValueError):
    """Raised when a URL is not safe to fetch server-side."""


def _ip_is_public(ip: str) -> bool:
    """True only for globally routable unicast addresses.

    ``is_global`` rather than a list of ``is_private``/``is_loopback``/...:
    that list missed 100.64.0.0/10 (carrier-grade NAT), which is exactly the
    range Tailscale hands out, so a job URL could reach every host on the
    tailnet (SSH on the other servers, the Coolify API) from the API
    container. An allowlist of "global" fails closed on any range nobody
    thought to name. IPv6 forms that carry an IPv4 address inside (mapped
    ``::ffff:a.b.c.d``, 6to4, Teredo, NAT64) are judged by that IPv4 address.
    """
    try:
        addr = ipaddress.ip_address(str(ip).split("%", 1)[0])
    except ValueError:
        return False
    if addr.version == 6:
        embedded = addr.ipv4_mapped or addr.sixtofour
        if embedded is None and addr.teredo:
            embedded = addr.teredo[1]
        if embedded is None and addr in _NAT64:
            embedded = ipaddress.IPv4Address(int(addr) & 0xFFFFFFFF)
        if embedded is not None and not _ip_is_public(str(embedded)):
            return False
    return bool(addr.is_global) and not addr.is_multicast


_NAT64 = ipaddress.ip_network("64:ff9b::/96")


def assert_public_url(url: str) -> str:
    """Return ``url`` if it is safe to fetch, else raise ``UnsafeURLError``.

    Blocks non-http(s) schemes and hosts that resolve to any non-public IP.
    Resolves every A/AAAA record so a hostname that maps to a private range
    (or to 169.254.169.254) is rejected rather than silently fetched.
    """
    if not url or not isinstance(url, str):
        raise UnsafeURLError("Empty URL")

    parsed = urlparse(url.strip())
    if parsed.scheme.lower() not in ("http", "https"):
        raise UnsafeURLError(f"Unsupported URL scheme: {parsed.scheme!r}")

    host = parsed.hostname
    if not host:
        raise UnsafeURLError("URL has no host")

    # If the host is a literal IP, validate it directly (no DNS).
    try:
        ipaddress.ip_address(host)
        is_ip_literal = True
    except ValueError:
        is_ip_literal = False

    if is_ip_literal:
        if not _ip_is_public(host):
            raise UnsafeURLError(f"URL host is not a public address: {host}")
        return url

    try:
        infos = socket.getaddrinfo(host, parsed.port or None, proto=socket.IPPROTO_TCP)
    except (socket.gaierror, UnicodeError) as e:
        # UnicodeError: the idna codec refuses the name before any lookup
        # (a label over 63 characters, an empty label). Same answer as a
        # name that does not resolve, not a 500.
        raise UnsafeURLError(f"Could not resolve host {host!r}: {e}")

    resolved = {info[4][0] for info in infos}
    if not resolved:
        raise UnsafeURLError(f"Host {host!r} did not resolve")
    for ip in resolved:
        if not _ip_is_public(ip):
            raise UnsafeURLError(f"Host {host!r} resolves to a non-public address: {ip}")
    return url
