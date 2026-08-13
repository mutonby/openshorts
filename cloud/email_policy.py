"""Email hygiene for sign-up: block disposable domains, normalize aliases,
verify the domain can actually receive mail.

Three abuse vectors this closes, now that the free plan is open to email (not
just Google) accounts:
  * temp-mail domains → free-minute farms (blocklist below, plus MX-target
    matching for the rotating front domains the blocklist can't keep up with).
  * provider aliases (gmail dots / +tags) → one address becomes infinite
    accounts → normalized to a single canonical form.
  * nonexistent addresses/domains → every magic link bounces back into the
    support inbox and burns sender reputation with PrivateEmail.
"""
import os
import time

_DOMAINS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "disposable_domains.txt")

# Providers where a "+tag" suffix and (for Google) dots are ignored by the
# mail server, so foo+1@ and f.oo@ all deliver to the same inbox.
_DOT_INSENSITIVE = {"gmail.com", "googlemail.com"}
_PLUS_ALIASING = _DOT_INSENSITIVE | {
    "outlook.com", "hotmail.com", "live.com", "icloud.com", "me.com",
    "fastmail.com", "protonmail.com", "proton.me", "yahoo.com",
}


def _load_list(path: str) -> set:
    domains = set()
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip().lower()
                if line and not line.startswith("#"):
                    domains.add(line)
    except FileNotFoundError:
        pass
    return domains


_DISPOSABLE = _load_list(_DOMAINS_FILE)


def is_disposable(email: str) -> bool:
    """True if the address's domain is a known disposable/temp-mail provider."""
    domain = (email or "").rsplit("@", 1)[-1].strip().lower()
    return domain in _DISPOSABLE


def normalize_email(email: str) -> str:
    """Canonical form used as the account key.

    Lowercases; strips a +tag from providers that ignore it; strips dots from
    the Gmail local part. Non-aliasing providers keep their local part intact
    so we never merge two genuinely different addresses.
    """
    email = (email or "").strip().lower()
    if "@" not in email:
        return email
    local, domain = email.rsplit("@", 1)
    if domain in _PLUS_ALIASING and "+" in local:
        local = local.split("+", 1)[0]
    if domain in _DOT_INSENSITIVE:
        local = local.replace(".", "")
    return f"{local}@{domain}"


def disposable_count() -> int:
    return len(_DISPOSABLE)


# --------------------------------------------------------------------------- #
# MX verification
# --------------------------------------------------------------------------- #
# Disposable providers rotate their front domains faster than any blocklist
# (onldm.net was one of 10minutemail's), but every front domain points its MX
# at the provider's real SMTP hosts, which stay put. Matching the MX target
# catches the whole rotation at once. Suffix match against the domains in
# disposable_mx.txt — same ops workflow as the sign-up blocklist above.
_MX_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "disposable_mx.txt")
_DISPOSABLE_MX = _load_list(_MX_FILE)

MX_OK = "ok"
MX_NONE = "no_mx"
MX_DISPOSABLE = "disposable_mx"

_MX_TIMEOUT = 4.0        # seconds; a slow resolver must not stall sign-in
_MX_CACHE_TTL = 3600.0   # verdicts computed from an actual DNS answer
_MX_ERROR_TTL = 60.0     # error-derived verdicts: a blip must not stick for 1h
_MX_CACHE_MAX = 10_000   # bots probing random domains must not grow it unbounded
_mx_cache = {}           # domain -> (verdict, monotonic expiry)
_resolver = None


def classify_mx_hosts(hosts) -> str:
    """Verdict from a domain's MX target hostnames."""
    hosts = [str(h).rstrip(".").lower() for h in hosts]
    # RFC 7505 null MX (a single "." target): the domain declares it takes no mail.
    if not any(hosts):
        return MX_NONE
    for host in hosts:
        if any(host == d or host.endswith("." + d) for d in _DISPOSABLE_MX):
            return MX_DISPOSABLE
    return MX_OK


def _get_resolver():
    # One shared resolver: reads resolv.conf once, and its LRU cache honors
    # each record's real TTL, so a junk-domain flood can't force live lookups
    # for the hot legit domains (gmail, outlook, …).
    global _resolver
    import dns.asyncresolver
    import dns.resolver

    if _resolver is None:
        r = dns.asyncresolver.Resolver()
        r.lifetime = _MX_TIMEOUT
        r.cache = dns.resolver.LRUCache()
        _resolver = r
    return _resolver


async def _resolve(domain: str, rtype: str):
    return await _get_resolver().resolve(domain, rtype)


async def _implicit_mx(domain: str):
    """RFC 5321 implicit MX: no MX record → deliverable if A or AAAA exists."""
    import dns.resolver

    for rtype in ("A", "AAAA"):
        try:
            await _resolve(domain, rtype)
            return MX_OK, _MX_CACHE_TTL
        except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer):
            continue
        except Exception:
            return MX_OK, _MX_ERROR_TTL
    return MX_NONE, _MX_ERROR_TTL


async def mx_verdict(email: str) -> str:
    """MX_OK / MX_NONE / MX_DISPOSABLE for the address's domain.

    Fails open (MX_OK) on resolver trouble — a DNS blip must never lock real
    users out of sign-in — and error-derived verdicts are cached only briefly
    so a transient NXDOMAIN can't pin a lockout for an hour.
    """
    import dns.resolver

    domain = (email or "").rsplit("@", 1)[-1].strip().lower()
    if not domain:
        return MX_NONE
    now = time.monotonic()
    hit = _mx_cache.get(domain)
    if hit and hit[1] > now:
        return hit[0]

    ttl = _MX_CACHE_TTL
    try:
        answers = await _resolve(domain, "MX")
        verdict = classify_mx_hosts(r.exchange for r in answers)
    except dns.resolver.NXDOMAIN:
        verdict, ttl = MX_NONE, _MX_ERROR_TTL
    except dns.resolver.NoAnswer:
        verdict, ttl = await _implicit_mx(domain)
    except Exception:
        verdict, ttl = MX_OK, _MX_ERROR_TTL

    # FIFO-evict one entry when full (dicts keep insertion order) — a flood of
    # junk domains must not wipe the whole cache at once.
    while len(_mx_cache) >= _MX_CACHE_MAX:
        _mx_cache.pop(next(iter(_mx_cache)))
    _mx_cache[domain] = (verdict, now + ttl)
    return verdict
