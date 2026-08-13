"""Disposable-domain blocking, email alias normalization and MX verification."""
import asyncio

import pytest

from cloud import email_policy as ep


class TestDisposable:
    def test_known_temp_mail_blocked(self):
        assert ep.is_disposable("x@mailinator.com") is True
        assert ep.is_disposable("y@10minutemail.com") is True

    def test_db_observed_domains_blocked(self):
        assert ep.is_disposable("a@luckfeed.com") is True
        assert ep.is_disposable("b@web-library.net") is True
        assert ep.is_disposable("c@onldm.net") is True

    def test_real_providers_allowed(self):
        for e in ("real@gmail.com", "u@outlook.com", "u@icloud.com", "u@yandex.ru"):
            assert ep.is_disposable(e) is False

    def test_case_insensitive(self):
        assert ep.is_disposable("X@MailInator.CoM") is True

    def test_list_loaded(self):
        assert ep.disposable_count() > 50


class TestNormalize:
    def test_gmail_dots_and_tags_stripped(self):
        assert ep.normalize_email("Foo.Bar+promo@gmail.com") == "foobar@gmail.com"
        assert ep.normalize_email("a.b.c@googlemail.com") == "abc@googlemail.com"

    def test_other_providers_strip_tag_keep_dots(self):
        assert ep.normalize_email("user+tag@outlook.com") == "user@outlook.com"
        # Dots are significant outside Gmail — must be preserved.
        assert ep.normalize_email("first.last@outlook.com") == "first.last@outlook.com"

    def test_non_aliasing_provider_untouched(self):
        assert ep.normalize_email("normal@yahoo.com") == "normal@yahoo.com"
        assert ep.normalize_email("a.b@protonmail.com") == "a.b@protonmail.com"

    def test_lowercased(self):
        assert ep.normalize_email("USER@Example.COM") == "user@example.com"


class TestClassifyMxHosts:
    def test_disposable_infrastructure_caught(self):
        # onldm.net-style rotating front domain: MX points at the real provider.
        assert ep.classify_mx_hosts(["prd-smtp.10minutemail.com."]) == ep.MX_DISPOSABLE
        assert ep.classify_mx_hosts(["mail.mailinator.com"]) == ep.MX_DISPOSABLE

    def test_null_mx_means_no_mail(self):
        assert ep.classify_mx_hosts(["."]) == ep.MX_NONE

    def test_real_provider_ok(self):
        assert ep.classify_mx_hosts(["aspmx.l.google.com."]) == ep.MX_OK
        assert ep.classify_mx_hosts(["mx1.privateemail.com", "mx2.privateemail.com"]) == ep.MX_OK

    def test_suffix_match_not_substring(self):
        assert ep.classify_mx_hosts(["not10minutemail.com"]) == ep.MX_OK

    def test_any_disposable_host_taints(self):
        hosts = ["aspmx.l.google.com.", "backup.guerrillamail.com."]
        assert ep.classify_mx_hosts(hosts) == ep.MX_DISPOSABLE


class TestMxVerdict:
    """The resolver is monkeypatched: no network in tests."""

    def setup_method(self):
        ep._mx_cache.clear()

    def test_fail_open_on_resolver_trouble(self, monkeypatch):
        pytest.importorskip("dns")

        async def boom(domain, rtype):
            raise RuntimeError("resolver down")

        monkeypatch.setattr(ep, "_resolve", boom)
        assert asyncio.run(ep.mx_verdict("u@example.com")) == ep.MX_OK

    def test_nxdomain_is_no_mx(self, monkeypatch):
        dns = pytest.importorskip("dns")
        import dns.resolver

        async def nx(domain, rtype):
            raise dns.resolver.NXDOMAIN()

        monkeypatch.setattr(ep, "_resolve", nx)
        assert asyncio.run(ep.mx_verdict("u@no-such-domain.example")) == ep.MX_NONE

    def test_no_mx_falls_back_to_a_record(self, monkeypatch):
        dns = pytest.importorskip("dns")
        import dns.resolver

        async def no_mx_but_a(domain, rtype):
            if rtype == "MX":
                raise dns.resolver.NoAnswer()
            return ["1.2.3.4"]

        monkeypatch.setattr(ep, "_resolve", no_mx_but_a)
        assert asyncio.run(ep.mx_verdict("u@a-record-only.example")) == ep.MX_OK

    def test_no_mx_falls_back_to_aaaa_record(self, monkeypatch):
        dns = pytest.importorskip("dns")
        import dns.resolver

        async def ipv6_only(domain, rtype):
            if rtype == "AAAA":
                return ["::1"]
            raise dns.resolver.NoAnswer()

        monkeypatch.setattr(ep, "_resolve", ipv6_only)
        assert asyncio.run(ep.mx_verdict("u@ipv6-only.example")) == ep.MX_OK

    def test_error_verdicts_cached_briefly(self, monkeypatch):
        dns = pytest.importorskip("dns")
        import dns.resolver

        async def nx(domain, rtype):
            raise dns.resolver.NXDOMAIN()

        monkeypatch.setattr(ep, "_resolve", nx)
        assert asyncio.run(ep.mx_verdict("u@transient-blip.example")) == ep.MX_NONE
        import time
        _, expiry = ep._mx_cache["transient-blip.example"]
        # A transient NXDOMAIN must not pin the lockout for the full hour.
        assert expiry - time.monotonic() <= ep._MX_ERROR_TTL + 1

    def test_mx_blocklist_loaded_from_file(self):
        assert "10minutemail.com" in ep._DISPOSABLE_MX
        assert len(ep._DISPOSABLE_MX) >= 10

    def test_verdict_is_cached_per_domain(self, monkeypatch):
        pytest.importorskip("dns")
        calls = []

        async def counting(domain, rtype):
            calls.append(domain)

            class R:
                exchange = "prd-smtp.10minutemail.com."

            return [R()]

        monkeypatch.setattr(ep, "_resolve", counting)
        assert asyncio.run(ep.mx_verdict("a@onldm.net")) == ep.MX_DISPOSABLE
        assert asyncio.run(ep.mx_verdict("b@onldm.net")) == ep.MX_DISPOSABLE
        assert calls == ["onldm.net"]

    def test_empty_email_is_no_mx(self):
        pytest.importorskip("dns")
        assert asyncio.run(ep.mx_verdict("")) == ep.MX_NONE
