"""The API lifts its soft open-file limit at startup so a 1024 default doesn't
make uvicorn refuse connections with EMFILE under load."""
import resource

import pytest

import fdlimit


class _FakeResource:
    RLIMIT_NOFILE = resource.RLIMIT_NOFILE
    RLIM_INFINITY = resource.RLIM_INFINITY

    def __init__(self, soft, hard, fail=False):
        self.limits = (soft, hard)
        self.fail = fail
        self.calls = []

    def getrlimit(self, which):
        assert which == self.RLIMIT_NOFILE
        return self.limits

    def setrlimit(self, which, limits):
        self.calls.append(limits)
        if self.fail:
            raise ValueError("not allowed")
        self.limits = limits


@pytest.fixture
def fake(monkeypatch):
    def make(soft, hard, fail=False):
        r = _FakeResource(soft, hard, fail)
        monkeypatch.setattr(fdlimit, "resource", r)
        return r
    return make


def test_raises_soft_limit_up_to_hard(fake):
    r = fake(1024, 4096)
    assert fdlimit.raise_nofile_limit() == (4096, 4096)
    assert r.calls == [(4096, 4096)]


def test_caps_a_huge_hard_limit(fake):
    r = fake(1024, 1048576)
    assert fdlimit.raise_nofile_limit() == (fdlimit.NOFILE_TARGET, 1048576)
    assert r.calls == [(fdlimit.NOFILE_TARGET, 1048576)]


def test_caps_an_unlimited_hard_limit(fake):
    r = fake(1024, resource.RLIM_INFINITY)
    assert fdlimit.raise_nofile_limit()[0] == fdlimit.NOFILE_TARGET
    assert r.calls == [(fdlimit.NOFILE_TARGET, resource.RLIM_INFINITY)]


def test_never_lowers_a_higher_soft_limit(fake):
    r = fake(200000, 1048576)
    assert fdlimit.raise_nofile_limit() == (200000, 1048576)
    assert r.calls == []


def test_refused_setrlimit_does_not_raise(fake):
    fake(1024, 4096, fail=True)
    assert fdlimit.raise_nofile_limit() == (1024, 4096)


def test_no_rlimits_on_platform(monkeypatch):
    monkeypatch.setattr(fdlimit, "resource", None)
    assert fdlimit.raise_nofile_limit() is None


def test_real_process_limit_is_lifted():
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(256, hard), hard))
        new_soft, new_hard = fdlimit.raise_nofile_limit()
        assert new_hard == hard
        assert new_soft == resource.getrlimit(resource.RLIMIT_NOFILE)[0]
        assert new_soft > 256 or hard <= 256
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))


def test_app_lifts_the_limit_at_import():
    src = open(fdlimit.__file__.replace("fdlimit.py", "app.py"), encoding="utf-8").read()
    assert "fdlimit.raise_nofile_limit()" in src
