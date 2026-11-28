"""Tenant-minute accounting does not need a live Postgres to be checked."""
import asyncio
import inspect
import uuid
from decimal import Decimal
from types import SimpleNamespace

import pytest

from cloud import metering, tenants
from cloud.config import DEFAULT_TENANT_ID


class _Result:
    def __init__(self, value):
        self.value = value

    def scalar_one_or_none(self):
        return self.value


class _Session:
    """Only the slice of AsyncSession the tenant accounting helpers use."""

    def __init__(self, tenant=None, ledger=None):
        self.tenant = tenant
        self.ledger = ledger
        self.added = []

    async def execute(self, _statement):
        return _Result(self.tenant)

    async def get(self, _model, _key):
        return self.ledger

    def add(self, row):
        self.added.append(row)

    async def flush(self):
        return None


def _tenant(balance="10"):
    return SimpleNamespace(id=uuid.UUID(DEFAULT_TENANT_ID),
                           credit_balance=Decimal(balance))


def test_reserve_deducts_and_writes_a_reserved_ledger_row():
    tenant = _tenant()
    session = _Session(tenant=tenant)
    ledger_id = asyncio.run(tenants.reserve_tenant_minutes(
        session, tenant.id, 4, "job-1"))

    assert tenant.credit_balance == Decimal("6")
    assert ledger_id
    row = session.added[0]
    assert str(row.tenant_id) == DEFAULT_TENANT_ID
    assert row.job_id == "job-1"
    assert row.minutes == Decimal("4")
    assert row.status == "reserved"
    assert row.balance_after == Decimal("6")


def test_reserve_refuses_an_overdraft_without_deducting():
    tenant = _tenant("2")
    session = _Session(tenant=tenant)
    with pytest.raises(tenants.TenantQuotaExceeded) as exc:
        asyncio.run(tenants.reserve_tenant_minutes(session, tenant.id, 3, "job-2"))

    assert exc.value.remaining == 2
    assert exc.value.required == 3
    assert tenant.credit_balance == Decimal("2")
    assert session.added == []


def test_release_refunds_the_exact_reservation():
    tenant = _tenant("6")
    row = SimpleNamespace(tenant_id=tenant.id, minutes=Decimal("4"),
                          status="reserved", balance_after=Decimal("6"))
    session = _Session(tenant=tenant, ledger=row)
    asyncio.run(tenants.settle_tenant_minutes(session, "ledger-1", "released"))

    assert row.status == "released"
    assert tenant.credit_balance == Decimal("10")
    assert row.balance_after == Decimal("10")


def test_trim_refunds_only_the_unused_minutes():
    tenant = _tenant("6")
    row = SimpleNamespace(tenant_id=tenant.id, minutes=Decimal("4"),
                          status="reserved", balance_after=Decimal("6"))
    session = _Session(tenant=tenant, ledger=row)
    refunded = asyncio.run(tenants.trim_tenant_minutes(session, "ledger-1", 1.5))

    assert refunded == 1.5
    assert tenant.credit_balance == Decimal("7.5")
    assert row.minutes == Decimal("2.5")
    assert row.status == "reserved"  # still live; only the source was shorter


def test_meter_reservation_accepts_a_tenant_scope():
    params = inspect.signature(metering.reserve_minutes).parameters
    assert "tenant_id" in params
    assert params["tenant_id"].default is None
