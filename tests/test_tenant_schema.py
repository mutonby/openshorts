"""Phase 1 schema guard: every media/worker table has a tenant boundary."""
from pathlib import Path

from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable

from cloud.config import DEFAULT_TENANT_ID, normalize_database_url
from cloud.database import Base
from cloud.models import (
    Tenant, TenantClip, TenantCreditLedger, TenantJob, TenantMember, UploadPostProfile,
    UserVideo, Project,
)


def _fk_targets(table):
    return {fk.target_fullname for fk in table.foreign_keys}


def test_tenant_tables_have_requested_white_label_fields():
    tenants = Base.metadata.tables["tenants"]
    for column in ("id", "owner_firebase_uid", "brand_name", "custom_domain", "logo_url",
                   "primary_color", "accent_color", "credit_balance", "created_at"):
        assert column in tenants.c
    assert tenants.c.credit_balance.nullable is False

    members = Base.metadata.tables["tenant_members"]
    for column in ("tenant_id", "firebase_uid", "role"):
        assert column in members.c
    assert "tenants.id" in _fk_targets(members)


def test_existing_media_tables_are_mandatorily_scoped():
    for model in (UserVideo, Project, UploadPostProfile):
        table = Base.metadata.tables[model.__tablename__]
        column = table.c.tenant_id
        assert column.nullable is False
        assert "tenants.id" in _fk_targets(table)


def test_worker_job_and_clip_tables_are_tenant_scoped():
    for model in (TenantJob, TenantClip):
        table = Base.metadata.tables[model.__tablename__]
        assert table.c.tenant_id.nullable is False
        assert "tenants.id" in _fk_targets(table)
    assert "jobs.job_id" in _fk_targets(Base.metadata.tables["clips"])


def test_credit_ledger_tracks_tenant_minutes():
    table = Base.metadata.tables[TenantCreditLedger.__tablename__]
    for column in ("tenant_id", "job_id", "minutes", "status", "balance_after"):
        assert column in table.c
    assert "tenants.id" in _fk_targets(table)


def test_tenant_uuid_defaults_are_quoted_sql_literals():
    dialect = postgresql.dialect()
    for table in ("tenants", "user_videos", "jobs"):
        ddl = str(CreateTable(Base.metadata.tables[table]).compile(dialect=dialect))
        assert f"DEFAULT '{DEFAULT_TENANT_ID}'" in ddl


def test_neon_urls_get_tls_without_rewriting_other_urls():
    url = "postgresql+asyncpg://user:pw@ep-cool.neon.tech/openshorts"
    assert "ssl=require" in normalize_database_url(url)
    assert normalize_database_url("postgresql+asyncpg://u:p@localhost/db") == \
        "postgresql+asyncpg://u:p@localhost/db"
    explicit = url + "?sslmode=verify-full"
    assert normalize_database_url(explicit) == explicit


def test_default_tenant_id_and_migration_present():
    assert DEFAULT_TENANT_ID == "00000000-0000-0000-0000-000000000001"
    migration = Path(__file__).parents[1] / "alembic" / "versions" / \
        "0001_whitelabel_neon.py"
    text = migration.read_text(encoding="utf-8")
    for fragment in ("create_table(", "tenant_members", "tenant_credit_ledger",
                     '"jobs"', '"clips"', "ondelete=\"RESTRICT\""):
        assert fragment in text
