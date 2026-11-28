"""White-label tenants, tenant-minute credits and Neon-ready schema.

Revision ID: 0001_whitelabel_neon
Revises: None

Neon does not accept non-TLS connections; cloud.config.normalize_database_url
adds ``ssl=require`` to ``*.neon.tech`` URLs before Alembic or the app opens
one. This revision adds the tenant boundary itself. Legacy rows are backfilled
into the default tenant so the mandatory FK can be enforced.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from cloud.config import DEFAULT_TENANT_BRAND, DEFAULT_TENANT_ID

revision = "0001_whitelabel_neon"
down_revision = None
branch_labels = None
depends_on = None

UUID = postgresql.UUID(as_uuid=False)
DEFAULT_TENANT_SQL = f"'{DEFAULT_TENANT_ID}'"
_SCOPED = ("user_videos", "projects", "upload_post_profiles")


def _inspector():
    return sa.inspect(op.get_bind())


def _has_table(name):
    return _inspector().has_table(name)


def _has_column(table, column):
    return any(c["name"] == column for c in _inspector().get_columns(table))


def _has_index(table, name):
    return any(i["name"] == name for i in _inspector().get_indexes(table))


def _has_fk(table, name):
    return any(f["name"] == name for f in _inspector().get_foreign_keys(table))


def _ensure_default_tenant():
    op.execute(sa.text(
        "INSERT INTO tenants (id, brand_name, credit_balance) "
        "VALUES (:id, :brand, 0) ON CONFLICT (id) DO NOTHING"),
        {"id": DEFAULT_TENANT_ID, "brand": DEFAULT_TENANT_BRAND})


def upgrade():
    # Firebase identity: Phase 2 verifies ID tokens and stores the same UID here.
    if _has_table("users"):
        if not _has_column("users", "firebase_uid"):
            op.add_column("users", sa.Column("firebase_uid", sa.Text(), nullable=True))
        if not _has_index("users", "ux_users_firebase_uid"):
            op.create_index("ux_users_firebase_uid", "users", ["firebase_uid"], unique=True)

    if not _has_table("tenants"):
        op.create_table(
            "tenants",
            sa.Column("id", UUID, primary_key=True, server_default=sa.text(DEFAULT_TENANT_SQL)),
            sa.Column("owner_firebase_uid", sa.Text(), nullable=True),
            sa.Column("brand_name", sa.Text(), nullable=False,
                      server_default=sa.text(f"'{DEFAULT_TENANT_BRAND}'")),
            sa.Column("custom_domain", sa.Text(), nullable=True),
            sa.Column("logo_url", sa.Text(), nullable=True),
            sa.Column("primary_color", sa.String(16), nullable=True),
            sa.Column("accent_color", sa.String(16), nullable=True),
            sa.Column("credit_balance", sa.Numeric(12, 2), nullable=False,
                      server_default=sa.text("0")),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        )
        op.create_index("ix_tenants_custom_domain", "tenants", ["custom_domain"], unique=True)
        op.create_index("ix_tenants_owner_firebase_uid", "tenants", ["owner_firebase_uid"])
    _ensure_default_tenant()

    if not _has_table("tenant_members"):
        op.create_table(
            "tenant_members",
            sa.Column("tenant_id", UUID, sa.ForeignKey("tenants.id", ondelete="CASCADE"),
                      primary_key=True),
            sa.Column("firebase_uid", sa.Text(), primary_key=True),
            sa.Column("role", sa.String(16), nullable=False, server_default=sa.text("'viewer'")),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        )
        op.create_index("ix_tenant_members_firebase_uid", "tenant_members", ["firebase_uid"])

    if not _has_table("tenant_credit_ledger"):
        op.create_table(
            "tenant_credit_ledger",
            sa.Column("id", UUID, primary_key=True),
            sa.Column("tenant_id", UUID, sa.ForeignKey("tenants.id", ondelete="CASCADE"),
                      nullable=False),
            sa.Column("job_id", sa.Text(), nullable=False),
            sa.Column("minutes", sa.Numeric(10, 2), nullable=False),
            sa.Column("status", sa.String(12), nullable=False,
                      server_default=sa.text("'reserved'")),
            sa.Column("balance_after", sa.Numeric(12, 2), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        )
        op.create_index("ix_tenant_credit_job", "tenant_credit_ledger", ["job_id"])
        op.create_index("ix_tenant_credit_status", "tenant_credit_ledger",
                        ["tenant_id", "status"])

    if not _has_table("jobs"):
        op.create_table(
            "jobs",
            sa.Column("job_id", sa.Text(), primary_key=True),
            sa.Column("tenant_id", UUID, sa.ForeignKey("tenants.id", ondelete="RESTRICT"),
                      nullable=False, server_default=sa.text(DEFAULT_TENANT_SQL)),
            sa.Column("user_id", sa.Text(), nullable=True),
            sa.Column("status", sa.String(24), nullable=False,
                      server_default=sa.text("'pending'")),
            sa.Column("source_key", sa.Text(), nullable=True),
            sa.Column("source_duration_seconds", sa.Numeric(10, 2), nullable=True),
            sa.Column("requested_minutes", sa.Numeric(10, 2), nullable=True),
            sa.Column("reservation_id", sa.Text(), nullable=True),
            sa.Column("error", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
            sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        )
        op.create_index("ix_jobs_tenant_id", "jobs", ["tenant_id"])
        op.create_index("ix_jobs_tenant_status", "jobs", ["tenant_id", "status"])

    if not _has_table("clips"):
        op.create_table(
            "clips",
            sa.Column("id", UUID, primary_key=True),
            sa.Column("tenant_id", UUID, sa.ForeignKey("tenants.id", ondelete="RESTRICT"),
                      nullable=False, server_default=sa.text(DEFAULT_TENANT_SQL)),
            sa.Column("job_id", sa.Text(), sa.ForeignKey("jobs.job_id", ondelete="CASCADE"),
                      nullable=False),
            sa.Column("clip_index", sa.Integer(), nullable=False),
            sa.Column("r2_key", sa.Text(), nullable=False),
            sa.Column("title", sa.Text(), nullable=True),
            sa.Column("size_bytes", sa.BigInteger(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
            sa.UniqueConstraint("tenant_id", "job_id", "clip_index",
                                name="uq_clips_tenant_job_index"),
        )
        op.create_index("ix_clips_tenant_id", "clips", ["tenant_id"])
        op.create_index("ix_clips_job", "clips", ["job_id"])

    if _has_table("usage_ledger") and not _has_column("usage_ledger", "tenant_reservation_id"):
        op.add_column("usage_ledger", sa.Column("tenant_reservation_id", sa.Text(), nullable=True))

    for table in _SCOPED:
        if not _has_table(table):
            continue
        if not _has_column(table, "tenant_id"):
            op.add_column(table, sa.Column("tenant_id", UUID, nullable=True))
        # Backfill before the NOT NULL constraint goes on.
        op.execute(sa.text(
            f"UPDATE {table} SET tenant_id = :id WHERE tenant_id IS NULL"),
            {"id": DEFAULT_TENANT_ID})
        op.alter_column(table, "tenant_id", nullable=False,
                        existing_type=UUID, server_default=sa.text(DEFAULT_TENANT_SQL))
        fk_name = f"fk_{table}_tenant_id_tenants"
        if not _has_fk(table, fk_name):
            op.create_foreign_key(fk_name, table, "tenants", ["tenant_id"], ["id"],
                                  ondelete="RESTRICT")
        index_name = f"ix_{table}_tenant_id"
        if not _has_index(table, index_name):
            op.create_index(index_name, table, ["tenant_id"])


def downgrade():
    for table in _SCOPED:
        if not _has_table(table):
            continue
        index_name = f"ix_{table}_tenant_id"
        if _has_index(table, index_name):
            op.drop_index(index_name, table_name=table)
        fk_name = f"fk_{table}_tenant_id_tenants"
        if _has_fk(table, fk_name):
            op.drop_constraint(fk_name, table, type_="foreign_key")
        if _has_column(table, "tenant_id"):
            op.drop_column(table, "tenant_id")

    if _has_table("usage_ledger") and _has_column("usage_ledger", "tenant_reservation_id"):
        op.drop_column("usage_ledger", "tenant_reservation_id")
    for name in ("clips", "jobs", "tenant_credit_ledger", "tenant_members", "tenants"):
        if _has_table(name):
            op.drop_table(name)
    if _has_table("users"):
        if _has_index("users", "ux_users_firebase_uid"):
            op.drop_index("ux_users_firebase_uid", table_name="users")
        if _has_column("users", "firebase_uid"):
            op.drop_column("users", "firebase_uid")
