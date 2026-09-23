"""session_provenance: designator, correction_mode, cv_backend, hashes, detections, status

Revision ID: 002
Revises: 001
Create Date: 2026-09-23
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "diagnostic_sessions",
        sa.Column("status", sa.String(), nullable=False, server_default="completed"),
    )
    op.add_column("diagnostic_sessions", sa.Column("error", sa.Text(), nullable=True))
    op.add_column("diagnostic_sessions", sa.Column("designator", sa.String(), nullable=True))
    op.add_column("diagnostic_sessions", sa.Column("correction_mode", sa.String(), nullable=True))
    op.add_column("diagnostic_sessions", sa.Column("cv_backend", sa.String(), nullable=True))
    op.add_column("diagnostic_sessions", sa.Column("image_sha256", sa.String(64), nullable=True))
    op.add_column("diagnostic_sessions", sa.Column("template_sha256", sa.String(64), nullable=True))
    op.add_column(
        "diagnostic_sessions",
        sa.Column("detections", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.add_column(
        "diagnostic_sessions",
        sa.Column("rag_documents_used", sa.Integer(), server_default="0"),
    )
    op.create_index(
        "ix_diagnostic_sessions_image_sha256", "diagnostic_sessions", ["image_sha256"]
    )


def downgrade() -> None:
    op.drop_index("ix_diagnostic_sessions_image_sha256", table_name="diagnostic_sessions")
    for col in (
        "rag_documents_used",
        "detections",
        "template_sha256",
        "image_sha256",
        "cv_backend",
        "correction_mode",
        "designator",
        "error",
        "status",
    ):
        op.drop_column("diagnostic_sessions", col)
