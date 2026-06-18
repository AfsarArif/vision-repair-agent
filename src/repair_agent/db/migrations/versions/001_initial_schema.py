"""initial_schema

Revision ID: 001
Revises:
Create Date: 2026-06-18
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create diagnostic_sessions table
    op.create_table(
        "diagnostic_sessions",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("defect_type", sa.String(), nullable=True),
        sa.Column("defect_confidence", sa.Float(), nullable=True),
        sa.Column("serial_number", sa.String(), nullable=True),
        sa.Column("diagnosis", sa.Text(), nullable=True),
        sa.Column("self_correction_triggered", sa.Boolean(), server_default="false"),
        sa.Column("correction_attempts", sa.Integer(), server_default="0"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create document_chunks table
    op.create_table(
        "document_chunks",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("source_file", sa.String(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("defect_type", sa.String(), nullable=True),
        sa.Column("ingested_at", sa.DateTime(), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("document_chunks")
    op.drop_table("diagnostic_sessions")
