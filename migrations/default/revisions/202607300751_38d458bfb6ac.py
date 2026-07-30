from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy import column

from lib.database import GUID, Timestamp

SCHEMA = 'public'

revision: str = '38d458bfb6ac'
down_revision: str | None = 'd125346e2b9c'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_KINDS = ('audio', 'image', 'summary', 'text', 'tool_call', 'tool_result')
_KINDS_BEFORE = ('summary', 'text', 'tool_call', 'tool_result')


def upgrade() -> None:
    op.create_table(
        'blobs',
        sa.Column('id', GUID(), nullable=False),
        sa.Column('sha256', sa.Text(), nullable=False),
        sa.Column('mime', sa.Text(), nullable=False),
        sa.Column('size', sa.Integer(), nullable=False),
        sa.Column('data', sa.LargeBinary(), nullable=False),
        sa.Column(
            'created_at',
            Timestamp(),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint('id', name='pk_blobs'),
        sa.UniqueConstraint('sha256', name='uq_blobs_sha256'),
        schema=SCHEMA,
    )
    op.drop_constraint('ck_parts_kind_known', 'parts', schema=SCHEMA, type_='check')
    op.create_check_constraint(
        'ck_parts_kind_known',
        'parts',
        column('kind').in_(_KINDS),
        schema=SCHEMA,
    )


def downgrade() -> None:
    op.drop_constraint('ck_parts_kind_known', 'parts', schema=SCHEMA, type_='check')
    op.create_check_constraint(
        'ck_parts_kind_known',
        'parts',
        column('kind').in_(_KINDS_BEFORE),
        schema=SCHEMA,
    )
    op.drop_table('blobs', schema=SCHEMA)
