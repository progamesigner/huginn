from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy import column

from lib.database import Timestamp

SCHEMA = 'public'

revision: str = '18d49d860b10'
down_revision: str | None = '9bfc8f1d6d51'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_BLOB_KINDS = ('audio', 'document', 'image', 'video')
_PART_KINDS = (
    'audio',
    'document',
    'image',
    'summary',
    'text',
    'tool_call',
    'tool_result',
    'video',
)
_PART_KINDS_BEFORE = ('audio', 'image', 'summary', 'text', 'tool_call', 'tool_result')


def upgrade() -> None:
    op.add_column('blobs', sa.Column('kind', sa.Text(), nullable=True), schema=SCHEMA)
    op.add_column(
        'blobs',
        sa.Column('filename', sa.Text(), nullable=True),
        schema=SCHEMA,
    )
    op.add_column(
        'blobs',
        sa.Column('expires_at', Timestamp(), nullable=True),
        schema=SCHEMA,
    )
    blobs = sa.table('blobs', column('mime'), column('kind'), schema=SCHEMA)
    op.execute(
        blobs.update().values(
            kind=sa.case(
                (blobs.c.mime.like('audio/%'), 'audio'),
                else_='image',
            ),
        ),
    )
    op.alter_column(
        'blobs',
        'kind',
        existing_type=sa.Text(),
        nullable=False,
        schema=SCHEMA,
    )
    op.create_check_constraint(
        'ck_blobs_kind_known',
        'blobs',
        column('kind').in_(_BLOB_KINDS),
        schema=SCHEMA,
    )
    op.create_index('ix_blobs_expires_at', 'blobs', ['expires_at'], schema=SCHEMA)

    op.drop_constraint('ck_parts_kind_known', 'parts', schema=SCHEMA, type_='check')
    op.create_check_constraint(
        'ck_parts_kind_known',
        'parts',
        column('kind').in_(_PART_KINDS),
        schema=SCHEMA,
    )


def downgrade() -> None:
    op.drop_constraint('ck_parts_kind_known', 'parts', schema=SCHEMA, type_='check')
    op.create_check_constraint(
        'ck_parts_kind_known',
        'parts',
        column('kind').in_(_PART_KINDS_BEFORE),
        schema=SCHEMA,
    )
    op.drop_index('ix_blobs_expires_at', table_name='blobs', schema=SCHEMA)
    op.drop_constraint('ck_blobs_kind_known', 'blobs', schema=SCHEMA, type_='check')
    op.drop_column('blobs', 'expires_at', schema=SCHEMA)
    op.drop_column('blobs', 'filename', schema=SCHEMA)
    op.drop_column('blobs', 'kind', schema=SCHEMA)
