from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from lib.database import GUID, Timestamp

SCHEMA = 'public'

revision: str = '192ab5159074'
down_revision: str | None = 'e3463639f380'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        'blob_grants',
        sa.Column('id', GUID(), nullable=False),
        sa.Column('blob_id', GUID(), nullable=False),
        sa.Column('owner', sa.Text(), nullable=False),
        sa.Column(
            'created_at',
            Timestamp(),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint('id', name='pk_blob_grants'),
        sa.ForeignKeyConstraint(
            ['blob_id'],
            [f'{SCHEMA}.blobs.id'],
            name='fk_blob_grants_blob_id_blobs',
            ondelete='CASCADE',
        ),
        sa.UniqueConstraint('blob_id', 'owner', name='uq_blob_grants_blob_id'),
        schema=SCHEMA,
    )


def downgrade() -> None:
    op.drop_table('blob_grants', schema=SCHEMA)
