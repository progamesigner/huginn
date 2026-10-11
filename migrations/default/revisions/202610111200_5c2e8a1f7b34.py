from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy import column

SCHEMA = 'public'

revision: str = '5c2e8a1f7b34'
down_revision: str | None = '192ab5159074'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        'blob_grants',
        sa.Column('filename', sa.Text(), nullable=True),
        schema=SCHEMA,
    )
    # NOTE: existing grants keep the name they were answered with until now
    blobs = sa.table('blobs', column('id'), column('filename'), schema=SCHEMA)
    grants = sa.table(
        'blob_grants',
        column('blob_id'),
        column('filename'),
        schema=SCHEMA,
    )
    op.execute(
        grants.update()
        .values(filename=blobs.c.filename)
        .where(blobs.c.id == grants.c.blob_id),
    )


def downgrade() -> None:
    op.drop_column('blob_grants', 'filename', schema=SCHEMA)
