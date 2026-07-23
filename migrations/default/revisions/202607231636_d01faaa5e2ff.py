from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from lib.database import Timestamp

SCHEMA = 'public'

revision: str = 'd01faaa5e2ff'
down_revision: str | None = '7df29f4bcbb1'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_COLUMNS = (
    ('users', 'created_at'),
    ('users', 'updated_at'),
    ('conversations', 'created_at'),
    ('conversations', 'updated_at'),
)


def upgrade() -> None:
    for table, column in _COLUMNS:
        op.alter_column(
            table,
            column,
            existing_type=Timestamp(),
            existing_nullable=False,
            server_default=None,
            schema=SCHEMA,
        )


def downgrade() -> None:
    for table, column in _COLUMNS:
        op.alter_column(
            table,
            column,
            existing_type=Timestamp(),
            existing_nullable=False,
            server_default=sa.func.now(),
            schema=SCHEMA,
        )
