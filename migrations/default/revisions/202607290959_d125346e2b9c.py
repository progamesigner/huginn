from collections.abc import Sequence

from alembic import op
from sqlalchemy import column

SCHEMA = 'public'

revision: str = 'd125346e2b9c'
down_revision: str | None = 'd01faaa5e2ff'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_ROLES = ('assistant', 'system', 'tool', 'user')
_KINDS = ('summary', 'text', 'tool_call', 'tool_result')


def upgrade() -> None:
    op.create_check_constraint(
        'ck_parts_role_known',
        'parts',
        column('role').in_(_ROLES),
        schema=SCHEMA,
    )
    op.create_check_constraint(
        'ck_parts_kind_known',
        'parts',
        column('kind').in_(_KINDS),
        schema=SCHEMA,
    )


def downgrade() -> None:
    op.drop_constraint(
        'ck_parts_kind_known',
        'parts',
        schema=SCHEMA,
        type_='check',
    )
    op.drop_constraint(
        'ck_parts_role_known',
        'parts',
        schema=SCHEMA,
        type_='check',
    )
