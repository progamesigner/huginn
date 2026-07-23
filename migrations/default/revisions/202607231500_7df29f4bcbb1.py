from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from lib.database import GUID

SCHEMA = 'public'

revision: str = '7df29f4bcbb1'
down_revision: str | None = '6b4de18291a7'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        'conversations',
        sa.Column(
            'compact_version',
            sa.Integer(),
            server_default=sa.text('0'),
            nullable=False,
        ),
        schema=SCHEMA,
    )
    op.add_column(
        'conversations',
        sa.Column('summarized_through_turn_id', GUID(), nullable=True),
        schema=SCHEMA,
    )
    op.create_foreign_key(
        'fk_conversations_summarized_through_turn_id_turns',
        'conversations',
        'turns',
        ['summarized_through_turn_id'],
        ['id'],
        source_schema=SCHEMA,
        referent_schema=SCHEMA,
        ondelete='SET NULL',
    )


def downgrade() -> None:
    op.drop_constraint(
        'fk_conversations_summarized_through_turn_id_turns',
        'conversations',
        schema=SCHEMA,
        type_='foreignkey',
    )
    op.drop_column('conversations', 'summarized_through_turn_id', schema=SCHEMA)
    op.drop_column('conversations', 'compact_version', schema=SCHEMA)
