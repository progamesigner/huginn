import sqlalchemy as sa
from alembic import op

from lib.database import GUID

SCHEMA = 'public'

revision = 'e3463639f380'
down_revision = '18d49d860b10'
branch_labels = None
depends_on = None

_FOREIGN_KEY = 'fk_conversations_reset_through_turn_id_turns'


def upgrade() -> None:
    op.add_column(
        'conversations',
        sa.Column('reset_through_turn_id', GUID(), nullable=True),
        schema=SCHEMA,
    )
    op.create_foreign_key(
        _FOREIGN_KEY,
        'conversations',
        'turns',
        ['reset_through_turn_id'],
        ['id'],
        source_schema=SCHEMA,
        referent_schema=SCHEMA,
        ondelete='SET NULL',
    )


def downgrade() -> None:
    op.drop_constraint(_FOREIGN_KEY, 'conversations', schema=SCHEMA, type_='foreignkey')
    op.drop_column('conversations', 'reset_through_turn_id', schema=SCHEMA)
