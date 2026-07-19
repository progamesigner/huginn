from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from lib.database import GUID, Json, Timestamp

revision: str = '0c7a74dd213c'
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        'conversations',
        sa.Column('id', GUID(), nullable=False),
        sa.Column('channel', sa.Text(), nullable=False),
        sa.Column('chat_id', sa.Text(), nullable=False),
        sa.Column('agent', sa.Text(), nullable=False),
        sa.Column(
            'created_at',
            Timestamp(),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            'updated_at',
            Timestamp(),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint('id', name='pk_conversations'),
        sa.UniqueConstraint(
            'channel',
            'chat_id',
            'agent',
            name='uq_conversations_channel',
        ),
    )

    op.create_table(
        'turns',
        sa.Column('id', GUID(), nullable=False),
        sa.Column('conversation_id', GUID(), nullable=False),
        sa.Column(
            'usage',
            Json(),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            'created_at',
            Timestamp(),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint('id', name='pk_turns'),
        sa.ForeignKeyConstraint(
            ['conversation_id'],
            ['conversations.id'],
            name='fk_turns_conversation_id_conversations',
            ondelete='CASCADE',
        ),
    )
    op.create_index('ix_turns_conversation_id', 'turns', ['conversation_id', 'id'])

    op.create_table(
        'parts',
        sa.Column('id', GUID(), nullable=False),
        sa.Column('turn_id', GUID(), nullable=False),
        sa.Column('seq', sa.Integer(), nullable=False),
        sa.Column('role', sa.Text(), nullable=False),
        sa.Column('kind', sa.Text(), nullable=False),
        sa.Column('payload', Json(), nullable=False),
        sa.PrimaryKeyConstraint('id', name='pk_parts'),
        sa.ForeignKeyConstraint(
            ['turn_id'],
            ['turns.id'],
            name='fk_parts_turn_id_turns',
            ondelete='CASCADE',
        ),
    )
    op.create_index('ix_parts_turn_id', 'parts', ['turn_id', 'seq'])


def downgrade() -> None:
    op.drop_table('parts')
    op.drop_table('turns')
    op.drop_table('conversations')
