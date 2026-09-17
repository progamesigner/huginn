from collections.abc import Sequence
from uuid import UUID

import sqlalchemy as sa
from alembic import op

SCHEMA = 'public'

revision: str = '9bfc8f1d6d51'
down_revision: str | None = '38d458bfb6ac'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_UNIQUE = 'uq_conversations_channel'

_conversations = sa.table(
    'conversations',
    sa.column('id'),
    sa.column('channel'),
    sa.column('chat_id'),
    sa.column('agent'),
    sa.column('created_at'),
    sa.column('compact_version'),
    sa.column('summarized_through_turn_id'),
    schema=SCHEMA,
)
_turns = sa.table(
    'turns',
    sa.column('id'),
    sa.column('conversation_id'),
    sa.column('agent'),
    schema=SCHEMA,
)
_parts = sa.table('parts', sa.column('turn_id'), sa.column('kind'), schema=SCHEMA)


def upgrade() -> None:
    op.add_column(
        'turns',
        sa.Column('agent', sa.Text(), nullable=True),
        schema=SCHEMA,
    )
    op.execute(
        _turns.update()
        .values(agent=_conversations.c.agent)
        .where(_conversations.c.id == _turns.c.conversation_id),
    )

    ranked = sa.select(
        _conversations.c.id,
        sa.func.first_value(_conversations.c.id)
        .over(
            partition_by=(_conversations.c.channel, _conversations.c.chat_id),
            order_by=(_conversations.c.created_at, _conversations.c.id),
        )
        .label('survivor'),
    ).subquery()
    # NOTE: read the merges before touching anything; once the duplicates are gone
    # the window can no longer tell which survivor absorbed them
    rows = op.get_bind().execute(
        sa.select(ranked).where(ranked.c.id != ranked.c.survivor),
    )
    merges: list[tuple[UUID, UUID]] = [(row.id, row.survivor) for row in rows]
    if merges:
        _merge(merges)

    op.alter_column(
        'turns',
        'agent',
        existing_type=sa.Text(),
        nullable=False,
        schema=SCHEMA,
    )
    op.drop_constraint(_UNIQUE, 'conversations', schema=SCHEMA, type_='unique')
    op.create_unique_constraint(
        _UNIQUE,
        'conversations',
        ['channel', 'chat_id'],
        schema=SCHEMA,
    )
    op.drop_column('conversations', 'agent', schema=SCHEMA)


def _merge(merges: Sequence[tuple[UUID, UUID]]) -> None:
    for merged, survivor in merges:
        op.execute(
            _turns.update()
            .where(_turns.c.conversation_id == merged)
            .values(conversation_id=survivor),
        )
    op.execute(
        _conversations.delete().where(
            _conversations.c.id.in_([merged for merged, _ in merges]),
        ),
    )

    # NOTE: each agent's summary only covered its own turns, so the merged history
    # starts unsummarised and the next compaction folds it as one
    survivors = sorted({survivor for _, survivor in merges})
    summaries = sa.select(_parts.c.turn_id).where(_parts.c.kind == 'summary')
    op.execute(
        _turns.delete().where(
            _turns.c.conversation_id.in_(survivors),
            _turns.c.id.in_(summaries),
        ),
    )
    op.execute(
        _conversations.update()
        .where(_conversations.c.id.in_(survivors))
        .values(
            summarized_through_turn_id=None,
            compact_version=_conversations.c.compact_version + 1,
        ),
    )


def downgrade() -> None:
    # NOTE: this restores the old schema, but cannot split the per-agent conversations
    # that upgrade merged once the database holds data
    op.add_column(
        'conversations',
        sa.Column('agent', sa.Text(), nullable=True),
        schema=SCHEMA,
    )
    latest = (
        sa.select(_turns.c.agent)
        .where(_turns.c.conversation_id == _conversations.c.id)
        .order_by(_turns.c.id.desc())
        .limit(1)
        .scalar_subquery()
    )
    op.execute(_conversations.update().values(agent=sa.func.coalesce(latest, '')))
    op.alter_column(
        'conversations',
        'agent',
        existing_type=sa.Text(),
        nullable=False,
        schema=SCHEMA,
    )
    op.drop_constraint(_UNIQUE, 'conversations', schema=SCHEMA, type_='unique')
    op.create_unique_constraint(
        _UNIQUE,
        'conversations',
        ['channel', 'chat_id', 'agent'],
        schema=SCHEMA,
    )
    op.drop_column('turns', 'agent', schema=SCHEMA)
