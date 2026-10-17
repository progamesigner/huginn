from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = '61a0cb94de32'
down_revision: str | None = '5c2e8a1f7b34'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_HANDLE = "handle = lower(btrim(handle)) AND handle ~ '^[a-z0-9._-]{1,64}$' AND handle NOT IN ('.', '..')"


def upgrade() -> None:
    # Refuse incompatible existing handles instead of silently changing identities
    # or renaming their vaults. This precedes every schema change.
    op.execute(
        sa.text("""
        DO $$ BEGIN
            IF EXISTS (SELECT 1 FROM public.users WHERE NOT (handle = lower(btrim(handle)) AND handle ~ '^[a-z0-9._-]{1,64}$' AND handle NOT IN ('.', '..'))) THEN
                RAISE EXCEPTION 'Invalid existing handles; review users before upgrading';
            END IF;
        END $$;
    """),
    )
    op.drop_constraint(
        'ck_users_handle_normalized',
        'users',
        type_='check',
        schema='public',
    )
    op.create_check_constraint(
        'ck_users_handle_normalized',
        'users',
        _HANDLE,
        schema='public',
    )
    op.add_column(
        'users',
        sa.Column('memory_scope', sa.Text(), nullable=True),
        schema='public',
    )
    op.execute('UPDATE public.users SET memory_scope = handle')
    op.alter_column('users', 'memory_scope', nullable=False, schema='public')
    op.create_unique_constraint(
        'uq_users_memory_scope',
        'users',
        ['memory_scope'],
        schema='public',
    )
    op.alter_column('identity_link_invites', 'user_id', nullable=True, schema='public')
    for column in (
        sa.Column('kind', sa.Text(), nullable=False, server_default='link'),
        sa.Column('memory_scope', sa.Text(), nullable=True),
        sa.Column('reserved_handle', sa.Text(), nullable=True),
        sa.Column('pending_external_user_id', sa.Text(), nullable=True),
    ):
        op.add_column('identity_link_invites', column, schema='public')


def downgrade() -> None:
    op.execute(
        sa.text("""
        DO $$ BEGIN
            IF EXISTS (SELECT 1 FROM public.users WHERE memory_scope <> handle) THEN
                RAISE EXCEPTION 'Downgrade would discard an approved memory binding';
            END IF;
        END $$;
    """),
    )
    # Pending registration invites have no account and cannot exist in the old schema.
    op.execute('DELETE FROM public.identity_link_invites WHERE user_id IS NULL')
    for name in ('pending_external_user_id', 'reserved_handle', 'memory_scope', 'kind'):
        op.drop_column('identity_link_invites', name, schema='public')
    op.alter_column('identity_link_invites', 'user_id', nullable=False, schema='public')
    op.drop_constraint(
        'uq_users_memory_scope',
        'users',
        type_='unique',
        schema='public',
    )
    op.drop_column('users', 'memory_scope', schema='public')
    op.drop_constraint(
        'ck_users_handle_normalized',
        'users',
        type_='check',
        schema='public',
    )
    op.create_check_constraint(
        'ck_users_handle_normalized',
        'users',
        'handle = lower(btrim(handle)) AND length(handle) > 0',
        schema='public',
    )
