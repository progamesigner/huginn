from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from lib.database import GUID, Json, Timestamp

SCHEMA = 'public'

revision: str = 'c2f96ac19f84'
down_revision: str | None = '0c7a74dd213c'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        'users',
        sa.Column('id', GUID(), nullable=False),
        sa.Column('handle', sa.Text(), nullable=False),
        sa.Column('display_name', sa.Text(), nullable=False),
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
        sa.CheckConstraint(
            'handle = lower(btrim(handle)) AND length(handle) > 0',
            name='ck_users_handle_normalized',
        ),
        sa.PrimaryKeyConstraint('id', name='pk_users'),
        sa.UniqueConstraint('handle', name='uq_users_handle'),
        schema=SCHEMA,
    )
    op.create_table(
        'external_identities',
        sa.Column('id', GUID(), nullable=False),
        sa.Column('user_id', GUID(), nullable=False),
        sa.Column('provider', sa.Text(), nullable=False),
        sa.Column('realm', sa.Text(), nullable=False),
        sa.Column('external_user_id', sa.Text(), nullable=False),
        sa.Column(
            'profile',
            Json(),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            'linked_at',
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
        sa.ForeignKeyConstraint(
            ['user_id'],
            [f'{SCHEMA}.users.id'],
            name='fk_external_identities_user_id_users',
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('id', name='pk_external_identities'),
        sa.UniqueConstraint(
            'provider',
            'realm',
            'external_user_id',
            name='uq_external_identities_provider',
        ),
        schema=SCHEMA,
    )
    op.create_index(
        'ix_external_identities_user_provider',
        'external_identities',
        ['user_id', 'provider'],
        schema=SCHEMA,
    )
    op.create_table(
        'identity_audit_events',
        sa.Column('id', GUID(), nullable=False),
        sa.Column('event_type', sa.Text(), nullable=False),
        sa.Column('user_id', GUID(), nullable=True),
        sa.Column('external_identity_id', GUID(), nullable=True),
        sa.Column('provider', sa.Text(), nullable=False),
        sa.Column('realm', sa.Text(), nullable=False),
        sa.Column(
            'occurred_at',
            Timestamp(),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            'metadata',
            Json(),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ['user_id'],
            [f'{SCHEMA}.users.id'],
            name='fk_identity_audit_events_user_id_users',
            ondelete='SET NULL',
        ),
        sa.PrimaryKeyConstraint('id', name='pk_identity_audit_events'),
        schema=SCHEMA,
    )
    op.create_index(
        'ix_identity_audit_events_user_occurred',
        'identity_audit_events',
        ['user_id', 'occurred_at'],
        schema=SCHEMA,
    )


def downgrade() -> None:
    op.drop_table('identity_audit_events', schema=SCHEMA)
    op.drop_table('external_identities', schema=SCHEMA)
    op.drop_table('users', schema=SCHEMA)
