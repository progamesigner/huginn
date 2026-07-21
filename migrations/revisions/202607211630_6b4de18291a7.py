from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from lib.database import GUID, Timestamp

revision: str = '6b4de18291a7'
down_revision: str | None = 'c2f96ac19f84'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        'identity_link_invites',
        sa.Column('id', GUID(), nullable=False),
        sa.Column('user_id', GUID(), nullable=False),
        sa.Column('provider', sa.Text(), nullable=False),
        sa.Column('realm', sa.Text(), nullable=False),
        sa.Column('token_digest', sa.LargeBinary(), nullable=False),
        sa.Column('expires_at', Timestamp(), nullable=False),
        sa.Column('consumed_at', Timestamp(), nullable=True),
        sa.Column('revoked_at', Timestamp(), nullable=True),
        sa.Column(
            'created_at',
            Timestamp(),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column('created_by', sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ['user_id'],
            ['users.id'],
            name='fk_identity_link_invites_user_id_users',
            ondelete='CASCADE',
        ),
        sa.CheckConstraint(
            'NOT (consumed_at IS NOT NULL AND revoked_at IS NOT NULL)',
            name='ck_identity_link_invites_terminal_state',
        ),
        sa.PrimaryKeyConstraint('id', name='pk_identity_link_invites'),
        sa.UniqueConstraint(
            'token_digest',
            name='uq_identity_link_invites_token_digest',
        ),
    )
    op.create_index(
        'ix_identity_link_invites_user_created',
        'identity_link_invites',
        ['user_id', 'created_at'],
    )


def downgrade() -> None:
    op.drop_table('identity_link_invites')
