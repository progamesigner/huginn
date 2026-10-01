from uuid import UUID

from app.features.identities.entities import ExternalIdentity, IdentityAuditEvent, User


class FakeIdentityRepository:
    counter: int
    users: dict[str, User]
    identities: dict[tuple[str, str, str], ExternalIdentity]
    audit: list[IdentityAuditEvent]

    def __init__(self) -> None:
        self.counter = 0
        self.users = {}
        self.identities = {}
        self.audit = []

    def next_id(self) -> UUID:
        self.counter += 1
        return UUID(int=self.counter)

    async def create_user(self, handle: str, display_name: str) -> User:
        user = User(self.next_id(), handle.strip().lower(), display_name)
        self.users[user.handle] = user
        return user

    async def find_user(self, handle: str) -> User | None:
        return self.users.get(handle.strip().lower())

    async def link(
        self,
        user: User,
        provider: str,
        realm: str,
        external_user_id: str,
    ) -> ExternalIdentity:
        identity = ExternalIdentity(
            self.next_id(),
            user.id,
            provider,
            realm,
            external_user_id,
        )
        self.identities[(provider, realm, external_user_id)] = identity
        return identity

    async def find_external(
        self,
        provider: str,
        realm: str,
        external_user_id: str,
    ) -> ExternalIdentity | None:
        return self.identities.get((provider, realm, external_user_id))


async def test_fake_resolves_normalized_users_and_realm_scoped_identities() -> None:
    repository = FakeIdentityRepository()
    user = await repository.create_user(' Alice ', 'Alice')
    await repository.link(user, 'telegram', 'primary', '42')

    assert await repository.find_user('ALICE') == user
    assert await repository.find_external('telegram', 'primary', '42') is not None
    assert await repository.find_external('telegram', 'other', '42') is None
