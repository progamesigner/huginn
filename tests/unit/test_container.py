from typing import override

import pytest

from lib.container import BindingError, Container, ContainerContract, ResolutionError


class Greeting:
    pass


class Greeter(Greeting):
    def __init__(self, name: str = 'huginn') -> None:
        self.name = name


class Farewell:
    def __init__(self, greeter: Greeter) -> None:
        self.greeter = greeter


class RecordingManager:
    def __init__(self, journal: list[str], name: str) -> None:
        self.journal = journal
        self.name = name

    async def start(self) -> None:
        self.journal.append(f'{self.name}:start')

    async def dispose(self) -> None:
        self.journal.append(f'{self.name}:dispose')


class OtherManager(RecordingManager):
    pass


class ExplodingManager(RecordingManager):
    @override
    async def start(self) -> None:
        raise RuntimeError(f'{self.name} refused to start')


def test_instance_binding_returns_same_object() -> None:
    container = Container()
    greeter = Greeter()
    container.bind(Greeter, instance=greeter)

    assert container.resolve(Greeter) is greeter


def test_transient_factory_returns_new_object_each_time() -> None:
    container = Container()
    container.bind(Greeter, factory=lambda _: Greeter())

    assert container.resolve(Greeter) is not container.resolve(Greeter)


def test_shared_factory_returns_cached_object() -> None:
    container = Container()
    container.bind(Greeter, factory=lambda _: Greeter(), shared=True)

    assert container.resolve(Greeter) is container.resolve(Greeter)


def test_factory_resolves_dependencies_from_container() -> None:
    container = Container()
    container.bind(Greeter, factory=lambda _: Greeter('wired'), shared=True)
    container.bind(
        Farewell,
        factory=lambda c: Farewell(c.resolve(Greeter)),
        shared=True,
    )

    assert container.resolve(Farewell).greeter.name == 'wired'


def test_duplicate_binding_raises() -> None:
    container = Container()
    container.bind(Greeter, instance=Greeter())

    with pytest.raises(BindingError):
        container.bind(Greeter, factory=lambda _: Greeter())


def test_unbound_resolve_raises() -> None:
    container = Container()

    with pytest.raises(ResolutionError):
        container.resolve(Greeter)


def test_binding_requires_exactly_one_source() -> None:
    container = Container()

    with pytest.raises(BindingError):
        container.bind(Greeter)
    with pytest.raises(BindingError):
        container.bind(Greeter, instance=Greeter(), factory=lambda _: Greeter())


def test_eager_requires_shared() -> None:
    container = Container()

    with pytest.raises(BindingError):
        container.bind(Greeter, factory=lambda _: Greeter(), eager=True)


def test_use_overrides_and_restores() -> None:
    container = Container()
    container.bind(Greeter, instance=Greeter('real'))
    fake = Greeter('fake')

    with container.use(Greeter, fake):
        assert container.resolve(Greeter) is fake
    assert container.resolve(Greeter).name == 'real'


def test_use_supports_nested_overrides() -> None:
    container = Container()
    container.bind(Greeter, instance=Greeter('real'))
    outer, inner = Greeter('outer'), Greeter('inner')

    with container.use(Greeter, outer):
        with container.use(Greeter, inner):
            assert container.resolve(Greeter) is inner
        assert container.resolve(Greeter) is outer
    assert container.resolve(Greeter).name == 'real'


async def test_start_prewarms_only_eager_bindings() -> None:
    journal: list[str] = []
    created: list[str] = []
    container = Container()
    container.bind(
        RecordingManager,
        factory=lambda _: RecordingManager(journal, 'eager'),
        shared=True,
        eager=True,
    )

    def make_lazy(_: ContainerContract) -> Greeter:
        created.append('lazy')
        return Greeter()

    container.bind(Greeter, factory=make_lazy, shared=True)

    await container.start()

    assert journal == ['eager:start']
    assert created == []


async def test_dispose_closes_in_reverse_creation_order() -> None:
    journal: list[str] = []
    container = Container()
    container.bind(
        RecordingManager,
        factory=lambda _: RecordingManager(journal, 'a'),
        shared=True,
    )
    container.bind(
        OtherManager,
        factory=lambda _: OtherManager(journal, 'b'),
        shared=True,
    )

    container.resolve(RecordingManager)
    container.resolve(OtherManager)
    await container.dispose()

    assert journal == ['b:dispose', 'a:dispose']


async def test_start_failure_closes_already_started() -> None:
    journal: list[str] = []
    container = Container()
    container.bind(
        RecordingManager,
        factory=lambda _: RecordingManager(journal, 'first'),
        shared=True,
        eager=True,
    )
    container.bind(
        ExplodingManager,
        factory=lambda _: ExplodingManager(journal, 'second'),
        shared=True,
        eager=True,
    )

    with pytest.raises(RuntimeError):
        await container.start()

    assert 'first:start' in journal
    assert journal[-1] == 'first:dispose'


def test_alias_resolves_through_original_binding() -> None:
    container = Container()
    container.bind(Greeter, factory=lambda _: Greeter('aliased'), shared=True)
    container.alias(Greeter, Greeting)

    assert container.resolve(Greeting) is container.resolve(Greeter)


def test_alias_requires_bound_target() -> None:
    container = Container()

    with pytest.raises(BindingError):
        container.alias(Greeter, Greeting)


def test_alias_key_cannot_collide_with_binding() -> None:
    container = Container()
    container.bind(Greeting, instance=Greeting())
    container.bind(Greeter, instance=Greeter())

    with pytest.raises(BindingError):
        container.alias(Greeter, Greeting)


def test_bound_reports_registrations_and_overrides() -> None:
    container = Container()

    assert not container.bound(Greeter)

    container.bind(Greeter, instance=Greeter())
    assert container.bound(Greeter)

    with container.use(Greeting, Greeting()):
        assert container.bound(Greeting)
    assert not container.bound(Greeting)


def test_to_copies_bindings_with_fresh_singleton_state() -> None:
    container = Container()
    container.bind(Greeter, factory=lambda _: Greeter('shared'), shared=True)
    original = container.resolve(Greeter)

    clone = container.to(Container)

    assert clone.resolve(Greeter) is not original
    assert clone.resolve(Greeter) is clone.resolve(Greeter)


def test_to_shares_instance_bindings_but_not_overrides() -> None:
    container = Container()
    real = Greeter('real')
    container.bind(Greeter, instance=real)

    with container.use(Greeter, Greeter('fake')):
        clone = container.to(Container)

    assert clone.resolve(Greeter) is real


async def test_async_with_starts_and_closes() -> None:
    journal: list[str] = []
    container = Container()
    container.bind(
        RecordingManager,
        factory=lambda _: RecordingManager(journal, 'managed'),
        shared=True,
        eager=True,
    )

    async with container:
        assert journal == ['managed:start']
    assert journal == ['managed:start', 'managed:dispose']
