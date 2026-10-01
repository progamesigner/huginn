from typing import Annotated, cast, override

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


class Ceremony:
    def __init__(self, farewell: Farewell, greeter: Greeter) -> None:
        self.farewell = farewell
        self.greeter = greeter


class Unannotated:
    def __init__(self, mystery) -> None:
        self.mystery = mystery


class Announcement:
    def __init__(self, greeter: Annotated[Greeter, 'primary']) -> None:
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


def test_resolve_detects_direct_cycle() -> None:
    container = Container()
    container.bind(Greeter, factory=lambda c: c.resolve(Greeter))

    with pytest.raises(ResolutionError, match='Greeter -> Greeter'):
        container.resolve(Greeter)


def test_resolve_detects_indirect_cycle_with_chain() -> None:
    container = Container()
    container.bind(Greeter, factory=lambda c: c.resolve(Farewell).greeter)
    container.bind(Farewell, factory=lambda c: Farewell(c.resolve(Greeter)))

    with pytest.raises(ResolutionError, match='Greeter -> Farewell -> Greeter'):
        container.resolve(Greeter)


def test_resolve_cleans_stack_after_cycle_error() -> None:
    container = Container()
    container.bind(Greeter, factory=lambda c: c.resolve(Greeter))

    with pytest.raises(ResolutionError, match=r'Greeter -> Greeter$'):
        container.resolve(Greeter)
    with pytest.raises(ResolutionError, match=r'Greeter -> Greeter$'):
        container.resolve(Greeter)


def test_use_override_breaks_cycle() -> None:
    container = Container()
    container.bind(Greeter, factory=lambda c: c.resolve(Farewell).greeter)
    container.bind(Farewell, factory=lambda c: Farewell(c.resolve(Greeter)))

    with container.use(Greeter, Greeter('fake')):
        assert container.resolve(Farewell).greeter.name == 'fake'


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
    journal = []
    created = []
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
    journal = []
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
    journal = []
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
    journal = []
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


def test_constructor_binding_builds_dependency_chain() -> None:
    container = Container()
    container.bind(Greeter, constructor=Greeter, shared=True)
    container.bind(Farewell, constructor=Farewell)
    container.bind(Ceremony, constructor=Ceremony)

    ceremony = container.resolve(Ceremony)

    assert ceremony.farewell.greeter is ceremony.greeter
    assert ceremony.greeter is container.resolve(Greeter)


def test_constructor_binding_supports_contract_key() -> None:
    container = Container()
    container.bind(Greeting, constructor=Greeter)

    assert isinstance(container.resolve(Greeting), Greeter)


def test_constructor_binding_skips_defaulted_parameters() -> None:
    container = Container()
    container.bind(Greeter, constructor=Greeter)

    assert container.resolve(Greeter).name == 'huginn'


def test_constructor_binding_unwraps_annotated_parameters() -> None:
    container = Container()
    container.bind(Greeter, instance=Greeter('annotated'))
    container.bind(Announcement, constructor=Announcement)

    assert container.resolve(Announcement).greeter.name == 'annotated'


def test_constructor_binding_requires_annotations() -> None:
    container = Container()

    with pytest.raises(BindingError, match='mystery'):
        container.bind(Unannotated, constructor=Unannotated)


def test_constructor_binding_requires_bound_dependencies() -> None:
    container = Container()
    container.bind(Farewell, constructor=Farewell)

    with pytest.raises(ResolutionError):
        container.resolve(Farewell)


def test_constructor_binding_excludes_other_sources() -> None:
    container = Container()

    with pytest.raises(BindingError):
        container.bind(Greeter, constructor=Greeter, factory=lambda _: Greeter())


def test_constructor_binding_resolves_use_overrides() -> None:
    container = Container()
    container.bind(Greeter, constructor=Greeter, shared=True)
    container.bind(Farewell, constructor=Farewell)

    with container.use(Greeter, Greeter('fake')):
        assert container.resolve(Farewell).greeter.name == 'fake'


def test_collect_runs_each_contribution_in_order() -> None:
    container = Container()
    container.bind(Greeter, instance=Greeter('shared'))
    container.contribute(Greeting, lambda c: Greeter(f'{c.resolve(Greeter).name}-a'))
    container.contribute(Greeting, lambda _: Greeter('b'))

    collected = container.collect(Greeting)

    assert [cast(Greeter, item).name for item in collected] == ['shared-a', 'b']


def test_collect_without_contributions_is_empty() -> None:
    assert Container().collect(Greeting) == ()


def test_contributions_wait_for_collection() -> None:
    container = Container()
    calls = []

    def factory(_: ContainerContract) -> Greeting:
        calls.append(1)
        return Greeting()

    container.contribute(Greeting, factory)

    assert calls == []
    container.collect(Greeting)
    assert calls == [1]


def test_to_copies_contributions() -> None:
    container = Container()
    container.contribute(Greeting, lambda _: Greeting())

    copied = container.to(Container)
    copied.contribute(Greeting, lambda _: Greeting())

    assert len(container.collect(Greeting)) == 1
    assert len(copied.collect(Greeting)) == 2
