from pydantic import SecretStr

from lib.decision import DecisionManager, DecisionProviderConfig, DecisionRef
from lib.decision.adapters.bifrost import BifrostAdapter
from lib.decision.adapters.typesafe import TypeSafeAdapter
from lib.http import HTTPClientManager


def _providers() -> tuple[DecisionProviderConfig, ...]:
    return (
        DecisionProviderConfig(
            name='bifrost',
            adapter='bifrost',
            api_key=SecretStr('secret'),
            models=('typesafe/jev-1.13.0',),
            url='http://bifrost:8080/v1/decisions',
        ),
        DecisionProviderConfig(
            name='openrouter',
            adapter='typesafe',
            api_key=SecretStr('secret'),
            models=('typesafe/jev-1.13',),
            url='https://openrouter.ai/api/alpha/decisions',
        ),
        DecisionProviderConfig(
            name='typesafe',
            adapter='typesafe',
            api_key=SecretStr('secret'),
            models=('jev-1.13.0',),
        ),
    )


def test_manager_builds_the_adapter_each_provider_names() -> None:
    manager = DecisionManager(_providers(), HTTPClientManager())

    bifrost = manager.client_for(DecisionRef(model='typesafe/jev-1.13.0'))
    openrouter = manager.client_for(DecisionRef(model='typesafe/jev-1.13'))
    typesafe = manager.client_for(DecisionRef(model='jev-1.13.0', provider='typesafe'))

    assert isinstance(bifrost, BifrostAdapter)
    assert isinstance(openrouter, TypeSafeAdapter)
    assert isinstance(typesafe, TypeSafeAdapter)


def test_manager_reuses_one_client_per_provider() -> None:
    manager = DecisionManager(_providers(), HTTPClientManager())
    ref = DecisionRef(model='jev-1.13.0')

    assert manager.client_for(ref) is manager.client_for(ref)


def test_manager_has_no_client_for_an_unserved_model() -> None:
    manager = DecisionManager(_providers(), HTTPClientManager())

    assert manager.client_for(DecisionRef(model='jev-0')) is None
    assert (
        manager.provider_for(DecisionRef(model='jev-1.13.0', provider='bifrost'))
        is None
    )


def test_manager_without_providers_serves_nothing() -> None:
    manager = DecisionManager((), HTTPClientManager())

    assert manager.client_for(DecisionRef(model='jev-1.13.0')) is None
