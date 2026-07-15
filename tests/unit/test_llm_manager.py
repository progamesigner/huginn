from pydantic import SecretStr

from lib.container import Disposable
from lib.llm import LLMManager, ProviderConfig


def test_manager_looks_up_provider_by_model() -> None:
    anthropic = ProviderConfig(
        name='anthropic',
        adapter='anthropic',
        api_key=SecretStr('secret'),
        models=('claude-opus-4-8',),
    )
    manager = LLMManager((anthropic,))

    assert manager.provider_for('claude-opus-4-8') is anthropic
    assert manager.provider_for('unknown-model') is None


def test_manager_satisfies_disposable_contract() -> None:
    manager = LLMManager(())

    assert isinstance(manager, Disposable)


async def test_manager_dispose_does_not_raise() -> None:
    manager = LLMManager(())

    await manager.dispose()
