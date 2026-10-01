from app.features.agents.entities import TurnScope


def test_document_scope_is_optional_and_defaults_empty() -> None:
    scope = TurnScope(actor='user', agent='agent', channel='web', chat='1')
    assert scope.document_blobs == frozenset()
