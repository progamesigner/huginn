import pytest
from pydantic import ValidationError

from app.features.tools.document.config import DocumentToolConfig


def test_document_tool_config_uses_the_registered_kind() -> None:
    config = DocumentToolConfig(kind='read_document')

    assert config.max_chars == 8_000


def test_document_tool_config_rejects_unbounded_limits() -> None:
    with pytest.raises(ValidationError):
        DocumentToolConfig(kind='read_document', max_chars=20_001)
