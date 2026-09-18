from datetime import timedelta

import pytest
from pydantic import ValidationError

from app.features.blobs.config import BLOBS, BlobsConfig, RetentionConfig


def test_retention_reads_durations_and_keeps_inbound_blobs_by_default() -> None:
    retention = RetentionConfig.model_validate({'generated': '7d'})

    assert retention.generated == timedelta(days=7)
    assert retention.inbound is None


def test_retention_refuses_a_duration_it_cannot_read() -> None:
    with pytest.raises(ValidationError, match='not a positive duration'):
        RetentionConfig.model_validate({'inbound': '2 weeks'})


def test_the_blobs_section_nests_retention() -> None:
    config = BlobsConfig.model_validate(
        {'retention': {'generated': '30d', 'inbound': None}},
    )

    assert config.retention.generated == timedelta(days=30)
    assert config.retention.inbound is None
    assert BLOBS.key == 'blobs'
