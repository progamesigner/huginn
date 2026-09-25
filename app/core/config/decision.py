from typing import Annotated, Literal, Self

from pydantic import Field, SecretStr, model_validator

from app.core.config.schema import ConfigValue
from lib.decision import Adapter


class DecisionProviderConfig(ConfigValue):
    type: Annotated[Literal['decision'], Field()]
    adapter: Annotated[Adapter, Field()]
    api_key: Annotated[SecretStr, Field()]
    models: Annotated[tuple[str, ...], Field(min_length=1)]
    url: Annotated[str | None, Field(min_length=1)] = None
    timeout: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 3.0

    @model_validator(mode='after')
    def addressable(self) -> Self:
        if self.adapter == 'bifrost' and not self.url:
            raise ValueError('bifrost requires the url of its decisions endpoint')
        return self
