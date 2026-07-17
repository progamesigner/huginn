from pydantic import BaseModel, ConfigDict


class ConfigValue(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra='forbid')
