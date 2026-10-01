from pydantic import BaseModel, ConfigDict


class AppRequest(BaseModel):
    model_config = ConfigDict(extra='ignore', frozen=True)


class AppResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, extra='forbid', frozen=True)


# NOTE: strict like the lib.llm values, so a domain value is the type it says it is
class AppEntity(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        extra='forbid',
        frozen=True,
        strict=True,
    )


class AppValue(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
