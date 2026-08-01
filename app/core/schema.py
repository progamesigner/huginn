from pydantic import BaseModel, ConfigDict


class AppRequest(BaseModel):
    model_config = ConfigDict(extra='ignore', frozen=True)


class AppResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, extra='forbid', frozen=True)
