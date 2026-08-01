from pydantic import BaseModel, ConfigDict


class AppRequest(BaseModel):
    model_config = ConfigDict(extra='ignore', frozen=True)
