from app.core.settings import HuginnSettings


def make_settings(**values: object) -> HuginnSettings:
    return HuginnSettings.model_validate(values)
