from app.core.settings import HuginnSettings


def make_settings(
    *,
    database_url: str | None = None,
    redis_url: str | None = None,
    **values: object,
) -> HuginnSettings:
    if database_url is not None:
        values['database_url'] = database_url

    if redis_url is not None:
        values['redis_url'] = redis_url

    return HuginnSettings.model_validate(values)
