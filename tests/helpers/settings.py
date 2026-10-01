from app.core.application import App
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


# NOTE: settings bound before build win over the environment, so a test pins its own
def app_with(settings: HuginnSettings) -> App:
    app = App()
    app.bind(HuginnSettings, instance=settings)
    return app
