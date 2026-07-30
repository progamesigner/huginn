class TelegramError(Exception):
    pass


class MediaTooLargeError(TelegramError):
    pass


class MediaUnavailableError(TelegramError):
    pass
