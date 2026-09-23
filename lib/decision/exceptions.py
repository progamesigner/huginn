class DecisionError(Exception):
    pass


class DecisionTransportError(DecisionError):
    pass


class DecisionTimeoutError(DecisionTransportError):
    pass


class DecisionHTTPError(DecisionError):
    status: int

    def __init__(self, status: int, message: str) -> None:
        super().__init__(f'decision request failed with {status}: {message}')
        self.status = status


class DecisionResponseError(DecisionError):
    pass
