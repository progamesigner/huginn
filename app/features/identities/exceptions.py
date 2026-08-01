class IdentityLinkError(Exception):
    pass


class InvalidIdentityLinkError(IdentityLinkError):
    reason: str

    def __init__(self, reason: str) -> None:
        super().__init__('This link is invalid or expired. Ask for a new invite.')
        self.reason = reason


class IdentityLinkSecretError(IdentityLinkError):
    pass
