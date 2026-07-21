class IdentityLinkError(Exception):
    pass


class InvalidIdentityLinkError(IdentityLinkError):
    def __init__(self) -> None:
        super().__init__('This link is invalid or expired. Ask for a new invite.')


class IdentityLinkSecretError(IdentityLinkError):
    pass
