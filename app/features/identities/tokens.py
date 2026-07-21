import base64
import hashlib
import hmac
import secrets
from collections.abc import Callable


class IdentityTokenCodec:
    TOKEN_BYTES = 16
    BASE32_LENGTH = (TOKEN_BYTES * 8 + 4) // 5

    def __init__(
        self,
        secret: bytes,
        random_bytes: Callable[[int], bytes] = secrets.token_bytes,
    ) -> None:
        if len(secret) < 32:
            raise ValueError('identity link secret must be at least 32 bytes')
        self.secret = secret
        self.random_bytes = random_bytes

    def issue(self) -> tuple[str, str, bytes]:
        raw = self.random_bytes(self.TOKEN_BYTES)
        token = base64.urlsafe_b64encode(raw).rstrip(b'=').decode()
        encoded = base64.b32encode(raw).decode().rstrip('=')
        checksum = hashlib.sha256(raw).hexdigest()[:4].upper()
        short_code = '-'.join(
            encoded[index : index + 4] for index in range(0, len(encoded), 4)
        )
        return token, f'{short_code}-{checksum}', self.digest(raw)

    def digest_credential(self, credential: str) -> bytes | None:
        raw = self._decode_token(credential) or self._decode_short_code(credential)
        return self.digest(raw) if raw is not None else None

    def digest(self, raw: bytes) -> bytes:
        return hmac.digest(self.secret, raw, 'sha256')

    def matches(self, digest: bytes, credential: str) -> bool:
        candidate = self.digest_credential(credential)
        return candidate is not None and hmac.compare_digest(digest, candidate)

    def _decode_token(self, credential: str) -> bytes | None:
        try:
            padding = '=' * (-len(credential) % 4)
            raw = base64.b64decode(
                credential + padding,
                altchars=b'-_',
                validate=True,
            )
        except ValueError, UnicodeEncodeError:
            return None
        return raw if len(raw) == self.TOKEN_BYTES else None

    def _decode_short_code(self, credential: str) -> bytes | None:
        compact = credential.replace('-', '').upper()
        encoded, checksum = compact[:-4], compact[-4:]
        if len(encoded) != self.BASE32_LENGTH or len(checksum) != 4:
            return None
        try:
            padding = '=' * (-len(encoded) % 8)
            raw = base64.b32decode(encoded + padding)
        except ValueError, UnicodeEncodeError:
            return None
        expected = hashlib.sha256(raw).hexdigest()[:4].upper()
        return raw if hmac.compare_digest(checksum, expected) else None
