from pathlib import PurePosixPath

_SIGNATURES = (
    (b'\x89PNG\r\n\x1a\n', 'image/png'),
    (b'\xff\xd8\xff', 'image/jpeg'),
    (b'GIF87a', 'image/gif'),
    (b'GIF89a', 'image/gif'),
    (b'%PDF-', 'application/pdf'),
    (b'OggS', 'audio/ogg'),
    (b'ID3', 'audio/mpeg'),
    (b'fLaC', 'audio/flac'),
    (b'\x1aE\xdf\xa3', 'video/webm'),
)
_RIFF = {b'WEBP': 'image/webp', b'WAVE': 'audio/wav'}
# NOTE: a zip or a text file says nothing about what it holds, so only the name can
_ZIPPED = {
    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
}
_TEXTS = {
    '.csv': 'text/csv',
    '.json': 'application/json',
    '.md': 'text/markdown',
    '.markdown': 'text/markdown',
}
_TEXT = 'text/plain'


def _suffix(filename: str | None) -> str:
    return '' if filename is None else PurePosixPath(filename).suffix.lower()


def _text(data: bytes) -> bool:
    if b'\x00' in data:
        return False

    try:
        data.decode('utf-8')
    except UnicodeDecodeError:
        return False

    return True


def sniff(data: bytes, filename: str | None = None) -> str | None:
    for signature, mime in _SIGNATURES:
        if data.startswith(signature):
            return mime

    if data.startswith(b'RIFF') and (riff := _RIFF.get(data[8:12])) is not None:
        return riff
    if data[4:8] == b'ftyp':
        return 'audio/mp4' if data[8:12] == b'M4A ' else 'video/mp4'
    # NOTE: a bare mpeg audio frame has no magic, only its sync bits
    if len(data) > 1 and data[0] == 0xFF and data[1] & 0xE0 == 0xE0:
        return 'audio/mpeg'
    if data.startswith(b'PK\x03\x04'):
        return _ZIPPED.get(_suffix(filename))
    if data and _text(data):
        return _TEXTS.get(_suffix(filename), _TEXT)

    return None
