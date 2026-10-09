import pytest

from lib.media import sniff


@pytest.mark.parametrize(
    ('data', 'mime'),
    [
        (b'\x89PNG\r\n\x1a\n rest', 'image/png'),
        (b'\xff\xd8\xff\xe0 jfif', 'image/jpeg'),
        (b'GIF89a rest', 'image/gif'),
        (b'RIFF\x00\x00\x00\x00WEBPVP8 ', 'image/webp'),
        (b'RIFF\x00\x00\x00\x00WAVEfmt ', 'audio/wav'),
        (b'%PDF-1.7 rest', 'application/pdf'),
        (b'OggS\x00 rest', 'audio/ogg'),
        (b'ID3\x04 rest', 'audio/mpeg'),
        (b'\xff\xfb\x90\x00 frame', 'audio/mpeg'),
        (b'\x00\x00\x00\x20ftypM4A rest', 'audio/mp4'),
        (b'\x00\x00\x00\x20ftypisom rest', 'video/mp4'),
    ],
)
def test_sniff_reads_the_format_off_the_bytes(data: bytes, mime: str) -> None:
    assert sniff(data, 'misleading.txt') == mime


def test_sniff_needs_the_name_to_tell_office_files_apart() -> None:
    zipped = b'PK\x03\x04 rest of the archive'

    assert sniff(zipped, 'report.XLSX') == (
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
    assert sniff(zipped, 'notes.docx') == (
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    )
    assert sniff(zipped, 'archive.zip') is None
    assert sniff(zipped) is None


def test_sniff_names_text_by_its_extension_and_defaults_to_plain() -> None:
    assert sniff(b'a,b\n1,2\n', 'table.csv') == 'text/csv'
    assert sniff(b'# title', 'notes.md') == 'text/markdown'
    assert sniff('中文'.encode(), None) == 'text/plain'


def test_sniff_gives_up_on_bytes_it_does_not_know() -> None:
    assert sniff(b'\x00\x01\x02 binary') is None
    assert sniff(b'') is None
