from app.features.channels.documents import DocumentReading, read_document


async def test_a_readable_document_previews_its_text_and_keeps_all_of_it() -> None:
    text = 'x' * 2500

    reading = await read_document('text/plain', text.encode(), 'notes.txt', '[doc]')

    assert reading == DocumentReading(f'[doc] (truncated)\n{"x" * 2000}', text)


async def test_an_unreadable_document_says_why_and_keeps_no_text() -> None:
    reading = await read_document('application/pdf', b'not a pdf', 'scan.pdf', '[doc]')

    assert reading == DocumentReading('[doc] (could not extract scan.pdf)')
