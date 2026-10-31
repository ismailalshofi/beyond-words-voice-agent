from collections.abc import AsyncIterator

PUNCTUATION = {".", ",", "?", "!", ";", ":", "—"}


async def chunk_for_tts(chunks: AsyncIterator[str]) -> AsyncIterator[str]:
    """Group LLM tokens into TTS-friendly pieces.

    The first piece is released after ~3 words so speech starts as early as possible;
    after that, pieces break on punctuation so the voice keeps natural phrasing.

    A token ending in punctuation only marks a *possible* break: the break happens if the
    next token starts a new word. That keeps "3.5", "e.g." and "U.S." in one piece.
    """
    buffer = ""
    first_chunk_yielded = False
    word_count = 0
    break_pending = False

    async for text in chunks:
        if not text:
            continue

        starts_new_word = text[0].isspace()
        if break_pending and starts_new_word and buffer.strip():
            yield buffer.strip() + " "
            buffer = ""
            first_chunk_yielded = True
            word_count = 0
        break_pending = False

        # A token that starts with whitespace starts a new word
        if text.strip() and starts_new_word:
            word_count += 1

        buffer += text
        stripped = text.strip()
        if stripped and stripped[-1] in PUNCTUATION:
            break_pending = True
        elif not first_chunk_yielded and word_count >= 3 and buffer.strip():
            yield buffer.strip() + " "
            buffer = ""
            first_chunk_yielded = True
            word_count = 0

    if buffer.strip():
        yield buffer.strip() + (" " if break_pending else "")
