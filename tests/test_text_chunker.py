from app.voice.text_chunker import chunk_for_tts


async def _stream(tokens):
    for token in tokens:
        yield token


async def _collect(tokens):
    return [chunk async for chunk in chunk_for_tts(_stream(tokens))]


async def test_first_chunk_released_after_three_words():
    chunks = await _collect(["Hi", " there", " my", " friend", " how", " are", " you", "?"])
    assert chunks[0] == "Hi there my friend "
    assert chunks[1] == "how are you? "


async def test_breaks_on_punctuation_after_first_chunk():
    chunks = await _collect(["Sure", ".", " I", " can", " help", ",", " just", " ask", "."])
    assert chunks == ["Sure. ", "I can help, ", "just ask. "]


async def test_flushes_remaining_text():
    chunks = await _collect(["Okay"])
    assert chunks == ["Okay"]


async def test_skips_empty_tokens():
    chunks = await _collect(["", "Yes", "", "."])
    assert chunks == ["Yes. "]


async def test_does_not_break_inside_numbers_or_hyphenated_words():
    chunks = await _collect(
        ["Sure", ".", " It", " costs", " 3", ".", "5", " for", " a", " well", "-", "known", " brand", "."]
    )
    assert chunks == ["Sure. ", "It costs 3.5 for a well-known brand. "]
