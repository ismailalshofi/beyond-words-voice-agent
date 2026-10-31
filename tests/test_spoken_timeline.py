from app.voice.spoken_timeline import PCM16_16K_BYTES_PER_MS, SpokenTimeline


def _timeline(*chunks):
    """Each chunk is spoken at 10 ms per character."""
    timeline = SpokenTimeline()
    for text in chunks:
        starts = [i * 10 for i in range(len(text))]
        timeline.add_audio(len(text) * 10 * PCM16_16K_BYTES_PER_MS, {"chars": list(text), "charStartTimesMs": starts})
    return timeline


def test_chunk_timings_are_laid_end_to_end():
    timeline = _timeline("Hello ", "there.")
    assert timeline.audio_ms == 120
    # "there." starts 60 ms in, so 75 ms covers "Hello" plus the "th" of "there"
    assert timeline.text_before(75) == "Hello"
    assert timeline.text_before(120) == "Hello there."


def test_cuts_back_to_the_last_whole_word():
    timeline = _timeline("Book a table")
    assert timeline.text_before(35) == "Book"
    assert timeline.text_before(15) == ""


def test_nothing_heard_before_playback():
    assert _timeline("Hi there").text_before(0) == ""


def test_audio_without_alignment_only_advances_time():
    timeline = SpokenTimeline()
    timeline.add_audio(3200, None)
    assert timeline.audio_ms == 100
    assert timeline.text_before(100) == ""
