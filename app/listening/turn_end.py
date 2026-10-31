from typing import Protocol

# A turn that ends on one of these is almost never finished: "I was going to say that, um"
TRAILING_WORDS = {
    "and", "but", "so", "or", "because", "if", "when", "that", "which", "then",
    "the", "a", "an", "to", "of", "with", "for", "my", "your", "is", "was",
    "um", "uh", "erm", "like",
}  # fmt: skip


class TurnEndModel(Protocol):
    def probability(self, text: str) -> float: ...


class TextTurnEnd:
    """How likely it is that the caller finished, judged from the words only.

    A small stand-in for a semantic turn detector (Deepgram Flux, LiveKit's turn model). It sits
    behind the same one-method interface, so either can replace it without touching the policy.
    """

    def probability(self, text: str) -> float:
        text = text.strip()
        if not text:
            return 0.0
        last_word = text.split()[-1].lower().strip(".,!?;:\"'")
        if last_word in TRAILING_WORDS:
            return 0.15
        if text.endswith("?"):
            return 0.95
        if text.endswith((".", "!")):
            return 0.85
        if text.endswith((",", "-", "—", ";", ":")):
            return 0.25
        return 0.5
