import random
from .impressionistic_filter import genesis2


def genesis4(
    text: str,
    depth: int = 3,
    glyph_prob: float = 0.3,
    shuffle_prob: float = 0.1,
    echo_chance: float = 0.3,
) -> str:
    """Return ``text`` recursively mutated by :func:`genesis2` with echoes.

    Each iteration applies ``genesis2`` to the current text. With a probability
    of ``echo_chance`` a snippet from an earlier iteration is appended to the
    result, creating a small resonance loop.
    """
    history = [text]
    current = text
    for _ in range(depth):
        current = genesis2(current, glyph_prob=glyph_prob, shuffle_prob=shuffle_prob)
        if random.random() < echo_chance and history:
            snippet = random.choice(history)
            current = f"{current} {snippet}"
        history.append(current)
    return current
