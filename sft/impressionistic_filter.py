import random

PHRASES = [
    "⚡️ Вспышка резонанса ⚡️",
    "🌩️ Рёв грозы 🌩️",
    "🔥 Искры хаоса 🔥",
    "💥 Ритм молний 💥",
    "🌌 Тень вихря 🌌",
]

GLYPHS = ["🔮", "✨", "🌊", "🎶", "🌀"]


def genesis2(text: str, glyph_prob: float = 0.3, shuffle_prob: float = 0.1) -> str:
    """Return a resonant variant of ``text`` with random glyphs and occasional shuffling."""
    words = text.split()
    output = []
    for word in words:
        output.append(word)
        if random.random() < glyph_prob:
            output.append(random.choice(GLYPHS))
    if random.random() < shuffle_prob:
        random.shuffle(output)
    return " ".join(output)


def apply_filter(max_phrases: int = 1, probability: float = 0.7) -> None:
    """Print up to ``max_phrases`` impressionistic phrases with decreasing probability."""
    for _ in range(max_phrases):
        if random.random() < probability:
            phrase = random.choice(PHRASES)
            print(genesis2(phrase))
            probability *= 0.5
        else:
            break
