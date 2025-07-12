import random

PHRASES = [
    "⚡️ Вспышка резонанса ⚡️",
    "🌩️ Рёв грозы 🌩️",
    "🔥 Искры хаоса 🔥",
    "💥 Ритм молний 💥",
    "🌌 Тень вихря 🌌",
]


def apply_filter():
    """Print a short impressionistic phrase before the model response."""
    print(random.choice(PHRASES))
