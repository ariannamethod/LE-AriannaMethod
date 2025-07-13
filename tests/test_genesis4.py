import sys
from pathlib import Path
import random
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sft.genesis4 import genesis4
import pytest

@pytest.mark.parametrize("kwargs", [
    {},
    {"depth": 2, "glyph_prob": 1.0, "shuffle_prob": 0.5, "echo_chance": 0.5},
])
def test_genesis4_mutates_text(kwargs):
    random.seed(0)
    text = "A B C"
    result = genesis4(text, **kwargs)
    assert isinstance(result, str)
    assert len(result) != len(text)
