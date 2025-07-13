import sys
from pathlib import Path
import importlib
import types
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

si = importlib.import_module("sft.simple_inference")


def test_simple_inference_cli(monkeypatch):
    captured = {}

    def fake_pipeline(*args, **kwargs):
        def run(prompt, **k):
            captured["prompt"] = prompt
            return [{"generated_text": "text"}]
        return run

    monkeypatch.setattr(si.transformers, "pipeline", fake_pipeline)
    monkeypatch.setattr(si.AutoTokenizer, "from_pretrained", lambda *a, **k: object())
    monkeypatch.setattr(si.time, "sleep", lambda *a, **k: None)
    monkeypatch.setattr(si, "apply_filter", lambda *a, **k: None)
    monkeypatch.setattr(si, "genesis4", lambda t: t + " follow")
    monkeypatch.setattr(sys, "argv", ["prog", "Prompt"])

    si.main()
    assert captured["prompt"].startswith("<|im_start|>")
