import pytest

pytest.importorskip("torch")
import torch
from lit_gpt.utils import incremental_save, lazy_load, NotYetLoadedTensor

def test_device_attributes(tmp_path):
    path = tmp_path / "weights.pth"
    with incremental_save(path) as saver:
        saver.save({"t": torch.tensor([1, 2, 3])})

    with lazy_load(path) as sd:
        t = sd["t"]
        assert isinstance(t, NotYetLoadedTensor)
        expected = torch.device("cpu")
        assert t.device == expected
        assert t.is_cpu
        assert not t.is_cuda
        loaded = t._load_tensor()
        for attr in ["device", "is_cuda", "is_cpu"]:
            assert getattr(t, attr) == getattr(loaded, attr)
