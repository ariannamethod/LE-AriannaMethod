import importlib
import types
import sys
from pathlib import Path

import pytest

# helper to import pretrain.utils with stub modules

def import_pretrain_utils(monkeypatch):
    torch_stub = types.ModuleType("torch")
    utils_stub = types.ModuleType("torch.utils")
    data_stub = types.ModuleType("torch.utils.data")

    class DummyDataLoader:
        def __init__(self, dataset, batch_size, shuffle=False, pin_memory=False):
            self.dataset = dataset
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.pin_memory = pin_memory

    data_stub.DataLoader = DummyDataLoader
    utils_stub.data = data_stub
    torch_stub.utils = utils_stub
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    monkeypatch.setitem(sys.modules, "torch.utils", utils_stub)
    monkeypatch.setitem(sys.modules, "torch.utils.data", data_stub)

    pd_stub = types.ModuleType("lit_gpt.packed_dataset")

    class DummyPackedDataset:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def __iter__(self):
            if False:
                yield None

    class DummyCombinedDataset:
        def __init__(self, datasets, seed, weights=None):
            self.datasets = datasets
            self.seed = seed
            self.weights = weights

        def __iter__(self):
            if False:
                yield None

    pd_stub.PackedDataset = DummyPackedDataset
    pd_stub.CombinedDataset = DummyCombinedDataset
    monkeypatch.setitem(sys.modules, "lit_gpt.packed_dataset", pd_stub)

    if "pretrain.utils" in sys.modules:
        del sys.modules["pretrain.utils"]
    return importlib.import_module("pretrain.utils")


def test_create_dataloader(monkeypatch, tmp_path):
    pu = import_pretrain_utils(monkeypatch)

    def fake_glob(pattern):
        prefix = Path(pattern).name[:-1]
        return [str(tmp_path / f"{prefix}1"), str(tmp_path / f"{prefix}2")]

    monkeypatch.setattr(pu.glob, "glob", fake_glob)
    monkeypatch.setattr(pu.random, "shuffle", lambda x: None)

    class Fabric:
        global_rank = 0
        world_size = 1

    config = [("data", 1.0), ("more", 2.0)]
    dl = pu.create_dataloader(
        batch_size=2,
        block_size=4,
        data_dir=tmp_path,
        fabric=Fabric(),
        data_config=config,
        shuffle=True,
        seed=42,
    )

    assert isinstance(dl.dataset, pu.CombinedDataset)
    assert len(dl.dataset.datasets) == 2
    assert dl.dataset.weights == [1 / 3, 2 / 3]

    d0_args = dl.dataset.datasets[0]
    assert d0_args.args[0] == [str(tmp_path / "data1"), str(tmp_path / "data2")]
    assert d0_args.kwargs["block_size"] == 4
    assert d0_args.kwargs["shuffle"] is True


def test_create_dataloader_no_data(monkeypatch, tmp_path):
    pu = import_pretrain_utils(monkeypatch)
    monkeypatch.setattr(pu.glob, "glob", lambda pattern: [])
    class Fabric:
        global_rank = 0
        world_size = 1
    with pytest.raises(RuntimeError):
        pu.create_dataloader(
            batch_size=1,
            block_size=2,
            data_dir=tmp_path,
            fabric=Fabric(),
            data_config=[],
        )


def test_create_dataloaders(monkeypatch, tmp_path):
    pu = import_pretrain_utils(monkeypatch)
    captured = {}

    def fake_create(**kwargs):
        key = kwargs["data_dir"].name
        captured[key] = kwargs["block_size"]
        return f"dl_{key}"

    monkeypatch.setattr(pu, "create_dataloader", fake_create)

    class Fabric:
        pass

    train_dl, val_dl = pu.create_dataloaders(
        batch_size=1,
        block_size=3,
        fabric=Fabric(),
        train_data_dir=Path("train"),
        val_data_dir=Path("val"),
        train_config=[("t", 1)],
        val_config=[("v", 1)],
    )

    assert train_dl == "dl_train"
    assert val_dl == "dl_val"
    assert captured["train"] == 4  # block_size + 1
    assert captured["val"] == 4
