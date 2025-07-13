import importlib
import types
import sys
import runpy

import pytest


def patch_common(monkeypatch):
    torch_stub = types.ModuleType("torch")
    utils_stub = types.ModuleType("torch.utils")
    data_stub = types.ModuleType("torch.utils.data")
    data_stub.DataLoader = object
    utils_stub.data = data_stub
    torch_stub.utils = utils_stub
    torch_stub.no_grad = lambda: (lambda f: f)
    nn_mod = types.ModuleType("torch.nn")
    class Module:
        pass

    nn_mod.Module = Module
    class Tensor:
        pass
    torch_stub.Tensor = Tensor
    torch_stub.set_float32_matmul_precision = lambda *a, **k: None
    torch_stub.nn = nn_mod
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    monkeypatch.setitem(sys.modules, "torch.nn", nn_mod)
    monkeypatch.setitem(sys.modules, "torch.utils", utils_stub)
    monkeypatch.setitem(sys.modules, "torch.utils.data", data_stub)

    lightning = types.ModuleType("lightning")
    fabric = types.ModuleType("lightning.fabric")
    strategies = types.ModuleType("lightning.fabric.strategies")
    strategies.FSDPStrategy = object
    strategies.XLAStrategy = object
    fabric.strategies = strategies
    lightning.fabric = fabric
    lightning.Fabric = object
    monkeypatch.setitem(sys.modules, "lightning", lightning)
    monkeypatch.setitem(sys.modules, "lightning.fabric", fabric)
    monkeypatch.setitem(sys.modules, "lightning.fabric.strategies", strategies)

    lg_model = types.ModuleType("lit_gpt.model")
    lg_model.GPT = object
    lg_model.Block = object
    lg_model.Config = object
    monkeypatch.setitem(sys.modules, "lit_gpt.model", lg_model)

    speed = types.ModuleType("lit_gpt.speed_monitor")
    speed.SpeedMonitorFabric = object
    speed.estimate_flops = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "lit_gpt.speed_monitor", speed)

    utils = types.ModuleType("lit_gpt.utils")
    utils.chunked_cross_entropy = lambda *a, **k: None
    utils.get_default_supported_precision = lambda *a, **k: None
    utils.num_parameters = lambda *a, **k: None
    utils.step_csv_logger = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "lit_gpt.utils", utils)

    pl_logger = types.ModuleType("pytorch_lightning.loggers")
    pl_logger.WandbLogger = object
    monkeypatch.setitem(sys.modules, "pytorch_lightning.loggers", pl_logger)

    lg_root = types.ModuleType("lit_gpt")
    lg_root.FusedCrossEntropyLoss = object
    monkeypatch.setitem(sys.modules, "lit_gpt", lg_root)

    pretrain_utils = types.ModuleType("pretrain.utils")
    pretrain_utils.create_dataloaders = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "pretrain.utils", pretrain_utils)


def check_cli_invocation(monkeypatch, module_name):
    called = {}

    def fake_cli(func):
        called["func"] = func

    json_mod = types.ModuleType("jsonargparse")
    json_mod.CLI = fake_cli
    monkeypatch.setitem(sys.modules, "jsonargparse", json_mod)
    monkeypatch.setattr("jsonargparse.CLI", fake_cli)
    runpy.run_module(module_name, run_name="__main__")
    assert called["func"].__name__ == "setup"


@pytest.mark.parametrize("mod", ["pretrain.leoleg", "pretrain.leoleg_code"])
def test_script_cli(monkeypatch, mod):
    patch_common(monkeypatch)
    check_cli_invocation(monkeypatch, mod)
