import torch

from im2sim.configs import ImageConvBlockConfig
from im2sim.configs.core import LayerConfig
from im2sim.layers import ImageConvBlock


def test_json_round_trip_keeps_int_dict_keys(tmp_path):
    cfg = ImageConvBlockConfig(depth=3, residual_connections={2: [0]}, residual_type="add")
    path = tmp_path / "cfg.json"
    cfg.save(path)
    loaded = ImageConvBlockConfig.load(path)

    assert loaded.residual_connections == {2: [0]}


def test_reloaded_config_builds_the_same_model(tmp_path):
    cfg = ImageConvBlockConfig(depth=3, residual_connections={2: [0]}, residual_type="add")
    path = tmp_path / "cfg.json"
    cfg.save(path)
    loaded = ImageConvBlockConfig.load(path)

    x = torch.randn(1, 4, 8, 8)
    torch.manual_seed(0)
    original = ImageConvBlock(4, 4, 2, cfg)
    torch.manual_seed(0)
    restored = ImageConvBlock(4, 4, 2, loaded)

    assert torch.allclose(original(x), restored(x))


def test_layer_config_kwargs_default_to_empty_dict():
    assert LayerConfig(name="Conv").kwargs == {}
    assert LayerConfig(name="Conv", kwargs=None).kwargs == {}


def test_layer_config_default_kwargs_are_not_shared():
    a, b = LayerConfig(name="Conv"), LayerConfig(name="Conv")
    a.kwargs["kernel_size"] = 3
    assert b.kwargs == {}


def test_layer_config_without_kwargs_builds_a_block():
    cfg = ImageConvBlockConfig(depth=1, conv_cfg=LayerConfig(name="Conv"))
    cfg.conv_cfg.kwargs.update({"kernel_size": 3, "padding": "same"})
    assert ImageConvBlock(2, 2, 2, cfg)(torch.randn(1, 2, 8, 8)).shape == (1, 2, 8, 8)
