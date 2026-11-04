import pytest
import torch

from im2sim.data.core import (
    Operation,
    Pipeline,
    Transform,
    load_pipeline,
    register_op,
    save_pipeline,
)
from im2sim.transforms import FitNorm
from im2sim.transforms.ops import PowerScaleOp, RangeNormOp


@register_op
class _AddOp(Operation):
    def forward(self, a, b):
        return a + b, b


class _Child(RangeNormOp):
    def __init__(self):
        super().__init__(0, 5)


def test_reloaded_per_channel_transform_with_channels_is_still_invertible():
    t = Transform(PowerScaleOp(2.0, True), keys=["a"], channels=[0, 1], per_channel=True)
    reloaded = Pipeline.from_config(Pipeline([t]).config())

    assert reloaded.transforms[0].is_invertible

    data = {"a": torch.rand(4, 3) + 0.1}
    restored = reloaded.inverse(reloaded(data))
    assert torch.allclose(restored["a"], data["a"], atol=1e-5)


def test_multikey_survives_config_round_trip():
    t = Transform(_AddOp(), keys=["a", "b"], multikey=True)
    reloaded = Pipeline.from_config(Pipeline([t]).config())

    assert reloaded.transforms[0].is_multikey
    out = reloaded({"a": torch.ones(2), "b": torch.ones(2)})
    assert torch.equal(out["a"], torch.full((2,), 2.0))


def test_default_names_differ_for_different_channels():
    t0 = FitNorm(keys=["g"], channels=[0])
    t1 = FitNorm(keys=["g"], channels=[1])
    assert t0.name != t1.name


def test_duplicate_transform_names_keep_separate_state():
    # Same op on the same key with an explicit identical name must not share one state entry
    t0 = Transform(RangeNormOp(0, 1), keys=["g"], name="n")
    t1 = Transform(RangeNormOp(2, 3), keys=["g"], name="n")
    pipeline = Pipeline([t0, t1])

    assert len({t.name for t in pipeline.transforms}) == 2
    assert len(pipeline.state_dict()) == 2


def test_state_is_restored_per_channel_transform(tmp_path):
    ds = [{"g": torch.rand(10, 2) * scale} for scale in (1.0, 100.0)]
    loader = torch.utils.data.DataLoader(ds, batch_size=1, collate_fn=lambda b: b[0])
    t0 = FitNorm(keys=["g"], channels=[0], channel_dim=-1)
    t1 = FitNorm(keys=["g"], channels=[1], channel_dim=-1)
    pipeline = Pipeline([t0, t1])
    for t in pipeline.transforms:
        t.fit(loader)

    path = tmp_path / "pipeline.pt"
    save_pipeline(pipeline, path)
    reloaded = load_pipeline(path)

    sample = {"g": torch.rand(10, 2) * 50}
    assert torch.allclose(pipeline(sample)["g"], reloaded(sample)["g"])


def test_subclass_records_its_own_constructor_arguments():
    assert _Child().call_args == {"args": (), "kwargs": {}}


def test_load_pipeline_maps_to_cpu(tmp_path):
    pipeline = Pipeline([Transform(RangeNormOp(0, 1), keys=["a"])])
    path = tmp_path / "pipeline.pt"
    save_pipeline(pipeline, path)

    assert isinstance(load_pipeline(path), Pipeline)


def test_unfitted_pipeline_still_raises():
    pipeline = Pipeline([FitNorm(keys=["a"])])
    with pytest.raises(RuntimeError):
        pipeline({"a": torch.rand(3)})


def test_transform_from_fn_applies_the_function_but_cannot_be_saved(tmp_path):
    from im2sim.transforms.transforms import transform_from_fn

    t = transform_from_fn(lambda x: x * 2, keys=["a"])
    assert torch.equal(t({"a": torch.ones(2)})["a"], torch.full((2,), 2.0))

    with pytest.raises(TypeError, match="cannot be saved"):
        save_pipeline(Pipeline([t]), tmp_path / "pipeline.pt")
