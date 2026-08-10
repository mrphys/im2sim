import copy

import pytest
import torch

from im2sim.src.data import (
    FittableOperation,
    InvertibleOperation,
    Operation,
    Pipeline,
    Transform,
    register_op,
)

# -----------------------------
# Helpers / Dummy Ops
# -----------------------------


class DummyOp(Operation):
    def forward(self, x):
        return x + 1


class DummyInvertibleOp(InvertibleOperation):
    def forward(self, x):
        return x + 1

    def inverse(self, x):
        return x - 1


@register_op
class DummyFittableOp(FittableOperation):
    def __init__(self):
        self.total = 0
        self.count = 0

    def fit_step(self, x):
        self.total += x.sum()
        self.count += x.numel()

    def complete_fit(self):
        self.mean = self.total / self.count

    def forward(self, x):
        return x - self.mean

    def inverse(self, x):
        return x + self.mean


# -----------------------------
# Transform Tests
# -----------------------------


def test_transform_single_key_forward():
    t = Transform(DummyOp(), keys="a")
    data = {"a": torch.tensor([1.0])}

    out = t.forward(data)
    assert torch.allclose(out["a"], torch.tensor([2.0]))


def test_transform_multiple_keys():
    t = Transform(DummyOp(), keys=["a", "b"])
    data = {"a": torch.tensor([1.0]), "b": torch.tensor([2.0])}

    out = t.forward(data)
    assert torch.allclose(out["a"], torch.tensor([2.0]))
    assert torch.allclose(out["b"], torch.tensor([3.0]))


def test_transform_missing_attr():
    t = Transform(DummyOp(), keys="a", attr="missing")
    data = {"a": object()}

    with pytest.raises(ValueError):
        t.forward(data)


# -----------------------------
# Channel Handling
# -----------------------------


def test_transform_channel_subset():
    t = Transform(DummyOp(), keys="a", channels=[0], channel_dim=0)
    data = {"a": torch.tensor([[1.0], [2.0]])}

    out = t.forward(data)
    assert out["a"][0] == 2.0
    assert out["a"][1] == 2.0  # unchanged


def test_transform_per_channel():
    t = Transform(DummyOp(), keys="a", per_channel=True, channel_dim=0)
    data = {"a": torch.tensor([[1.0], [2.0]])}

    out = t.forward(data)
    assert torch.allclose(out["a"], torch.tensor([[2.0], [3.0]]))


# -----------------------------
# Fittable Transform
# -----------------------------


def test_fittable_requires_fit():
    t = Transform(DummyFittableOp(), keys="a")
    data = {"a": torch.tensor([1.0])}

    with pytest.raises(RuntimeError):
        t.forward(data)


def test_fittable_fit_and_forward():
    t = Transform(DummyFittableOp(), keys="a")

    dataset = [{"a": torch.tensor([1.0])}, {"a": torch.tensor([3.0])}]

    class DummyLoader:
        def __iter__(self):
            return iter(dataset)

    t.fit(DummyLoader())

    out = t.forward({"a": torch.tensor([3.0])})
    assert torch.allclose(out["a"], torch.tensor([3.0 - 2.0]))


# -----------------------------
# Invertibility
# -----------------------------


def test_invertible_transform():
    t = Transform(DummyInvertibleOp(), keys="a")
    t.is_invertible = True  # simulate isinstance check

    data = {"a": torch.tensor([5.0])}

    out = t.forward(copy.deepcopy(data))
    inv = t.inverse(out)

    assert torch.allclose(inv["a"], data["a"])


def test_non_invertible_raises():
    t = Transform(DummyOp(), keys="a")
    data = {"a": torch.tensor([1.0])}

    with pytest.raises(RuntimeError):
        t.inverse(data)


# -----------------------------
# Pipeline Tests
# -----------------------------


def test_pipeline_sequential():
    t1 = Transform(DummyOp(), keys="a")
    t2 = Transform(DummyOp(), keys="a")

    p = Pipeline([t1, t2])
    data = {"a": torch.tensor([1.0])}

    out = p(data)
    assert torch.allclose(out["a"], torch.tensor([3.0]))


def test_pipeline_inverse_order():
    t1 = Transform(DummyInvertibleOp(), keys="a")
    t1.is_invertible = True
    t2 = Transform(DummyInvertibleOp(), keys="a")
    t2.is_invertible = True

    p = Pipeline([t1, t2])
    data = {"a": torch.tensor([1.0])}

    out = p(copy.deepcopy(data))
    inv = p.inverse(out)

    assert torch.allclose(inv["a"], data["a"])


def test_pipeline_skips_missing_keys():
    t = Transform(DummyOp(), keys="b")
    p = Pipeline([t])

    data = {"a": torch.tensor([1.0])}
    out = p(data)

    assert "a" in out
    assert "b" not in out


# -----------------------------
# Pipeline Fit
# -----------------------------


def test_pipeline_fit_only_fittable():
    t1 = Transform(DummyOp(), keys="a")
    t2 = Transform(DummyFittableOp(), keys="a")

    dataset = [{"a": torch.tensor([1.0])}, {"a": torch.tensor([3.0])}]

    class DummyDataset:
        def __init__(self):
            self.data = dataset
            self.transforms = []

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            for t in self.transforms:
                item = t(item)
            return item

    p = Pipeline([t1, t2])
    p.fit(DummyDataset())

    assert t2.fitted is True


# -----------------------------
# Serialization
# -----------------------------


def test_transform_state_dict_roundtrip():
    t = Transform(DummyFittableOp(), keys="a")

    dataset = [{"a": torch.tensor([2.0])}]

    class DummyLoader:
        def __iter__(self):
            return iter(dataset)

    t.fit(DummyLoader())
    state = t.state_dict()

    t2 = Transform(DummyFittableOp(), keys="a")
    t2.load_state_dict(state)

    assert t2.fitted is True


def test_pipeline_state_dict_roundtrip(tmp_path):
    t = Transform(DummyFittableOp(), keys="a")

    dataset = [{"a": torch.tensor([2.0])}]

    class DummyLoader:
        def __iter__(self):
            return iter(dataset)

    t.fit(DummyLoader())

    p = Pipeline([t])
    path = tmp_path / "pipeline.pt"

    torch.save({"config": p.config(), "state": p.state_dict()}, path)

    obj = torch.load(path)
    p2 = Pipeline.from_config(obj["config"])
    p2.load_state_dict(obj["state"])

    assert list(p2.state_dict().keys()) == list(p.state_dict().keys())


# -----------------------------
# Edge Cases
# -----------------------------


def test_empty_pipeline():
    p = Pipeline([])
    data = {"a": torch.tensor([1.0])}

    out = p(data)
    assert out == data


def test_no_keys_transform():
    with pytest.raises(ValueError):
        Transform(DummyOp(), keys=[])
