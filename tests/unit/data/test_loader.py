import pytest
import torch
from torch_geometric.data import Batch, Data

from im2sim.data import DataLoader, Dataset, collate

# -------------------------
# Fixtures
# -------------------------


@pytest.fixture
def tensor_sample():
    return {
        "x": torch.randn(3, 4),
        "y": torch.randn(1),
    }


@pytest.fixture
def pyg_sample():
    return {"graph": Data(x=torch.randn(5, 3), edge_index=torch.tensor([[0, 1], [1, 2]]))}


@pytest.fixture
def mixed_sample():
    return {
        "x": torch.randn(3, 4),
        "graph": Data(x=torch.randn(5, 3), edge_index=torch.tensor([[0, 1], [1, 2]])),
    }


# -------------------------
# collate tests
# -------------------------


def test_collate_tensors(tensor_sample):
    batch = [tensor_sample, tensor_sample]
    out = collate(batch)

    assert isinstance(out, dict)
    assert all(isinstance(v, torch.Tensor) for v in out.values())
    assert out["x"].shape[0] == 2  # batched dimension


def test_collate_pyg(pyg_sample):
    batch = [pyg_sample, pyg_sample]
    out = collate(batch)

    assert isinstance(out["graph"], Batch)
    assert out["graph"].num_graphs == 2


def test_collate_mixed(mixed_sample):
    batch = [mixed_sample, mixed_sample]
    out = collate(batch)

    assert isinstance(out["x"], torch.Tensor)
    assert isinstance(out["graph"], Batch)


def test_collate_invalid_type():
    batch = [{"bad": "not allowed"}, {"bad": "still not allowed"}]

    with pytest.raises(TypeError):
        collate(batch)


# -------------------------
# Dataset tests
# -------------------------


def dummy_load_fn(case: str):
    return {"x": torch.tensor([len(case)])}


def test_dataset_len():
    cases = ["a", "bb", "ccc"]
    ds = Dataset(load_fn=dummy_load_fn, cases=cases)

    assert len(ds) == 3


def test_dataset_getitem():
    cases = ["a"]
    ds = Dataset(load_fn=dummy_load_fn, cases=cases)

    sample = ds[0]
    assert isinstance(sample, dict)
    assert torch.equal(sample["x"], torch.tensor([1]))


def test_dataset_no_transforms():
    ds = Dataset(load_fn=dummy_load_fn, cases=["a"], transforms=None)
    sample = ds[0]

    assert "x" in sample  # should still work


# -------------------------
# DataLoader tests
# -------------------------


def test_dataloader_basic():
    ds = Dataset(load_fn=dummy_load_fn, cases=["a", "bb", "ccc"])

    loader = DataLoader(ds, batch_size=2)

    batch = next(iter(loader))
    assert isinstance(batch, dict)
    assert "x" in batch
    assert batch["x"].shape[0] == 2


def test_dataloader_with_pyg():
    def load_fn(case):
        return {"graph": Data(x=torch.randn(3, 2), edge_index=torch.tensor([[0, 1], [1, 2]]))}

    ds = Dataset(load_fn=load_fn, cases=["a", "b", "c"])
    loader = DataLoader(ds, batch_size=2)

    batch = next(iter(loader))
    assert isinstance(batch["graph"], Batch)
    assert batch["graph"].num_graphs == 2
