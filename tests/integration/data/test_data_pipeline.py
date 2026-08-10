import copy

import pytest
import torch
from torch_geometric.data import Data

from im2sim.src.data import Dataset, Pipeline, transforms


def make_toy_dataset():
    """
    Mimics:
      input.x -> model inputs
      gt.x    -> targets

    but with tiny synthetic data.
    """

    samples = {}
    cases = ["1", "2", "3", "4", "5"]
    for case in cases:
        num_nodes = 10

        sample = {
            "input": Data(
                x=torch.randn(num_nodes, 7),
                edge_index=torch.tensor(
                    [
                        [0, 1, 2, 3],
                        [1, 2, 3, 4],
                    ]
                ),
            ),
            "gt": Data(
                x=torch.randn(num_nodes, 4) * 10,
                edge_index=torch.tensor(
                    [
                        [0, 1, 2, 3],
                        [1, 2, 3, 4],
                    ]
                ),
            ),
        }

        samples[case] = sample

    def load(case):
        return samples[case]

    ds = Dataset(load, cases)

    return ds


@pytest.fixture
def pipeline():

    dataset = make_toy_dataset()

    pipeline = Pipeline(
        [
            transforms.PowerScaling(
                exp=1 / 3,
                preserve_sign=True,
                keys=["gt"],
                attr="x",
                channels=[0],
            ),
            transforms.FitZScore(
                keys=["gt"],
                attr="x",
                channels=[0, 1, 2, 3],
                per_channel=True,
            ),
            transforms.FitZScore(
                keys=["input"],
                attr="x",
                channels=[0, 1, 2, 3, 4, 5, 6],
                per_channel=True,
            ),
        ]
    )

    pipeline.fit(dataset)

    return pipeline


def test_pipeline_forward(pipeline):

    sample = make_toy_dataset()[0]

    transformed = pipeline(sample)

    assert transformed["input"].x.shape == (10, 7)
    assert transformed["gt"].x.shape == (10, 4)

    assert torch.isfinite(transformed["input"].x).all()
    assert torch.isfinite(transformed["gt"].x).all()


def test_pipeline_inverse(pipeline):

    sample = make_toy_dataset()[0]

    # keep a copy before modification
    original = {key: value.x.clone() for key, value in sample.items()}

    transformed = pipeline(sample)

    recovered = pipeline.inverse(transformed)

    for key in ["input", "gt"]:
        torch.testing.assert_close(
            recovered[key].x,
            original[key],
            rtol=1e-5,
            atol=1e-6,
        )


def test_integrated_pipeline(pipeline):

    dataset1 = make_toy_dataset()
    dataset2 = copy.deepcopy(dataset1)

    dataset1.add_transforms(pipeline)

    sample1 = dataset1[0]
    recovered = pipeline.inverse(sample1)

    sample2 = dataset2[0]
    transformed = pipeline(sample2)

    for key in ["input", "gt"]:
        torch.testing.assert_close(
            recovered[key].x,
            sample2[key].x,
            rtol=1e-5,
            atol=1e-6,
        )

        torch.testing.assert_close(
            sample1[key].x,
            transformed[key].x,
            rtol=1e-5,
            atol=1e-6,
        )
