import torch
from torch.autograd import gradcheck

from im2sim.layers.custom_graph_layers import DefaultGraphNorm


def test_default_graph_norm_gradcheck():
    torch.manual_seed(42)

    norm = DefaultGraphNorm(in_channels=4).double()

    x = torch.randn(
        8,
        4,
        dtype=torch.double,
        requires_grad=True,
    )

    batch = torch.tensor(
        [0, 0, 0, 0, 1, 1, 1, 1],
        dtype=torch.long,
    )

    assert gradcheck(
        lambda x: norm(x, batch),
        (x,),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
    )