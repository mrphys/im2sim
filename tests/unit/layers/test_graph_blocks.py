import pytest
import torch
from torch_geometric.data import Data

from im2sim.src.layers import GraphConvBlock, GraphConvResBlock


@pytest.fixture
def graph():
    """
    Simple graph:
        0 -- 1
        |    |
        2 -- 3
    """
    x = torch.randn(4, 8)

    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 0, 2],
            [1, 0, 3, 2, 2, 0],
        ],
        dtype=torch.long,
    )

    return Data(
        x=x,
        edge_index=edge_index,
    )


graph_blocks = {"GraphConv": GraphConvBlock, "GraphRes": GraphConvResBlock}


@pytest.mark.parametrize(
    "conv_type",
    [
        "GCNConv",
        "GATConv",
    ],
)
@pytest.mark.parametrize(
    "graph_block_type",
    [
        "GraphConv",
        "GraphRes",
    ],
)
def test_forward_shape(graph, conv_type, graph_block_type):
    filters = 16

    model = graph_blocks[graph_block_type](
        in_channels=8,
        filters=filters,
        depth=2,
        conv_type=conv_type,
        conv_kwargs={},
    )

    print(type(model), type(graph))
    out = model(graph)

    assert out.x.shape == (graph.num_nodes, filters)


@pytest.mark.parametrize("depth", [1, 2, 4])
@pytest.mark.parametrize(
    "graph_block_type",
    [
        "GraphConv",
        "GraphRes",
    ],
)
def test_depth_changes_number_of_layers(depth, graph_block_type):
    model = graph_blocks[graph_block_type](
        in_channels=8,
        filters=16,
        depth=depth,
        conv_type="GCNConv",
        conv_kwargs={},
    )

    assert len(model.convs) == depth
    assert len(model.norms) == depth


@pytest.mark.parametrize(
    "graph_block_type",
    [
        "GraphConv",
        "GraphRes",
    ],
)
def test_graph_structure_is_preserved(graph, graph_block_type):
    model = graph_blocks[graph_block_type](
        in_channels=8,
        filters=16,
        conv_type="GCNConv",
        conv_kwargs={},
    )

    out = model(graph)

    assert torch.equal(
        out.edge_index,
        graph.edge_index,
    )


@pytest.mark.parametrize(
    "graph_block_type",
    [
        "GraphConv",
        "GraphRes",
    ],
)
def test_input_graph_is_not_modified(graph, graph_block_type):
    model = graph_blocks[graph_block_type](
        in_channels=8,
        filters=16,
        conv_type="GCNConv",
        conv_kwargs={},
    )

    original_x = graph.x.clone()

    _ = model(graph)

    assert torch.allclose(
        graph.x,
        original_x,
    )


@pytest.mark.parametrize(
    "graph_block_type",
    [
        "GraphConv",
        "GraphRes",
    ],
)
def test_output_is_finite(graph, graph_block_type):
    model = graph_blocks[graph_block_type](
        in_channels=8,
        filters=16,
        conv_type="GCNConv",
        conv_kwargs={},
    )

    out = model(graph)

    assert torch.isfinite(out.x).all()


@pytest.mark.parametrize(
    "graph_block_type",
    [
        "GraphConv",
        "GraphRes",
    ],
)
def test_gradients_flow(graph, graph_block_type):
    graph.x.requires_grad_(True)

    model = graph_blocks[graph_block_type](
        in_channels=8,
        filters=16,
        depth=2,
        conv_type="GCNConv",
        conv_kwargs={},
    )

    out = model(graph)

    loss = out.x.sum()
    loss.backward()

    assert graph.x.grad is not None
    assert torch.isfinite(graph.x.grad).all()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} has no gradient"
        assert torch.isfinite(param.grad).all()


@pytest.mark.parametrize(
    "graph_block_type",
    [
        "GraphConv",
        "GraphRes",
    ],
)
@pytest.mark.parametrize(
    "activation",
    ["ReLU", "leakyrelu", "gelu", "sigmoid", None],
)
def test_supported_activations(graph, activation, graph_block_type):
    model = graph_blocks[graph_block_type](
        in_channels=8,
        filters=16,
        activation=activation,
        conv_type="GCNConv",
        conv_kwargs={},
    )

    out = model(graph)

    assert out.x.shape == (4, 16)


@pytest.mark.parametrize(
    "graph_block_type",
    [
        "GraphConv",
        "GraphRes",
    ],
)
def test_no_normalisation(graph, graph_block_type):
    model = graph_blocks[graph_block_type](
        in_channels=8,
        filters=16,
        norm_type=None,
        conv_type="GCNConv",
        conv_kwargs={},
    )

    assert isinstance(
        model.norms[0],
        torch.nn.Identity,
    )

    out = model(graph)

    assert out.x.shape == (4, 16)


@pytest.mark.parametrize(
    "graph_block_type",
    [
        "GraphConv",
        "GraphRes",
    ],
)
def test_serialisation(graph, tmp_path, graph_block_type):
    model = graph_blocks[graph_block_type](
        in_channels=8,
        filters=16,
        conv_type="GCNConv",
        conv_kwargs={},
    )

    model.eval()

    before = model(graph).x

    path = tmp_path / "model.pt"

    torch.save(
        model.state_dict(),
        path,
    )

    model2 = graph_blocks[graph_block_type](
        in_channels=8,
        filters=16,
        conv_type="GCNConv",
        conv_kwargs={},
    )

    model2.load_state_dict(torch.load(path))

    model2.eval()

    after = model2(graph).x

    assert torch.allclose(
        before,
        after,
    )
