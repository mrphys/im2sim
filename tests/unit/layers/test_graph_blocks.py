import pytest
import torch
import torch_geometric as pyg

from im2sim.layers.graph_blocks import GraphConvBlock
from im2sim.configs.core import LayerConfig
from im2sim.configs.graph_blocks import GraphConvBlockConfig


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def graph():
    """Small deterministic graph used throughout the tests."""
    return pyg.data.Data(
        x=torch.randn(10, 16),
        edge_index=torch.tensor(
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
            ],
            dtype=torch.long,
        ),
    )


@pytest.fixture
def default_cfg():
    return GraphConvBlockConfig()


def make_cfg(**kwargs):
    """Create a config while keeping the test cases concise."""
    return GraphConvBlockConfig(**kwargs)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_default_construction(default_cfg):
    """The block should construct successfully with its default config."""
    block = GraphConvBlock(
        in_channels=16,
        out_channels=32,
        cfg=default_cfg,
    )

    assert block.in_channels == 16
    assert block.out_channels == 32
    assert block.depth == 1
    assert len(block.layers) == 1


@pytest.mark.parametrize("depth", [1, 2, 3, 4])
def test_depth(depth):
    """The number of generated layers should equal the configured depth."""
    cfg = make_cfg(depth=depth)

    block = GraphConvBlock(
        in_channels=16,
        out_channels=32,
        cfg=cfg,
    )

    assert len(block.layers) == depth


def test_default_configs_are_set():
    """
    None configs should be replaced by the appropriate defaults.

    This is particularly useful if users construct configs manually with
    conv_cfg/norm_cfg/etc. set to None.
    """
    cfg = GraphConvBlockConfig(
        conv_cfg=None,
        norm_cfg=None,
        dropout_cfg=None,
        attn_cfg=None,
    )

    block = GraphConvBlock(
        in_channels=16,
        out_channels=32,
        cfg=cfg,
    )

    assert block.conv_cfg is not None
    assert block.norm_cfg is not None
    assert block.dropout_cfg is not None
    assert block.attn_cfg is not None


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def test_forward_output_shape(graph):
    """Output should have the requested number of channels."""
    cfg = make_cfg(
        depth=3,
        activation="ReLU",
    )

    block = GraphConvBlock(
        in_channels=16,
        out_channels=32,
        cfg=cfg,
    )

    output = block(graph)

    assert output.x.shape == (10, 32)
    assert output.edge_index.shape == graph.edge_index.shape


def test_forward_does_not_modify_input_graph(graph):
    """forward() clones the input graph, so the input should remain unchanged."""
    original_x = graph.x.clone()
    original_edge_index = graph.edge_index.clone()

    cfg = make_cfg(depth=2)

    block = GraphConvBlock(
        in_channels=16,
        out_channels=32,
        cfg=cfg,
    )

    _ = block(graph)

    assert torch.equal(graph.x, original_x)
    assert torch.equal(graph.edge_index, original_edge_index)


def test_edge_index_is_preserved(graph):
    """The output should retain the input edge_index."""
    cfg = make_cfg(depth=2)

    block = GraphConvBlock(
        in_channels=16,
        out_channels=32,
        cfg=cfg,
    )

    output = block(graph)

    assert torch.equal(output.edge_index, graph.edge_index)


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "activation",
    ["ReLU", "LeakyReLU", None],
)
def test_activation_options(graph, activation):
    cfg = make_cfg(
        depth=2,
        activation=activation,
    )

    block = GraphConvBlock(
        in_channels=16,
        out_channels=32,
        cfg=cfg,
    )

    output = block(graph)

    assert output.x.shape == (10, 32)
    assert torch.isfinite(output.x).all()


def test_output_activation(graph):
    """Sigmoid output activation should constrain values to [0, 1]."""
    cfg = make_cfg(
        depth=2,
        activation="ReLU",
        out_activation="sigmoid",
    )

    block = GraphConvBlock(
        in_channels=16,
        out_channels=32,
        cfg=cfg,
    )

    output = block(graph)

    assert torch.all(output.x >= 0)
    assert torch.all(output.x <= 1)


# ---------------------------------------------------------------------------
# Different convolution layers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "conv_name",
    ["GCNConv", "GATConv"],
)
def test_different_graph_convolutions(graph, conv_name):
    cfg = make_cfg(
        depth=2,
        conv_cfg=LayerConfig(
            name=conv_name,
            kwargs={},
        ),
    )

    block = GraphConvBlock(
        in_channels=16,
        out_channels=32,
        cfg=cfg,
    )

    output = block(graph)

    assert output.x.shape == (10, 32)
    assert torch.isfinite(output.x).all()


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def test_without_normalisation(graph):
    cfg = make_cfg(
        depth=2,
        norm_cfg=LayerConfig(
            name=None,
            kwargs={},
        ),
    )

    block = GraphConvBlock(
        in_channels=16,
        out_channels=32,
        cfg=cfg,
    )

    output = block(graph)

    assert output.x.shape == (10, 32)
    assert torch.isfinite(output.x).all()


def test_with_normalisation(graph):
    cfg = make_cfg(
        depth=2,
        norm_cfg=LayerConfig(
            name="DefaultGraphNorm",
            kwargs={},
        ),
    )

    block = GraphConvBlock(
        in_channels=16,
        out_channels=32,
        cfg=cfg,
    )

    output = block(graph)

    assert output.x.shape == (10, 32)
    assert torch.isfinite(output.x).all()


# ---------------------------------------------------------------------------
# Dropout
# ---------------------------------------------------------------------------

def test_dropout_can_be_enabled(graph):
    cfg = make_cfg(
        depth=3,
        dropout_cfg=LayerConfig(
            name="EdgeDropout",
            kwargs={"p": 0.5},
        ),
        dropout_position=[1],
    )

    block = GraphConvBlock(
        in_channels=16,
        out_channels=32,
        cfg=cfg,
    )

    output = block(graph)

    assert output.x.shape == (10, 32)
    assert torch.isfinite(output.x).all()


def test_multiple_dropout_positions(graph):
    cfg = make_cfg(
        depth=4,
        dropout_cfg=LayerConfig(
            name="EdgeDropout",
            kwargs={"p": 0.2},
        ),
        dropout_position=[1, 3],
    )

    block = GraphConvBlock(
        in_channels=16,
        out_channels=32,
        cfg=cfg,
    )

    output = block(graph)

    assert output.x.shape == (10, 32)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "attention",
    ["EfficientChannelAttn", "SqueezeExcite"],
)
def test_attention(graph, attention):
    cfg = make_cfg(
        depth=2,
        attn_cfg=LayerConfig(
            name=attention,
            kwargs={},
        ),
    )

    block = GraphConvBlock(
        in_channels=16,
        out_channels=32,
        cfg=cfg,
    )

    output = block(graph)

    assert output.x.shape == (10, 32)
    assert torch.isfinite(output.x).all()


@pytest.mark.parametrize(
    "attention",
    ["InvalidAttention", "MultiHeadAttention"],
)
def test_invalid_attention_raises(attention):
    cfg = make_cfg(
        attn_cfg=LayerConfig(
            name=attention,
            kwargs={},
        ),
    )

    with pytest.raises(AssertionError, match="Unsupported attention type"):
        GraphConvBlock(
            in_channels=16,
            out_channels=32,
            cfg=cfg,
        )


# ---------------------------------------------------------------------------
# Residual connections
# ---------------------------------------------------------------------------

def test_add_input_residual(graph):
    cfg = make_cfg(
        depth=2,
        residual_connections={1: [0]},
        residual_type="add",
    )

    block = GraphConvBlock(
        in_channels=16,
        out_channels=32,
        cfg=cfg,
    )

    with pytest.raises((RuntimeError, ValueError)):
        block(graph)



def test_residual_from_previous_layer(graph):
    """
    Test a residual connection between two layers with matching output
    channel dimensions.
    """
    cfg = make_cfg(
        depth=3,
        residual_connections={2: [1]},
        residual_type="add",
    )

    block = GraphConvBlock(
        in_channels=16,
        out_channels=32,
        cfg=cfg,
    )

    output = block(graph)

    assert output.x.shape == (10, 32)
    assert torch.isfinite(output.x).all()


def test_concat_residual(graph):
    """
    Concatenation should increase the input channel count to the target
    convolution.
    """
    cfg = make_cfg(
        depth=3,
        residual_connections={2: [1, 0]},
        residual_type="concat",
    )

    block = GraphConvBlock(
        in_channels=16,
        out_channels=32,
        cfg=cfg,
    )

    output = block(graph)

    assert output.x.shape == (10, 32)
    assert torch.isfinite(output.x).all()


def test_concat_residual_changes_conv_input_channels():
    """
    Verify the channel bookkeeping performed during construction.
    """
    cfg = make_cfg(
        depth=3,
        residual_connections={2: [1, 0]},
        residual_type="concat",
    )

    block = GraphConvBlock(
        in_channels=16,
        out_channels=32,
        cfg=cfg,
    )

    # Layer 0: 16 -> 32
    # Layer 1: 32 -> 32
    # Layer 2 receives:
    #   output[1]*2 = 32*2= 64
    #   output[0] = 16
    # therefore 80 channels.
    if hasattr(block.layers[2][0], "pyg_module"):
        assert block.layers[2][0].pyg_module.in_channels == 80
    elif hasattr(block.layers[2][0], "in_channels"):
        assert block.layers[2][0].in_channels == 80
    else:
        pytest.skip("Cannot determine in_channels for the last layer; skipping test.")


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------

def test_invalid_concat_residual_target():
    cfg = make_cfg(
        depth=3,
        residual_connections={3: [0]},
        residual_type="concat",
    )

    with pytest.raises(
        ValueError,
        match="cannot be created on the last layer",
    ):
        GraphConvBlock(
            in_channels=16,
            out_channels=32,
            cfg=cfg,
        )


def test_dropout_position_validation():
    cfg = make_cfg(
        depth=3,
        dropout_cfg=LayerConfig(
            name="EdgeDropout",
            kwargs={"p": 0.5},
        ),
        dropout_position=[3],
    )

    # Current implementation checks max(position) < depth, so this is invalid.
    with pytest.raises(AssertionError):
        GraphConvBlock(
            in_channels=16,
            out_channels=32,
            cfg=cfg,
        )


# ---------------------------------------------------------------------------
# Gradient tests
# ---------------------------------------------------------------------------

def test_input_gradients(graph):
    """Gradients should propagate all the way back to the input."""
    graph.x.requires_grad_(True)

    cfg = make_cfg(
        depth=3,
        activation="LeakyReLU",
        out_activation="sigmoid",
    )

    block = GraphConvBlock(
        in_channels=16,
        out_channels=32,
        cfg=cfg,
    )

    output = block(graph)
    loss = output.x.sum()

    loss.backward()

    assert graph.x.grad is not None
    assert torch.isfinite(graph.x.grad).all()


def test_parameter_gradients(graph):
    """All trainable parameters participating in the forward pass should
    receive finite gradients."""
    graph.x.requires_grad_(True)

    cfg = make_cfg(
        depth=3,
        activation="LeakyReLU",
    )

    block = GraphConvBlock(
        in_channels=16,
        out_channels=32,
        cfg=cfg,
    )

    output = block(graph)
    loss = output.x.square().mean()

    loss.backward()

    trainable_parameters = [
        (name, parameter)
        for name, parameter in block.named_parameters()
        if parameter.requires_grad
    ]

    assert trainable_parameters

    for name, parameter in trainable_parameters:
        assert parameter.grad is not None, (
            f"Parameter {name} has no gradient"
        )
        assert torch.isfinite(parameter.grad).all(), (
            f"Parameter {name} has NaN/Inf gradient"
        )


def test_gradients_with_attention_and_residuals(graph):
    """A more complex configuration should still be differentiable."""
    graph.x.requires_grad_(True)

    cfg = make_cfg(
        depth=3,
        activation="LeakyReLU",
        attn_cfg=LayerConfig(
            name="SqueezeExcite",
            kwargs={},
        ),
        residual_connections={2: [1]},
        residual_type="add",
    )

    block = GraphConvBlock(
        in_channels=16,
        out_channels=32,
        cfg=cfg,
    )

    output = block(graph)
    loss = output.x.square().mean()

    loss.backward()

    assert graph.x.grad is not None
    assert torch.isfinite(graph.x.grad).all()

    for name, parameter in block.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, (
                f"{name} gradient is None"
            )
            assert torch.isfinite(parameter.grad).all(), (
                f"{name} gradient contains NaN/Inf"
            )


# ---------------------------------------------------------------------------
# Train / eval behaviour
# ---------------------------------------------------------------------------

def test_train_eval_modes(graph):
    """The block should support the standard PyTorch train/eval modes."""
    cfg = make_cfg(
        depth=2,
        dropout_cfg=LayerConfig(
            name="EdgeDropout",
            kwargs={"p": 0.5},
        ),
        dropout_position=[1],
    )

    block = GraphConvBlock(
        in_channels=16,
        out_channels=32,
        cfg=cfg,
    )

    block.train()
    assert block.training

    output_train = block(graph)

    block.eval()
    assert not block.training

    output_eval = block(graph)

    assert output_train.x.shape == output_eval.x.shape
    assert torch.isfinite(output_train.x).all()
    assert torch.isfinite(output_eval.x).all()


# ---------------------------------------------------------------------------
# State dict
# ---------------------------------------------------------------------------

def test_state_dict_round_trip(graph):
    """Saving and loading the state_dict should reproduce the output."""
    cfg = make_cfg(
        depth=2,
        activation="LeakyReLU",
    )

    block1 = GraphConvBlock(
        in_channels=16,
        out_channels=32,
        cfg=cfg,
    )

    block2 = GraphConvBlock(
        in_channels=16,
        out_channels=32,
        cfg=cfg,
    )

    block1.eval()
    block2.eval()

    output1 = block1(graph).x

    block2.load_state_dict(block1.state_dict())

    output2 = block2(graph).x

    assert torch.allclose(output1, output2)

