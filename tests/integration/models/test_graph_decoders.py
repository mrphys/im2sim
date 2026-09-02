# tests/models/test_simple_graph_decoder.py

import pytest
import torch
import torch_geometric as pyg

from im2sim.models.graph_decoders import (
    SimpleGraphDecoder,
    SimpleGraphDecoderConfig,
)
from im2sim.configs.graph_blocks import GraphConvBlockConfig


@pytest.fixture
def graph():
    return pyg.data.Data(
        x=torch.randn(6, 4),
        edge_index=torch.tensor([
            [0, 1, 2, 3, 4],
            [1, 2, 3, 4, 5],
        ]),
    )


@pytest.fixture
def block_cfg():
    return GraphConvBlockConfig(
        # Use whatever minimal configuration is required
        # by your actual GraphConvBlockConfig.
        depth=1,
        activation="ReLU",
    )


def test_update_protocol(graph, block_cfg):
    """SimpleGraphDecoder with update protocol modifies graph.x."""

    decoder_cfg = SimpleGraphDecoderConfig(
        protocol="update",
        pred_feature_key="x",
    )

    decoder = SimpleGraphDecoder(
        in_channels=4,
        out_channels=4,
        cfg=decoder_cfg,
    )

    original_x = graph.x.clone()

    output = decoder(graph)

    assert output.x.shape == graph.x.shape

    # The updater should have modified the graph.
    assert not torch.allclose(output.x, original_x)

    # Input graph must remain unchanged.
    assert torch.equal(graph.x, original_x)


def test_predict_protocol(graph, block_cfg):
    """SimpleGraphDecoder with predict protocol replaces graph.x."""

    decoder_cfg = SimpleGraphDecoderConfig(
        protocol="predict",
        pred_feature_key="x",
    )

    decoder = SimpleGraphDecoder(
        in_channels=4,
        out_channels=4,
        cfg=decoder_cfg,
    )

    original_x = graph.x.clone()

    output = decoder(graph)

    assert output.x.shape == graph.x.shape
    assert not torch.allclose(output.x, original_x)

    # Input is unchanged.
    assert torch.equal(graph.x, original_x)


def test_projected_features_are_used(graph):
    """Projected image features are concatenated before the GNN."""

    projected_features = torch.randn(
        graph.num_nodes,
        8,
        requires_grad=True,
    )

    decoder_cfg = SimpleGraphDecoderConfig(
        protocol="predict",
        pred_feature_key="x",
    )

    decoder = SimpleGraphDecoder(
        in_channels=4 + 8,
        out_channels=4,
        cfg=decoder_cfg,
    )

    output = decoder(
        graph,
        projected_features=projected_features,
    )

    # Decoder removes the temporary concatenated channels.
    assert output.x.shape == graph.x.shape

    # Make sure the projected features participate in the computation.
    loss = output.x.sum()
    loss.backward()

    assert projected_features.grad is not None
    assert torch.isfinite(projected_features.grad).all()


def test_projected_features_have_correct_number_of_nodes(
    graph,
):
    """Each graph node must have a corresponding image feature."""

    projected_features = torch.randn(
        graph.num_nodes + 1,
        8,
    )

    decoder_cfg = SimpleGraphDecoderConfig(
        protocol="predict",
        pred_feature_key="x",
    )

    decoder = SimpleGraphDecoder(
        in_channels=12,
        out_channels=4,
        cfg=decoder_cfg,
    )

    with pytest.raises(RuntimeError):
        decoder(
            graph,
            projected_features=projected_features,
        )


def test_predict_selected_channels(graph):
    """Only pred_feature_channels are replaced by the predictor."""

    graph.pred = torch.randn(6, 6)

    decoder_cfg = SimpleGraphDecoderConfig(
        protocol="predict",
        pred_feature_key="pred",
        pred_feature_channels=[1, 3],
    )

    decoder = SimpleGraphDecoder(
        in_channels=6,
        out_channels=2,
        cfg=decoder_cfg,
    )

    original_pred = graph.pred.clone()

    output = decoder(graph)

    assert output.pred.shape == graph.pred.shape

    # Channels not being predicted should be untouched.
    assert torch.equal(
        output.pred[:, [0, 2, 4, 5]],
        original_pred[:, [0, 2, 4, 5]],
    )

    # Selected channels should have been changed.
    assert not torch.allclose(
        output.pred[:, [1, 3]],
        original_pred[:, [1, 3]],
    )


def test_update_selected_channels(graph):
    """Only pred_feature_channels are updated by the updater."""

    graph.pred = torch.randn(6, 6)

    decoder_cfg = SimpleGraphDecoderConfig(
        protocol="update",
        pred_feature_key="pred",
        pred_feature_channels=[1, 3],
    )

    decoder = SimpleGraphDecoder(
        in_channels=6,
        out_channels=2,
        cfg=decoder_cfg,
    )

    original_pred = graph.pred.clone()

    output = decoder(graph)

    assert output.pred.shape == graph.pred.shape

    # Non-selected channels are unchanged.
    assert torch.equal(
        output.pred[:, [0, 2, 4, 5]],
        original_pred[:, [0, 2, 4, 5]],
    )

    # Selected channels are updated.
    assert not torch.allclose(
        output.pred[:, [1, 3]],
        original_pred[:, [1, 3]],
    )


def test_include_ids(graph):
    """Only nodes specified by include_ids are modified."""

    graph.include_nodes = torch.tensor([1, 3, 5])

    decoder_cfg = SimpleGraphDecoderConfig(
        protocol="predict",
        pred_feature_key="x",
        include_ids=["include_nodes"],
    )

    decoder = SimpleGraphDecoder(
        in_channels=4,
        out_channels=4,
        cfg=decoder_cfg,
    )

    original_x = graph.x.clone()

    output = decoder(graph)

    # Included nodes should change.
    assert not torch.allclose(
        output.x[[1, 3, 5]],
        original_x[[1, 3, 5]],
    )

    # Other nodes should remain unchanged.
    assert torch.equal(
        output.x[[0, 2, 4]],
        original_x[[0, 2, 4]],
    )


def test_exclude_ids(graph):
    """Nodes specified by exclude_ids are not modified."""

    graph.exclude_nodes = torch.tensor([1, 3, 5])

    decoder_cfg = SimpleGraphDecoderConfig(
        protocol="predict",
        pred_feature_key="x",
        exclude_ids=["exclude_nodes"],
    )

    decoder = SimpleGraphDecoder(
        in_channels=4,
        out_channels=4,
        cfg=decoder_cfg,
    )

    original_x = graph.x.clone()

    output = decoder(graph)

    # Excluded nodes should remain unchanged.
    assert torch.equal(
        output.x[[1, 3, 5]],
        original_x[[1, 3, 5]],
    )

    # Other nodes should change.
    assert not torch.allclose(
        output.x[[0, 2, 4]],
        original_x[[0, 2, 4]],
    )


def test_include_takes_precedence_over_exclude(graph):
    """If both are supplied, include_ids should take precedence."""

    graph.include_nodes = torch.tensor([0, 1])
    graph.exclude_nodes = torch.tensor([0, 1, 2, 3, 4, 5])

    decoder_cfg = SimpleGraphDecoderConfig(
        protocol="predict",
        pred_feature_key="x",
        include_ids=["include_nodes"],
        exclude_ids=["exclude_nodes"],
    )

    decoder = SimpleGraphDecoder(
        in_channels=4,
        out_channels=4,
        cfg=decoder_cfg,
    )

    original_x = graph.x.clone()

    output = decoder(graph)

    # include_ids wins.
    assert not torch.allclose(
        output.x[[0, 1]],
        original_x[[0, 1]],
    )

    assert torch.equal(
        output.x[[2, 3, 4, 5]],
        original_x[[2, 3, 4, 5]],
    )


def test_non_x_prediction_key(graph):
    """Decoder can predict a graph attribute other than x."""

    graph.pred = torch.zeros(6, 2)

    decoder_cfg = SimpleGraphDecoderConfig(
        protocol="predict",
        pred_feature_key="pred",
        pred_feature_channels=[0, 1],
    )

    # x (4) + pred (2) are fed to the GNN.
    decoder = SimpleGraphDecoder(
        in_channels=6,
        out_channels=2,
        cfg=decoder_cfg,
    )

    output = decoder(graph)

    assert output.pred.shape == (6, 2)

    # x should retain its original dimensions.
    assert output.x.shape == (6, 4)

    # Prediction should have been written to pred.
    assert not torch.allclose(
        output.pred,
        graph.pred,
    )


def test_non_x_update_key(graph):
    """Decoder can update a graph attribute other than x."""

    graph.pred = torch.zeros(6, 2)

    decoder_cfg = SimpleGraphDecoderConfig(
        protocol="update",
        pred_feature_key="pred",
        pred_feature_channels=[0, 1],
    )

    decoder = SimpleGraphDecoder(
        in_channels=6,
        out_channels=2,
        cfg=decoder_cfg,
    )

    output = decoder(graph)

    assert output.pred.shape == (6, 2)
    assert output.x.shape == (6, 4)

    assert not torch.allclose(
        output.pred,
        graph.pred,
    )


def test_decoder_preserves_graph_structure(graph):
    """Decoder should not alter edge_index or other graph attributes."""

    graph.some_attribute = torch.randn(6)

    decoder_cfg = SimpleGraphDecoderConfig(
        protocol="predict",
        pred_feature_key="x",
    )

    decoder = SimpleGraphDecoder(
        in_channels=4,
        out_channels=4,
        cfg=decoder_cfg,
    )

    output = decoder(graph)

    assert torch.equal(
        output.edge_index,
        graph.edge_index,
    )

    assert torch.equal(
        output.some_attribute,
        graph.some_attribute,
    )


def test_decoder_does_not_modify_input(graph):
    """The original graph must remain completely unchanged."""

    original_x = graph.x.clone()
    original_edge_index = graph.edge_index.clone()

    decoder_cfg = SimpleGraphDecoderConfig(
        protocol="predict",
        pred_feature_key="x",
    )

    decoder = SimpleGraphDecoder(
        in_channels=4,
        out_channels=4,
        cfg=decoder_cfg,
    )

    decoder(graph)

    assert torch.equal(graph.x, original_x)
    assert torch.equal(graph.edge_index, original_edge_index)