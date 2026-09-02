from abc import ABC, abstractmethod

import torch
import torch_geometric as pyg

def _get_selected_nodes(graph: pyg.data.Data, 
                        include_ids: list[str] = None, 
                        exclude_ids: list[str] = None) -> torch.Tensor:
    """
    Helper function to get the selected nodes based on include_ids and exclude_ids.

    Args:
        graph (pyg.data.Data): The input graph data.
        include_ids (list[str], optional): List of keys in the graph to include. If provided, only nodes corresponding to these keys will be selected.
        exclude_ids (list[str], optional): List of keys in the graph to exclude. If provided, nodes corresponding to these keys will be excluded.
    """

    def get_ids_from_keys(keys):
        ids = []
        for id_key in keys:
            if id_key in graph:
                ids.append(graph[id_key])
            else:
                raise ValueError(f"ID key '{id_key}' not found in graph.")
        return torch.cat(ids, dim=0).unique()
    
    if include_ids is not None:
        ids = get_ids_from_keys(include_ids)

    elif exclude_ids is not None:
        ids = get_ids_from_keys(exclude_ids)
        all_ids = torch.arange(graph.num_nodes)
        ids = torch.tensor([i for i in all_ids if i not in ids])

    else:
        ids = torch.arange(graph.num_nodes)
    
    return ids

class _GraphFeatureWrapper(torch.nn.Module, ABC):
    """
    Base class for graph feature wrappers.

    The wrapped module is applied to a graph and its output is written to
    ``graph.<pred_feature_key>`` for the selected nodes and channels.

    Subclasses define how the predicted values are written.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        pred_feature_key: str = "x",
        pred_feature_channels: list[int] | None = None,
        include_ids: list[str] | None = None,
        exclude_ids: list[str] | None = None,
    ):
        super().__init__()

        if include_ids is not None and exclude_ids is not None:
            print(
                "Warning: Both include_ids and exclude_ids are provided. "
                "Only include_ids will be used."
            )

        self.module = module
        self.pred_feature_key = pred_feature_key
        self.pred_feature_channels = pred_feature_channels
        self.include_ids = include_ids
        self.exclude_ids = exclude_ids

    def _prepare_graph(
        self,
        in_graph: pyg.data.Data,
    ) -> tuple[pyg.data.Data, pyg.data.Data, torch.Tensor, list[int]]:
        """
        Prepare the input and output graphs before applying the module.

        Returns:
            graph:
                Graph passed to the wrapped module.
            out_graph:
                Clone of the input graph that will receive the result.
            selected_nodes:
                Node indices to update.
            pred_feature_channels:
                Channels to update.
        """
        graph = in_graph.clone()
        out_graph = in_graph.clone()

        # Create the prediction feature if it does not already exist.
        if self.pred_feature_key not in graph:
            graph[self.pred_feature_key] = torch.zeros(
                (graph.num_nodes, self.module.out_channels),
                device=graph.x.device,
                dtype=graph.x.dtype,
            )

            out_graph[self.pred_feature_key] = torch.zeros(
                (graph.num_nodes, self.module.out_channels),
                device=graph.x.device,
                dtype=graph.x.dtype,
            )

        # Determine which prediction channels are being operated on.
        if self.pred_feature_channels is None:
            pred_feature_channels = list(
                range(self.module.out_channels)
            )
        else:
            pred_feature_channels = self.pred_feature_channels

        # Determine which nodes are being operated on.
        selected_nodes = _get_selected_nodes(
            graph,
            self.include_ids,
            self.exclude_ids,
        )

        # If predicting/updating a feature other than x, provide the
        # selected existing feature channels as input to the module.
        if self.pred_feature_key != "x":
            graph.x = torch.cat(
                [
                    graph.x,
                    graph[self.pred_feature_key][:, pred_feature_channels],
                ],
                dim=-1,
            )

        return (
            graph,
            out_graph,
            selected_nodes,
            pred_feature_channels,
        )

    @abstractmethod
    def _write_output(
        self,
        out_graph: pyg.data.Data,
        output: torch.Tensor,
        selected_nodes: torch.Tensor,
        pred_feature_channels: list[int],
    ) -> pyg.data.Data:
        """Write the module output into the output graph."""

    def forward(self, in_graph: pyg.data.Data) -> pyg.data.Data:
        graph, out_graph, selected_nodes, pred_feature_channels = (
            self._prepare_graph(in_graph)
        )

        output = self.module(graph)

        return self._write_output(
            out_graph,
            output,
            selected_nodes,
            pred_feature_channels,
        )


class GraphUpdater(_GraphFeatureWrapper):
    """
    Apply a graph module to update existing graph features.

    The update is:

        graph.<pred_feature_key>[:, pred_feature_channels] += module(graph)

    If ``pred_feature_key`` does not exist, it is initialized to zeros.

    Args:
        module:
            Graph convolutional block used to generate the update.

        pred_feature_key:
            Graph attribute containing the features to update.

        pred_feature_channels:
            Channels of the feature to update. If ``None``, all channels
            are used.

        include_ids:
            Graph keys identifying nodes to include.

        exclude_ids:
            Graph keys identifying nodes to exclude. Ignored when
            ``include_ids`` is provided.
    """

    def _write_output(
        self,
        out_graph: pyg.data.Data,
        output: torch.Tensor,
        selected_nodes: torch.Tensor,
        pred_feature_channels: list[int],
    ) -> pyg.data.Data:

        out_graph[self.pred_feature_key][
            selected_nodes[:, None],
            pred_feature_channels,
        ] += output.x[selected_nodes]

        return out_graph


class GraphPredictor(_GraphFeatureWrapper):
    """
    Apply a graph module to predict graph features.

    The prediction is:

        graph.<pred_feature_key>[:, pred_feature_channels] = module(graph)

    If ``pred_feature_key`` does not exist, it is initialized to zeros.

    Args:
        module:
            Graph convolutional block used to predict the features.

        pred_feature_key:
            Graph attribute containing the features to predict.

        pred_feature_channels:
            Channels of the feature to predict. If ``None``, all channels
            are used.

        include_ids:
            Graph keys identifying nodes to include.

        exclude_ids:
            Graph keys identifying nodes to exclude. Ignored when
            ``include_ids`` is provided.
    """

    def _write_output(
        self,
        out_graph: pyg.data.Data,
        output: torch.Tensor,
        selected_nodes: torch.Tensor,
        pred_feature_channels: list[int],
    ) -> pyg.data.Data:

        out_graph[self.pred_feature_key][
            selected_nodes[:, None],
            pred_feature_channels,
        ] = output.x[selected_nodes]

        return out_graph
    
GNN_PROTOCOLS = {
    "update": GraphUpdater,
    "predict": GraphPredictor
}