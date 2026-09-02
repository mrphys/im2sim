import torch
import torch.nn.functional as F
from torch_geometric.nn import knn_interpolate


class KnnFeatureLoss(torch.nn.Module):
    """
    Computes the loss between features of two graphs using k-nearest neighbors interpolation based on their coordinates.
    This loss can be used to compare the features of a predicted graph against a ground truth graph when they are not in point-to-point correspondence.

    Args:
        mode (str): The type of loss to compute. Options are 'l1' for L1 loss and 'l2' for L2 loss. Default is 'l1'.
        k (int): The number of nearest neighbors to use for interpolation. Default is 3.
        feature_key (str): The attribute name in the graph data that contains the features to compare. Default is 'x'.
        feature_channels (list or None): A list of channel indices to select from the features. If None, all channels are used. Default is None.
    """

    def __init__(self, mode="l1", k=3, feature_key="x", feature_channels=None):
        super().__init__()
        if mode not in ["l1", "l2"]:
            raise ValueError("Mode must be either 'l1' or 'l2'.")
        self.mode = mode
        self.k = k
        self.feature_key = feature_key
        self.feature_channels = feature_channels

    def forward(self, true_graph, pred_graph):
        if not hasattr(true_graph, "coords") or not hasattr(pred_graph, "coords"):
            raise ValueError("Both true_graph and pred_graph must have 'coords' attribute.")

        if not hasattr(true_graph, self.feature_key) or not hasattr(pred_graph, self.feature_key):
            raise ValueError(
                f"Both true_graph and pred_graph must have '{self.feature_key}' attribute."
            )

        if (
            getattr(true_graph, self.feature_key) is None
            or getattr(pred_graph, self.feature_key) is None
        ):
            raise ValueError(
                f"Both true_graph and pred_graph must have non-None '{self.feature_key}' attribute."
            )

        c1 = true_graph.coords
        c2 = pred_graph.coords

        if self.feature_channels is not None:
            f1 = getattr(true_graph, self.feature_key)[:, self.feature_channels]
            f2 = getattr(pred_graph, self.feature_key)[:, self.feature_channels]
        else:
            f1 = getattr(true_graph, self.feature_key)
            f2 = getattr(pred_graph, self.feature_key)

        if not hasattr(true_graph, "batch") or not hasattr(pred_graph, "batch"):
            b1 = torch.zeros(c1.size(0), dtype=torch.long, device=c1.device)
            b2 = torch.zeros(c2.size(0), dtype=torch.long, device=c2.device)
        else:
            b1 = true_graph.batch
            b2 = pred_graph.batch

        f1_interp = knn_interpolate(f1, c1, c2, b1, b2, k=self.k)

        if self.mode == "l1":
            return F.l1_loss(f1_interp, f2)
        else:
            return F.mse_loss(f1_interp, f2)
