import torch
from torch_geometric.nn import knn_interpolate


def mse(x1, x2):
    return torch.mean((x1 - x2) ** 2)


class KnnMSE(torch.nn.Module):
    """
    KNN-based Mean Squared Error loss for graph data.
    Allows comparison between node features of two graphs without requiring point-to-point correspondence. 
    The features of the true graph are interpolated to the coordinates of the predicted graph using K-Nearest Neighbors (KNN) interpolation, 
    and then the MSE is computed between the interpolated features and the predicted features.

    Args:
        k (int): Number of nearest neighbors to consider for interpolation. Default is 3.

    """
    def __init__(self, k=3):
        """"""
        super().__init__()
        self.k = k

    def forward(self, true_graph, pred_graph):
        """
        Computes the KNN-based Mean Squared Error loss between two graphs.

        Args:
            true_graph (torch_geometric.data.Data): The ground truth graph, containing node features and coordinates.
            pred_graph (torch_geometric.data.Data): The predicted graph, containing node features and coordinates.
        """
        c1 = true_graph.x[:, :3]
        c2 = pred_graph.x[:, :3]

        f1 = true_graph.x[:, 3:]
        f2 = pred_graph.x[:, 3:]

        b1 = true_graph.batch
        b2 = pred_graph.batch

        f1_interp = knn_interpolate(f1, c1, c2, b1, b2, k=self.k)

        return mse(f1_interp, f2)


def mae(x1, x2):
    return torch.mean(torch.abs(x1 - x2))


class KnnMAE(torch.nn.Module):
    """
    KNN-based Mean Absolute Error loss for graph data.
    Allows comparison between node features of two graphs without requiring point-to-point correspondence. 
    The features of the true graph are interpolated to the coordinates of the predicted graph using K-Nearest Neighbors (KNN) interpolation, 
    and then the MAE is computed between the interpolated features and the predicted features.

    Args:
        k (int): Number of nearest neighbors to consider for interpolation. Default is 3.
    """

    def __init__(self, k=3):
        """"""
        super().__init__()
        self.k = k

    def forward(self, true_graph, pred_graph):
        """
        Computes the KNN-based Mean Absolute Error loss between two graphs.

        Args:
            true_graph (torch_geometric.data.Data): The ground truth graph, containing node features and coordinates.
            pred_graph (torch_geometric.data.Data): The predicted graph, containing node features and coordinates.
        """
        c1 = true_graph.x[:, :3]
        c2 = pred_graph.x[:, :3]

        f1 = true_graph.x[:, 3:]
        f2 = pred_graph.x[:, 3:]

        b1 = true_graph.batch
        b2 = pred_graph.batch

        f1_interp = knn_interpolate(f1, c1, c2, b1, b2, k=self.k)

        return mae(f1_interp, f2)