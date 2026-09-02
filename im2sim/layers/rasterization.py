import torch
import torch.nn.functional as F
import torch_geometric.nn as gnn

from im2sim.utils.layer_util import register_image_layer


def pointcloud_to_mask(points, im_shape, vox_sizes):
    """
    Fast voxelization
    """
    device = points.device
    
    # Convert world coords → voxel indices
    idx = (points / torch.tensor(vox_sizes, device=device)).long()
    
    # Clamp to valid range
    for d in range(len(im_shape)):
        idx[:, d] = idx[:, d].clamp(0, im_shape[d] - 1)
    
    # Create mask
    mask = torch.zeros(im_shape, device=device)
    mask[idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0
    
    return mask

def dilate_mask(mask, kernel_size=3):
    """
    mask dilation to fill holes
    """
    kernel = torch.ones(1, 1, kernel_size, kernel_size, kernel_size, device=mask.device)
    mask = mask.unsqueeze(0).unsqueeze(0)
    dilated = F.conv3d(mask, kernel, padding=kernel_size//2)
    return (dilated > 0).float().squeeze()

def rasterise_feats(coords, feats, domain_size):
        x = torch.arange(0, domain_size[0])
        y = torch.arange(0, domain_size[1])
        z = torch.arange(0, domain_size[2])

        X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")

        img_coords = torch.stack(
                        [X.flatten(), Y.flatten(), Z.flatten()],
                        dim=-1
                    )
        # create mask in global coordinates
        mask = dilate_mask(
            pointcloud_to_mask(
                coords,
                domain_size,
                [1, 1, 1]
            )
        )

        # interpolate only active voxels
        rasterised_features = gnn.knn_interpolate(
            feats,
            coords,
            img_coords[mask].to(torch.float32),
            k=1
        )

        img = torch.zeros(
            (*domain_size, feats.shape[-1]),
            device=feats.device,
            dtype=torch.float32
        )

        img.view(-1, feats.shape[-1])[mask] = rasterised_features.float()

        # fill background with channel minima
        return img



class MaskRasterizer(torch.nn.Module):
    """
    Rasterizes a point cloud graph into a 3D mask image.

    Args:
        voxel_sizes (tuple): Voxel sizes for the rasterization process. Default is (1.0, 1.0, 1.0).
    
    """

    def __init__(self, voxel_sizes=(1.0, 1.0, 1.0)):
        super().__init__()
        self.voxel_sizes = voxel_sizes

    def forward(self, graph, image_input):
        """
        Rasterizes a point cloud graph into a 3D mask image.

        Args:
            graph (pyg.data.Data): Input graph data containing 'coords' attribute.
            image_input (torch.Tensor): Input image tensor to determine the shape and voxel sizes.

        Returns:
            torch.Tensor: Image tensor with the last channel replaced by the rasterized mask.
        """
        if 'coords' not in graph:
            raise ValueError("Graph must have 'coords' attribute for rasterization.")
        
        # Determine the shape of the output mask from the input image
        im_shape = image_input.shape[-3:]

        masks = []
        for b in graph.batch.unique():
            mask = dilate_mask(pointcloud_to_mask(graph.coords[graph.batch==b], im_shape, vox_sizes=self.voxel_sizes))
            masks.append(mask.unsqueeze(0))

        masks = torch.stack(masks, dim=0)

        image_input = torch.cat([image_input[:,:-1], masks], dim=1)

        return image_input
    

class FeatureRasterizer(torch.nn.Module):
    """
    Rasterizes a point cloud graph into a 3D feature image.

    Args:
        voxel_sizes (tuple): Voxel sizes for the rasterization process. Default is (1.0, 1.0, 1.0).
        feature_key (str): Key in the graph data where the features are stored. Default is 'x'.
        feature_channels (list[int]): List of channel indices to rasterize. If None, all channels will be rasterized. Default is None.
    """

    def __init__(self, 
                 voxel_sizes=(1.0, 1.0, 1.0),
                 feature_key='x',
                 feature_channels=None):
        super().__init__()
        self.voxel_sizes = voxel_sizes
        self.feature_key = feature_key
        self.feature_channels = feature_channels
    
    def forward(self, graph, image_input):
        """
        Rasterizes a point cloud graph into a 3D feature image.

        Args:
            graph (pyg.data.Data): Input graph data containing 'coords' attribute.
            image_input (torch.Tensor): Input image tensor to determine the shape and voxel sizes.

        Returns:
            torch.Tensor: Image tensor with the last channels replaced by the rasterized features.
        """
        if 'coords' not in graph:
            raise ValueError("Graph must have 'coords' attribute for rasterization.")
        
        # Determine the shape of the output mask from the input image
        im_shape = image_input.shape[-3:]

        feats = graph[self.feature_key]
        if self.feature_channels is not None:
            feats = feats[:, self.feature_channels]

        rasterised_feats = []
        for b in graph.batch.unique():
            rasterised_feat = rasterise_feats(
                graph.coords[graph.batch==b],
                feats[graph.batch==b],
                im_shape
            )
            rasterised_feats.append(rasterised_feat.unsqueeze(0))

        rasterised_feats = torch.cat(rasterised_feats, dim=0)

        image_input = torch.cat([image_input[:,:-rasterised_feats.shape[1]], rasterised_feats], dim=1)

        return image_input
