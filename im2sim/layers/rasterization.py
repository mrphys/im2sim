# ==============================================================================
# Copyright 2026 University College London.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import torch
import torch.nn.functional as F

from im2sim.utils.pyg_ops import knn_interpolate


def pointcloud_to_mask(points, im_shape, vox_sizes):
    """
    Fast voxelization.

    A point is assigned to voxel ``floor(point / vox_size)``, i.e. voxel ``i`` covers the half-open
    interval ``[i * vox_size, (i + 1) * vox_size)`` in the units of ``points``. Note that
    `TrilinearProjection` instead treats voxel ``i`` as centred at coordinate ``i``, so a point is
    rasterised half a voxel differently to how it is sampled from the image features.
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
    dilated = F.conv3d(mask, kernel, padding=kernel_size // 2)
    # only drop the batch and channel dims: spatial dims of size 1 must be kept
    return (dilated > 0).float().squeeze(0).squeeze(0)


def _graph_batch(graph, image_input):
    """Batch vector of the graph, and the number of images it must match."""
    n_images = image_input.shape[0]
    batch = graph.batch
    if batch is None:
        batch = torch.zeros(graph.coords.shape[0], dtype=torch.long, device=graph.coords.device)
    if batch.numel() > 0 and int(batch.max()) >= n_images:
        raise ValueError(
            f"The graph has {int(batch.max()) + 1} graphs but image_input has a batch size of "
            f"{n_images}."
        )
    return batch, n_images


def rasterise_feats(coords, feats, domain_size, vox_sizes=(1.0, 1.0, 1.0)):
    device = coords.device
    x = torch.arange(0, domain_size[0], device=device)
    y = torch.arange(0, domain_size[1], device=device)
    z = torch.arange(0, domain_size[2], device=device)

    X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")

    # voxel positions in the units of `coords`
    img_coords = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=-1).to(torch.float32)
    img_coords = img_coords * torch.tensor(vox_sizes, dtype=torch.float32, device=device)
    # create mask in global coordinates, flattened to index img_coords
    mask = dilate_mask(pointcloud_to_mask(coords, domain_size, vox_sizes)).flatten().bool()

    # interpolate only active voxels
    rasterised_features = knn_interpolate(feats, coords, img_coords[mask], k=1)

    img = torch.zeros((*domain_size, feats.shape[-1]), device=feats.device, dtype=torch.float32)

    img.view(-1, feats.shape[-1])[mask] = rasterised_features.float()

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
            graph (pyg.data.Data): Input graph data containing 'coords' and 'batch' attributes.
            image_input (torch.Tensor): Input image tensor of shape (B, C, D, H, W) used to determine the shape of the mask.

        Returns:
            torch.Tensor: Image tensor with the last channel replaced by the rasterized mask.
        """
        if "coords" not in graph:
            raise ValueError("Graph must have 'coords' attribute for rasterization.")

        # Determine the shape of the output mask from the input image
        im_shape = image_input.shape[-3:]

        batch, n_images = _graph_batch(graph, image_input)

        masks = []
        for b in range(n_images):
            mask = dilate_mask(
                pointcloud_to_mask(graph.coords[batch == b], im_shape, vox_sizes=self.voxel_sizes)
            )
            masks.append(mask.unsqueeze(0))

        masks = torch.stack(masks, dim=0).to(image_input.dtype)

        image_input = torch.cat([image_input[:, :-1], masks], dim=1)

        return image_input


class FeatureRasterizer(torch.nn.Module):
    """
    Rasterizes a point cloud graph into a 3D feature image.

    Args:
        voxel_sizes (tuple): Voxel sizes for the rasterization process. Default is (1.0, 1.0, 1.0).
        feature_key (str): Key in the graph data where the features are stored. Default is 'x'.
        feature_channels (list[int]): List of channel indices to rasterize. If None, all channels will be rasterized. Default is None.
    """

    def __init__(self, voxel_sizes=(1.0, 1.0, 1.0), feature_key="x", feature_channels=None):
        super().__init__()
        self.voxel_sizes = voxel_sizes
        self.feature_key = feature_key
        self.feature_channels = feature_channels

    def forward(self, graph, image_input):
        """
        Rasterizes a point cloud graph into a 3D feature image.

        Args:
            graph (pyg.data.Data): Input graph data containing 'coords', 'batch' and `feature_key` attributes.
            image_input (torch.Tensor): Input image tensor used to determine the shape of the rasterized image.

        Returns:
            torch.Tensor: Image tensor with the last channels replaced by the rasterized features.
        """
        if "coords" not in graph:
            raise ValueError("Graph must have 'coords' attribute for rasterization.")

        # Determine the shape of the output mask from the input image
        im_shape = image_input.shape[-3:]

        feats = graph[self.feature_key]
        if self.feature_channels is not None:
            feats = feats[:, self.feature_channels]

        batch, n_images = _graph_batch(graph, image_input)

        rasterised_feats = []
        for b in range(n_images):
            rasterised_feat = rasterise_feats(
                graph.coords[batch == b], feats[batch == b], im_shape, self.voxel_sizes
            )
            # (D, H, W, C) -> (1, C, D, H, W)
            rasterised_feats.append(rasterised_feat.permute(3, 0, 1, 2).unsqueeze(0))

        rasterised_feats = torch.cat(rasterised_feats, dim=0)

        image_input = torch.cat(
            [image_input[:, : -rasterised_feats.shape[1]], rasterised_feats], dim=1
        )

        return image_input
