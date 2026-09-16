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

from im2sim.losses.confusion_losses import (
    ConfusionLoss,
    DiceLoss,
    FocalTverskyLoss,
    IoULoss,
    TverskyLoss,
)
from im2sim.losses.feature import KnnFeatureLoss
from im2sim.losses.mesh import (
    AspectRatioLoss,
    EdgeLengthDeviationLoss,
    FaceNormalLoss,
    InversionLoss,
)
from im2sim.losses.pointcloud import ChamferLoss
from im2sim.losses.ssim import SSIMLoss

__all__ = [
    "ConfusionLoss",
    "FocalTverskyLoss",
    "TverskyLoss",
    "DiceLoss",
    "IoULoss",
    "KnnFeatureLoss",
    "AspectRatioLoss",
    "EdgeLengthDeviationLoss",
    "InversionLoss",
    "FaceNormalLoss",
    "ChamferLoss",
    "SSIMLoss",
]
