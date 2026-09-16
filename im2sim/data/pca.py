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


class PCA:
    """
    Principal Component Analysis

    This class can be used for PCA-related operations such as computing PCs and saving PC projection matrices

    Parameters
    ----------
    data: np.ndarraye
        data to PCA
    axis: int
        axis to conduct the PCA over

    Attributes
    ----------
    S: np.ndarray
        PC matrix
    V: np.ndarray
        Variance explained by each PC
    """

    def __init__(self, data, axis=-1):
        """
        takes the data and axis, computes the PCs matrix and stores in object attributes
        """
        raise NotImplementedError

    def save(self):
        """saves the PC data"""
        raise NotImplementedError

    def load(self, fname):
        """loads saved PC data"""
        raise NotImplementedError

    def forward_transform(data):
        """forward transform"""
        raise NotImplementedError

    def inverse_transform(data):
        """inverse transform"""
        raise NotImplementedError

    def forward_transform_tf(data):
        """forward transform"""
        raise NotImplementedError

    def inverse_transform_tf(data):
        """inverse transform"""
        raise NotImplementedError
