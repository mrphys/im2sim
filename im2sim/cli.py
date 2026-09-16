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

import urllib.request


def install_pyg():
    import subprocess
    import sys

    import torch
    import torch_geometric

    version = torch.__version__.split("+")[0]
    cuda = torch.version.cuda

    cuda_tag = "cpu" if cuda is None else f"cu{cuda.replace('.', '')}"

    print(f"Detected PyTorch version: {version}, CUDA version: {cuda_tag}")

    url = f"https://data.pyg.org/whl/torch-{version}+{cuda_tag}.html"

    print(f"Installing PyTorch Geometric dependencies from {url}...")

    try:
        urllib.request.urlopen(url)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise RuntimeError(
                f"No PyG wheels available for {url}, either PyTorch/CUDA combination is not supported or data.pyg.org is down"
                + "\n Please check https://data.pyg.org/whl/index.html for supported versions."
                + "\n If https://data.pyg.org/whl/index.html doesn't load, the PyG server is down, please try again later."
            ) from e
        raise

    pyg_version = torch_geometric.__version__
    pyg_version = int(pyg_version.replace(".", ""))

    if pyg_version >= 280:
        print(
            "Detected PyTorch Geometric version >= 2.8.0, installing pyg-lib and torch-scatter..."
        )
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "pyg-lib",
                "torch-scatter",
                "-f",
                url,
            ]
        )
    else:
        print(
            "Detected PyTorch Geometric version < 2.8.0, installing torch-scatter, torch-sparse, torch-cluster, and torch-spline-conv..."
        )
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "torch-scatter",
                "torch-sparse",
                "torch-cluster",
                "torch-spline-conv",
                "-f",
                url,
            ]
        )


if __name__ == "__main__":
    install_pyg()
