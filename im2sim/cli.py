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
"""
Installs the compiled PyTorch Geometric extensions used by im2sim (see ``im2sim.utils.pyg_ops``):

- The backend for ``knn``, ``knn_interpolate`` and ``graclus``:

  - ``torch_geometric >= 2.8``: ``pyg-lib >= 0.6.0`` (wheels exist for ``torch >= 2.8``)
  - ``torch_geometric < 2.8``: ``torch-cluster`` (wheels exist for ``torch < 2.12``)

- ``torch-scatter``, which PyTorch Geometric uses in every version for faster scatter ops in
  message passing (pyg-lib does not replace it). It is installed when a wheel exists for the
  installed PyTorch, otherwise PyTorch Geometric falls back to native PyTorch scatter ops.

``torch-sparse`` and ``torch-spline-conv`` are not used by im2sim.
"""

import re
import subprocess
import sys
import urllib.error
import urllib.request


def _major_minor(version: str) -> tuple[int, int]:
    major, minor = re.match(r"(\d+)\.(\d+)", version).groups()
    return int(major), int(minor)


def _cluster_backend(torch_version: str, pyg_version: str) -> str:
    """Returns the package providing knn/graclus, exiting if no wheels exist for this setup."""
    torch_mm = _major_minor(torch_version)
    if _major_minor(pyg_version) >= (2, 8):
        if torch_mm < (2, 8):
            sys.exit(
                f"PyTorch Geometric {pyg_version} needs pyg-lib>=0.6.0, which requires "
                f"PyTorch>=2.8 (found {torch_version}). Either upgrade PyTorch or install "
                "'torch-geometric<2.8' and rerun this command."
            )
        return "pyg-lib>=0.6.0"

    if torch_mm >= (2, 12):
        sys.exit(
            f"PyTorch Geometric {pyg_version} needs torch-cluster, which has no wheels for "
            f"PyTorch>=2.12 (found {torch_version}). Upgrade with "
            "'pip install -U torch-geometric' and rerun this command."
        )
    return "torch-cluster"


def _fetch_wheel_index(url: str) -> str | None:
    """Returns the wheel index page at ``url``, or None if it does not exist."""
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return response.read().decode()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    except urllib.error.URLError as e:
        sys.exit(f"Could not reach {url} ({e.reason}). data.pyg.org may be down, try again later.")


def _find_wheel_index(torch_version: str, cuda_tag: str) -> tuple[str, str]:
    """
    Returns the url and page of the PyG wheel index for this PyTorch/CUDA combination.

    PyG builds wheels per PyTorch minor version (e.g. ``pt214``) and usually only publishes an
    index for the ``X.Y.0`` release, so patch releases such as 2.14.1 fall back to it.
    """
    major, minor = _major_minor(torch_version)
    versions = [torch_version]
    if torch_version != f"{major}.{minor}.0":
        versions.append(f"{major}.{minor}.0")

    urls = [f"https://data.pyg.org/whl/torch-{version}+{cuda_tag}.html" for version in versions]
    for url in urls:
        page = _fetch_wheel_index(url)
        if page is not None:
            return url, page
    sys.exit(
        f"No PyG wheels available at {' or '.join(urls)}: this PyTorch/CUDA combination is not "
        "supported. Check https://data.pyg.org/whl/index.html for supported versions."
    )


def install_pyg():
    import torch
    import torch_geometric

    from im2sim.utils import pyg_ops

    torch_version = torch.__version__.split("+")[0]
    pyg_version = torch_geometric.__version__
    print(f"Detected PyTorch {torch.__version__}, PyTorch Geometric {pyg_version}")

    need_cluster = not (pyg_ops.HAS_COMPILED_KNN and pyg_ops.HAS_COMPILED_GRACLUS)
    need_scatter = not pyg_ops.HAS_TORCH_SCATTER
    if not need_cluster and not need_scatter:
        print("All compiled extensions are already available, nothing to install.")
        return

    if torch.version.hip is not None:
        sys.exit(
            "PyG does not publish ROCm wheels. im2sim will use its pure-PyTorch fallbacks; "
            "to use the compiled ops, build pyg-lib/torch-cluster and torch-scatter from source."
        )

    packages = [_cluster_backend(torch_version, pyg_version)] if need_cluster else []

    cuda = torch.version.cuda
    cuda_tag = "cpu" if cuda is None else f"cu{cuda.replace('.', '')}"
    url, wheel_index = _find_wheel_index(torch_version, cuda_tag)

    if need_scatter:
        if "torch_scatter-" in wheel_index:
            packages.append("torch-scatter")
        else:
            print(
                f"torch-scatter has no wheels for PyTorch {torch_version}+{cuda_tag}, skipping "
                "it. PyTorch Geometric will use native PyTorch scatter ops, which can be slower."
            )

    if not packages:
        return

    names = [re.split(r"[<>=!~]", package)[0] for package in packages]
    print(f"Installing {', '.join(packages)} from {url}...")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            *packages,
            "--only-binary",
            ",".join(names),
            "-f",
            url,
        ]
    )


if __name__ == "__main__":
    install_pyg()
