import io
import urllib.error

import pytest
import torch
import torch_geometric

from im2sim import cli
from im2sim.utils import pyg_ops

INDEX_WITH_SCATTER = "torch_scatter-2.1.2-cp310 pyg_lib-0.9.0 torch_cluster-1.6.3"
INDEX_WITHOUT_SCATTER = "pyg_lib-0.9.0"


@pytest.fixture
def env(monkeypatch):
    """Configures the detected versions/backends and records the urls and pip commands used."""
    calls = {"urls": [], "pip": []}
    index = {"page": INDEX_WITH_SCATTER, "error": None}

    def urlopen(url, timeout=None):
        calls["urls"].append(url)
        if index["error"] is not None:
            raise index["error"]
        return io.BytesIO(index["page"].encode())

    monkeypatch.setattr(cli.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(cli.subprocess, "check_call", lambda cmd: calls["pip"].append(cmd))

    def configure(
        torch_version="2.10.0",
        pyg_version="2.8.0",
        cuda=None,
        hip=None,
        compiled=False,
        scatter=False,
        page=INDEX_WITH_SCATTER,
        error=None,
    ):
        monkeypatch.setattr(torch, "__version__", torch_version)
        monkeypatch.setattr(torch.version, "cuda", cuda)
        monkeypatch.setattr(torch.version, "hip", hip)
        monkeypatch.setattr(torch_geometric, "__version__", pyg_version)
        monkeypatch.setattr(pyg_ops, "HAS_COMPILED_KNN", compiled)
        monkeypatch.setattr(pyg_ops, "HAS_COMPILED_GRACLUS", compiled)
        monkeypatch.setattr(pyg_ops, "HAS_TORCH_SCATTER", scatter)
        index.update(page=page, error=error)
        return calls

    return configure


def _installed(cmd):
    """Returns the packages and --only-binary names from a recorded pip command."""
    args = cmd[cmd.index("install") + 1 :]
    packages = args[: args.index("--only-binary")]
    only_binary = args[args.index("--only-binary") + 1].split(",")
    return packages, only_binary, args[args.index("-f") + 1]


def test_nothing_to_install(env):
    calls = env(compiled=True, scatter=True)
    cli.install_pyg()
    assert calls["urls"] == []
    assert calls["pip"] == []


def test_pyg_28_installs_pyg_lib_and_scatter(env):
    calls = env(torch_version="2.10.0+cu128", pyg_version="2.8.0", cuda="12.8")
    cli.install_pyg()
    url = "https://data.pyg.org/whl/torch-2.10.0+cu128.html"
    assert calls["urls"] == [url]
    assert _installed(calls["pip"][0]) == (
        ["pyg-lib>=0.6.0", "torch-scatter"],
        ["pyg-lib", "torch-scatter"],
        url,
    )


def test_pyg_27_installs_torch_cluster_and_scatter(env):
    calls = env(torch_version="2.3.1", pyg_version="2.7.0")
    cli.install_pyg()
    url = "https://data.pyg.org/whl/torch-2.3.1+cpu.html"
    assert _installed(calls["pip"][0]) == (
        ["torch-cluster", "torch-scatter"],
        ["torch-cluster", "torch-scatter"],
        url,
    )


def test_pyg_dev_version_uses_pyg_lib(env):
    calls = env(pyg_version="2.9.0.dev20260901")
    cli.install_pyg()
    assert _installed(calls["pip"][0])[0][0] == "pyg-lib>=0.6.0"


def test_only_missing_scatter_is_installed(env):
    calls = env(compiled=True, scatter=False)
    cli.install_pyg()
    assert _installed(calls["pip"][0])[:2] == (["torch-scatter"], ["torch-scatter"])


def test_only_missing_cluster_backend_is_installed(env):
    calls = env(compiled=False, scatter=True)
    cli.install_pyg()
    assert _installed(calls["pip"][0])[0] == ["pyg-lib>=0.6.0"]


def test_scatter_skipped_without_wheels(env, capsys):
    calls = env(torch_version="2.14.0", page=INDEX_WITHOUT_SCATTER)
    cli.install_pyg()
    assert _installed(calls["pip"][0])[0] == ["pyg-lib>=0.6.0"]
    assert "torch-scatter has no wheels" in capsys.readouterr().out


def test_scatter_only_without_wheels_installs_nothing(env, capsys):
    calls = env(torch_version="2.14.0", compiled=True, page=INDEX_WITHOUT_SCATTER)
    cli.install_pyg()
    assert calls["pip"] == []
    assert "torch-scatter has no wheels" in capsys.readouterr().out


@pytest.mark.parametrize(
    "torch_version, pyg_version, message",
    [
        ("2.3.1", "2.8.0", "requires PyTorch>=2.8"),
        ("2.12.0", "2.7.0", "no wheels for PyTorch>=2.12"),
    ],
)
def test_unsupported_combinations_exit(env, torch_version, pyg_version, message):
    calls = env(torch_version=torch_version, pyg_version=pyg_version)
    with pytest.raises(SystemExit, match=message):
        cli.install_pyg()
    assert calls["pip"] == []


def test_unsupported_cluster_combination_ignored_when_backend_present(env):
    calls = env(torch_version="2.12.0", pyg_version="2.7.0", compiled=True)
    cli.install_pyg()
    assert _installed(calls["pip"][0])[0] == ["torch-scatter"]


def test_rocm_exits(env):
    calls = env(torch_version="2.10.0+rocm6.4", hip="6.4")
    with pytest.raises(SystemExit, match="ROCm"):
        cli.install_pyg()
    assert calls["urls"] == []


def test_missing_wheel_index_exits(env):
    error = urllib.error.HTTPError("url", 404, "Not Found", None, None)
    calls = env(error=error)
    with pytest.raises(SystemExit, match="No PyG wheels available"):
        cli.install_pyg()
    assert calls["pip"] == []


def test_unreachable_wheel_index_exits(env):
    env(error=urllib.error.URLError("connection refused"))
    with pytest.raises(SystemExit, match="Could not reach"):
        cli.install_pyg()


def test_other_http_errors_are_raised(env):
    env(error=urllib.error.HTTPError("url", 500, "Server Error", None, None))
    with pytest.raises(urllib.error.HTTPError):
        cli.install_pyg()
