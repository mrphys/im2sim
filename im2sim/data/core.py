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


import copy
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import torch
from torch_geometric.data import Batch, Data

logger = logging.getLogger(__name__)


class Operation(ABC):
    """
    Abstract base class for making operations.
    To make a new simple operation, subclass this and overwrite the forward method.
    Operations are saved by name and constructor arguments, so subclasses that should be saved
    must be registered with `register_op` and take serialisable arguments. Set
    `serializable = False` on operations that cannot be saved (e.g. ones wrapping a function).
    """

    serializable = True

    def __init_subclass__(cls):
        original_init = cls.__init__

        def new_init(self, *args, **kwargs):
            # Keep the arguments of the outermost (most derived) constructor call: a subclass that
            # calls ``super().__init__(...)`` must not have its arguments overwritten by the parent's.
            if "call_args" not in self.__dict__:
                self.call_args = {"args": args, "kwargs": kwargs}
            original_init(self, *args, **kwargs)

        cls.__init__ = new_init

    def __call__(self, x):
        return self.forward(x)

    @abstractmethod
    def forward(self, x):
        """forward pass of the operation"""
        pass

    def state_dict(self):
        """
        Returns the state of all attributes
        Default: return all tensor-like / basic attributes.
        Override for custom behavior.
        """
        state = {}

        for k, v in self.__dict__.items():
            # only save simple / tensor state
            if self._is_serializable(v):
                state[k] = v

        return state

    def load_state_dict(self, state):
        """
        Load saved state into the operation.
        """
        for k, v in state.items():
            setattr(self, k, v)

    def _is_serializable(self, v):
        """
        helper function to check if attr is serializable
        """
        return isinstance(v, (int, float, str, bool, torch.Tensor)) or v is None

    def to(self, device):
        """
        method to move attrs to device for torch training
        """
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                setattr(self, k, v.to(device))
        return self


class InvertibleOperation(Operation):
    """
    Subclass of Operation that additionally enforces implementation of an inverse method

    """

    @abstractmethod
    def inverse(self, x):
        """inverse pass of the operation"""
        pass


class FittableOperation(InvertibleOperation):
    """
    Subclass of InvertibleOperation that additionally enforces implementation of fit_step and complete_fit for fitting attrs to full dataset

    """

    @abstractmethod
    def fit_step(self, x):
        """single step of fitting the operation to a batch of data"""
        pass

    @abstractmethod
    def complete_fit(self):
        """finalize fitting the operation after all batches have been processed"""
        pass


class Transform:
    """
    A wrapper for operations that allows the selective application of the operation by dict key, object attribute and channel

    Args:
        op (Operation):
            Operation to wrap

        keys (str | list[str]):
            Key or list of keys in the data dict for the op to operate over

        multikey (bool, optional):
            If True, the values of all keys are passed to the op together as separate arguments,
            and the op should return one output per key. If False, the op is applied to each key separately.
            (default: False)

        attr (str, optional):
            If data[key] is an object, attr is the attribute of that object to perform the op over.
            If the op needs the full object, set attr to 'all'
            If data[key] is not an object attr=None
            (default: None)

        channels (int | list[int], optional):
            Channel(s) to apply the op to. If None, the op is applied to all channels. (default: None)

        per_channel (bool, optional):
            If True, a separate copy of the op is used for each channel. (default: False)

        channel_dim (int, optional):
            Dimension of the channel axis. (default: -1)

        name (str, optional):
            Name of the transform. If None, defaults to `<OpClassName>_<keys>`. (default: None)
    """

    def __init__(
        self,
        op,
        keys,
        multikey=False,
        attr=None,
        channels=None,
        per_channel=False,
        channel_dim=-1,
        name=None,
    ):

        if keys is None or keys == []:
            raise ValueError("keys must be specified")

        self.keys = keys if isinstance(keys, list) else [keys]
        self.attr = attr

        if channels is None:
            self.channels = None
        elif isinstance(channels, list):
            self.channels = channels
        else:
            self.channels = [channels]

        self.channel_dim = channel_dim
        self.per_channel = per_channel

        self.op = op

        # ``op`` is a list of per-channel copies when restored from a config with explicit channels
        base_op = op[0] if isinstance(op, list) else op

        if name is None:
            name = f"{base_op.__class__.__name__}_{'_'.join(self.keys)}"
            if self.attr not in (None, "all"):
                name += f"_{self.attr}"
            if self.channels is not None:
                name += "_ch" + "-".join(str(c) for c in self.channels)
        self.name = name

        self.is_multikey = multikey
        self.is_invertible = isinstance(base_op, InvertibleOperation)
        self.is_fittable = isinstance(base_op, FittableOperation)

        self.fitted = False

    # -----------------------------
    # Internal helpers
    # -----------------------------

    def _get_target(self, data, key):
        if self.attr is None or self.attr == "all":
            return data[key]

        elif hasattr(data[key], self.attr):
            return getattr(data[key], self.attr)

        else:
            raise ValueError(f"{key} has no attribute {self.attr}")

    def _set_target(self, data, key, value):
        if self.attr is None or self.attr == "all":
            data[key] = value
        else:
            setattr(data[key], self.attr, value)

    def _ensure_per_channel_ops(self, x):

        if not self.per_channel:
            return

        if isinstance(self.op, list):
            return

        x_moved = torch.moveaxis(x, self.channel_dim, 0)

        channels = range(x_moved.shape[0]) if self.channels is None else self.channels

        self.op = [copy.deepcopy(self.op) for _ in channels]

    def _apply_channel_op(self, x, fn, no_return=False):

        if self.channels is None and not self.per_channel:
            return getattr(self.op, fn)(x)

        x_moved = torch.moveaxis(x, self.channel_dim, 0)

        # initialize per-channel ops lazily
        if self.per_channel:
            self._ensure_per_channel_ops(x)

        channels = range(x_moved.shape[0]) if self.channels is None else self.channels

        if self.per_channel:
            for i, c in enumerate(channels):
                op = self.op[i]
                fn_op = getattr(op, fn)

                if no_return:
                    fn_op(x_moved[c])
                else:
                    x_moved = _assign_channels(x_moved, c, fn_op(x_moved[c]))

        else:
            idx = self.channels if self.channels is not None else slice(None)
            fn_op = getattr(self.op, fn)

            if no_return:
                fn_op(x_moved[idx])
            else:
                x_moved = _assign_channels(x_moved, idx, fn_op(x_moved[idx]))

        if no_return:
            return None

        return torch.moveaxis(x_moved, 0, self.channel_dim)

    # -----------------------------
    # Core apply
    # -----------------------------

    def _apply_op(self, data, fn, no_return=False):

        # --- MULTI-KEY ---
        if self.is_multikey:
            # ---- multikey op ----
            inputs = [self._get_target(data, k) for k in self.keys]

            if no_return:
                getattr(self.op, fn)(*inputs)
                return

            outputs = getattr(self.op, fn)(*inputs)

            if not isinstance(outputs, (list, tuple)):
                outputs = [outputs]

            for k, out in zip(self.keys, outputs, strict=True):
                self._set_target(data, k, out)

        elif len(self.keys) > 1:
            # ---- single key op over mutliple keys ----
            inputs = [self._get_target(data, k) for k in self.keys]

            if no_return:
                for i in inputs:
                    self._apply_channel_op(i, fn, no_return)
                return

            for i, k in zip(inputs, self.keys, strict=True):
                out = self._apply_channel_op(i, fn, no_return)
                self._set_target(data, k, out)

        else:
            # --- single key ---
            key = self.keys[0]

            x = self._get_target(data, key)

            x = self._apply_channel_op(x, fn, no_return)

            if no_return:
                return None

            self._set_target(data, key, x)

        return data

    # -----------------------------
    # Public API
    # -----------------------------

    def __call__(self, data):
        return self.forward(data)

    def forward(self, data):
        data = copy.deepcopy(data)
        if self.is_fittable and not self.fitted:
            raise RuntimeError(f"Fittable transform {self.name} has not been fit")
        return self._apply_op(data, "forward")

    def inverse(self, data):
        data = copy.deepcopy(data)
        if not self.is_invertible:
            raise RuntimeError(f"{self.name} is not invertible")
        if self.is_fittable and not self.fitted:
            raise RuntimeError(f"Fittable transform {self.name} has not been fit")
        return self._apply_op(data, "inverse")

    # -----------------------------
    # Fitting
    # -----------------------------

    def fit(self, dataloader):
        """
        Fit the transform.

        Args:
            dataloader (Iterable[dict]): Iterable of samples in the same layout that `forward`
                receives. A batching ``torch.utils.data.DataLoader`` adds a leading batch
                dimension, so a non-negative ``channel_dim`` would then point at the wrong axis.
                `Pipeline.fit` passes unbatched samples.
        """

        if not self.is_fittable:
            return

        for batch in dataloader:
            self._apply_op(batch, "fit_step", no_return=True)

        # finalize fit
        if self.per_channel:
            for op in self.op:
                op.complete_fit()
        else:
            self.op.complete_fit()

        self.fitted = True

    # -----------------------------
    # Serialization
    # -----------------------------

    def _base_op(self):
        return self.op[0] if isinstance(self.op, list) else self.op

    def config(self):
        if not getattr(self._base_op(), "serializable", True):
            raise TypeError(
                f"Transform {self.name!r} wraps {self._base_op().__class__.__name__}, which cannot "
                "be saved. Define a registered `Operation` subclass (see `register_op`) instead."
            )
        return {
            "type": self.__class__.__name__,
            "op": self._base_op().__class__.__name__,
            "op_args": self._base_op().call_args,
            "name": self.name,
            "keys": self.keys,
            "multikey": self.is_multikey,
            "attr": self.attr,
            "channels": self.channels,
            "per_channel": self.per_channel,
            "channel_dim": self.channel_dim,
        }

    def state_dict(self):
        if self.per_channel:
            # Per-channel ops that were never expanded (never used or fit) have no state yet
            if not isinstance(self.op, list):
                return []
            return [op.state_dict() for op in self.op]

        if hasattr(self.op, "state_dict"):
            return self.op.state_dict()
        return {}

    def load_state_dict(self, state):

        if self.per_channel:
            if not isinstance(self.op, list):
                if len(state) == 0:
                    return
                self.op = [copy.deepcopy(self.op) for _ in state]
            for op, s in zip(self.op, state, strict=True):
                op.load_state_dict(s)
        else:
            self.op.load_state_dict(state)

        self.fitted = True

    def to(self, device):
        if self.per_channel and isinstance(self.op, list):
            for op in self.op:
                op.to(device)
        else:
            self.op.to(device)


class Pipeline:
    def __init__(self, transforms):
        # State is saved and restored by transform name, so names must be unique. Repeated names
        # (e.g. the same op applied twice to one key) get a positional suffix instead of silently
        # sharing one state entry. The names are saved in `config()`, so reloads stay consistent.
        # A new name must not clash with any given name (including later ones) or earlier renames.
        reserved = {t.name for t in transforms}
        used: set[str] = set()
        for t in transforms:
            if t.name in used:
                n = 2
                while f"{t.name}_{n}" in reserved or f"{t.name}_{n}" in used:
                    n += 1
                new_name = f"{t.name}_{n}"
                logger.info("Renaming duplicate transform %r to %r", t.name, new_name)
                t.name = new_name
            used.add(t.name)
        self.transforms = transforms

    # -----------------------------
    # Forward
    # -----------------------------
    def __call__(self, data):
        transforms = self._get_applicable_transforms(data.keys())
        for t in transforms:
            logger.debug(f"before {t.name}")
            _log_data(data)

            data = t.forward(data)

            logger.debug(f"after {t.name}")
            _log_data(data)
        return data

    # -----------------------------
    # Inverse (reverse order)
    # -----------------------------
    def inverse(self, data):
        transforms = self._get_applicable_transforms(data.keys())
        for t in reversed(transforms):
            if t.is_invertible:
                logger.debug(f"before {t.name} inverse")
                _log_data(data)

                data = t.inverse(data)

                logger.debug(f"after {t.name} inverse")
                _log_data(data)

        return data

    def _get_applicable_transforms(self, keys: str) -> list[Transform]:
        transforms = []
        for t in self.transforms:
            skip = False
            for key in t.keys:
                if key not in keys:
                    skip = True
                    break
            if not skip:
                transforms.append(t)
        return transforms

    # -----------------------------
    # Fit (only fittable transforms)
    # -----------------------------
    def fit(self, dataset):
        n = len(self.transforms)
        temp_dataset = copy.deepcopy(dataset)
        for i in range(n):
            if self.transforms[i].is_fittable:
                # Feed unbatched samples: batching would prepend a dimension and shift any
                # non-negative ``channel_dim``.
                temp_dataset.transforms = [Pipeline(self.transforms[:i])]
                self.transforms[i].fit(iter_samples(temp_dataset))
        return self

    # -----------------------------
    # Serialization
    # -----------------------------
    def config(self):
        return {"transforms": [t.config() for t in self.transforms]}

    def state_dict(self):
        return {t.name: t.state_dict() for t in self.transforms}

    # -----------------------------
    # Loading
    # -----------------------------
    @classmethod
    def from_config(cls, config):

        transforms = []

        for tconf in config["transforms"]:
            op_cls = TRANSFORM_REGISTRY[tconf["op"]]
            op_args = tconf["op_args"]

            # Per-channel ops with no listed channels are created from a single op once the
            # number of channels is known (in load_state_dict or on first use).
            op = (
                op_cls(*op_args["args"], **op_args["kwargs"])
                if not tconf["per_channel"] or tconf["channels"] is None
                else [
                    op_cls(*op_args["args"], **op_args["kwargs"])
                    for _ in range(len(tconf["channels"]))
                ]
            )

            t = Transform(
                op=op,
                keys=tconf["keys"],
                multikey=tconf.get("multikey", False),
                attr=tconf["attr"],
                channels=tconf["channels"],
                per_channel=tconf["per_channel"],
                channel_dim=tconf["channel_dim"],
                name=tconf["name"],
            )

            transforms.append(t)

        return cls(transforms)

    def load_state_dict(self, state_dict):

        for t in self.transforms:
            if t.name in state_dict:
                t.load_state_dict(state_dict[t.name])

    def to(self, device):
        for i in range(len(self.transforms)):
            self.transforms[i].to(device)


def _assign_channels(x, idx, value):
    """
    Write ``value`` into ``x[idx]``, promoting ``x`` first if needed (e.g. a float result of a
    normalisation written into an integer image).
    """
    if value.dtype != x.dtype:
        x = x.to(torch.promote_types(x.dtype, value.dtype))
    x[idx] = value
    return x


def iter_samples(dataset):
    """Yield the samples of ``dataset`` one at a time, without batching."""
    for i in range(len(dataset)):
        yield dataset[i]


def save_pipeline(pipeline, path):

    obj = {"config": pipeline.config(), "state": pipeline.state_dict()}

    torch.save(obj, path)


def load_pipeline(path):

    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # torch < 1.13 has no ``weights_only``
        obj = torch.load(path, map_location="cpu")

    pipeline = Pipeline.from_config(obj["config"])
    pipeline.load_state_dict(obj["state"])

    return pipeline


def _log_data(data: dict[str, Any]) -> None:
    if logger.isEnabledFor(logging.DEBUG):
        for k, v in data.items():
            logger.debug(k)

            if isinstance(v, Data):
                for c in range(v.x.shape[-1]):
                    logger.debug(f"channel {c}- max:{v.x[..., c].max()}, min:{v.x[..., c].min()}")
            else:
                logger.debug(v.shape)


# ------------------------------------------------------------------------------------
# TRANSFORM REGISTRY DEFINITION
# ------------------------------------------------------------------------------------


TRANSFORM_REGISTRY = {}


def register_op(cls):
    TRANSFORM_REGISTRY[cls.__name__] = cls
    return cls


# ------------------------------------------------------------------------------------
# DATA LOADING
# ------------------------------------------------------------------------------------


LoadFn = Callable[[str], dict[str, torch.Tensor | Data]]


def collate(batch):
    """
    Collates a batch of data in the form of dict{str: torch.Tensor/PyG.Data}
    Tensors are batched by stacking in the 0 dim and PyG Data are batched by PyG (https://pytorch-geometric.readthedocs.io/en/2.5.2/advanced/batching.html)
    """
    out_dict = {}
    for key, val in batch[0].items():
        if isinstance(val, torch.Tensor):
            out_dict[key] = torch.utils.data.default_collate([b[key] for b in batch])
        elif isinstance(val, Data):
            out_dict[key] = Batch.from_data_list([b[key] for b in batch])
        else:
            raise TypeError(
                f"{key} is type {type(val)}. Generator outputs must be either torch.Tensor or torch_geometric.data.Data object"
            )
    return out_dict


def DataLoader(dataset, **kwargs):
    """
    Create a DataLoader for an im2sim dataset.

    Args:
        dataset (im2sim.data.Dataset):
            Dataset to wrap in a DataLoader.

        **kwargs:
            Additional keyword arguments passed to ``torch.utils.data.DataLoader``.
            Common options include:

            - batch_size (int, optional):
                Number of samples per batch (default: 1).
            - shuffle (bool, optional):
                Whether to reshuffle the data at every epoch (default: False).
            - num_workers (int, optional):
                Number of subprocesses used for data loading. ``0`` means data
                is loaded in the main process (default: 0).
            - pin_memory (bool, optional):
                If True, tensors are copied into CUDA pinned memory before returning.

            See https://docs.pytorch.org/docs/stable/data.html for the full list
            of supported arguments.

    Returns:
        torch.utils.data.DataLoader:
            Configured DataLoader instance that collates samples with `im2sim.data.collate`.
    """
    return torch.utils.data.DataLoader(dataset, collate_fn=collate, **kwargs)


class Dataset(torch.utils.data.Dataset):
    """
    Template for building im2sim datasets.

    To build a custom dataset, create a new Dataset with a custom load function,
    case files, and transforms.

    Args:
        load_fn (Callable[[str], dict[str, Tensor | PyGData]]):
            Function that loads all files needed for a specific case and returns
            a dictionary containing the data for that case.

        cases (list[str]):
            List of case names. These names are passed to `load_fn` to load data.

        transforms (Transform | list[Transform] | Pipeline | None):
            Transforms or pipeline applied to each sample. Default is None (no transforms).

    Examples:


        .. code-block:: python

            import torch

            cases = ['case1', 'case2', 'case3', 'case4']

            def load(case):
                img = torch.load(f'images/{case}.pt')
                graph = torch.load(f'graphs/{case}.pt')
                template = torch.load(f'template/{case}.pt')
                return {'image': img, 'template': template, 'out_graph': graph}

            dataset = im2sim.data.Dataset(load_fn=load, cases=cases)
    """

    def __init__(
        self,
        load_fn: LoadFn,
        cases: list[str],
        transforms: list[Transform] | Pipeline | None = None,
    ):
        self.load_fn = load_fn
        self.cases = cases
        self.add_transforms(transforms)

    def add_transforms(self, transforms: Transform | list[Transform] | Pipeline | None = None):
        if transforms is None:
            self.transforms = []
        elif isinstance(transforms, (Pipeline, Transform)):
            self.transforms = [transforms]
        elif isinstance(transforms, (list, tuple)):
            self.transforms = list(transforms)
        else:
            raise TypeError(
                "transforms must be a Transform, a list of Transforms, a Pipeline or None, "
                f"got {type(transforms).__name__}."
            )

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | Data]:
        case = self.cases[idx]
        sample = self.load_fn(case)

        for transform in self.transforms:
            sample = transform(sample)
        return sample
