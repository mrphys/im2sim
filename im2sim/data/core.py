from abc import ABC, abstractmethod
import copy
import torch
from torch_geometric.data import Data, Batch
import logging

logger = logging.getLogger(__name__)




def collate(batch):
    '''
    Collates a batch of data in the form of dict{str: torch.Tensor/PyG.Data} 
    Tensors are batched by stacking in the 0 dim and PyG Data are batched by PyG (https://pytorch-geometric.readthedocs.io/en/2.5.2/advanced/batching.html)
    '''
    out_dict = {}
    for key,val in batch[0].items():
        if isinstance(val, torch.Tensor):
            out_dict[key] = torch.utils.data.default_collate([b[key] for b in batch])
        elif isinstance(val, Data):
            out_dict[key] = Batch.from_data_list([b[key] for b in batch])
        else:
            raise TypeError(f"{key} is type {type(val)}. Generator outputs must be either torch.Tensor or torch_geometric.data.Data object")
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
        im2sim.data.DataLoader:
            Configured DataLoader instance.
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

        transforms (list[Transform] | Pipeline):
            Transforms or pipeline applied to each sample.

    Example:
        >>> import torch
        >>>
        >>> cases = ['case1', 'case2', 'case3', 'case4']
        >>>
        >>> def load(case):
        ...     img = torch.load(f'images/{case}.pt')
        ...     graph = torch.load(f'graphs/{case}.pt')
        ...     template = torch.load(f'template/{case}.pt')
        ...     return {'image': img, 'template': template, 'out_graph': graph}
        >>>
        >>> dataset = im2sim.data.Dataset(load_fn=load, cases=cases)
    """
    def __init__(self, load_fn, cases, transforms=[]):
        self.load_fn = load_fn
        self.cases = cases
        self.transforms = transforms

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case = self.cases[idx]
        sample = self.load_fn(case)

        for transform in self.transforms:
            sample = transform(sample)
        return sample
  

class Operation(ABC):
    """
    Abstract base class for making operations. 
    To make a new simple operation, subclass this and overwrite the forward method.
    """
    def __init_subclass__(cls):
        original_init = cls.__init__

        def new_init(self, *args, **kwargs):
            self.call_args = {"args": args, "kwargs": kwargs}
            original_init(self, *args, **kwargs)

        cls.__init__ = new_init

    def __call__(self, x):
        return self.forward(x)

    @abstractmethod
    def forward(self, x):
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
        return (
            isinstance(v, (int, float, str, bool))
            or isinstance(v, torch.Tensor)
            or v is None
        )
    
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
        pass


class FittableOperation(InvertibleOperation):
    """
    Subclass of InvertibleOperation that additionally enforces implementation of fit_step and complete_fit for fitting attrs to full dataset

    """

    @abstractmethod
    def fit_step(self, x):
        pass

    @abstractmethod
    def complete_fit(self):
        pass



class Transform:
    """
    A wrapper for operations that allows the selective application of the operation by dict key, object attribute and channel

    Args:
        op (Operation):
            Operation to wrap

        keys (list[str]):
            List of keys in the data dict for the op to operate over
        
        attr (str, optional): 
            If data[key] is an object, attr is the attribute of that object to perfrom the op over. 
            If the op needs the full object, set attr to 'all' 
            If data[key] is not an object attr=None 
            (default:None)
    """

    def __init__(
        self,
        op,
        keys,
        multikey = False,
        attr=None,
        channels=None,
        per_channel=False,
        channel_dim=-1,
        name=None
    ):
       
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

        self.name = name if name is not None else f"{op.__class__.__name__}_{'_'.join(self.keys)}"

        self.is_multikey = multikey
        self.is_invertible = isinstance(op, InvertibleOperation)
        self.is_fittable = isinstance(op, FittableOperation)

        self.fitted = False

    # -----------------------------
    # Internal helpers
    # -----------------------------

    def _get_target(self, data, key):
        if self.attr is None:
            return data[key]

        elif self.attr == "all":
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

        channels = (
            range(x_moved.shape[0])
            if self.channels is None
            else self.channels
        )

        self.op = [copy.deepcopy(self.op) for _ in channels]


    def _apply_channel_op(self, x, fn, no_return=False):

        if self.channels is None and not self.per_channel:
            return getattr(self.op, fn)(x)

        x_moved = torch.moveaxis(x, self.channel_dim, 0)

        # initialize per-channel ops lazily
        if self.per_channel:
            self._ensure_per_channel_ops(x)

        channels = (
            range(x_moved.shape[0])
            if self.channels is None
            else self.channels
        )

        if self.per_channel:

            for i, c in enumerate(channels):

                op = self.op[i]
                fn_op = getattr(op, fn)

                if no_return:
                    fn_op(x_moved[c])
                else:
                    x_moved[c] = fn_op(x_moved[c])

        else:

            idx = self.channels if self.channels is not None else slice(None)
            fn_op = getattr(self.op, fn)

            if no_return:
                fn_op(x_moved[idx])
            else:
                x_moved[idx] = fn_op(x_moved[idx])

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

            
            for k, out in zip(self.keys, outputs):
                self._set_target(data, k, out)

        elif len(self.keys)>1:
        # ---- single key op over mutliple keys ----
            inputs = [self._get_target(data, k) for k in self.keys]

            if no_return:
                for i in inputs:
                    self._apply_channel_op(i, fn, no_return)
                return

            for i,k in zip(inputs, self.keys):
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

    def forward(self, data):
        if self.is_fittable and not self.fitted:
            raise RuntimeError(f"Fittable transform {self.name} has not been fit")
        return self._apply_op(data, "forward")

    def inverse(self, data):
        if not self.is_invertible:
            raise RuntimeError(f"{self.name} is not invertible")
        if self.is_fittable and not self.fitted:
            raise RuntimeError(f"Fittable transform {self.name} has not been fit")
        return self._apply_op(data, "inverse")

    # -----------------------------
    # Fitting
    # -----------------------------

    def fit(self, dataloader):
        print("in fit")
        print(self.op)
        if not self.is_fittable:
            return
        
        for batch in dataloader:
            self._apply_op(batch, "fit_step", no_return=True)

        # finalize fit
        if self.per_channel:
            for op in self.op: op.complete_fit()
        else:
            self.op.complete_fit()

        self.fitted = True

    # -----------------------------
    # Serialization
    # -----------------------------

    def config(self):
        return {
            "type": self.__class__.__name__,
            "op": self.op.__class__.__name__ if not self.per_channel else self.op[0].__class__.__name__,
            "op_args": self.op.call_args if not self.per_channel else self.op[0].call_args,
            "name": self.name,
            "keys": self.keys,
            "attr": self.attr,
            "channels": self.channels,
            "per_channel": self.per_channel,
            "channel_dim": self.channel_dim,
        }

    def state_dict(self):
        if self.per_channel:
            return [op.state_dict() for op in self.op]

        if hasattr(self.op, "state_dict"):
            return self.op.state_dict()
        return {}

    def load_state_dict(self, state):

        if self.per_channel:
            for op, s in zip(self.op, state):
                op.load_state_dict(s)
        else:
            self.op.load_state_dict(state)

        self.fitted = True

    def to(self, device):
        if self.per_channel:
            for op in self.op: op.to(device)
        else:
            self.op.to(device)


class Pipeline:

    def __init__(self, transforms):
        self.transforms = transforms

    # -----------------------------
    # Forward
    # -----------------------------
    def __call__(self, data):
        for t in self.transforms:
            if logger.isEnabledFor(logging.DEBUG):
                logging.debug(f' before {t.name}')
                for k,v in data.items():
                    logging.debug(k)
                    
                    if isinstance(v, Data):
                        for c in range(v.x.shape[-1]):
                            logging.debug(f'channel {c}- max:{v.x[...,c].max()}, min:{v.x[...,c].min()}')
                    else:
                        logging.debug(v.shape)
            data = t.forward(data)
        return data

    # -----------------------------
    # Inverse (reverse order)
    # -----------------------------
    def inverse(self, data):
        for t in reversed(self.transforms):
            if t.is_invertible:
                if logger.isEnabledFor(logging.DEBUG):
                    logging.debug(f' before {t.name}')
                    for k,v in data.items():
                        logging.debug(k)
                    
                        if isinstance(v, Data):
                            for c in range(v.x.shape[-1]):
                                logging.debug(f'channel {c}- max:{v.x[...,c].max()}, min:{v.x[...,c].min()}')
                        else:
                            logging.debug(v.shape)
                data = t.inverse(data)
        return data

    # -----------------------------
    # Fit (only fittable transforms)
    # -----------------------------
    def fit(self, dataset):
        n = len(self.transforms)
        temp_dataset = copy.deepcopy(dataset)
        for i in range(n):
            if self.transforms[i].is_fittable:
                temp_dataset.transforms = [t.forward for t in self.transforms[:i]]
                dataloader = DataLoader(temp_dataset,batch_size=1)
                self.transforms[i].fit(dataloader)
        return self
        


    # -----------------------------
    # Serialization
    # -----------------------------
    def config(self):
        return {
            "transforms": [t.config() for t in self.transforms]
        }

    def state_dict(self):
        return {
            t.name: t.state_dict()
            for t in self.transforms
        }

    # -----------------------------
    # Loading
    # -----------------------------
    @classmethod
    def from_config(cls, config):

        transforms = []

        for tconf in config["transforms"]:

            op_cls = TRANSFORM_REGISTRY[tconf["op"]]
            op_args = tconf["op_args"]

            op = op_cls(*op_args["args"], **op_args["kwargs"]) if not tconf["per_channel"]\
                  else [op_cls(*op_args["args"], **op_args["kwargs"]) for _ in range(len(tconf["channels"]))]

            t = Transform(
                op=op,
                keys=tconf["keys"],
                attr=tconf["attr"],
                channels=tconf["channels"],
                per_channel=tconf["per_channel"],
                channel_dim=tconf["channel_dim"],
                name=tconf["name"]
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


def save_pipeline(pipeline, path):

    obj = {
        "config": pipeline.config(),
        "state": pipeline.state_dict()
    }

    torch.save(obj, path)


def load_pipeline(path):

    obj = torch.load(path)

    pipeline = Pipeline.from_config(obj["config"])
    pipeline.load_state_dict(obj["state"])

    return pipeline

# ------------------------------------------------------------------------------------
# TRANSFORM REGISTRY DEFINITION
# ------------------------------------------------------------------------------------


TRANSFORM_REGISTRY = {}

def register_op(cls):
    TRANSFORM_REGISTRY[cls.__name__] = cls
    return cls