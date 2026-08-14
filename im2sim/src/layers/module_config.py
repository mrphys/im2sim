import json
from dataclasses import dataclass, fields
from enum import Enum
from types import UnionType
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

from im2sim.src.utils import api_util

T = TypeVar("T", bound="Config")


@api_util.export("_internal.Config")
@dataclass
class Config:
    """
    Base class for recursively serialisable configuration objects.

    This class provides recursive (de)serialization to/from dictionaries and JSON

    Subclasses should be defined as dataclasses.

    Example:
        ```python

        cfg = MyConfig(...)

        cfg.save("config.json")

        cfg2 = MyConfig().load("config.json")

        ```
    """

    def __init_subclass__(cls):
        """Automatically initialise a preset registry for each subclass."""
        super().__init_subclass__()
        cls._presets = {}

    def mod(self, **kwargs):
        """
        Create a modified copy of the config with updated fields.

        Args:
            **kwargs: Field names and their new values.

        Returns:
            Config: A new config instance with updated fields.
        """
        return self.__class__(**{**self.as_kwargs(), **kwargs})

    def as_kwargs(self):
        """
        Convert config fields into keyword arguments.

        Returns:
            dict: Mapping of field names to values.
        """
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def to_dict(self):
        """
        Recursively convert config into a serialisable dictionary.

        Returns:
            dict: Serialized representation."""
        return {
            "__class__": self.__class__.__name__,
            **{f.name: self._serialize_value(getattr(self, f.name)) for f in fields(self)},
        }

    @classmethod
    def from_dict(cls: type[T], data: dict[str, Any]) -> T:
        """
        Reconstruct a config object from a dictionary.

        Args:
            data: Serialized config dictionary.

        Returns:
            Reconstructed config object.
        """

        hints = get_type_hints(cls)

        kwargs = {}

        for field in fields(cls):
            if field.name not in data:
                continue

            value = data[field.name]
            field_type = hints.get(field.name)

            if field_type is not None:
                value = cls._deserialize_value(value, field_type)

            kwargs[field.name] = value

        return cls(**kwargs)

    @staticmethod
    def _serialize_value(value):
        """Recursively serialize values into JSON-compatible structures."""
        if isinstance(value, Config):
            return value.to_dict()

        if isinstance(value, Enum):
            return {
                "__enum__": value.__class__.__name__,
                "value": value.value,
            }

        if isinstance(value, list):
            return [Config._serialize_value(v) for v in value]

        if isinstance(value, dict):
            return {k: Config._serialize_value(v) for k, v in value.items()}

        return value

    @staticmethod
    def _deserialize_value(value, typ):
        """Recursively deserialize values based on type hints."""

        if value is None:
            return None

        origin = get_origin(typ)
        args = get_args(typ)

        # Optional[T] / T | None / Union[T, None]
        if origin in (Union, UnionType):
            for subtype in args:
                if subtype is type(None):
                    continue

                try:
                    return Config._deserialize_value(value, subtype)
                except (TypeError, ValueError):
                    continue

            return value

        # list[T]
        if origin is list:
            subtype = args[0]

            return [Config._deserialize_value(v, subtype) for v in value]

        # dict[K, V]
        if origin is dict:
            key_type, value_type = args

            return {
                Config._deserialize_value(k, key_type): Config._deserialize_value(v, value_type)
                for k, v in value.items()
            }

        # Nested Config
        if isinstance(typ, type) and issubclass(typ, Config):
            return typ.from_dict(value)

        # Enum
        if isinstance(typ, type) and issubclass(typ, Enum):
            return typ(value)

        return value

    def save(self, filepath):
        """
        Save config to a JSON file.

        Args:
            filepath (str): Path to file.
        """
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load(cls, filepath):
        """
        Load config from a JSON file.

        Args:
            filepath (str): Path to file.

        Returns:
            Config: Loaded config instance.
        """
        with open(filepath) as f:
            data = json.load(f)

        return cls.from_dict(data)


CONFIG_REGISTRY = {}


@api_util.export("_internal.register_config")
def register_config(cls):
    """
    Register a config class globally.
    Useful for dynamic lookup and deserialization.

    Args:
        cls (type): Config class.

    Returns:
        type: Registered class.
    """
    CONFIG_REGISTRY[cls.__name__] = cls
    return cls


@api_util.export("configs.LayerConfig")
@register_config
@dataclass
class LayerConfig(Config):
    """
    Configuration for a single layer/module.

    Args:
        name (str):
            The name of the layer/module. (e.g., 'Conv', 'Linear', 'BatchNorm', etc.)

        kwargs (dict[str, Any]):
            A dictionary of keyword arguments for the layer/module. (e.g. {'kernel_size': 3, 'stride': 1, 'padding': 1})

    Examples:
        >>> conv_cfg = LayerConfig(name='Conv', kwargs={'kernel_size': 3, 'stride': 1, 'padding': 1})
        >>> batchnorm_cfg = LayerConfig(name='BatchNorm', kwargs={'affine': True})

    Note:
    If being used within an im2sim model, inferred inputs like `in_channels` and `out_channels` should not be included and
    will be automatically set based on the model's architecture.
    The rank will also be inferred from the model's architecture and should not be included in the configuration.

    """

    name: str
    kwargs: dict[str, Any] = None
