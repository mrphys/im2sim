import json
from enum import Enum
from copy import deepcopy
from dataclasses import dataclass, fields
from typing import Any, TypeVar, get_args, get_origin, get_type_hints
import inspect

from im2sim.src.utils import api_util


T = TypeVar("T", bound="Config")


@api_util.export("_internal.Config")
@dataclass
class Config:
    """
    Base class for recursively serialisable configuration objects. 

    This class provides:  
    - Recursive (de)serialization to/from dictionaries and JSON  
    - Preset system for modifying configs declaratively  
    - Type-aware reconstruction using type hints  

    Subclasses should be defined as dataclasses.  

    Example:  
        ```python
        cfg = MyConfig(...)  
        cfg = cfg.apply_presets(["fast", "lightweight"])  
        cfg.save("config.json")  

        cfg2 = MyConfig().load("config.json") 
        ```
    """


    def __init_subclass__(cls):
        """ Automatically initialise a preset registry for each subclass. """
        super().__init_subclass__()
        cls._presets = {}
    
    @classmethod
    def register_preset(cls, name):
        """ 
        Register a preset function for this config class. 
        
        A preset is a function that takes a config instance and modifies it. 
        Args: 
            name (str): Name of the preset. 
            Returns: decorator: Function decorator. 
        
        Example: 
            ```python
            @MyConfig.register_preset("small") 
            def small(cfg): 
                cfg.hidden_dim = 32 
                return cfg 
            ```
        """
        def decorator(fn):
            cls._presets[name] = fn
            return fn
        return decorator

    def apply_presets(self, names: list[str]):
        """ 
        Apply a sequence of presets to a copy of this config. 

        Presets are applied in order. 
        
        Args: 
            names (list[str]): List of preset names. 

        Returns: 
            Config: Modified config instance. 
        """
        
        cfg = deepcopy(self)
        for name in names:
            cfg = self._presets[name](cfg)
        return cfg
    

    def as_kwargs(self):
        """ 
        Convert config fields into keyword arguments. 

        Returns: 
            dict: Mapping of field names to values. 
        """
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
        }

    def to_dict(self):
        """ 
        Recursively convert config into a serialisable dictionary. 

        Returns: 
            dict: Serialized representation. """
        return {
            "__class__": self.__class__.__name__,
            **{
                f.name: self._serialize_value(getattr(self, f.name))
                for f in fields(self)
            }
        }

    @classmethod
    def from_dict(cls: type[T], data: dict[str, Any]) -> T:
        """ 
        Reconstruct a config object from a dictionary. 
        Uses type hints to correctly deserialize nested configs, enums, lists, etc. 

        Args: 
            data (dict): Serialized config. 

        Returns: 
            Config: Reconstructed config object. """
        
        hints = get_type_hints(cls)

        kwargs = {}

        for key, value in data.items():
            if key == "__class__":
                continue

            field_type = hints.get(key)

            if field_type is not None:
                value = cls._deserialize_value(value, field_type)

            kwargs[key] = value

        return cls(**kwargs)
    
    

    @staticmethod
    def _serialize_value(value):
        """ Recursively serialize values into JSON-compatible structures. """
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
            return {
                k: Config._serialize_value(v)
                for k, v in value.items()
            }

        return value

    @staticmethod
    def _deserialize_value(value, typ):
        """ Recursively deserialize values based on type hints. """
        origin = get_origin(typ)
        args = get_args(typ)

        # Optional / Union
        if origin is type(None):
            return value

        if origin is list:
            subtype = args[0]
            return [
                Config._deserialize_value(v, subtype)
                for v in value
            ]

        if origin is dict:
            key_type, val_type = args
            return {
                k: Config._deserialize_value(v, val_type)
                for k, v in value.items()
            }

        # Handle Optional[T] / Union[T, None]
        if origin is not None and origin.__name__ == "Union":
            for subtype in args:
                if subtype is type(None):
                    continue
                try:
                    return Config._deserialize_value(value, subtype)
                except Exception:
                    pass
            return value

        # Nested configs
        if isinstance(typ, type) and issubclass(typ, Config):
            return typ.from_dict(value)

        # Enums
        if isinstance(typ, type) and issubclass(typ, Enum):
            return typ(value)

        return value
    
    def save(self, filepath):
        """ 
        Save config to a JSON file. 

        Args: 
            filepath (str): Path to file. 
        """
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
        
    def load(self, filepath):
        """ 
        Load config from a JSON file. 

        Args: 
            filepath (str): Path to file. 
        
        Returns: 
            Config: Loaded config instance. 
        """
        with open(filepath) as f:
            data = json.load(f)
        return self.from_dict(data)
    
    @classmethod
    def generate_documentation(cls) -> str:
        """ 
        Generate documentation for all registered presets. 

        Returns: 
            str: Formatted documentation string. 
        """
        cls_doc = inspect.getdoc(fn) or "No class docstring provided."
        preset_docs = []
        for name, fn in cls._presets.items():
            doc = inspect.getdoc(fn) or "No description provided."
            preset_docs.append(f"""
                        {name}
                        {'-' * len(name)}

                        {doc}
                        """.strip()
                                )
        preset_text = "\n\n".join(preset_docs)
        return f"""
                {cls_doc}

                Preset Configurations
                =====================

                {preset_text}
                """.strip()


    
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
    Attributes:
        name (str): The name of the layer/module. (e.g., 'Conv', 'Linear', 'BatchNorm', etc.)
        kwargs (dict[str, Any]): A dictionary of keyword arguments for the layer/module. (e.g. {'kernel_size': 3, 'stride': 1, 'padding': 1})
    """
    name: str
    kwargs: dict[str, Any] = None


@api_util.export("_internal.ConfigurableModule")
class ConfigurableModule:
    """
    Base class for modules that can be constructed from a Config.
    Provides a standard interface for building modules from configs.
    """
    @classmethod
    def build(cls, 
            rank:int, 
            in_channels:int, 
            out_channels:int, 
            cfg: Config):
        """ 
        Instantiate a module using a config object. 

        Args: 
            rank (int): Spatial rank (1D, 2D, 3D). 
            in_channels (int): Input channels. 
            out_channels (int): Output channels. 
            cfg (Config): Configuration object. 

        Returns: 
            nn.Module: Instantiated module. 
        """
        return cls(in_channels, out_channels, **cfg.as_kwargs(), rank=rank)



