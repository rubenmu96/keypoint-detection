import torch
import inspect
from types import SimpleNamespace

def update_cfg_from_args(cfg, args):
    """Combine parser args and config"""
    for key, value in vars(args).items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg

def dict_to_config(d):
    """Convert dict to config"""
    config = SimpleNamespace()
    for k, v in d.items():
        setattr(config, k, v)
    
    if not hasattr(config, 'device'):
        config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    return config

def config_to_dict(config):
    """Serialize a config class (or instance) to a plain dict."""
    if inspect.isclass(config):
        # Get class and subclass, subclass parameters wins if conflict
        attrs: dict = {}
        for cls in reversed(config.__mro__):
            if cls is object:
                continue
            for k, v in vars(cls).items():
                if not k.startswith('__') and not callable(v):
                    attrs[k] = v
    else:
        attrs = {k: v for k, v in vars(config).items()
                 if not k.startswith('__') and k != 'base' and not callable(v)}

    # Normalise non-JSON-serialisable types
    for k, v in list(attrs.items()):
        if isinstance(v, torch.device):
            attrs[k] = str(v)
        elif isinstance(v, torch.nn.Module):
            attrs[k] = v.__class__.__name__

    return attrs