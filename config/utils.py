import torch
import inspect
from types import SimpleNamespace

def update_cfg_from_args(cfg, args):
    for key, value in vars(args).items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg

def dict_to_config(d):
    config = SimpleNamespace()
    for k, v in d.items():
        setattr(config, k, v)
    
    if not hasattr(config, 'device'):
        config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    return config

def config_to_dict(config):
    if inspect.isclass(config):
        attrs = {k: v for k, v in vars(config).items() 
                if not k.startswith('__') and not callable(v)}
    else:
        attrs = {k: v for k, v in vars(config).items() 
                if not k.startswith('__') and k != 'base' and not callable(v)}
    
    if hasattr(config, 'base'):
        attrs['__base_config__'] = config_to_dict(config.base)
    
    for k, v in attrs.items():
        if isinstance(v, torch.device):
            attrs[k] = str(v)
        elif isinstance(v, torch.nn.Module):
            attrs[k] = v.__class__.__name__
    
    return attrs