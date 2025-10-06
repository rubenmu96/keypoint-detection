import torch
import inspect
from types import SimpleNamespace

class BaseConfig:
    path = "dataset/tennis_court_det_dataset/data/"
    train_json = path + "data_train.json"
    valid_json = path + "data_val.json"
    img_dir = path + "images"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 8 # 16 for resnet/heatmap, 
    epochs = 35
    patience = None
    learn_rate = 5e-4
    weight_decay = 1e-4
    warmup_ratio = 0.15
    reset = True
    train_aug = None

    # For testing model on images during training
    display_examples = True
    display_names = ["clay", "fed", "synframe", "synthetic"]
    # make it possible to just search for all .jpg or .png inside a folder
    sample_image_path = [
        "dataset/sample_images/clay.jpg",
        "dataset/sample_images/fed.jpg",
        "dataset/sample_images/synframe.jpg",
        "dataset/sample_images/synthetic.jpg"
    ]
    if len(sample_image_path) == 0:
        display_examples = False

class ModelConfig:
    def __init__(self, base_config=None, **kwargs):
        self.base = base_config

        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def __getattr__(self, name):
        return getattr(self.base, name)
    
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