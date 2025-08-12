import torch

class ModelConfig:
    def __init__(self, base_config=None, **kwargs):
        self.base = base_config

        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def __getattr__(self, name):
        return getattr(self.base, name)

class BaseConfig:
    path = "dataset/tennis_court_det_dataset/data/"
    train_json = path + "data_train.json"
    valid_json = path + "data_val.json"
    img_dir = path + "images"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    width = 672
    height = 448
    batch_size = 16
    epochs = 35
    patience = None
    learn_rate = 5e-4
    weight_decay = 1e-4
    warmup_ratio = 0.15
    num_kps = 14
    num_coords = 28
    reset = True

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_aug = None
    pretrained = True

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