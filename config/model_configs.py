from config import BaseConfig, ModelConfig
from torchvision import models
import torch 

# TODO: move to config.py?
# NOTE: these will overwrite BaseConfig if they got any of the same names

# ResNetConfig = ModelConfig(
#     base_config=BaseConfig,
#     model=models.resnet50,
#     weights="IMAGENET1K_V2",
#     pretrained=True,
#     criterion=torch.nn.SmoothL1Loss(),
#     scale=(0, 1),
#     mean=(0.485, 0.456, 0.406),
#     std=(0.229, 0.224, 0.225),
#     save_path="resnet.pth",
# )

ResNetConfig = ModelConfig(
    base_config=BaseConfig,
    model=models.resnet34,
    weights="IMAGENET1K_V1",
    pretrained=True,
    criterion=torch.nn.SmoothL1Loss(),
    scale=(0, 1),
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    num_kps=14,
    num_coords=28,
    width=672,
    height=448,
    save_path="resnet.pth",
)


# HeatmapConfig = ModelConfig(
#     base_config=BaseConfig,
#     model=models.resnet50,
#     model_size=2048, # rename to something else?
#     input_size=2048,
#     weights="IMAGENET1K_V2",
#     pretrained=True,
#     criterion=torch.nn.BCEWithLogitsLoss(),
#     sigma=1,
#     scale=(0, 1),
#     mean=(0.485, 0.456, 0.406),
#     std=(0.229, 0.224, 0.225),
#     save_path="heatmap.pth",
# )

HeatmapConfig = ModelConfig(
    base_config=BaseConfig,
    # model=models.resnet34,
    # model_size=512,
    # input_size=512,
    model="resnet34",
    weights="IMAGENET1K_V1",
    pretrained=True,
    # criterion=torch.nn.BCEWithLogitsLoss(),
    criterion="bcelogitloss",
    sigma=1.4,
    scale=(0, 1),
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    num_kps=14,
    num_coords=28,
    width=672,
    height=448,
    save_path="heatmap_pretrained_resnet34_672x448.pth",
)

RCNNConfig = ModelConfig(
    base_config=BaseConfig,
    mean=(0, 0, 0),
    std=(1, 1, 1),
    criterion=torch.nn.MSELoss(),
    scale=None,
    num_kps=14,
    num_coords=28,
    width=672,
    height=448,
    save_path="keypoint_rcnn.pth",
)