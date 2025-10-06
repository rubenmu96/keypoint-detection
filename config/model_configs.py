from config import BaseConfig, ModelConfig

ResNetConfig = ModelConfig(
    base_config=BaseConfig,
    model="resnet18",
    pretrained=True,
    criterion="smoothl1loss",
    scale=(0, 1),
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    num_kps=14,
    num_coords=28,
    width=672,
    height=448,
    save_path="resnet18.pth",
)

HeatmapConfig = ModelConfig(
    base_config=BaseConfig,
    model="resnet18",
    pretrained=True,
    criterion="bcelogitloss",
    sigma=1.4,
    scale=(0, 1),
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    num_kps=14,
    num_coords=28,
    width=672,
    height=448,
    # create a function for save-name, maybe in parserargs to config function
    save_path="heatmap_pretrained_resnet18_672x448.pth",
)

RCNNConfig = ModelConfig(
    base_config=BaseConfig,
    mean=(0, 0, 0),
    std=(1, 1, 1),
    criterion="mseloss",
    scale=None,
    num_kps=14,
    num_coords=28,
    width=672,
    height=448,
    save_path="keypoint_rcnn.pth",
)