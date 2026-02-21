import os
import glob
from pathlib import Path

from torchvision import models

from config import (
    ResNetConfig,
    HeatmapConfig,
    RCNNConfig,
)
from src.models import (
    ResNetKeypoint,
    ResNetHeatmap,
    KeypointRCNN
)

model_dictionary = {
    "resnet18": {"model": models.resnet18, "input_size": 512, "weights": "IMAGENET1K_V1"}, 
    "resnet34": {"model": models.resnet34, "input_size": 512, "weights": "IMAGENET1K_V1"},
    "resnet50": {"model": models.resnet50, "input_size": 2048, "weights": "IMAGENET1K_V2"}
}

def get_model_and_config(name="resnet", classes=None):
    name = name.lower()

    valid_models = ["resnet", "heatmap", "rcnn"]
    assert name in valid_models, print(
        f"name should be one of: {valid_models}"
    )
    
    if name == "resnet":
        cfg = ResNetConfig
        model = ResNetKeypoint(
            model=model_dictionary[cfg.model]["model"],
            weights=model_dictionary[cfg.model]["weights"],
            input_size=model_dictionary[cfg.model]["input_size"],
            num_kps=cfg.num_kps,
            pretrained=cfg.pretrained
        )
    elif name == "heatmap":
        cfg = HeatmapConfig
        model = ResNetHeatmap(
            model=model_dictionary[cfg.model]["model"],
            weights=model_dictionary[cfg.model]["weights"],
            num_kps=cfg.num_kps,
            input_size=model_dictionary[cfg.model]["input_size"],
            pretrained=cfg.pretrained
        )
    elif name == "rcnn":
        cfg = RCNNConfig
        model = KeypointRCNN(cfg.num_kps)
    cfg.model_name = model.__class__.__name__
    return model, cfg


def load_model_inference(name, config):
    valid_models = ["resnet", "heatmap", "rcnn"]
    assert name in valid_models, print(
        f"name should either be {valid_models}"
    )
    
    if name == "heatmap":
        model = ResNetHeatmap(
            model=model_dictionary[config.model]["model"],
            weights=model_dictionary[config.model]["weights"],
            num_kps=config.num_kps,
            input_size=model_dictionary[config.model]["input_size"],
        )
    elif name == "resnet":
        model = ResNetKeypoint(
            model=model_dictionary[config.model]["model"],
            weights=model_dictionary[config.model]["weights"],
            input_size=model_dictionary[config.model]["input_size"],
            num_kps=config.num_kps,
            pretrained=config.pretrained
        )
    elif "rcnn":
        model = KeypointRCNN(config.num_kps)

    return model

def get_file_type(filename):
    """Determine file type based on extension"""
    filename = filename.lower()
    
    image_extensions = {
        '.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp'
    }
    video_extensions = {
        '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'
    }
    
    ext = os.path.splitext(filename)[1]
    
    if ext in image_extensions:
        return 'image'
    elif ext in video_extensions:
        return 'video'
    else:
        return 'unknown'

def get_images_from_folder(folder_path):
    """Get all image files from a folder."""
    image_extensions = {
        '.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp'
    }
    images = []
    for ext in image_extensions:
        images.extend(glob.glob(os.path.join(folder_path, f"*{ext}")))
        images.extend(glob.glob(os.path.join(folder_path, f"*{ext.upper()}")))
    return sorted(images)

def predict_folder(predictor, folder_path, batch_size=4, output_dir="predictions/"):
    """Process all images in a folder with batch inference."""
    images = list(set(
        get_images_from_folder(folder_path)))
    
    if not images:
        print(f"No images found in {folder_path}")
        return
    
    print(f"Found {len(images)} images in {folder_path}")
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = {}
    for i in range(0, len(images), batch_size):
        batch_paths = images[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1}/{(len(images) + batch_size - 1) // batch_size}")
        
        batch_keypoints = predictor.predict_batch(batch_paths)
        
        for img_path, keypoints in zip(batch_paths, batch_keypoints):
            filename = Path(img_path).stem
            output_path = os.path.join(output_dir, f"{filename}.png")
            predictor.draw_keypoints(img_path, keypoints, output_path, show=False)
            all_results[img_path] = keypoints

    return all_results