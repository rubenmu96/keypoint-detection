import os
import glob

import torch

class BaseConfig:
    """Dataset and training parameters"""
    # Dataset parameters
    path = "dataset/tennis_court_det_dataset/data/"
    train_json = path + "data_train.json"
    valid_json = path + "data_val.json"
    img_dir = path + "images"
    clean_dataset = True
    train_aug = None
    num_kps = 14
    num_coords = 28

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training parameters
    epochs = 35
    batch_size = 4 # 4 for rcnn, 16 for heatmap/resnet
    learn_rate = 1e-5 # 5e-4 for heatmap/resnet
    weight_decay = 1e-4
    warmup_ratio = 0.15

    # Early stopping
    patience = None
    reset = False

    # Best model
    greater_is_better = False # True if higher value is better (e.g., pck@0.05), False if lower is better (e.g., loss, mpjpe)
    metric_tracker = "valid_loss" # supports valid_loss, valid_mpjpe, valid_pck@0.05

    """
    For testing model on images during training at each epoch
    - catching potential errors during training
    - see how the model improves or worsen for each epoch
    """
    display_examples = True
    
    # if not same length as sample_images_path, will use img_i instead as image name
    display_names = ["clay", "fed", "synframe", "synthetic"] # optional 
    images = (
        glob.glob(os.path.join("dataset/sample_images/", '*.jpg')) + 
        glob.glob(os.path.join("dataset/sample_images/", '*.png'))
    )
    sample_image_path = images[:4] # limit to first 4 images

    if len(sample_image_path) == 0:
        display_examples = False