"""
Testing a subset of the training data to check if code is working.
"""
import os
import math
import argparse
import json

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset
import numpy as np

from config import (
    config_to_dict,
    update_cfg_from_args
)
from src.utils import get_model_and_config
from src.core import KeypointData, CollateFunction
from src.trainer import (
    Trainer,
    convert_onnx_fp16,
    convert_onnx_fp32
)

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)

def main(args, use_amp, test_size=750):
    # Get model and config
    model, cfg = get_model_and_config(args.name)
    cfg = update_cfg_from_args(cfg, args)

    print(f"Using {'FP16' if use_amp else 'FP32'} precision for {cfg.model_name}")

    model = model.to(cfg.device)
    collate_fn = CollateFunction(cfg.model_name)

    # Get keypoint dataset
    train_data, valid_data = KeypointData(cfg, cfg.clean_dataset).get_data()

    indices = np.arange(test_size)

    train_data = Subset(train_data, indices)
    valid_data = Subset(valid_data, indices)

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=args.num_workers,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=2 if args.num_workers > 0 else None
    )
    valid_loader = DataLoader(
        dataset=valid_data,
        batch_size=cfg.batch_size,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=args.num_workers,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=2 if args.num_workers > 0 else None
    )

    # Optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learn_rate, weight_decay=cfg.weight_decay
    )
    num_train_steps = len(train_loader) * cfg.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(num_train_steps * cfg.warmup_ratio),
        num_training_steps=num_train_steps
    )

    config_dict = config_to_dict(cfg)
    
    os.makedirs(cfg.folder, exist_ok=True)
    with open(f'{cfg.folder}/{args.name}_config.json', 'w') as f:
        json.dump(config_dict, f, indent=4)

    train = Trainer(
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        use_amp=use_amp
    )

    train.train(train_loader, valid_loader)

    folder = os.path.join(cfg.folder, "")
    if cfg.onnx:
        success = convert_onnx_fp32(folder=folder)
        
        if success and use_amp:
            convert_onnx_fp16(folder)

if __name__ == "__main__":
    """
    How to run:
    python test.py --name MODEL_NAME --test_size NUMBER_OF_TEST_SAMPLES --num_workers NUMBER_OF_WORKERS [--fp32]
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default="rcnn", help="resnet, heatmap, or rcnn")
    parser.add_argument('--test_size', type=int, default=750, help="How many samples in training set")
    parser.add_argument('--num_workers', type=int, default=0, help="Number of workers")
    parser.add_argument('--fp32', action="store_true", help="Use FP32 instead of FP16")
    args = parser.parse_args()

    use_amp = not args.fp32

    if not torch.cuda.is_available():
        use_amp = False

    main(args, use_amp, test_size=args.test_size)