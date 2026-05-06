import math
import os
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import argparse
import json


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)

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

def main(args, use_amp):
    # Get model and config
    model, cfg = get_model_and_config(args.name)
    cfg = update_cfg_from_args(cfg, args)

    print(f"Using {'FP16' if use_amp else 'FP32'} precision for {cfg.model_name}")

    model = model.to(cfg.device)
    collate_fn = CollateFunction(cfg.model_name)

    # Get keypoint dataset
    train_data, valid_data = KeypointData(cfg, cfg.clean_dataset).get_data()

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

    print("Config is saved")

    train = Trainer(
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        use_amp=use_amp
    )

    history = train.train(train_loader, valid_loader)

    folder = os.path.join(cfg.folder, "")
    if cfg.onnx:
        success = convert_onnx_fp32(folder=folder)
        
        if success and use_amp:
            convert_onnx_fp16(folder)

if __name__ == "__main__":
    """
    Example usage:
    Train using fp16: python main.py --name "heatmap" --num_workers 4
    Train using fp32: python main.py --name "heatmap" --num_workers 4 --fp32

    RCNN currently requires FP32.

    Other parameters can be changed in the config (config/config.py and model_configs.py).

    Be careful with num_workers on Windows, num_workers > 0 might give unpredictable results or not work.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default="rcnn", help="resnet, heatmap, or rcnn") # find a different name?
    parser.add_argument('--fp32', action="store_true", help="Use FP32 instead of FP16")
    parser.add_argument('--num_workers', type=int, default=0, help="Number of workers")
    args = parser.parse_args()

    use_amp = not args.fp32

    if not torch.cuda.is_available():
        use_amp = False

    main(args, use_amp)