import torch
import transformers
import math 
from torch.utils.data import DataLoader
from src.utils import get_model_and_config, update_cfg_from_args
from src.core import KeypointData, CollateFunction
from src.trainer import Trainer
import argparse
from config import BaseConfig

def main(args):
    model, cfg = get_model_and_config(args.model_name)

    # Update config based on args
    cfg = update_cfg_from_args(cfg, args)

    print(cfg.model_name, cfg.criterion, cfg.learn_rate)

    model = model.to(cfg.device)
    collate_fn = CollateFunction(cfg.model_name)

    train_data, valid_data = KeypointData(cfg).get_data(cfg.model_name)

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    valid_loader = DataLoader(
        dataset=valid_data,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learn_rate, weight_decay=cfg.weight_decay)
    num_train_steps = math.ceil(len(train_data) / cfg.batch_size) * cfg.epochs
    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_train_steps * cfg.warmup_ratio),
        num_training_steps=num_train_steps
    )

    train = Trainer(
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        criterion=cfg.criterion,
        scheduler=scheduler,
    )
    # TODO: save model and config 
    # TODO: fix index warning in torch.meshgrid()
    # TODO: add display sample images
    train.train(train_loader, valid_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="heatmap", help="resnet, heatmap, or rcnn")
    parser.add_argument('--batch_size', type=int, default=BaseConfig().batch_size, help='Batch size')
    parser.add_argument('--learn_rate', type=float, default=BaseConfig().learn_rate, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=BaseConfig().epochs, help='Epochs')
    args = parser.parse_args()
    print(args)

    main(args)