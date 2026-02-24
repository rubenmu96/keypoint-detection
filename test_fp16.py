import torch, numpy as np, transformers, math
from src.utils import get_model_and_config
from src.core import KeypointData, CollateFunction
from src.trainer import Trainer
from torch.utils.data import DataLoader, Subset
import os

model, cfg = get_model_and_config('rcnn')
cfg.model_name = model.__class__.__name__
cfg.epochs = 2
cfg.display_examples = False
os.makedirs(cfg.folder, exist_ok=True)

model = model.cuda()
train_data, valid_data = KeypointData(cfg, cfg.clean_dataset).get_data(cfg.model_name)
train_data = Subset(train_data, np.arange(750))
valid_data = Subset(valid_data, np.arange(750))

collate_fn = CollateFunction(cfg.model_name)
train_loader = DataLoader(train_data, batch_size=4, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_data, batch_size=4, collate_fn=collate_fn)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
num_steps = len(train_loader) * cfg.epochs
scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, int(num_steps*0.15), num_steps)

trainer = Trainer(cfg=cfg, model=model, optimizer=optimizer, criterion=cfg.criterion,
                  scheduler=scheduler, use_amp=True)

history = trainer.train(train_loader, valid_loader)
for h in history:
    tl = h['training']['loss']
    vl = h['validation']['loss']
    print(f"train_loss={tl:.4f} valid_loss={vl:.4f} nan_train={math.isnan(tl)} nan_valid={math.isnan(vl)}")
print(f"Final scaler scale: {trainer.scaler.get_scale()}")
