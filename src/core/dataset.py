import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from src.utils import (
    are_keypoints_valid,
    keypoints_with_visibility,
    keypoints_to_bbox,
    keypoint_scaler
)
import cv2
import numpy as np

class KeypointPyTorch(Dataset):
    """Maybe make a Dataset class for each model, rather than one global?"""
    def __init__(self, data, cfg, transform=None):
        self.img_dir = cfg.img_dir
        self.data = data
        self.width = cfg.width
        self.height = cfg.height
        self.scale = cfg.scale
        self.cfg = cfg

        p = 0.4 # add to cfg
        if transform:
            self.transform = A.Compose([
                A.Resize(width=cfg.width, height=cfg.height),
                A.MotionBlur(blur_limit=3, p=0.2),
                A.MedianBlur(blur_limit=3, p=0.2),
                A.PixelDropout(dropout_prob=0.01, p=p),
                A.RandomGamma(p=p),
                A.Normalize(mean=cfg.mean, std=cfg.std),
                ToTensorV2(p=1.0),
                # should one use "remove_invisible" or not?
                ], keypoint_params=A.KeypointParams(format="xy"))
        else:
            self.transform = A.Compose([
                A.Resize(width=cfg.width, height=cfg.height),
                A.Normalize(mean=cfg.mean, std=cfg.std),
                ToTensorV2(p=1.0),
                # should one use "remove_invisible" or not?
                ], keypoint_params=A.KeypointParams(format="xy"))

    def __len__(self):
        return len(self.data)
    
    def clip_kps(self, kps, width, height):
        kps[:, 0] = np.clip(kps[:, 0], 0, width - 1)
        kps[:, 1] = np.clip(kps[:, 1], 0, height - 1)
        return kps

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        
        # Load image
        image_path = f"{self.img_dir}/{item['id']}.png"
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        
        # will this work when imread is inside?
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        h, w, _ = image.shape

        keypoints = np.array(item["kps"], dtype=np.float32).reshape(-1, 2)

        # need to clip based on actual image size
        keypoints = self.clip_kps(keypoints, width=w, height=h)
        
        transformed = self.transform(image=image, keypoints=keypoints)
        image = transformed["image"]
        keypoints = np.array(transformed["keypoints"], dtype=np.float32)

        # keep or remove? 
        # keypoints = self.clip_kps(keypoints, width=self.width, height=self.height)

        if self.cfg.model_name == "KeypointRCNN":
            keypoints_tensor = keypoints_with_visibility(keypoints)
            bbox = keypoints_to_bbox(
                keypoints, 
                offset=10,
                width=self.width, 
                height=self.height
            )
    
            target = {
                'boxes': bbox,
                'labels': torch.ones((1,), dtype=torch.int64), 
                'keypoints': keypoints_tensor.unsqueeze(0)
            }
            return image, target
        else:
            # make flatten an option in __init__?
            target = keypoint_scaler(keypoints.flatten(), self.width, self.height, self.scale)
            target = torch.tensor(target, dtype=torch.float32)
            return image, target

class KeypointData:
    def __init__(self, cfg, clean=True): # cfg the clean param
        self.train_data = pd.read_json(cfg.train_json)
        self.valid_data = pd.read_json(cfg.valid_json)
        self.cfg = cfg
        self.clean = clean

    def clean_keypoints(self, df, img_path):
        def get_shape(path):
            image = cv2.imread(path + ".png")
            if image is None:
                return (None, None)
            return image.shape[:2]
        
        def kps_outside(coords, height=720, width=1280):
            is_outside = False
            x = [x[0] for x in coords]
            y = [y[1] for y in coords]
        
            # combine np.min for x and y
            if np.max(x) > width or np.min(x) < 0 or np.max(y) > height or np.min(y) < 0:
                is_outside = True
            return is_outside
    
        df["img_path"] = img_path + "/" + df["id"]
        df[["height", "width"]] = pd.DataFrame(
            df["img_path"].apply(get_shape).tolist(),
            index=df.index
        )
        df["kps_outside"] = df.apply(
            lambda x: kps_outside(x["kps"], x["height"], x["width"]), axis=1
        )
        new_df = df[df["kps_outside"] != True]
        return new_df
    
    def _get_data_rcnn(self):
        train_data = self.train_data[
            self.train_data['kps'].apply(are_keypoints_valid)
        ]
        valid_data = self.valid_data[
            self.valid_data['kps'].apply(are_keypoints_valid)
        ]
        return train_data, valid_data

    def get_data(self, model_name):
        # any downsides by doing the PyTorch dataset inside here?
        if model_name == "KeypointRCNN":
            train, valid = self._get_data_rcnn()
        else:
            train, valid = self.train_data, self.valid_data

        # kinda does the same as _get_data_rcnn()?
        if self.clean:
            train = self.clean_keypoints(train, self.cfg.img_dir)
            valid = self.clean_keypoints(valid, self.cfg.img_dir)

        train_dataset = KeypointPyTorch(train, self.cfg, transform=self.cfg.train_aug)
        valid_dataset = KeypointPyTorch(valid, self.cfg, transform=False)
        
        return train_dataset, valid_dataset
    

class CollateFunction:
    def __init__(self, model_name):
        self.model_name = model_name

    @staticmethod
    def collate_rcnn(batch):
        images = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        
        images = torch.stack(images, dim=0)
        return images, targets
        
    @staticmethod
    def collate_fn(batch):
        images = [item[0] for item in batch]
        keypoints = [item[1] for item in batch]
        images = torch.stack(images)
    
        max_length = max(len(kps) for kps in keypoints)
        padded_keypoints = [torch.cat([kps, torch.zeros(max_length - len(kps))]) for kps in keypoints]
        keypoints = torch.stack(padded_keypoints)
        return images, keypoints

    def __call__(self, batch):
        if self.model_name == "KeypointRCNN":
            return self.collate_rcnn(batch)
        else:
            return self.collate_fn(batch)