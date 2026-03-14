import cv2
import numpy as np
import pandas as pd

import imagesize
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from src.utils import (
    keypoints_with_visibility,
    keypoints_region,
    keypoint_scaler
)

class KeypointPyTorch(Dataset):
    """Maybe make a Dataset class for each model, rather than one global?"""
    def __init__(self, data, cfg, transform=None):
        self.img_dir = cfg.img_dir
        self.data = data.copy()
        # Turn into np.arrays() here instead of __getitem_()
        self.data["kps"] = data["kps"].apply(lambda x: np.array(x, dtype=np.float32).reshape(-1, 2))
        self.width = cfg.width
        self.height = cfg.height
        self.scale = cfg.scale

        self.model_name = cfg.model_name

        # Get albumentation transformation
        if cfg.model_name == "KeypointRCNN":
            # KeypointRCNN's GeneralizedRCNNTransform expects [0, 1] float tensors
            if transform:
                self.transform = self._train_transform_rcnn(cfg)
            else:
                self.transform = self._val_transform_rcnn(cfg)
        else:
            if transform:
                self.transform = self._train_transform(cfg)
            else:
                self.transform = self._val_transform(cfg)

    @staticmethod
    def _train_transform(cfg, p=0.4):
        return A.Compose([
            A.Resize(width=cfg.width, height=cfg.height),
            A.MotionBlur(blur_limit=3, p=0.2),
            A.MedianBlur(blur_limit=3, p=0.2),
            A.PixelDropout(dropout_prob=0.01, p=p),
            A.RandomGamma(p=p),
            A.Normalize(mean=cfg.mean, std=cfg.std),
            ToTensorV2(p=1.0),
            ], keypoint_params=A.KeypointParams(format="xy"))

    @staticmethod
    def _val_transform(cfg):
        return A.Compose([
            A.Resize(width=cfg.width, height=cfg.height),
            A.Normalize(mean=cfg.mean, std=cfg.std),
            ToTensorV2(p=1.0),
            ], keypoint_params=A.KeypointParams(format="xy"))

    @staticmethod
    def _train_transform_rcnn(cfg, p=0.4):
        """
        No A.Normalize — model expects [0, 1] float tensors.
        A.ToFloat(max_value=255) divides by 255 to get [0, 1] float32.
        ToTensorV2 then does HWC → CHW without any further scaling.
        """
        return A.Compose([
            A.Resize(width=cfg.width, height=cfg.height),
            A.MotionBlur(blur_limit=3, p=0.2),
            A.MedianBlur(blur_limit=3, p=0.2),
            A.PixelDropout(dropout_prob=0.01, p=p),
            A.RandomGamma(p=p),
            A.ToFloat(max_value=255),
            ToTensorV2(p=1.0),
            ], keypoint_params=A.KeypointParams(format="xy"))

    @staticmethod
    def _val_transform_rcnn(cfg):
        return A.Compose([
            A.Resize(width=cfg.width, height=cfg.height),
            A.ToFloat(max_value=255),
            ToTensorV2(p=1.0),
            ], keypoint_params=A.KeypointParams(format="xy"))

    def __len__(self):
        return len(self.data)
    
    def clip_kps(self, kps, width, height):
        kps = kps.copy()
        kps[:, 0] = np.clip(kps[:, 0], 0, width - 1)
        kps[:, 1] = np.clip(kps[:, 1], 0, height - 1)
        return kps

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        image_path = f"{self.img_dir}/{item['id']}.png"
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        h, w, _ = image.shape

        keypoints = item["kps"]

        keypoints = self.clip_kps(keypoints, width=w, height=h)
        
        # Apply albumentations transformation
        transformed = self.transform(image=image, keypoints=keypoints)
        image = transformed["image"]
        keypoints = np.array(transformed["keypoints"], dtype=np.float32)

        keypoints = self.clip_kps(keypoints, width=self.width, height=self.height)

        if self.model_name == "KeypointRCNN":
            keypoints_tensor = keypoints_with_visibility(keypoints)
            bbox = keypoints_region(
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
            target = keypoint_scaler(keypoints.flatten(), self.width, self.height, self.scale)
            target = torch.tensor(target, dtype=torch.float32)
            return image, target

class KeypointData:
    def __init__(self, cfg, clean=True):
        self.train_data = pd.read_json(cfg.train_json)
        self.valid_data = pd.read_json(cfg.valid_json)
        self.cfg = cfg
        self.clean = clean
    
    def clean_keypoints(self, df, img_path):
        """Remove keypoints that are outside image boundary."""
        def get_shape(path):
            try:
                width, height = imagesize.get(path + ".png")
                return height, width
            except Exception:
                return None, None
        
        def kps_outside(coords, height, width):
            if height is None:
                return True
            coords = np.asarray(coords)
            x, y = coords[:, 0], coords[:, 1]
            return x.max() > width or x.min() < 0 or y.max() > height or y.min() < 0
        
        df = df.copy()
        df["img_path"] = f"{img_path}/" + df["id"]
        
        df["height"], df["width"] = zip(*df["img_path"].map(get_shape))
        
        mask = ~df.apply(lambda r: kps_outside(r["kps"], r["height"], r["width"]), axis=1)
        
        return df.loc[mask]
    
    def get_data(self, model_name):
        train, valid = self.train_data, self.valid_data

        # clean_keypoints filters rows whose keypoints fall outside the actual
        # image boundaries (using real per-image dimensions via imagesize).
        if self.clean:
            train = self.clean_keypoints(train, self.cfg.img_dir)
            valid = self.clean_keypoints(valid, self.cfg.img_dir)

        train_dataset = KeypointPyTorch(train, self.cfg, transform=self.cfg.train_aug)
        valid_dataset = KeypointPyTorch(valid, self.cfg, transform=False)
        
        return train_dataset, valid_dataset
    

class CollateFunction:
    """Collate function for stacking samples into batch"""
    def __init__(self, model_name):
        self.model_name = model_name

    @staticmethod
    def collate_rcnn(batch):
        """Collate function for Keypoint R-CNN"""
        images = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        
        images = torch.stack(images, dim=0)
        return images, targets
        
    @staticmethod
    def collate_fn(batch):
        """Collate function for Resnet/Heatmap"""
        images = [item[0] for item in batch]
        keypoints = [item[1] for item in batch]
        images = torch.stack(images)
        keypoints = torch.stack(keypoints)
        return images, keypoints

    def __call__(self, batch):
        if self.model_name == "KeypointRCNN":
            return self.collate_rcnn(batch)
        else:
            return self.collate_fn(batch)