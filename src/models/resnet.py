import torch
from torchvision import models
import torch.nn as nn

class ResNetKeypoint(nn.Module):
    def __init__(self, model=models.resnet34, weights="IMAGENET1K_V1", num_kps=14, pretrained=True): # add support for all resnet models!
        # create a bit better classification head?
        super().__init__()
        if pretrained:
            self.model = model(weights=weights)
        else:
            self.model = model(weights=None)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_kps*2) # 14 keypoints with (x, y)

    def forward(self, x):
        return self.model(x)