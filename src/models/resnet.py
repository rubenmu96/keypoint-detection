import torch
from torchvision import models
import torch.nn as nn

class ResNetKeypoint(nn.Module):
    def __init__(self, model=models.resnet34, weights="IMAGENET1K_V1", input_size=512, num_kps=14, pretrained=True):
        super().__init__()
        if pretrained:
            self.model = model(weights=weights)
        else:
            self.model = model(weights=None)
        # self.model.fc = torch.nn.Linear(input_size, num_kps*2) # 14 keypoints with (x, y)

        # self.model.fc = nn.Identity()
        self.backbone = nn.Sequential(*list(self.model.children())[:-2])

        self.regression_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(input_size, 256),
            # nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, num_kps * 2)
        )

    def forward(self, x):
        features = self.backbone(x)
        features = self.regression_head(features)
        return features