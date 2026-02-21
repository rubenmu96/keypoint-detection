from torchvision import models
import torch.nn as nn

class ResNetHeatmap(nn.Module):
    def __init__(
            self, model=models.resnet34, weights="IMAGENET1K_V1", 
            num_kps=14, input_size=512, pretrained=True
        ):
        super().__init__()
        if pretrained:
            self.backbone = model(weights=weights)
        else:
            self.backbone = model(weights=None)
        self.model_size = self.backbone.fc.in_features

        self.backbone.avgpool = nn.Identity()
        self.backbone.fc = nn.Identity()

        self.backbone.layer4 = self._make_dilated(self.backbone.layer4)

        upsample_factor = input_size // 32
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(self.model_size, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_kps, kernel_size=1),
            nn.ConvTranspose2d(num_kps, num_kps, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(num_kps, num_kps, kernel_size=upsample_factor//2 * 2, stride=upsample_factor//2, padding=1),
        )

    def _make_dilated(self, layer):
        layers = list(layer.children())
        layers[-1] = nn.Conv2d(
            self.model_size, self.model_size, kernel_size=3, stride=1, dilation=2, padding=2
        )
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.heatmap_head(x)
        return x