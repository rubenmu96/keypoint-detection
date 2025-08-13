import torch.nn as nn
import torchvision
from torchvision.models.detection.keypoint_rcnn import KeypointRCNNPredictor

class KeypointRCNN(nn.Module):
    def __init__(self, num_kps, num_classes=2):
        super().__init__()
        self.model = torchvision.models.detection.keypointrcnn_resnet50_fpn(
            weights='DEFAULT',
            num_classes=num_classes
        )
        
        in_features = self.model.roi_heads.keypoint_predictor.kps_score_lowres.in_channels
        self.model.roi_heads.keypoint_predictor = KeypointRCNNPredictor(
            in_channels=in_features,
            num_keypoints=num_kps
        )

    def forward(self, images, targets=None):
        # might have to fix for targets=None?
        return self.model(images, targets)