"""Keypoint R-CNN model"""
import torch
from torch import nn
import torchvision
from torchvision.models.detection.keypoint_rcnn import KeypointRCNNPredictor


class KeypointRCNN(nn.Module):
    """Keypoint R-CNN (ResNet-50 FPN) with its keypoint head resized to num_kps.

    pretrained=True loads the COCO keypoint weights. pretrained=False builds the
    model with random weights and downloads nothing. Note that the two variants are
    not checkpoint-compatible: torchvision uses FrozenBatchNorm2d for a pretrained
    backbone and BatchNorm2d otherwise, so load a checkpoint into a model built
    with the same pretrained setting it was trained with.

    Extra keyword arguments (e.g. min_size, max_size) are passed through to
    torchvision's keypointrcnn_resnet50_fpn.
    """
    def __init__(self, num_kps, num_classes=2, score_thresh=0.005, pretrained=True, **kwargs):
        super().__init__()
        self.num_kps = num_kps
        self.model = torchvision.models.detection.keypointrcnn_resnet50_fpn(
            weights="DEFAULT" if pretrained else None,
            # weights=None alone still downloads ImageNet backbone weights.
            weights_backbone="DEFAULT" if pretrained else None,
            num_classes=num_classes,
            box_score_thresh=score_thresh,
            **kwargs,
        )

        in_features = self.model.roi_heads.keypoint_predictor.kps_score_lowres.in_channels
        self.model.roi_heads.keypoint_predictor = KeypointRCNNPredictor(
            in_channels=in_features,
            num_keypoints=num_kps
        )


    def forward(self, images, targets=None):
        """Run the wrapped torchvision model in its current mode.

        In train mode, targets are required and the loss dict is returned. In eval
        mode, a list with one prediction dict per image is returned.
        """
        return self.model(images, targets)


    def get_loss(self, images, targets):
        """Return the loss dict for images and targets. Leaves the model in train mode."""
        self.train()
        loss_dict = self.model(images, targets)
        return loss_dict


    def predict(self, images):
        """Return one prediction dict per image, computed without gradients.

        Leaves the model in eval mode.
        """
        self.eval()
        with torch.no_grad():
            return self.model(images)
