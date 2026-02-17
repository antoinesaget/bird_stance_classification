from __future__ import annotations

import torch
import torch.nn as nn


class MultiHeadAttributeModel(nn.Module):
    def __init__(self, backbone_name: str = "convnextv2_small", pretrained: bool = True) -> None:
        super().__init__()
        import timm

        self.backbone_name = backbone_name
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        feat_dim = int(getattr(self.backbone, "num_features", 768))

        self.readability = nn.Linear(feat_dim, 2)
        self.activity = nn.Linear(feat_dim, 3)
        self.support = nn.Linear(feat_dim, 3)
        self.resting_back = nn.Linear(feat_dim, 2)
        self.legs = nn.Linear(feat_dim, 3)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.backbone(x)
        return {
            "readability": self.readability(feat),
            "activity": self.activity(feat),
            "support": self.support(feat),
            "resting_back": self.resting_back(feat),
            "legs": self.legs(feat),
        }

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True


class ImageStatusModel(nn.Module):
    def __init__(self, backbone_name: str = "convnext_tiny", pretrained: bool = True) -> None:
        super().__init__()
        import timm

        self.backbone_name = backbone_name
        self.model = timm.create_model(backbone_name, pretrained=pretrained, num_classes=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
