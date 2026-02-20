from __future__ import annotations

import torch
import torch.nn as nn


class MultiHeadAttributeModel(nn.Module):
    def __init__(
        self,
        backbone_name: str = "convnextv2_tiny.fcmae_ft_in1k",
        pretrained: bool = True,
        heads: list[str] | None = None,
    ) -> None:
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

        # Default to all heads if not specified
        if heads is None:
            heads = ["readability", "specie", "behavior", "substrate", "legs"]
        
        self.heads = heads
        head_configs = {
            "readability": 3,
            "specie": 3,
            "behavior": 7,
            "substrate": 4,
            "legs": 4,
        }

        for head in heads:
            if head not in head_configs:
                raise ValueError(f"Unknown head: {head}. Must be one of {list(head_configs.keys())}")
            setattr(self, head, nn.Linear(feat_dim, head_configs[head]))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.backbone(x)
        return {head: getattr(self, head)(feat) for head in self.heads}

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
