import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor

class VisionFeatureExtractor(nn.Module):
    """
    Extracts both pooled and spatial features from a Vision Backbone (ResNet50).
    Pooled features: (B, 2048)
    Spatial features: (B, 49, 2048) - where 49 = 7x7 grid
    """
    def __init__(self, model_name="resnet50", pretrained=True):
        super().__init__()
        if model_name == "resnet50":
            base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
            # We want the layer before global avg pool
            return_nodes = {
                'layer4': 'spatial_features',
                'avgpool': 'pooled_features'
            }
            self.feature_extractor = create_feature_extractor(base_model, return_nodes=return_nodes)
            self.d_output = 2048
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def forward(self, x):
        """
        x: (B, 3, H, W)
        Returns:
            pooled: (B, d_output)
            spatial: (B, L, d_output) where L is spatial grid size
        """
        features = self.feature_extractor(x)
        
        # Pooled features come as (B, 2048, 1, 1) -> (B, 2048)
        pooled = features['pooled_features'].flatten(1)
        
        # Spatial features come as (B, 2048, 7, 7) -> (B, 49, 2048)
        spatial = features['spatial_features']
        B, C, H, W = spatial.shape
        spatial = spatial.view(B, C, -1).transpose(1, 2) # (B, 49, 2048)
        
        return pooled, spatial
