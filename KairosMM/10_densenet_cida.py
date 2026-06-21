import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet121, DenseNet121_Weights
from cida_plugin import CIDAPlugin, CIDAPluginConfig

NIH_CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

def make_cida_config_densenet(
    num_classes: int = 14,
    multi_label: bool = True,
    d_hidden: int = 512,
    max_rounds: int = 3,
    num_agents: int = 4,
    **kwargs,
) -> CIDAPluginConfig:
    return CIDAPluginConfig(
        d_input=1024, # DenseNet121 outputs 1024 channels
        d_hidden=d_hidden,
        d_message=d_hidden,
        num_agents=num_agents,
        num_classes=num_classes,
        max_rounds=max_rounds,
        multi_label=multi_label,
        num_attn_heads=8,
        role_prior_blend=0.30,
        positive_class_idx=1,
        lambda_cal=0.4,
        lambda_ac=0.2,
        min_disagreement=0.05,
        comm_dropout=0.1,
        early_stop_threshold=0.88,
        use_perspective_projector=True,
        **kwargs,
    )

class DenseNetCIDA(nn.Module):
    """
    Compact, End-to-End trainable DenseNet121 + CIDA Reasoning Engine.
    Designed specifically to maximize AUC on NIH ChestX-ray14.
    """
    def __init__(
        self,
        num_classes: int = 14,
        d_hidden: int = 512,
        max_rounds: int = 3,
        num_agents: int = 4,
        freeze_encoder: bool = False
    ):
        super().__init__()
        # Load pre-trained DenseNet121 (ImageNet weights)
        self.encoder = densenet121(weights=DenseNet121_Weights.DEFAULT)
        # We don't need the classifier head
        self.features = self.encoder.features
        
        # DenseNet121 output channels = 1024
        self.d_output = 1024
        
        if freeze_encoder:
            for param in self.features.parameters():
                param.requires_grad = False
                
        # Configuration for CIDA
        cida_config = make_cida_config_densenet(
            num_classes=num_classes,
            multi_label=True,
            d_hidden=d_hidden,
            max_rounds=max_rounds,
            num_agents=num_agents
        )
        
        self.cida = CIDAPlugin(cida_config)
        self.num_classes = num_classes
        
    def forward_scale(self, x: torch.Tensor, target_spatial_size: int = None):
        """Extract features from DenseNet"""
        out = self.features(x)
        # out is (B, 1024, H, W)
        
        # Global pooled representation
        pooled = F.adaptive_avg_pool2d(out, (1, 1)).flatten(1) # (B, 1024)
        
        # Spatial representation (flatten H, W into sequence)
        B, C, H, W = out.shape
        spatial = out.view(B, C, H * W).transpose(1, 2).contiguous() # (B, H*W, 1024)
        
        # We can optionally interpolate the spatial sequence length if needed
        if target_spatial_size is not None and spatial.size(1) != target_spatial_size:
            # Interpolate in 1D
            spatial = spatial.transpose(1, 2) # (B, 1024, seq_len)
            spatial = F.interpolate(spatial, size=target_spatial_size, mode='linear', align_corners=False)
            spatial = spatial.transpose(1, 2).contiguous()
            
        return pooled, spatial

    def forward(self, x_224: torch.Tensor, x_448: torch.Tensor = None):
        """
        Multi-Scale DenseNet encoding.
        If x_448 is None, uses only x_224 (for faster inference/training).
        """
        pooled_224, spatial_224 = self.forward_scale(x_224)
        
        if x_448 is not None:
            pooled_448, spatial_448 = self.forward_scale(x_448)
            pooled = 0.5 * pooled_224 + 0.5 * pooled_448
            spatial = torch.cat([spatial_224, spatial_448], dim=1) # (B, seq1 + seq2, 1024)
        else:
            pooled = pooled_224
            spatial = spatial_224
            
        # Run CIDA reasoning
        out = self.cida(
            pooled,
            seq_output=spatial
        )
        return out
