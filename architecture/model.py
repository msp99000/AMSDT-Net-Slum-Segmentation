import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from architecture.multiscale_feature import MultiScaleFeatureExtraction
from architecture.residual import ResidualBlock
from architecture.dynamic_conv import DynamicConv
from architecture.encoder import TransformerEncoder
from architecture.boundary_refine import BoundaryRefinementModule
from architecture.cbam import CBAM
from architecture.fpn import FPN

class SlumSegNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.multi_scale = MultiScaleFeatureExtraction(
            config['input_channels'],
            config['multi_scale_channels']
        )
        
        self.residual1 = ResidualBlock(config['multi_scale_channels'], config['residual_channels'])
        
        self.transformer_encoder = TransformerEncoder(
            dim=config['transformer']['dim'],
            depth=config['transformer']['depth'],
            heads=config['transformer']['heads'],
            mlp_dim=config['transformer']['mlp_dim'],
            dropout=config['transformer']['dropout']
        )
        
        self.residual2 = ResidualBlock(config['residual_channels'], config['residual_channels'])
        
        # FPN layers
        self.fpn_layers = nn.ModuleList([
            FPN(config['fpn_channels'][i], config['fpn_channels'][i+1])
            for i in range(len(config['fpn_channels']) - 1)
        ])
        
        # CBAM attention layers
        self.cbam_layers = nn.ModuleList([
            CBAM(channels, config["channel_attention_reduction_ratio"], config["spactial_attention_kernel_size"]) for channels in config['fpn_channels'][1:]
        ])
        
        self.decoder = nn.ModuleList([
            nn.Sequential(
                DynamicConv(config['decoder_channels'][i], config['decoder_channels'][i+1], kernel_size=3, padding=1),
                ResidualBlock(config['decoder_channels'][i+1], config['decoder_channels'][i+1])
            ) for i in range(len(config['decoder_channels']) - 1)
        ])
        
        self.boundary_refinement = BoundaryRefinementModule(config['decoder_channels'][-1])
        
        self.final_conv = nn.Conv2d(config['decoder_channels'][-1], config['num_classes'], kernel_size=1)
    
    def forward(self, satellite):
        x = self.multi_scale(satellite)
        
        x = self.residual1(x)
        
        # Transformer encoder
        x_res = x
        x = rearrange(x, 'b c h w -> (h w) b c')
        x = self.transformer_encoder(x)
        x = rearrange(x, '(h w) b c -> b c h w', h=satellite.shape[2], w=satellite.shape[3])
        x = x + x_res  # Residual connection around transformer
        
        x = self.residual2(x)
        
        # FPN and CBAM
        fpn_features = [x]
        for fpn, cbam in zip(self.fpn_layers, self.cbam_layers):
            x = fpn(x, fpn_features[-1] if len(fpn_features) > 1 else None)
            x = cbam(x)
            fpn_features.append(x)
        
        # Decoder
        for conv in self.decoder:
            x = conv(x)
        
        # Boundary refinement
        x = self.boundary_refinement(x)
        
        # Final convolution
        x = self.final_conv(x)
        
        return x
