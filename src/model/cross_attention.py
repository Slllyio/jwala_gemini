import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionFusion(nn.Module):
    \"\"\"
    Cross-Attention Fusion for Optical (S2) and SAR (S1) data.
    Allows the Radar signal to 'query' the Optical features to filter out seasonal senescence.
    \"\"\"
    def __init__(self, embed_dim=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.layer_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, optical_feats, sar_feats):
        # SAR queries Optical
        # optical_feats: (B, Seq, Dim)
        # sar_feats: (B, Seq, Dim)
        attn_out, _ = self.multihead_attn(query=sar_feats, key=optical_feats, value=optical_feats)
        
        # Add & Norm
        out1 = self.layer_norm(sar_feats + attn_out)
        
        # FFN
        ffn_out = self.ffn(out1)
        
        # Add & Norm
        out2 = self.layer_norm2(out1 + ffn_out)
        
        return out2
