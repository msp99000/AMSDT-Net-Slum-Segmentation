from torch import nn

class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([[
                nn.LayerNorm(dim),
                nn.MultiheadAttention(dim, heads, dropout=dropout),
                nn.LayerNorm(dim),
                nn.Sequential(
                    nn.Linear(dim, mlp_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(mlp_dim, dim),
                    nn.Dropout(dropout)
                )
            ]
            for _ in range(depth)])
    
    def forward(self, x):
        for norm1, attn, norm2, mlp in self.layers:
            x = x + attn(norm1(x), norm1(x), norm1(x))[0]
            x = x + mlp(norm2(x))
        return x