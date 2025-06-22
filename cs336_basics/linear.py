import torch
import torch.nn as nn
import math
from einops import rearrange, einsum

class Linear(nn.Module):
    
    def __init__(self, 
                in_features: int, 
                out_features: int, 
                device: torch.device | None = None,
                dtype: torch.dtype | None = None):
        
        super().__init__()
        std = math.sqrt(2.0 / (in_features + out_features))
        w = torch.empty((out_features, in_features), dtype=dtype, device=device)
        nn.init.trunc_normal_(w, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)
        self.W = nn.Parameter(w)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor : 
        output = einsum(self.W, x, "out_features in_features, ... in_features -> ... out_features")
        return output

