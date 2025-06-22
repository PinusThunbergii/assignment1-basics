import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, einsum

# deactivate
# conda activate base
# uv run pytest -k test_linear
# uv run pytest -k test_embedding

class Linear(nn.Module):
    
    def __init__(self, 
                in_features: int, 
                out_features: int, 
                device: torch.device | None = None,
                dtype: torch.dtype | None = None):
        
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        std = math.sqrt(2.0 / (in_features + out_features))
        w = torch.empty((out_features, in_features), dtype=dtype, device=device)
        nn.init.trunc_normal_(w, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)
        self.W = nn.Parameter(w)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor : 
        output = einsum(self.W, x, "out_features in_features, ... in_features -> ... out_features")
        return output

class Emmbedding(nn.Module):
    
    def __init__(self, 
                vocab_size: int, 
                d_model: int, 
                device: torch.device | None = None, 
                dtype: torch.dtype | None = None  ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        std = math.sqrt(2.0 / (vocab_size + d_model))
        e = torch.empty((vocab_size, d_model), dtype=dtype, device=device)
        nn.init.trunc_normal_(e, mean=0.0, std=1.0, a=-3.0, b=3.0)
        self.E = nn.Parameter(e)
        # (vocab_size, d_model)
        
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor: 
        # token_ids (batch_size, sequence_length)
        # output (batch, sequence_length, embeddings_dim)
        assert token_ids.dtype == torch.long, "Expeted only long tensor"
        return self.E[token_ids]
        
        # one_hot = F.one_hot(token_ids, num_classes=self.vocab_size).float()
        # e = one_hot @ self.E
        # return e
        
    
    