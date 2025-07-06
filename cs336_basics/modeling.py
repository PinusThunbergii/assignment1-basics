import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, einsum, reduce

# deactivate
# conda activate base
# uv run pytest -k test_linear
# uv run pytest -k test_embedding
# uv run pytest -k test_rmsnorm
# uv run pytest -k test_swiglu
# uv run pytest -k test_rope

# uv run pytest -k test_softmax_matches_pytorch

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
        self.W = nn.Parameter(torch.empty((out_features, in_features), dtype=dtype, device=device))
        nn.init.trunc_normal_(self.W.data, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor : 
        output = einsum(self.W, x, "out_features in_features, ... in_features -> ... out_features")
        return output

class Emmbedding(nn.Module):
    
    def __init__(self, 
                vocab_size: int, 
                d_model: int, 
                device: torch.device | None = None, 
                dtype: torch.dtype | None = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.E = nn.Parameter(torch.empty((vocab_size, d_model), dtype=dtype, device=device))
        nn.init.trunc_normal_(self.E.data, mean=0.0, std=1.0, a=-3.0, b=3.0)
        # (vocab_size, d_model)
        
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor: 
        # token_ids (batch_size, sequence_length)
        # output (batch, sequence_length, embeddings_dim)
        assert token_ids.dtype == torch.long, "Expeted only long tensor"
        return self.E[token_ids]
        
        # one_hot = F.one_hot(token_ids, num_classes=self.vocab_size).float()
        # e = one_hot @ self.E
        # return e
        
    
class RMSNorm(nn.Module):
    
    def __init__(self, 
                 d_model: int, 
                 eps: float = 1e-5, 
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.G = nn.Parameter(torch.ones(d_model, dtype=dtype, device=device))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(reduce(x**2, "batch sequnce_length d_model -> batch sequnce_length 1", "mean") + self.eps)
        output = einsum(x / rms, self.G, "batch sequnce_length d_model, d_model ->  batch sequnce_length d_model")
        # output = x / rms * self.G
        return output.to(in_dtype)
    
    # (batch_size, sequence_length, d_model)
    

def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)    

class SwiGLU(nn.Module):
    
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        
        self.W1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.W2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.W3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # " ... d_model"
        return self.W2(silu(self.W1(x)) * self.W3(x))
    
class RotaryPositionalEmbedding(nn.Module):
    
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        theta_i = self.theta ** (-1.0 * (torch.arange(0, self.d_k, 2, device=device, dtype=dtype)) / self.d_k)
        positions = torch.arange(0, self.max_seq_len, 1, device=device, dtype=dtype).unsqueeze(1)

        theta_i = theta_i.repeat_interleave(2).unsqueeze(0)
        print(f"{theta_i.shape=}")
        angles = positions * theta_i
        print(f"{angles.shape=}")
        cos = angles.cos() # (seq_len, d_k)
        sin = angles.sin() # (seq_len, d_k)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
    
    def forward(self, x : torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # " ... sequence_length d_k"    " ... sequence_length"
        # (..., seq_len)
        
        cos = self.cos[token_positions] # (seq_len, d_k)
        sin = self.sin[token_positions]
        
        x_cos = x
        x_sin = torch.empty_like(x)
        x_sin[...,1::2] = x[...,0::2] 
        x_sin[...,0::2] = -x[...,1::2]
 
        x = (x_cos * cos) + (x_sin * sin)
         
        return x
    
def softmax(x: torch.Tensor, dim: int) -> torch.Tensor :
    x = x - x.max(dim=dim, keepdim=True).values
    exp_x = x.exp()
    sum_exp = exp_x.sum(dim=dim, keepdim=True)
    softmax = exp_x / sum_exp.unsqueeze(dim)    
    return softmax