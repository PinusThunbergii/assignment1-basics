from typing import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, einsum, reduce
import cs336_basics.modeling as m

def cross_entropy_loss(inputs, targets) -> torch.Tensor :
    max = inputs.max(dim=-1, keepdim=True).values
    shifted = inputs - max
    shifted_proj = shifted.gather(1, targets.unsqueeze(1)).squeeze(1)
    log_exp_sum = torch.log(shifted.exp().sum(dim=-1))
    result = (log_exp_sum - shifted_proj).mean()
    return result