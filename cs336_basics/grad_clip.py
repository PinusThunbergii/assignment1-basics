from collections.abc import Iterable
from typing import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, einsum, reduce

def grad_clip(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    
    epsilon = 10e-6


    l2_norm = torch.sqrt(
        torch.sum(
            torch.stack(
                [torch.sum(torch.pow(parameter.grad, 2)) for parameter in parameters if parameter.grad is not None])
            )
        )
    
    for parameter in parameters:
        if parameter.grad is not None:
            

            if l2_norm > max_l2_norm:
                parameter.grad = parameter.grad * (max_l2_norm / (l2_norm + epsilon))

    return
