from typing import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, einsum, reduce

def lr_shed(    
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int):

    learning_rate = None

    if it < warmup_iters:
        learning_rate = it / warmup_iters * max_learning_rate
    elif warmup_iters <= it and it <= cosine_cycle_iters:
        learning_rate = min_learning_rate + 0.5 * (1 + math.cos((it - warmup_iters)/(cosine_cycle_iters - warmup_iters)*math.pi)) * (max_learning_rate - min_learning_rate)
    elif it > cosine_cycle_iters:
        learning_rate = min_learning_rate

    return learning_rate