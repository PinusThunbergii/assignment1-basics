
from typing import Callable, Optional
from torch.optim.optimizer import Optimizer
import torch


class AdamWOptimizer(Optimizer):

    def __init__(self, params, lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8):
        
        defaults = { 
                    "lr" : lr, 
                    "weight_decay" : weight_decay,
                    "betas" : betas,
                    "eps" : eps,
                    }
        super().__init__(params, defaults)
        
    # def step(self, closure: Optional[Callable] | None = None):
    # def step(self, closure: Callable[[], float]) -> float:
    # def step(self, closure: None = None) -> float:
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None if closure is None else closure()
        
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            betas = group["betas"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m", torch.zeros_like(p))
                v = state.get("v", torch.zeros_like(p))
                grad = p.grad.data
                
                m.data = betas[0] * m.data + (1.0 - betas[0]) * grad
                v.data = betas[1] * v.data + (1.0 - betas[1]) * torch.pow(grad, 2)
                
                lr_t = lr * (1.0 - betas[1] ** t) ** 0.5 / (1.0 - betas[0] ** t)
                  
                p.data = p.data - lr_t * m / (torch.sqrt(v) + eps)
                p.data = p.data - lr * weight_decay * p.data
                
                state["t"] = t + 1
                state["m"] = m     
                state["v"] = v     
    
        return loss