import os
import typing

import numpy.typing as npt
import numpy as np
import torch

def save_checkpoint(
        model: torch.nn.Module, 
        optimizer: torch.optim.Optimizer, 
        iteration: int, 
        out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    
    state_dict = {
        "model" : model.state_dict(),
        "optimizer" : optimizer.state_dict(),
        "iteration" : iteration 
    }
    torch.save(state_dict, out)


def load_checkpoint(
        src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], 
        model: torch.nn.Module, 
        optimizer: torch.optim.Optimizer) -> int: 
    
    state_dict = torch.load(src)
    model.load_state_dict(state_dict["model"])
    optimizer.load_state_dict(state_dict["optimizer"])
    iteration = state_dict["iteration"]
    return iteration
