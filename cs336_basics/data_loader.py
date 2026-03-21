import numpy.typing as npt
import numpy as np
import torch

def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    
    max_start_id = (dataset.shape[0] - context_length) - 1

    if max_start_id < 0:
        raise ValueError("dataset too small for provided context length")


    start_sampling_ids = np.random.randint(0, max_start_id+1, batch_size)

    x_batch = np.zeros((batch_size, context_length), dtype=np.long)
    y_batch = np.zeros((batch_size, context_length), dtype=np.long)

    for i, start_id in enumerate(start_sampling_ids):
        x_batch[i] = dataset[start_id: start_id + context_length]
        y_batch[i] = dataset[start_id + 1: start_id + context_length + 1]

    x_batch = torch.tensor(x_batch, device=device)
    y_batch = torch.tensor(y_batch, device=device)

    return (x_batch, y_batch)