import os
import sys
import torch
import torch.distributed as dist


def init_process(rank, size, port, backend="nccl"):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend, rank=rank, world_size=size)


def is_master():
    if dist.is_initialized():
        return dist.get_rank() <= 0
    else:
        return True


def barrier():
    if dist.is_initialized():
        dist.barrier()


def set_master_only_print():
    if dist.get_rank() > 0:
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")


def get_world_size():
    if dist.is_initialized():
        return dist.get_world_size()
    else:
        return 0


def gather_uneven_tensors(tensor):
    tensor = tensor.contiguous()
    chunk_size = [
        torch.empty([1], dtype=torch.int64, device=tensor.device)
        for _ in range(get_world_size())
    ]

    dist.all_gather(
        chunk_size,
        torch.tensor([tensor.shape[0]], device=tensor.device),
    )
    max_size = torch.cat(chunk_size).max()

    padded = torch.empty(
        max_size, *tensor.shape[1:], dtype=tensor.dtype, device=tensor.device
    )
    padded[: tensor.shape[0]] = tensor
    storage = [
        torch.empty(
            (max_size, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device
        )
        for _ in range(get_world_size())
    ]
    dist.all_gather(storage, padded)

    ret = torch.cat([s[:l] for s, l in zip(storage, chunk_size)], dim=0)
    return ret
