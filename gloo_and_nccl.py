# torchrun --nproc_per_node=2 mixed_group.py

import os
import torch
import torch.distributed as dist


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )

    torch.cuda.set_device(local_rank)

    # NCCL 그룹 생성
    pg_nccl = dist.new_group(
        ranks=list(range(world_size)),
        backend="nccl",
    )

    #################################################################
    # 1. GLOO point-to-point (CPU tensor)
    #################################################################

    if rank == 0:
        x = torch.tensor([123], dtype=torch.long)

        req = dist.isend(
            tensor=x,
            dst=1,
            group=None,      # default group (gloo)
        )

        req.wait()
        print(f"[rank0] sent cpu tensor")

    elif rank == 1:
        buf = torch.empty(1, dtype=torch.long)

        req = dist.irecv(
            tensor=buf,
            src=0,
            group=None,      # default group (gloo)
        )

        req.wait()
        print(f"[rank1] received cpu tensor: {buf.item()}")

    dist.barrier()

    #################################################################
    # 2. NCCL collective (GPU tensor)
    #################################################################

    x_gpu = torch.tensor(
        [rank + 1],
        dtype=torch.float32,
        device=f"cuda:{local_rank}",
    )

    dist.all_reduce(
        x_gpu,
        op=dist.ReduceOp.SUM,
        group=pg_nccl,
    )

    print(
        f"[rank{rank}] "
        f"all_reduce result = {x_gpu.item()}"
    )

    #################################################################

    dist.destroy_process_group()


if __name__ == "__main__":
    main()