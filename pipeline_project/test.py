#OMP_NUM_THREADS=4 torchrun --nproc_per_node=2 __main__.py
import torch
import os
import torch.distributed as dist
import torch.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MAX_INFLIGHT = 4
N = 20

def pipe1():
    sender_queue = []
    for i in range(N):

        if len(sender_queue) >= MAX_INFLIGHT:
            req = sender_queue.pop(0)
            req.wait()                 #if finished, it will pop it. You don't need to handle this.
        x = torch.tensor([i])
        req = dist.isend(x, dst=1)
        sender_queue.append(req)

    for req in sender_queue:
        req.wait()






def pipe2():
    receiver_queue = []
    initial = min(MAX_INFLIGHT, N)

    #first give irecv to receive from queue asynchronously. Keep the queue size
    for _ in range(initial):
        buf = torch.empty(1, dtype=torch.long)
        req = dist.irecv(buf, src=0)
        receiver_queue.append((req, buf))

    received = 0
    while received < N:
        req, buf = receiver_queue.pop(0)
        req.wait()

        print(f"rank1 got {buf.item()}")
        received+=1

        if received + len(receiver_queue) < N:
            new_buf = torch.empty(1, dtype=torch.long)
            new_req = dist.irecv(new_buf, src=0)
            receiver_queue.append((new_req, new_buf))



def main():
    dist.init_process_group(backend="gloo")
    
    rank = dist.get_rank()
    if rank == 0:
        pipe1()

    elif rank == 1:
        pipe2()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()