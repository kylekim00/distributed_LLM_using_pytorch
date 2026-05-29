import os
import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoTokenizer
from safetensors.torch import load_file
from .parser import MyParser
from .download import download_model
from .division_model import Model1, Model2
from .pipe import PipeReceiver, PipeSender
from .node import FullNode


def load_model1(
        input_dir:str,
        device:str="cpu"
):
    config = AutoConfig.from_pretrained(input_dir)
    model1 = Model1(config)
    state = load_file(os.path.join(input_dir, "model1.safetensors"))
    model1.load_state_dict(state, strict=True)
    model1.to(device=device)
    model1.eval()
    return model1
    


def load_model2(
        input_dir:str,
        device:str="cpu"
):
    config = AutoConfig.from_pretrained(input_dir)
    model2 = Model2(config)
    state = load_file(os.path.join(input_dir, "model2.safetensors"))
    model2.load_state_dict(state, strict=True)
    model2.to(device=device)
    model2.eval()
    return model2



def run_dist(
    
):
    pass




# MAX_INFLIGHT = 4
# N = 20

# def pipe1():
#     sender_queue = []
#     for i in range(N):

#         if len(sender_queue) >= MAX_INFLIGHT:
#             req = sender_queue.pop(0)
#             req.wait()                 #if finished, it will pop it. You don't need to handle this.
#         x = torch.tensor([i])
#         req = dist.isend(x, dst=1)
#         sender_queue.append(req)

#     for req in sender_queue:
#         req.wait()






# def pipe2():
#     receiver_queue = []
#     initial = min(MAX_INFLIGHT, N)

#     #first give irecv to receive from queue asynchronously. Keep the queue size
#     for _ in range(initial):
#         buf = torch.empty(1, dtype=torch.long)
#         req = dist.irecv(buf, src=0)
#         receiver_queue.append((req, buf))

#     received = 0
#     while received < N:
#         req, buf = receiver_queue.pop(0)
#         req.wait()

#         print(f"rank1 got {buf.item()}")
#         received+=1

#         if received + len(receiver_queue) < N:
#             new_buf = torch.empty(1, dtype=torch.long)
#             new_req = dist.irecv(new_buf, src=0)
#             receiver_queue.append((new_req, new_buf))

