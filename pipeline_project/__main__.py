#OMP_NUM_THREADS=4 torchrun --nproc_per_node=2 __main__.py
# import torch
# import os
# import torch.distributed as dist
# import torch.functional as F
# from transformers import AutoModelForCausalLM, AutoTokenizer

from .parser import MyParser
from .download import download_model
from .run_dist import *
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_ROOT = PROJECT_ROOT/"models"
REPO_ID = "unsloth/Llama-3.2-3B-Instruct"
LOCAL_DIR = MODEL_ROOT / "Llama-3.2-3B-Instruct"
SPLIT_DIR = MODEL_ROOT / "Llama-3.2-3B-Instruct-split"



def run_main():
    print("main")
    print(PROJECT_ROOT)
    print(REPO_ID)
    print(LOCAL_DIR)
    print(SPLIT_DIR)

    # dist.init_process_group(backend="gloo")
    
    # rank = dist.get_rank()
    # if rank == 0:
    #     # pipe1()
    #     pass

    # elif rank == 1:
    #     pass
    #     # pipe2()
    pass

def main():
    parser = MyParser()
    args = parser.parse_args()

    if args.download:
        download_model(
            repo_id=REPO_ID,
            local_dir=LOCAL_DIR,
            output_dir=SPLIT_DIR,
            split_idx=14,
            torch_dtype=torch.float32,
            device='cpu'
        )
        # def download_model(
        #     repo_id: str = "unsloth/Llama-3.2-3B-Instruct",
        #     local_dir: str = "./models/Llama-3.2-3B-Instruct",
        #     output_dir: str = "./models/Llama-3.2-3B-Instruct-split",
        #     split_idx: int = 14,
        #     torch_dtype=torch.float32,
        #     device: str = "cpu",
        # ):
    else:
        run_main()





if __name__ == "__main__":
    main()


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

