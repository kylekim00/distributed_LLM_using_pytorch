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
    else:
        run_main(SPLIT_DIR)





if __name__ == "__main__":
    main()


