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
    model1.check_attn_backend()
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
    model2.check_attn_backend()
    model2.eval()
    return model2




def run_main(
    input_dir:str
):
    dist.init_process_group('gloo')
    rank = dist.get_rank()

    # if rank==1 or rank==2:
    #     torch.cuda.set_device(0)
    pg_nccl = dist.new_group(ranks=[1, 2], backend="nccl")

    if rank==0:
        pass
    elif rank==1:
        pass
    