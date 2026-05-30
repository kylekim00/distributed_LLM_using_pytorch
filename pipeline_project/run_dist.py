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



def run_dist(
    input_dir:str
):
    import subprocess
    import sys

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=3",
        "-m",
        "pipeline_project",
    ]

    subprocess.run(cmd, check=True)

