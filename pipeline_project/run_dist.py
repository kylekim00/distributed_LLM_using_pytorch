import os
import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoTokenizer
from safetensors.torch import load_file
from .parser import MyParser
from .download import download_model
from .division_model import Model1, Model2



def load_model1(
        input_dir:str

):
    config = AutoConfig.from_pretrained(input_dir)
    model1 = Model1(config)
    state = load_file(os.path.join(input_dir, "model1.safetensors"))
    model1.load_state_dict(state, strict=True)
    model1.to(input_dir)
    model1.eval()
    

    pass

def load_model2(
        input_dir:str
):
    config = AutoConfig.from_pretrained(input_dir)
    model2 = Model2(config)
    state = load_file(os.path.join(input_dir, "model2.safetensors"))
    
    pass



def run_dist(
        input_dir: str="./models/Llama-3.2-3B-Instruct-split"
):
    pass