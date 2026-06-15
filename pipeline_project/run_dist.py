import os
import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoTokenizer
from safetensors.torch import load_file
from .parser import MyParser
from .download import download_model
from .division_model import Model1, Model2
from .pipe import PipeReceiver, PipeSender
from .node import *

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
    input_dir: Path
):
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if world_size != 3:
        raise RuntimeError(f"expected world_size=3, got {world_size}")

    config = AutoConfig.from_pretrained(input_dir)
    hidden_size = config.hidden_size

    tokenizer = AutoTokenizer.from_pretrained(input_dir)
    eos_token_id = tokenizer.eos_token_id

    batch_size = 1

    
    next_token_dim = [batch_size, 1]

    
    hidden_dim = [batch_size, 1, hidden_size]

    
    token_comm_dtype = torch.long 
    hidden_comm_dtype = torch.float32

    # device setting
    use_cuda = torch.cuda.is_available()

    if rank == 0:
        model_device = "cpu" 
    elif use_cuda:
        torch.cuda.set_device(0) 
        model_device = "cuda:0" 
    else:
        model_device = "cpu"

    

    # pg_12 = dist.new_group(ranks=[1, 2], backend="nccl")

    # if rank not in (1, 2):
    #     pg_12 = None
    pg_12 = None
    
    try:
        if rank == 0:
            node = LLMPromptNode(
                tokenizer_path=input_dir,

                layer1_node=1,

                layer2_node=2,
                layer2_next_receiving_dim=next_token_dim,

                # prompt -> layer1 : token ids, dynamic
                layer1_send_data_group=None,
                layer1_send_data_device="cpu",

                # layer2 -> prompt : next token, fixed
                layer2_recv_data_group=None,
                layer2_recv_data_device="cpu",

                input_dtype=token_comm_dtype, 
                output_dtype=token_comm_dtype, 
                queue_size=4,
            )

            node.run()

        elif rank == 1:
            model1 = load_model1(
                input_dir=input_dir,
                device=model_device,
            )

            node = LLMLayerNode1(
                layer1=model1,

                prompt_node=0,

                layer2_node=2,
                layer2_hidden_sending_dim=hidden_dim,
                layer2_next_receiving_dim=next_token_dim,

                # prompt -> layer1 는 CPU/GLOO token
                prompt_recv_data_group=None,
                prompt_recv_data_device="cpu",

                # layer1 -> layer2 hidden 통신
                layer2_send_data_group=pg_12,
                layer2_send_data_device=model_device if use_cuda else "cpu",

                # layer2 -> layer1 next token
                layer2_recv_data_group=None,
                layer2_recv_data_device="cpu",

                input_dtype=token_comm_dtype,
                output_dtype=hidden_comm_dtype, 
                model_device=model_device,

                queue_size=4,
            )

            node.run()

        elif rank == 2:
            model2 = load_model2(
                input_dir=input_dir,
                device=model_device,
            )

            node = LLMLayerNode2(
                layer2=model2,

                prompt_node=0,

                layer1_node=1,
                layer1_hidden_receiving_dim=hidden_dim,

                next_token_sending_dim=next_token_dim,

                eos_token_id=eos_token_id,

                # layer2 -> prompt next token
                prompt_send_data_group=None,
                prompt_send_data_device="cpu",

                # layer1 -> layer2 hidden
                layer1_recv_data_group=pg_12,
                layer1_recv_data_device=model_device if use_cuda else "cpu",

                # layer2 -> layer1 next token
                layer1_send_data_group=None,
                layer1_send_data_device="cpu",

                input_dtype=hidden_comm_dtype, 
                output_dtype=token_comm_dtype, 
                model_device=model_device,

                queue_size=4,
            )

            node.run()

    finally:
        dist.destroy_process_group()