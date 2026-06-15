import torch
import torch.distributed as dist
from .pipe import *
import torch.nn as nn
import torch
import torch.distributed as dist
from .pipe import *
import torch.nn as nn


class LLMLayerNode1:
    def __init__(
        self,
        layer1: nn.Module,

        prompt_node: int,



        layer2_node: int,
        layer2_hidden_sending_dim,
        layer2_next_receiving_dim,

        extra_control_keys=None,


        prompt_recv_data_group: dist.ProcessGroup | None = None,
        prompt_recv_data_device: str = "cpu",

        layer2_send_data_group: dist.ProcessGroup | None = None,
        layer2_send_data_device: str = "cpu",


        layer2_recv_data_group: dist.ProcessGroup | None = None,
        layer2_recv_data_device: str = "cpu",

        # prompt_send_data_group: dist.ProcessGroup | None = None,
        # prompt_send_data_device: str = "cpu",


        input_dtype: torch.dtype = torch.float32, 
        output_dtype: torch.dtype = torch.float32,
        model_device: str = "cpu",

        # send_data_group: dist.ProcessGroup | None = None,
        # send_data_device: str = "cpu",

        queue_size: int = 4,
    ):
        self.model = layer1.to(model_device)
        self.model.eval()
        self.model_device = model_device

        self.input_dtype = input_dtype 
        self.output_dtype = output_dtype 

        self.prompt_recv = PipeReceiver.dynamic( # 0 -> 1 dynamic
            source=prompt_node, 
            data_dtype=input_dtype, 
            extra_control_keys=extra_control_keys,
            data_group=prompt_recv_data_group, 
            data_device=prompt_recv_data_device, 
            pipe_tag=0,
            )

        self.layer2_dynamic_send = PipeSender.dynamic(# 1->2 dynamic
            dest=layer2_node,
            extra_control_keys=extra_control_keys,
            data_group=layer2_send_data_group,
            data_device=layer2_send_data_device,
            data_dtype=output_dtype, 
            pipe_tag=0
        )


        self.layer2_fixed_recv = PipeReceiver.fixed( # 2->1
            source=layer2_node,
            data_dim=layer2_next_receiving_dim,
            extra_control_keys=extra_control_keys,
            # queue_size=queue_size,
            data_group=layer2_recv_data_group,
            data_device=layer2_recv_data_device,
            data_dtype=input_dtype, 
        )

        self.layer2_fixed_send = PipeSender.fixed( # 1->2
            dest=layer2_node,
            data_dim=layer2_hidden_sending_dim,
            extra_control_keys=extra_control_keys,
            queue_size=queue_size,
            data_group=layer2_send_data_group,
            data_device=layer2_send_data_device,
            data_dtype=output_dtype, 
            pipe_tag=1
        )
        self.layer2_send_data_device_is_cpu = layer2_send_data_device == "cpu"
        self.layer2_recv_data_device_is_cpu = layer2_recv_data_device == "cpu"
        # self.prompt_send_data_device_is_cpu = prompt_send_data_device == "cpu"
        self.prompt_recv_data_device_is_cpu = prompt_recv_data_device == "cpu"

        self.state = True

    def run(self) -> None:
        self.model.eval()
        past_key_values = None
        cache_len = 0
        while True:
            if self.state:
                ctl, X = self.prompt_recv.recv()
                if ctl['end']:
                    self.layer2_dynamic_send.send(ctl, None) 
                    self.prompt_recv.release(X)
                    break
                if ctl['data'] == 0:
                    self.prompt_recv.release(X) 
                    continue
                input1 = X.squeeze(-1).long().to(self.model_device) 

                cache_position = torch.arange(cache_len, input1.shape[1] + cache_len, device=input1.device) 
                cache_len = cache_len + input1.shape[1] 

                _res = self.model(
                    input_ids = input1,
                    past_key_values = past_key_values,
                    cache_position=cache_position,
                    use_cache = True
                    )

                past_key_values = _res["past_key_values"]

                # _out = _res["logits"][:, -1].argmax(dim=-1, keepdim=True) #why the fxxk did you do this???
                if self.layer2_send_data_device_is_cpu:
                    out = _res["hidden_states"].to(device="cpu", dtype=self.output_dtype) 
                else:
                    #if model and layer2_send node exists in a different GPU device, then this might get an error.
                    out = _res["hidden_states"].to(dtype=self.output_dtype) 


                self.layer2_dynamic_send.send(ctl,out) #doesn't need get_buffer()
                self.prompt_recv.release(X) 
                self.state = False
            else:
                #prompt recv
                # ctl_prompt, _ = self.prompt_recv.recv()
                # if ctl_prompt['end']:
                #     break

                #next_token recv
                ctl, next_token = self.layer2_fixed_recv.recv()
                if ctl['end']:
                    self.layer2_fixed_recv.release(next_token)
                    self.layer2_fixed_send.send(ctl)
                    break
                if ctl['eop']:
                    self.state = True
                    self.layer2_fixed_recv.release(next_token)
                    continue


                input1 = next_token.long().to(self.model_device)
                cache_position = torch.tensor([cache_len], device=input1.device)
                cache_len = cache_len + 1

                _res = self.model(
                    input_ids = input1,
                    past_key_values = past_key_values,
                    cache_position = cache_position,
                    use_cache = True
                )

                past_key_values = _res["past_key_values"]
                _out = _res["hidden_states"]

                if self.layer2_send_data_device_is_cpu:
                    _out = _out.to(device="cpu", dtype=self.output_dtype) 
                else:
                    #if model and layer2_send node exists in a different GPU device, then this might get an error... But who the hell would do that?
                    _out = _out.to(dtype=self.output_dtype) 

                out = self.layer2_fixed_send.get_buffer()

                out.copy_(_out)
                self.layer2_fixed_send.send(ctl, out)
                self.layer2_fixed_recv.release(next_token)
    def close(self):
        self.layer2_dynamic_send.close()
        self.layer2_fixed_send.close()
        self.layer2_fixed_recv.close()
        self.prompt_recv.close()

class LLMLayerNode2:
        def __init__(
                self,
                layer2: nn.Module,

                prompt_node:int,

                layer1_node:int,
                layer1_hidden_receiving_dim,

                next_token_sending_dim,

                eos_token_id,
                extra_control_keys = None,

                prompt_send_data_group: dist.ProcessGroup | None = None,
                prompt_send_data_device: str = "cpu",

                layer1_recv_data_group: dist.ProcessGroup | None = None,
                layer1_recv_data_device: str = "cpu",

                layer1_send_data_group: dist.ProcessGroup | None = None,
                layer1_send_data_device: str = "cpu",

                input_dtype: torch.dtype = torch.float32, 
                output_dtype: torch.dtype = torch.float32,
                model_device: str = "cpu",

                queue_size:int = 4,
                ):
            self.model = layer2.to(model_device)
            self.model.eval()
            self.model_device = model_device
            self.eos_token_id = eos_token_id

            self.input_dtype = input_dtype 
            self.output_dtype = output_dtype 

            self.layer1_dynamic_recv = PipeReceiver.dynamic( # 1 -> 2 dynamic
                source=layer1_node,
                extra_control_keys=extra_control_keys,
                data_group=layer1_recv_data_group,
                data_device=layer1_recv_data_device,
                data_dtype=input_dtype,
                pipe_tag=0,
            )
            self.layer1_fixed_recv = PipeReceiver.fixed( # 1 -> 2
                source=layer1_node,
                data_dim=layer1_hidden_receiving_dim,
                extra_control_keys=extra_control_keys,
                data_group=layer1_recv_data_group,
                data_device=layer1_recv_data_device,
                data_dtype=input_dtype,
                pipe_tag=1
            )
            self.prompt_send= PipeSender.fixed( # 2 -> 0
                dest=prompt_node,
                data_dim=next_token_sending_dim,
                extra_control_keys=extra_control_keys,
                queue_size=queue_size,
                data_group=prompt_send_data_group,
                data_device=prompt_send_data_device,
                data_dtype=output_dtype 
            )


            self.layer1_send = PipeSender.fixed( # 2 -> 1
                dest= layer1_node,
                data_dim= next_token_sending_dim,
                extra_control_keys=extra_control_keys,
                queue_size=queue_size,
                data_group= layer1_send_data_group,
                data_device= layer1_send_data_device,
                data_dtype=output_dtype, 
            )

            self.layer1_send_data_device_is_cpu = layer1_send_data_device == "cpu"
            self.layer1_recv_data_device_is_cpu = layer1_recv_data_device == "cpu"
            self.prompt_send_data_device_is_cpu = prompt_send_data_device == "cpu"

            self.state = True

        def run(self):
            self.model.eval()
            past_key_values = None
            cache_len = 0
            while True:
                if self.state:
                    ctl, X = self.layer1_dynamic_recv.recv()
                    if ctl['end']:
                        self.layer1_dynamic_recv.release(X)#just for intuition
                        break
                    if ctl['data'] == 0:
                        self.layer1_dynamic_recv.release(X)
                        continue


                    input2 = X.to(device=self.model_device, dtype=self.input_dtype) 

                    cache_position = torch.arange(cache_len, input2.shape[1] + cache_len, device=input2.device) # 이거 추가했어요!!!!
                    cache_len = cache_len + input2.shape[1] 

                    _res = self.model(
                        hidden_states = input2, 
                        past_key_values = past_key_values,
                        cache_position = cache_position,
                        use_cache = True
                        )
                    past_key_values = _res["past_key_values"]
                    _out = _res["logits"][:,-1].argmax(dim=-1, keepdim=True)

                    if _out.item() == self.eos_token_id:
                        ctl['eop'] = True
                        
                        self.layer1_send.send(ctl)
                        self.prompt_send.send(ctl)
                        self.layer1_dynamic_recv.release(X) 
                        continue

                    out1 = self.layer1_send.get_buffer()
                    out2 = self.prompt_send.get_buffer()

                    if self.layer1_send_data_device_is_cpu:
                        _out1 = _out.to(device="cpu", dtype=self.output_dtype) 
                    else:
                        _out1 = _out.to(dtype=self.output_dtype) 

                    out1.copy_(_out1)

                    if self.prompt_send_data_device_is_cpu:
                        _out2 = _out.to(device="cpu", dtype=self.output_dtype)
                    else:
                        _out2= _out.to(dtype=self.output_dtype) 

                    out2.copy_(_out2)

                    self.layer1_send.send(ctl, out1)
                    self.prompt_send.send(ctl, out2)

                    self.layer1_dynamic_recv.release(X)
                    self.state = False
                else:
                    
                    ctl, X = self.layer1_fixed_recv.recv() # get next state token

                    if ctl['end']:
                        self.layer1_fixed_recv.release(X)
                        break
                    if ctl['eop']:
                        self.state = True
                        self.layer1_fixed_recv.release(X)
                        continue

                    input2 = X.to(device=self.model_device, dtype=self.input_dtype) 
                    cache_position = torch.tensor([cache_len], device=input2.device)
                    cache_len = cache_len + 1

                    _res = self.model( 
                        hidden_states = input2, 
                        past_key_values = past_key_values,
                        cache_position = cache_position,
                        use_cache = True
                    )

                    past_key_values = _res["past_key_values"]

                    _out = _res["logits"][:,-1].argmax(dim=-1, keepdim=True)

                    out1 = self.layer1_send.get_buffer()
                    out2 = self.prompt_send.get_buffer()

                    if self.layer1_send_data_device_is_cpu:
                        _out1 = _out.to(device="cpu", dtype=self.output_dtype) 
                    else:
                        _out1 = _out.to(dtype=self.output_dtype)

                    out1.copy_(_out1)

                    if self.prompt_send_data_device_is_cpu:
                        _out2 = _out.to(device="cpu", dtype=self.output_dtype) 
                    else:
                        _out2 = _out.to(dtype=self.output_dtype)

                    out2.copy_(_out2)

                    if _out.item() == self.eos_token_id:
                        ctl['eop'] = True
                        self.state = True


                    self.layer1_send.send(ctl, out1)
                    self.prompt_send.send(ctl, out2)

                    self.layer1_fixed_recv.release(X)

        def close(self):
            self.layer1_dynamic_recv.close()
            self.layer1_fixed_recv.close()
            self.prompt_send.close()
            self.layer1_send.close()

from transformers import AutoTokenizer
from pathlib import Path

class LLMPromptNode:
        def __init__(
                self,
                tokenizer_path:Path,
                layer1_node:int,

                layer2_node:int,
                layer2_next_receiving_dim,

                extra_control_keys = None,

                layer1_send_data_group:dist.ProcessGroup | None = None,
                layer1_send_data_device:str = "cpu",

                layer2_recv_data_group:dist.ProcessGroup | None = None,
                layer2_recv_data_device:str = "cpu",

                input_dtype : torch.dtype = torch.float32, 
                output_dtype : torch.dtype = torch.float32,

                queue_size: int = 4
                ):

            self.input_dtype = input_dtype 
            self.output_dtype = output_dtype 

            self.layer1_send = PipeSender.dynamic(
                dest=layer1_node,
                extra_control_keys=extra_control_keys,
                queue_size=queue_size,
                data_group=layer1_send_data_group,
                data_device=layer1_send_data_device,
                data_dtype=output_dtype
            )

            self.layer2_recv = PipeReceiver.fixed(
                source=layer2_node,
                data_dim=layer2_next_receiving_dim,
                queue_size=queue_size,
                data_group=layer2_recv_data_group,
                data_device=layer2_recv_data_device,
                data_dtype=input_dtype, 
            )


            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            self.layer_data_is_cpu = layer1_send_data_device == "cpu"
            self.layer1_send_data_device = layer1_send_data_device
            self.state = True


        def run(self)->None:
            prompt = ""
            while True:
                if self.state:
                    print("user: ", end="")
                    p = input()
                    if p.lower() in ("q","quit", "exit"):
                        break
                    new_text = (
                        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                        + p
                        + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                    )   
                    prompt += new_text
                    inputs = self.tokenizer(new_text, return_tensors="pt")

                    # attention_mask = inputs.get("attention_mask", None)
                    # ##sxxt...... I didn't thought about sending.. this....
                    # Well.... sdpa seems to apply causal mask automatically.
                    # and since my implementation receive only one prompt, it's okay for now.

                    # if attention_mask is not None:
                    #     attention_mask = attention_mask.to(device)
                    input_ids = inputs['input_ids'].to(dtype=self.output_dtype).unsqueeze(-1)
                    if not self.layer_data_is_cpu:
                        input_ids = input_ids.to(device=self.layer1_send_data_device, dtype=self.output_dtype)

                    self.layer1_send.send({'end':False,'eop':False}, input_ids)

                    self.state = False
                    print("llama: ", end="")
                else:
                    ctl, next_token = self.layer2_recv.recv()

                    if ctl['eop']:
                        self.layer2_recv.release(next_token) 
                        self.state = True
                        print() 
                        continue

                    next_token = next_token.long() 
                    token_text = self.tokenizer.decode(next_token[0].tolist()) 
                    print(token_text, end="")
                    prompt += token_text
                    self.layer2_recv.release(next_token)


            self.layer1_send.send({'end':True})

        def close(self):
            self.layer1_send.close()
            self.layer2_recv.close()