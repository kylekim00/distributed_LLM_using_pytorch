import torch
import torch.distributed as dist
import torch.nn as nn



class Buffer_Send:
    def __init__(self, 
                 tensor_dim:list | torch.Size,
                 target:int, 
                 tag:int,
                 group:dist.ProcessGroup|None=None,
                 device:str='cpu',
                 dtype:torch.dtype=torch.float32,
                 queue_size:int=4,
                ):
        self.pending_queue = list()
        self.target = target
        self.tag = tag
        self.free_tensor = [torch.empty(size=tensor_dim, dtype=dtype, device=device) for _ in range(queue_size)]
        self.group = group
        self.queue_size = queue_size


    def get_empty_tensor(self):
        if not self.free_tensor:
            req, ten= self.pending_queue.pop(0)
            req.wait()
            self.free_tensor.append(ten)
            
        return self.free_tensor.pop(0)
    


    def send_tensor(self, ten:torch.Tensor):
        req = dist.isend(tensor=ten, dst=self.target, tag=self.tag, group=self.group)
        self.pending_queue.append((req, ten))

    #this is called when end signal is sent by send_tensor.
    def close(self):
        for i in range(self.queue_size):
            self.send_tensor(self.get_empty_tensor().fill_(-1))
        while self.pending_queue:
            req, ten = self.pending_queue.pop(0)
            req.wait()
            self.free_tensor.append(ten)
        
        

class Buffer_Recv:
    def __init__(self, 
                 tensor_dim:list | torch.Size, 
                 target:int,
                 tag:int, 
                 group:dist.ProcessGroup|None=None,
                 device:str='cpu',
                 dtype:torch.dtype=torch.float32,
                 queue_size:int=4
                 ):
        self.pending_queue = list()
        self.target = target
        self.tag = tag
        self.group = group
        self.queue_size = queue_size
        for _ in range(queue_size):#fill pending queue
            ten = torch.empty(size=tensor_dim, dtype=dtype, device=device)
            res = dist.irecv(ten, src=self.target, tag=self.tag, group=group)
            self.pending_queue.append((res, ten))

    def peep_next_tensor_available(self)->bool:
        res = self.pending_queue[0][0]
        return res.is_completed()

    #when starting computation, it gets the next tensor from pending queue to get data.
    def get_next_tensor(self)->torch.Tensor:
        res, ten = self.pending_queue.pop(0)
        res.wait()
        return ten

    #when computation is done, it posts used tensor back to pending_queue
    def free_sent_tensor(self, ten:torch.Tensor)->None:
        res = dist.irecv(ten,src=self.target, tag=self.tag, group=self.group)
        self.pending_queue.append((res, ten))

    def close(self):
        # for i in range(self.queue_size - len(self.pending_queue)):
        #     self.free_sent_tensor(self.get_next_tensor())
        #     pass
        while self.pending_queue:
            req, _ = self.pending_queue.pop(0)
            req.wait()


class PipeSender:
    def __init__(
        self,
        destination: int,
        data_dim: list | torch.Size,
        control_config: dict | None = None,
        queue_size: int = 4,
        pipe_tag: int = 0,

        data_group: dist.ProcessGroup | None = None,
        data_device: str = "cpu",
        data_dtype: torch.dtype = torch.float32,
    ):
        self.control = Control_Sender(
            dest=destination,
            control_config=control_config,
            queue_size=queue_size,
            tag=pipe_tag * 2 + 1,
        )

        self.data = Buffer_Send(
            tensor_dim=data_dim,
            target=destination,
            tag=pipe_tag * 2 + 0,
            group=data_group,
            device=data_device,
            dtype=data_dtype,
            queue_size=queue_size,
        )
        

    def getBuffer(self) -> torch.Tensor:
        return self.data.get_empty_tensor()

    def send(self, control_config: dict, data: torch.Tensor|None = None) -> None:
        has_data = data is not None
        control_config['data'] = int(has_data)
        self.control.send(control_config)
        if has_data:               
            self.data.send_tensor(data)

    def close(self) -> None:
        self.control.close()
        self.data.close()


class PipeReceiver:
    def __init__(
        self,
        source: int,
        data_dim: list | torch.Size,
        control_config: dict | None = None,
        queue_size: int = 4,
        pipe_tag: int = 0,

        data_group: dist.ProcessGroup | None = None,
        data_device: str = "cpu",
        data_dtype: torch.dtype = torch.float32,
    ):
        self.control = Control_Receiver(
            source=source,
            control_config=control_config,
            queue_size=queue_size,
            tag=pipe_tag * 2 + 1,
        )

        self.data = Buffer_Recv(
            tensor_dim=data_dim,
            target=source,
            tag=pipe_tag * 2 + 0,
            group=data_group,
            device=data_device,
            dtype=data_dtype,
            queue_size=queue_size,
        )

    def recv(self) -> tuple[dict, torch.Tensor | None]:
        control_config = self.control.recv()

        if control_config['end'] == 1:
            return control_config, None
        
        if control_config['data'] == 0:
            return control_config, None

        data = self.data.get_next_tensor()
        return control_config, data

    def release(self, data: torch.Tensor | None):
        if data is not None:
            self.data.free_sent_tensor(data)

    def close(self):
        # self.control.close()
        self.data.close()



class Control_Sender:
    def __init__(
            self,
            dest:int,
            control_config:dict|None = None,
            queue_size:int=4,
            tag:int = 0
            ):
        
        self.dest = dest
        self.control_config = control_config
        self.queue_size = queue_size

        self.control_config = {'end':False, 'append_state':True, 'data':1}
        if control_config is not None:
            for k, v in control_config.items():
                if k in self.control_config.keys():
                    raise KeyError(f"{self.control_config.keys()} can not be in control_config. These are main keys")
                self.control_config[k] = v
        
        self.control_len = len(self.control_config)
        self.send_buffer = Buffer_Send(
            tensor_dim=[self.control_len],
            target=dest,
            tag=tag,
            dtype=torch.int32,
            queue_size=queue_size
        )
    def send(
            self,
            control_config:dict
        )->None:
        keys = self.control_config.keys()
        for key, value in control_config.items():
            if key not in keys:
                raise KeyError(f"control_config keys does not match: {key}")
            self.control_config[key] = value
        
        ctl = self.send_buffer.get_empty_tensor()
        for i, v in enumerate(self.control_config.values()):
            ctl[i] = v
        self.send_buffer.send_tensor(ctl)

    def close(self):
        self.send_buffer.close()
            
            
class Control_Receiver:
    def __init__(
            self,
            source:int,
            control_config:dict|None = None,
            queue_size:int=4,
            tag:int=0
        ):
        self.source = source
        self.control_config = control_config
        self.queue_size = queue_size
        self.tag = tag

        self.control_config = {'end':False, 'append_state':True, "data":1}
        if control_config is not None:
            for k, v in control_config.items():
                if k in self.control_config.keys():
                    raise KeyError(f"{self.control_config.keys()} can not be in control_config. These are main keys")
                self.control_config[k] = v
        
        self.control_len = len(self.control_config)
        self.recv_buffer = Buffer_Recv(
            tensor_dim=[self.control_len],
            target=source,
            tag=tag,
            dtype=torch.int32,
            queue_size=queue_size
        )
    def recv(self)->dict:
        ctl = self.recv_buffer.get_next_tensor()
        for i, k in enumerate(self.control_config.keys()):
            self.control_config[k] = int(ctl[i].item())
        self.recv_buffer.free_sent_tensor(ctl)
        return dict(self.control_config)
    
    def close(self)->None:
        self.recv_buffer.close()
    


    




#.copy_() version 
#this node simply
# 1. receives fixed data, 
# 2. process through model in designated device 
# 3. and sends to another node.
class FullNode:
    def __init__(
        self,
        model: nn.Module,

        receiving_node: int,
        receiving_dim: list | torch.Size,

        sending_node: int,
        sending_dim: list | torch.Size,

        control_config: dict | None = None,
        queue_size: int = 4,

        recv_data_group: dist.ProcessGroup | None = None,
        send_data_group: dist.ProcessGroup | None = None,

        recv_data_device: str = "cpu",
        send_data_device: str = "cpu",
        model_device: str = "cpu",

        data_dtype: torch.dtype = torch.float32,
    ):
        self.model = model.to(model_device)
        self.model.eval()
        self.model_device = model_device

        if control_config is None:
            control_config = {}

        # end / append_state / data 는 Control class의 main key라 넣지 않음
        self.control_config = dict(control_config)

        for key, value in self.control_config.items():
            if not isinstance(value, int):
                raise ValueError(
                    f"control_config[{key}] must be int, got {type(value)}"
                )

        recv_queue_size = queue_size
        send_queue_size = queue_size

        if recv_data_group is not None:
            recv_queue_size = 1
        if send_data_group is not None:
            send_queue_size = 1

        self.send = PipeSender(
            destination=sending_node,
            data_dim=sending_dim,
            control_config=self.control_config,
            queue_size=send_queue_size,
            data_group=send_data_group,
            data_device=send_data_device,
            data_dtype=data_dtype,
        )

        self.recv = PipeReceiver(
            source=receiving_node,
            data_dim=receiving_dim,
            control_config=self.control_config,
            queue_size=recv_queue_size,
            data_group=recv_data_group,
            data_device=recv_data_device,
            data_dtype=data_dtype,
        )

    def run(self) -> None:
        while True:
            r_ctl, r_ten = self.recv.recv()

            if r_ctl["end"] == 1:
                # end control forwarding
                self.send.send(r_ctl, None)
                break

            if r_ctl["data"] == 0:
                # data 없는 control만 forwarding
                self.send.send(r_ctl, None)
                continue

            if r_ten is None:
                raise RuntimeError("control data flag is 1 but received tensor is None")

            s_ten = self.send.getBuffer()

            with torch.no_grad():
                inp = r_ten.to(self.model_device)
                out = self.model(inp)
                s_ten.copy_(out.to(s_ten.device))

            self.recv.release(r_ten)
            self.send.send(r_ctl, s_ten)

        self.send.close()
        self.recv.close()



#this node has two states. Append state and run state. 
# Append : It synchronously receives first tensor of data which has big size to compute all the tokens that it received and store it in kv cache.
# run : It generates a token and recursively computes each time. since this process has a fixed size of a tensor, it uses Pipe comm made above.

class LLMNode1:
    def __init__(
            self,
            model: nn.Module,
            
            prompt_node: int,

            receiving_node: int,
            receiving_dim: list | torch.Size,

            sending_node: int,
            sending_dim: list | torch.Size,
            
            # control_config: dict | None = None,
            queue_size: int = 4,
            
            recv_data_group:dist.ProcessGroup | None = None,
            send_data_group:dist.ProcessGroup | None = None,
            
            recv_data_device: str = "cpu",
            send_data_device:str = "cpu",
            model_device:str = "cpu",
            
            data_dtype: torch.dtype = torch.float32
            ):
        self.model = model.to(model_device)
        self.model_device = model_device
        self.prompt_node = prompt_node
        self.sending_node = sending_node
        self.data_dtype = data_dtype

        self.recv_data_group = recv_data_group
        self.send_data_group = send_data_group

        self.recv_data_device = recv_data_device
        self.send_data_device = send_data_device


        self.control_config = {
            'end':False,
            'append':True
        }
        
        for key, value in self.control_config.items():
            if not isinstance(value, int):
                raise ValueError(f"control_config{key} must be int, got {type(value)}")
        control_dim = [len(self.control_config)]

        #keep the queue of NCCL to 1(this nccl keeps getting problemssssssssss!!!!!!!!!)
        recv_queue_size = queue_size
        send_queue_size = queue_size

        if recv_data_group is not None:
            recv_queue_size = 1
        if send_data_group is not None:
            send_queue_size = 1



        #these are all for append=False.
        self.send = PipeSender(
            destination=sending_node,
            data_dim=sending_dim,
            control_dim=control_dim,
            control_queue_size=queue_size,
            data_queue_size=send_queue_size,
            control_group=None,
            data_group=send_data_group,
            control_device="cpu",
            data_device=send_data_device,
            control_dtype=torch.int32,
            data_dtype=data_dtype,
        )

        self.recv = PipeReceiver(
            source=receiving_node,
            control_dim=control_dim,
            data_dim=receiving_dim,
            control_queue_size=queue_size,
            data_queue_size=recv_queue_size,
            control_group=None,
            data_group=recv_data_group,
            control_device="cpu",
            data_device=recv_data_device,
            control_dtype=torch.int32,
            data_dtype=data_dtype,
        )

    def run(self)->None:
        running=True
        append_state = True
        while running:
            if append_state:

                size = torch.empty([3], dtype=torch.int32)
                dist.recv(size, src=self.receiving_node, tag=99)
                if size[0]==0:
                    dist.send(size, dst=self.sending_node, tag=99)
                    running=False
                    break

                ten = torch.empty(size, dtype=self.data_dtype)
                dist.recv(ten, src=self.receiving_node, group=self.recv_data_group, tag=99)
                
                with torch.no_grad():
                    inp = ten.to(self.model_device)
                    out_= self.model(inp)
                    ten = out_.to(self.send_data_device)
                
                size = torch.tensor(ten.shape, dtype=torch.int32)
                dist.send(size, dst=self.sending_node)
                dist.send(ten, dst=self.send_data_device, group=self.send_data_group)
                append_state = False
                
            else:
                r_ctl, r_ten = self.recv.recv()
                s_ctl, s_ten = self.send.getBuffer()

                
                pass




        self.send.close()
        self.recv.close()


############################################3class AddOne(nn.Module):
    def forward(self, x):
        return x + 1


buffer_dim = [1, 1]

dist.init_process_group("gloo")
rank = dist.get_rank()

if rank == 1:
    torch.cuda.set_device(0)
elif rank == 2:
    torch.cuda.set_device(1)

pg_nccl = dist.new_group(ranks=[1, 2], backend="nccl")


if rank == 0:
    send = PipeSender(
        destination=1,
        data_dim=buffer_dim,
        control_config={"idx": 0},
        queue_size=4,
        data_group=None,
        data_device="cpu",
        data_dtype=torch.float32,
    )

    recv = PipeReceiver(
        source=2,
        data_dim=buffer_dim,
        control_config={"idx": 0},
        queue_size=4,
        data_group=None,
        data_device="cpu",
        data_dtype=torch.float32,
    )

    for i in range(10):
        s_ten = send.getBuffer()
        s_ten[0, 0] = i

        print(f"Node 0 -{s_ten.item()}-> Node 1")

        send.send(
            {"end": 0, "append_state": 0, "idx": i},
            s_ten,
        )

        r_ctl, r_ten = recv.recv()

        print(f"result: {r_ctl['idx']} -> {r_ten.item()}")

        recv.release(r_ten)

    # end control만 보냄
    send.send(
        {"end": 1, "append_state": 0, "idx": 10},
        None,
    )

    r_ctl, r_ten = recv.recv()
    print(f"end received: {r_ctl}, data={r_ten}")

    send.close()
    recv.close()


elif rank == 1:
    node = FullNode(
        model=AddOne(),
        receiving_node=0,
        receiving_dim=buffer_dim,
        sending_node=2,
        sending_dim=buffer_dim,
        control_config={"idx": 0},
        queue_size=4,
        recv_data_group=None,
        send_data_group=pg_nccl,
        recv_data_device="cpu",
        send_data_device="cuda:0",
        model_device="cuda:0",
        data_dtype=torch.float32,
    )
    node.run()


elif rank == 2:
    node = FullNode(
        model=AddOne(),
        receiving_node=1,
        receiving_dim=buffer_dim,
        sending_node=0,
        sending_dim=buffer_dim,
        control_config={"idx": 0},
        queue_size=4,
        recv_data_group=pg_nccl,
        send_data_group=None,
        recv_data_device="cuda:1",
        send_data_device="cpu",
        model_device="cuda:1",
        data_dtype=torch.float32,
    )
    node.run()


dist.barrier()
dist.destroy_process_group()
# size = [2, 3]

# if rank==0:
#     data= torch.arange(size[0] * size[1]).reshape(size)

#     dist.send(torch.tensor(size), dst=1)
#     dist.send(data, dst=1)

# elif rank==1:
#     size = torch.empty(2, dtype=torch.int64)
#     dist.recv(size, src=0)
#     data = torch.empty(size.tolist(), dtype=torch.int64)
#     dist.recv(data, src=0)
#     print(data)






# class DynamicPipe:
#     def __init__(self, ):
#         pass

    