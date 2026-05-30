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
            destination:int, 
            data_dim:list | torch.Size, 
            control_dim:list | torch.Size, 
            control_queue_size:int, 
            data_queue_size:int,
            pipe_tag:int=0,                 #if connecting same node, then it should have a different pipe.

            control_group:dist.ProcessGroup|None = None,
            data_group:dist.ProcessGroup|None = None,
            
            control_device:str="cpu",
            data_device:str="cpu",

            control_dtype:torch.dtype = torch.int32,
            data_dtype:torch.dtype = torch.float32

            ):
        self.control = Buffer_Send(
            tensor_dim=control_dim, 
            target=destination, 
            tag=pipe_tag*2 + 1, 
            group=control_group,
            device=control_device,
            dtype=control_dtype,
            queue_size=control_queue_size
            )
        self.data = Buffer_Send(
            tensor_dim=data_dim, 
            target=destination, 
            tag=pipe_tag*2 + 0, 
            group=data_group,
            device=data_device,
            dtype=data_dtype,
            queue_size=data_queue_size
            )

    def getBuffer(self)->tuple[torch.Tensor, torch.Tensor]:
        return self.control.get_empty_tensor(), self.data.get_empty_tensor()

    def send(self, ctl:torch.Tensor, data:torch.Tensor)->None:
        self.control.send_tensor(ctl)
        self.data.send_tensor(data)

    def close(self)->None:
        self.control.close()
        self.data.close()


class PipeReceiver:
    def __init__(
            self, 
            source:int, 
            control_dim:list, 
            data_dim:list|torch.Size, 
            control_queue_size:int=4, 
            data_queue_size:int=4,
            pipe_tag=0,

            control_group:dist.ProcessGroup | None=None,
            data_group:dist.ProcessGroup | None = None,

            control_device:str = 'cpu',
            data_device:str = "cpu",

            control_dtype:torch.dtype = torch.int32,
            data_dtype:torch.dtype = torch.float32
            ):
        
        self.control = Buffer_Recv(
            tensor_dim=control_dim,
            target=source,
            tag=pipe_tag * 2 + 1,
            group=control_group,
            device=control_device,
            dtype=control_dtype,
            queue_size=control_queue_size,
        )

        self.data = Buffer_Recv(
            tensor_dim=data_dim,
            target=source,
            tag=pipe_tag * 2 + 0,
            group=data_group,
            device=data_device,
            dtype=data_dtype,
            queue_size=data_queue_size,
        )
    
    def recv(self)->tuple[torch.Tensor, torch.Tensor]:
        return self.control.get_next_tensor(), self.data.get_next_tensor()

    def release(self, ctl:torch.Tensor, data:torch.Tensor):
        self.control.free_sent_tensor(ctl)
        self.data.free_sent_tensor(data)
    
    def close(self):
        self.control.close()
        self.data.close()







#.copy_() version 
class FullNode:
    def __init__(self,
        model: nn.Module,
        
        receiving_node: int,
        receiving_dim: list | torch.Size,

        sending_node: int,
        sending_dim: list | torch.Size,
        
        control_config: dict | None = None,
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


        #chekc control config
        if control_config is None:
            control_config = dict()
            
        self.control_config = control_config.copy()
        self.control_config['end'] = 0

        for key, value in self.control_config.items():
            if not isinstance(value, int):
                raise ValueError(f"control_config{key} must be int, got {type(value)}")
        
        control_dim = [len(self.control_config)] 
        
        #control set to CPU for now
        self.send = PipeSender(
            destination=sending_node,
            data_dim=sending_dim,
            control_dim=control_dim,
            control_queue_size=queue_size,
            data_queue_size=queue_size,
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
            data_queue_size=queue_size,
            control_group=None,
            data_group=recv_data_group,
            control_device="cpu",
            data_device=recv_data_device,
            control_dtype=torch.int32,
            data_dtype=data_dtype,
        )
        
        # self.send_control = Buffer_Send(control_dim, sending_node, 1, queue_size)
        # self.send_buffer = Buffer_Send(sending_dim, sending_node, 0, queue_size)

    
        # self.recv_control = Buffer_Recv(control_dim, receiving_node, 1, queue_size)
        # self.recv_buffer = Buffer_Recv(receiving_dim, receiving_node, 0, queue_size)

    def _configDecoder(self, control_buffer:torch.Tensor)->dict:
        for inx, key in enumerate(self.control_config.keys()):
            self.control_config[key] = int(control_buffer[inx].item())
        return self.control_config
    
    # def _configEncoder(self)->torch.Tensor:
    #     return torch.tensor(self.control_config.values())
    

    def run(self)->None:
        running = True

        while running:

            r_ctl, r_ten = self.recv.recv()
            s_ctl, s_ten = self.send.getBuffer()

            if self._configDecoder(r_ctl)['end'] == 1:
                running = False

            s_ctl.copy_(r_ctl)

        
            with torch.no_grad():
                inp = r_ten.to(self.model.device)
                out_ = self.model(inp)
                s_ten.copy_(out_.to(s_ten.device)) #this part should be moderated.(as well as other buffers)

            
            self.recv.release(r_ctl, r_ten)
            self.send.send(s_ctl, s_ten)

        
        self.send.close()
        self.recv.close()


############################################ main ###################################################

class AddOne(nn.Module):
    def forward(self, x):
        return x + 1
        
control_dim = [1]
buffer_dim = [1,1]

dist.init_process_group('gloo')
rank = dist.get_rank()


if rank==1:
    torch.cuda.set_device(0)
elif rank==2:
    torch.cuda.set_device(1)

pg_nccl = dist.new_group(ranks=[1, 2], backend="nccl")


if rank == 0:
    
    send = PipeSender(
        destination=1,
        data_dim=buffer_dim,
        control_dim=control_dim,
        control_queue_size=4,
        data_queue_size=4,
        control_group=None,
        data_group=None,
        control_device="cpu",
        data_device="cpu",
        control_dtype=torch.int32,
        data_dtype=torch.float32,
    )

    recv = PipeReceiver(
        source=2,
        data_dim=buffer_dim,
        control_dim=control_dim,
        control_queue_size=4,
        data_queue_size=4,
        control_group=None,
        data_group=None,
        control_device="cpu",
        data_device="cpu",
        control_dtype=torch.int32,
        data_dtype=torch.float32,
    )

    
    # recv_control = Buffer_Recv(control_dim, 2, 1)
    # recv_buf = Buffer_Recv(buffer_dim, 2, 0)

    for i in range(10):
        # s_ctl = send_control.get_empty_tensor()
        # ten = send_buf.get_empty_tensor()

        s_ctl, s_ten = send.getBuffer()


        s_ctl[0] = 0 if i<9 else 1
        s_ten[0,0] = i
        print(f"Node 0 -{s_ten.item()}-> Node 1")

        send.send(s_ctl, s_ten)
#-----------------------------------------------------
        # r_ctl = recv_control.get_next_tensor()
        # out = recv_buf.get_next_tensor()
        
        r_ctl, r_ten = recv.recv()

        suffix = " | end!" if r_ctl[0].item() == 1 else ""
        print(f"result :{i} -> {r_ten.item()}{suffix}")

        recv.release(r_ctl, r_ten)
        # time.sleep(1)

    send.close()
    recv.close()



elif rank == 1:
    # 0 -> 1 : Gloo CPU
    # 1 -> 2 : NCCL CUDA cuda:0
    node = FullNode(
        model=AddOne(),
        receiving_node=0,
        receiving_dim=buffer_dim,
        sending_node=2,
        sending_dim=buffer_dim,
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
    # 1 -> 2 : NCCL CUDA cuda:1
    # 2 -> 0 : Gloo CPU
    node = FullNode(
        model=AddOne(),
        receiving_node=1,
        receiving_dim=buffer_dim,
        sending_node=0,
        sending_dim=buffer_dim,
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

