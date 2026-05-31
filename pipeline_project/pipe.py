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


