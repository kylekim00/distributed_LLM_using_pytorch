import torch
import torch.distributed as dist
import torch.nn as nn

class Buffer_Send:
    def __init__(self, 
                 tensor_dim:list | torch.Size,
                 target:int, 
                 tag:int,
                 queue_size:int=4,
                ):
        self.pending_queue = list()
        self.target = target
        self.tag = tag
        self.free_tensor = [torch.empty(tensor_dim) for _ in range(queue_size)]
        self.queue_size = queue_size

    def get_empty_tensor(self):
        if not self.free_tensor:
            req, ten= self.pending_queue.pop(0)
            req.wait()
            self.free_tensor.append(ten)
            
        return self.free_tensor.pop(0)
    


    def send_tensor(self, ten:torch.Tensor):
        req = dist.isend(ten, self.target, tag=self.tag)
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
                 queue_size:int=4
                 ):
        self.pending_queue = list()
        self.tag = tag
        self.queue_size = queue_size
        self.target = target
        for _ in range(queue_size):#fill pending queue
            ten = torch.empty(tensor_dim)
            res = dist.irecv(ten, src=self.target, tag=self.tag)
            self.pending_queue.append((res, ten))

    #when starting computation, it gets the next tensor from pending queue to get data.
    def get_next_tensor(self)->torch.Tensor:
        res, ten = self.pending_queue.pop(0)
        res.wait()
        return ten

    #when computation is done, it posts used tensor back to pending_queue
    def free_sent_tensor(self, ten:torch.Tensor)->None:
        res = dist.irecv(ten,src=self.target, tag=self.tag)
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
            pipe_tag=0
            ):
        self.control = Buffer_Send(control_dim, destination, tag=pipe_tag*2 + 1, queue_size=control_queue_size)
        self.data = Buffer_Send(data_dim, destination, tag=pipe_tag*2 + 0, queue_size=data_queue_size)

    def getBuffer(self)->tuple:
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
            pipe_tag=0
            ):
        self.control = Buffer_Recv(control_dim, source, tag=pipe_tag*2 + 1, queue_size=control_queue_size)
        self.data = Buffer_Recv(data_dim, source, tag=pipe_tag*2 + 0, queue_size=data_queue_size)
    
    def recv(self)->list:
        return self.control.get_next_tensor(), self.data.get_next_tensor()

    def release(self, ctl:torch.Tensor, data:torch.Tensor):
        self.control.free_sent_tensor(ctl)
        self.data.free_sent_tensor(data)
    
    def close(self):
        self.control.close()
        self.data.close()





