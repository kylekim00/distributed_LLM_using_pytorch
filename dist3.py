import torch
import torch.distributed as dist
import torch.nn as nn
import time
from functools import wraps

dist.init_process_group('gloo')
rank = dist.get_rank()


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



#.copy_() version 
class FullNode:
    def __init__(self,
        model: nn.Module,
        receiving_node: int,
        receiving_dim: list | torch.Size,
        sending_node: int,
        sending_dim: list | torch.Size,
        control_config: dict = {'end':0},
        queue_size: int = 4,
    ):



        self.send_control = None
        self.send_buffer = None
        self.recv_control = None
        self.recv_buffer = None

        self.control_config = control_config.copy()
        control_dim = [len(control_config)] 

        
        self.send_control = Buffer_Send(control_dim, sending_node, 1, queue_size)
        self.send_buffer = Buffer_Send(sending_dim, sending_node, 0, queue_size)

    
        self.recv_control = Buffer_Recv(control_dim, receiving_node, 1, queue_size)
        self.recv_buffer = Buffer_Recv(receiving_dim, receiving_node, 0, queue_size)

    def _configDecoder(self, control_buffer:torch.Tensor)->dict:
        for inx, key in enumerate(self.control_config.keys()):
            self.control_config[key] = control_buffer[inx]
        return self.control_config
    
    # def _configEncoder(self)->torch.Tensor:
    #     return torch.tensor(self.control_config.values())
    

    def run(self)->None:
        tmp = True

        while tmp:
            r_ctl = self.recv_control.get_next_tensor()
            s_ctl = self.send_control.get_empty_tensor()

            
            if self._configDecoder(r_ctl)['end'] == 1:
                tmp = False
            s_ctl.copy_(r_ctl)
            self.recv_control.free_sent_tensor(r_ctl)

            ten = self.recv_buffer.get_next_tensor()
            out = self.send_buffer.get_empty_tensor()
            with torch.no_grad():
                out_ = self.model(ten)
                out.copy_(out_) #this part should be moderated.(as well as other buffers)
            self.recv_buffer.free_sent_tensor(ten)
            self.send_control.send_tensor(s_ctl)
            self.send_buffer.send_tensor(out)
        
        self.send_control.close()
        self.recv_control.close()

        self.send_buf.close()
        self.recv_buf.close()

class AddOne(nn.Module):
    def forward(self, x):
        return x + 1

        
control_dim = [1]
buffer_dim = [1,1]



if rank == 0:
    send_control = Buffer_Send(control_dim, 1, 1)
    recv_control = Buffer_Recv(control_dim, 2, 1)
    
    send_buf = Buffer_Send(buffer_dim, 1, 0)
    recv_buf = Buffer_Recv(buffer_dim, 2, 0)

    for i in range(10):
        s_ctl = send_control.get_empty_tensor()
        ten = send_buf.get_empty_tensor()

        s_ctl[0] = 0 if i<9 else 1
        ten[0,0] = i
        print(f"Node 0 -{ten}-> Node 1")

        send_control.send_tensor(ten=s_ctl)
        send_buf.send_tensor(ten=ten)
#-----------------------------------------------------
        r_ctl = recv_control.get_next_tensor()
        out = recv_buf.get_next_tensor()

        suffix = " | end!" if r_ctl[0].item() == 1 else ""
        print(f"result :{i} -> {out}{suffix}")

        recv_control.free_sent_tensor(r_ctl)
        recv_buf.free_sent_tensor(out)
        # time.sleep(1)

    send_control.close()
    recv_control.close()
    send_buf.close()
    recv_buf.close()


elif rank == 1:
    node = FullNode(
        model=AddOne(),
        receiving_node=0,
        receiving_dim=buffer_dim,
        sending_node=2,
        sending_dim=buffer_dim,
        queue_size=4
    )
    node.run()
        

elif rank == 2:
    node = FullNode(
        model=AddOne(),
        receiving_node=1,
        receiving_dim=buffer_dim,
        sending_node=0,
        sending_dim=buffer_dim,
        queue_size=4
    )

    node.run()


dist.barrier()
dist.destroy_process_group()

