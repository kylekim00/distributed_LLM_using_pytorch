import torch
import torch.nn as nn
from .pipe import PipeReceiver, PipeSender


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

        self.model = model

        # self.send_control = None
        # self.send_buffer = None
        # self.recv_control = None
        # self.recv_buffer = None

        self.control_config = control_config.copy()
        control_dim = [len(control_config)] 

        self.send = PipeSender(
            destination=sending_node,
            data_dim=sending_dim,
            control_dim=control_dim,
            control_queue_size=queue_size,
            data_queue_size=queue_size
        )
        self.recv = PipeReceiver(
            source=receiving_node,
            control_dim=control_dim,
            data_dim= receiving_dim,
            control_queue_size=queue_size,
            data_queue_size=queue_size
        )
        
        # self.send_control = Buffer_Send(control_dim, sending_node, 1, queue_size)
        # self.send_buffer = Buffer_Send(sending_dim, sending_node, 0, queue_size)

    
        # self.recv_control = Buffer_Recv(control_dim, receiving_node, 1, queue_size)
        # self.recv_buffer = Buffer_Recv(receiving_dim, receiving_node, 0, queue_size)

    def _configDecoder(self, control_buffer:torch.Tensor)->dict:
        for inx, key in enumerate(self.control_config.keys()):
            self.control_config[key] = control_buffer[inx]
        return self.control_config
    
    # def _configEncoder(self)->torch.Tensor:
    #     return torch.tensor(self.control_config.values())
    

    def run(self)->None:
        tmp = True

        while tmp:

            r_ctl, r_ten = self.recv.recv()
            s_ctl, s_ten = self.send.getBuffer()

            if self._configDecoder(r_ctl)['end'] == 1:
                tmp = False

            s_ctl.copy_(r_ctl)

        
            with torch.no_grad():
                out_ = self.model(r_ten.to(self.model.device))
                s_ten.copy_(out_.to(s_ten.device)) #this part should be moderated.(as well as other buffers)

            
            self.recv.release(r_ctl, r_ten)
            self.send.send(s_ctl, s_ten)

        
        self.send.close()
        self.recv.close()
