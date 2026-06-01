import torch
import torch.nn as nn
import torch.distributed as dist
from .pipe import PipeReceiver, PipeSender




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


        #keep the queue of NCCL to 1(this nccl keeps getting problemssssssssss!!!!!!!!!)
        recv_queue_size = queue_size
        send_queue_size = queue_size

        if recv_data_group is not None:
            recv_queue_size = 1
        if send_data_group is not None:
            send_queue_size = 1

        
        #control set to CPU for now
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
                inp = r_ten.to(self.model_device)
                out_ = self.model(inp)
                s_ten.copy_(out_.to(s_ten.device)) #this part should be moderated.(as well as other buffers)

            
            self.recv.release(r_ctl, r_ten)
            self.send.send(s_ctl, s_ten)

        
        self.send.close()
        self.recv.close()




class LayerNode:
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


        #keep the queue of NCCL to 1(this nccl keeps getting problemssssssssss!!!!!!!!!)
        recv_queue_size = queue_size
        send_queue_size = queue_size

        if recv_data_group is not None:
            recv_queue_size = 1
        if send_data_group is not None:
            send_queue_size = 1

        
        #control set to CPU for now
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
                inp = r_ten.to(self.model_device)
                out_ = self.model(inp)
                s_ten.copy_(out_.to(s_ten.device)) #this part should be moderated.(as well as other buffers)

            
            self.recv.release(r_ctl, r_ten)
            self.send.send(s_ctl, s_ten)

        
        self.send.close()
        self.recv.close()



class PromptNode:
    def __init__(self):
        pass