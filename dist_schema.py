import torch
import torch.distributed as dist
import torch.nn as nn


# Buffer  : async buffer for socket transmission
# Schema  : dict <-> tensor 변환
# Channel : 한 종류의 통신 관리
# Pipe    : control + data 묶기
# Node    : 계산 흐름

# -------------------------
# Buffer layer
# -------------------------

class Buffer_Send:
    def __init__(
        self,
        tensor_dim,
        target: int,
        tag: int,
        group=None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        queue_size: int = 4,
    ):
        self.pending_queue = []
        self.target = target
        self.tag = tag
        self.group = group
        self.queue_size = queue_size
        self.free_tensor = [
            torch.empty(size=tensor_dim, dtype=dtype, device=device)
            for _ in range(queue_size)
        ]

    def get_empty_tensor(self):
        if not self.free_tensor:
            req, ten = self.pending_queue.pop(0)
            req.wait()
            self.free_tensor.append(ten)

        return self.free_tensor.pop(0)

    def send_tensor(self, ten: torch.Tensor):
        req = dist.isend(
            tensor=ten,
            dst=self.target,
            tag=self.tag,
            group=self.group,
        )
        self.pending_queue.append((req, ten))

    def close(self):
        for _ in range(self.queue_size):
            self.send_tensor(self.get_empty_tensor().fill_(-1))

        while self.pending_queue:
            req, ten = self.pending_queue.pop(0)
            req.wait()
            self.free_tensor.append(ten)


class Buffer_Recv:
    def __init__(
        self,
        tensor_dim,
        target: int,
        tag: int,
        group=None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        queue_size: int = 4,
    ):
        self.pending_queue = []
        self.target = target
        self.tag = tag
        self.group = group
        self.queue_size = queue_size

        for _ in range(queue_size):
            ten = torch.empty(size=tensor_dim, dtype=dtype, device=device)
            req = dist.irecv(
                ten,
                src=self.target,
                tag=self.tag,
                group=self.group,
            )
            self.pending_queue.append((req, ten))

    def get_next_tensor(self):
        req, ten = self.pending_queue.pop(0)
        req.wait()
        return ten

    def free_sent_tensor(self, ten: torch.Tensor):
        req = dist.irecv(
            ten,
            src=self.target,
            tag=self.tag,
            group=self.group,
        )
        self.pending_queue.append((req, ten))

    def close(self):
        while self.pending_queue:
            req, _ = self.pending_queue.pop(0)
            req.wait()


# -------------------------
# Control layer
# -------------------------

class ControlSchema:
    MAIN_KEYS = ["end", "eop", "data"]

    def __init__(self, extra_keys=None):
        self.keys = list(self.MAIN_KEYS)

        if extra_keys is not None:
            if isinstance(extra_keys, dict):
                extra_keys = extra_keys.keys()

            for key in extra_keys:
                if key not in self.keys:
                    self.keys.append(key)

    def __len__(self):
        return len(self.keys)

    def encode(self, msg: dict, out: torch.Tensor):
        for key in msg.keys():
            if key not in self.keys:
                raise KeyError(f"unknown control key: {key}, expected={self.keys}")

        for i, key in enumerate(self.keys):
            if key not in msg:
                raise KeyError(f"missing control key: {key}, expected={self.keys}")
            out[i] = int(msg[key])

    def decode(self, ten: torch.Tensor) -> dict:
        return {
            key: int(ten[i].item())
            for i, key in enumerate(self.keys)
        }


class ControlSender:
    def __init__(
        self,
        dest: int,
        schema: ControlSchema,
        queue_size: int = 4,
        tag: int = 1,
    ):
        self.schema = schema
        self.buffer = Buffer_Send(
            tensor_dim=[len(schema)],
            target=dest,
            tag=tag,
            dtype=torch.int32,
            queue_size=queue_size,
        )

    def send(self, msg: dict):
        ten = self.buffer.get_empty_tensor()
        self.schema.encode(msg, ten)
        self.buffer.send_tensor(ten)

    def close(self):
        self.buffer.close()


class ControlReceiver:
    def __init__(
        self,
        source: int,
        schema: ControlSchema,
        queue_size: int = 4,
        tag: int = 1,
    ):
        self.schema = schema
        self.buffer = Buffer_Recv(
            tensor_dim=[len(schema)],
            target=source,
            tag=tag,
            dtype=torch.int32,
            queue_size=queue_size,
        )

    def recv(self) -> dict:
        ten = self.buffer.get_next_tensor()
        msg = self.schema.decode(ten)
        self.buffer.free_sent_tensor(ten)
        return msg

    def close(self):
        self.buffer.close()


# -------------------------
# Data channels
# -------------------------

class FixedDataSender:
    def __init__(
        self,
        dest: int,
        data_dim,
        tag: int,
        group=None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        queue_size: int = 4,
    ):
        #this is for gpu that doesn't support irecv(as long as I know)
        if group is not None:
            queue_size = 1

        self.buffer = Buffer_Send(
            tensor_dim=data_dim,
            target=dest,
            tag=tag,
            group=group,
            device=device,
            dtype=dtype,
            queue_size=queue_size,
        )

    def prepare_control(self, msg: dict, data: torch.Tensor | None):
        return

    def get_buffer(self):
        return self.buffer.get_empty_tensor()

    def send(self, data: torch.Tensor):
        self.buffer.send_tensor(data)

    def close(self):
        self.buffer.close()


class FixedDataReceiver:
    def __init__(
        self,
        source: int,
        data_dim,
        tag: int,
        group=None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        queue_size: int = 4,
    ):
        #this is for gpu that doesn't support irecv(as long as I know)
        if group is not None:
            queue_size = 1

        self.buffer = Buffer_Recv(
            tensor_dim=data_dim,
            target=source,
            tag=tag,
            group=group,
            device=device,
            dtype=dtype,
            queue_size=queue_size,
        )

    def recv(self, msg: dict):
        return self.buffer.get_next_tensor()

    def release(self, data: torch.Tensor | None):
        if data is not None:
            self.buffer.free_sent_tensor(data)

    def close(self):
        self.buffer.close()


class DynamicDataSender:
    SHAPE_KEYS = ["dim0", "dim1", "dim2"]

    def __init__(
        self,
        dest: int,
        tag: int,
        group=None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.dest = dest
        self.tag = tag
        self.group = group
        self.device = device
        self.dtype = dtype

    def prepare_control(self, msg: dict, data: torch.Tensor | None):
        if data is None:
            for key in self.SHAPE_KEYS:
                msg[key] = 0
            return

        if len(data.shape) != 3:
            raise ValueError(f"dynamic data must be rank 3, got shape={tuple(data.shape)}")

        msg["dim0"] = data.shape[0]
        msg["dim1"] = data.shape[1]
        msg["dim2"] = data.shape[2]

    def get_buffer(self):
        raise RuntimeError("DynamicDataSender does not use preallocated buffers")

    def send(self, data: torch.Tensor):
        dist.send(
            data,
            dst=self.dest,
            tag=self.tag,
            group=self.group,
        )

    def close(self):
        pass


class DynamicDataReceiver:
    SHAPE_KEYS = ["dim0", "dim1", "dim2"]

    def __init__(
        self,
        source: int,
        tag: int,
        group=None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.source = source
        self.tag = tag
        self.group = group
        self.device = device
        self.dtype = dtype

    def recv(self, msg: dict):
        shape = [
            msg["dim0"],
            msg["dim1"],
            msg["dim2"],
        ]

        ten = torch.empty(
            shape,
            dtype=self.dtype,
            device=self.device,
        )

        dist.recv(
            ten,
            src=self.source,
            tag=self.tag,
            group=self.group,
        )

        return ten

    def release(self, data: torch.Tensor | None):
        pass

    def close(self):
        pass


# -------------------------
# Pipe layer
# -------------------------

class PipeSender:
    def __init__(
        self,
        dest: int,
        schema: ControlSchema,
        data_sender,
        queue_size: int = 4,
        control_tag: int = 1,
    ):
        self.control = ControlSender(
            dest=dest,
            schema=schema,
            queue_size=queue_size,
            tag=control_tag,
        )
        self.data = data_sender

    @classmethod
    def fixed(
        cls,
        dest: int,
        data_dim,
        extra_control_keys=None,
        queue_size: int = 4,
        pipe_tag: int = 0,
        data_group=None,
        data_device: str = "cpu",
        data_dtype: torch.dtype = torch.float32,
    ):
        schema = ControlSchema(extra_control_keys)

        data_sender = FixedDataSender(
            dest=dest,
            data_dim=data_dim,
            tag=pipe_tag * 2 + 0,
            group=data_group,
            device=data_device,
            dtype=data_dtype,
            queue_size=queue_size,
        )

        return cls(
            dest=dest,
            schema=schema,
            data_sender=data_sender,
            queue_size=queue_size,
            control_tag=pipe_tag * 2 + 1,
        )

    @classmethod
    def dynamic(
        cls,
        dest: int,
        extra_control_keys=None,
        queue_size: int = 1,
        pipe_tag: int = 0,
        data_group=None,
        data_device: str = "cpu",
        data_dtype: torch.dtype = torch.float32,
    ):
        keys = []
        if extra_control_keys is not None:
            if isinstance(extra_control_keys, dict):
                keys.extend(extra_control_keys.keys())
            else:
                keys.extend(extra_control_keys)

        keys.extend(DynamicDataSender.SHAPE_KEYS)

        schema = ControlSchema(keys)

        data_sender = DynamicDataSender(
            dest=dest,
            tag=pipe_tag * 2 + 0,
            group=data_group,
            device=data_device,
            dtype=data_dtype,
        )

        return cls(
            dest=dest,
            schema=schema,
            data_sender=data_sender,
            queue_size=queue_size,
            control_tag=pipe_tag * 2 + 1,
        )

    def get_buffer(self):
        return self.data.get_buffer()

    def send(self, msg: dict, data: torch.Tensor | None = None):
        msg = dict(msg)

        msg["end"] = int(msg.get("end", 0))
        msg["eop"] = int(msg.get("eop", 0))
        msg["data"] = int(data is not None)

        self.data.prepare_control(msg, data)

        self.control.send(msg)

        if data is not None:
            self.data.send(data)

    def close(self):
        self.control.close()
        self.data.close()


class PipeReceiver:
    def __init__(
        self,
        source: int,
        schema: ControlSchema,
        data_receiver,
        queue_size: int = 4,
        control_tag: int = 1,
    ):
        self.control = ControlReceiver(
            source=source,
            schema=schema,
            queue_size=queue_size,
            tag=control_tag,
        )
        self.data = data_receiver

    @classmethod
    def fixed(
        cls,
        source: int,
        data_dim,
        extra_control_keys=None,
        queue_size: int = 4,
        pipe_tag: int = 0,
        data_group=None,
        data_device: str = "cpu",
        data_dtype: torch.dtype = torch.float32,
    ):
        schema = ControlSchema(extra_control_keys)

        data_receiver = FixedDataReceiver(
            source=source,
            data_dim=data_dim,
            tag=pipe_tag * 2 + 0,
            group=data_group,
            device=data_device,
            dtype=data_dtype,
            queue_size=queue_size,
        )

        return cls(
            source=source,
            schema=schema,
            data_receiver=data_receiver,
            queue_size=queue_size,
            control_tag=pipe_tag * 2 + 1,
        )

    @classmethod
    def dynamic(
        cls,
        source: int,
        extra_control_keys=None,
        queue_size: int = 1,
        pipe_tag: int = 0,
        data_group=None,
        data_device: str = "cpu",
        data_dtype: torch.dtype = torch.float32,
    ):
        keys = []
        if extra_control_keys is not None:
            if isinstance(extra_control_keys, dict):
                keys.extend(extra_control_keys.keys())
            else:
                keys.extend(extra_control_keys)

        keys.extend(DynamicDataReceiver.SHAPE_KEYS)

        schema = ControlSchema(keys)

        data_receiver = DynamicDataReceiver(
            source=source,
            tag=pipe_tag * 2 + 0,
            group=data_group,
            device=data_device,
            dtype=data_dtype,
        )

        return cls(
            source=source,
            schema=schema,
            data_receiver=data_receiver,
            queue_size=queue_size,
            control_tag=pipe_tag * 2 + 1,
        )

    def recv(self):
        msg = self.control.recv()

        if msg["end"] == 1:
            return msg, None

        if msg["data"] == 0:
            return msg, None

        data = self.data.recv(msg)
        return msg, data

    def release(self, data: torch.Tensor | None):
        self.data.release(data)

    def close(self):
        self.data.close()



class FullNode:
    def __init__(
        self,
        model: nn.Module,

        receiving_node: int,
        receiving_dim,

        sending_node: int,
        sending_dim,

        extra_control_keys=None,

        recv_data_group: dist.ProcessGroup | None = None,
        recv_data_device: str = "cpu",

        data_dtype: torch.dtype = torch.float32,
        model_device: str = "cpu",
        
        send_data_group: dist.ProcessGroup | None = None,
        send_data_device: str = "cpu",

        queue_size: int = 4,
    ):
        self.model = model.to(model_device)
        self.model.eval()
        self.model_device = model_device

        self.recv = PipeReceiver.fixed(
            source=receiving_node,
            data_dim=receiving_dim,
            extra_control_keys=extra_control_keys,
            queue_size=queue_size,
            data_group=recv_data_group,
            data_device=recv_data_device,
            data_dtype=data_dtype,
        )

        self.send = PipeSender.fixed(
            dest=sending_node,
            data_dim=sending_dim,
            extra_control_keys=extra_control_keys,
            queue_size=queue_size,
            data_group=send_data_group,
            data_device=send_data_device,
            data_dtype=data_dtype,
        )

    def run(self) -> None:
        while True:
            ctl, x = self.recv.recv()

            if ctl["end"] == 1:
                self.send.send(ctl, None)
                break

            if ctl["data"] == 0:
                self.send.send(ctl, None)
                continue

            if x is None:
                raise RuntimeError("control says data=1 but tensor is None")

            y_buf = self.send.get_buffer()

            with torch.no_grad():
                x_model = x.to(self.model_device)
                y = self.model(x_model)
                y_buf.copy_(y.to(y_buf.device))

            self.recv.release(x)
            self.send.send(ctl, y_buf)

        self.send.close()
        self.recv.close()

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

        input_dtype: torch.dtype = torch.float32,
        output_dtype: torch.dtype = torch.float32,

        model_device: str = "cpu",

        queue_size: int = 4,
    ):
        self.model = layer1.to(model_device)
        self.model.eval()
        self.model_device = model_device

        self.input_dtype = input_dtype
        self.output_dtype = output_dtype

        self.prompt_recv = PipeReceiver.dynamic(  # 0 -> 1 dynamic
            source=prompt_node,
            data_dtype=input_dtype,
            extra_control_keys=extra_control_keys,
            data_group=prompt_recv_data_group,
            data_device=prompt_recv_data_device,
            pipe_tag=0,
        )

        self.layer2_dynamic_send = PipeSender.dynamic(  # 1 -> 2 dynamic
            dest=layer2_node,
            extra_control_keys=extra_control_keys,
            data_group=layer2_send_data_group,
            data_device=layer2_send_data_device,
            data_dtype=output_dtype,
            pipe_tag=0,
        )

        self.layer2_fixed_recv = PipeReceiver.fixed(  # 2 -> 1
            source=layer2_node,
            data_dim=layer2_next_receiving_dim,
            extra_control_keys=extra_control_keys,
            data_group=layer2_recv_data_group,
            data_device=layer2_recv_data_device,
            data_dtype=input_dtype,
        )

        self.layer2_fixed_send = PipeSender.fixed(  # 1 -> 2
            dest=layer2_node,
            data_dim=layer2_hidden_sending_dim,
            extra_control_keys=extra_control_keys,
            queue_size=queue_size,
            data_group=layer2_send_data_group,
            data_device=layer2_send_data_device,
            data_dtype=output_dtype,
            pipe_tag=1,
        )

        self.layer2_send_data_device_is_cpu = layer2_send_data_device == "cpu"
        self.layer2_recv_data_device_is_cpu = layer2_recv_data_device == "cpu"
        self.prompt_recv_data_device_is_cpu = prompt_recv_data_device == "cpu"

        self.state = True

    def run(self) -> None:
        self.model.eval()
        past_key_values = None
        cache_len = 0

        while True:
            if self.state:
                ctl, X = self.prompt_recv.recv()

                if ctl["end"]:
                    self.prompt_recv.release(X)
                    break

                if ctl["data"] == 0:
                    continue

                input1 = X.to(
                    device=self.model_device,
                    dtype=self.input_dtype,
                )

                cache_position = torch.arange(
                    cache_len,
                    X.shape[1] + cache_len,
                    device=input1.device,
                )
                cache_len = cache_len + X.shape[1]

                _res = self.model(
                    input_ids=input1,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    use_cache=True,
                )

                past_key_values = _res["past_key_values"]

                if self.layer2_send_data_device_is_cpu:
                    out = _res["hidden_states"][:, -1].to(
                        device="cpu",
                        dtype=self.output_dtype,
                    )
                else:
                    out = _res["hidden_states"].to(
                        dtype=self.output_dtype,
                    )

                self.layer2_dynamic_send.send(ctl, out)
                self.state = False

            else:
                ctl, next_token = self.layer2_fixed_recv.recv()

                if ctl["end"]:
                    self.layer2_fixed_recv.release(next_token)
                    self.layer2_fixed_send.send(ctl)
                    break

                if ctl["eop"]:
                    self.state = True
                    self.layer2_fixed_recv.release(next_token)
                    continue

                input1 = next_token.to(
                    device=self.model_device,
                    dtype=self.input_dtype,
                )

                cache_position = torch.tensor(
                    [cache_len],
                    device=input1.device,
                )
                cache_len = cache_len + 1

                _res = self.model(
                    input_ids=input1,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    use_cache=True,
                )

                past_key_values = _res["past_key_values"]
                _out = _res["hidden_states"]

                if self.layer2_send_data_device_is_cpu:
                    _out = _out.to(
                        device="cpu",
                        dtype=self.output_dtype,
                    )
                else:
                    _out = _out.to(
                        dtype=self.output_dtype,
                    )

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

                data_dtype: torch.dtype = torch.float32,
                model_device: str = "cpu",

                queue_size:int = 4,
                ):
            self.model = layer2.to(model_device)
            self.model.eval()
            self.model_device = model_device
            self.eos_token_id = eos_token_id
            self.layer1_dynamic_recv = PipeReceiver.dynamic( # 1 -> 2 dynamic
                source=layer1_node,
                extra_control_keys=extra_control_keys,
                data_group=layer1_recv_data_group,
                data_device=layer1_recv_data_device,
                data_dtype=data_dtype,
                pipe_tag=0,
            )
            self.layer1_fixed_recv = PipeReceiver.fixed( # 1 -> 2
                source=layer1_node,
                data_dim=layer1_hidden_receiving_dim,
                extra_control_keys=extra_control_keys,
                data_group=layer1_recv_data_group,
                data_device=layer1_recv_data_device,
                data_dtype=data_dtype,
                pipe_tag=1
            )
            self.prompt_send= PipeSender.fixed( # 2 -> 0
                dest=prompt_node,
                data_dim=next_token_sending_dim,
                extra_control_keys=extra_control_keys,
                queue_size=queue_size,
                data_group=prompt_send_data_group,
                data_device=prompt_send_data_device,
                data_dtype=data_dtype
            )
            

            self.layer1_send = PipeSender.fixed( # 2 -> 1
                dest= layer1_node,
                data_dim= next_token_sending_dim,
                extra_control_keys=extra_control_keys,
                queue_size=queue_size,
                data_group= layer1_send_data_group,
                data_device= layer1_send_data_device,
                data_dtype=data_dtype,
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


                    input2 = X.to(self.model_device)

                    cache_position = torch.arange(cache_len, X.shape[1] + cache_len, device=input2.device)
                    cache_len = cache_len + X.shape[1]

                    _res = self.model(
                        input_ids = input2,
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
                        continue

                    out1 = self.layer1_send.get_buffer()
                    out2 = self.prompt_send.get_buffer()

                    if self.layer1_send_data_device_is_cpu:
                        _out1 = _out.to("cpu")
                    else:
                        _out1 = _out

                    out1.copy_(_out1)

                    if self.prompt_send_data_device_is_cpu:
                        _out2 = _out.to("cpu")
                    else:
                        _out2= _out
                    
                    out2.copy_(_out2)

                    self.layer1_send.send(ctl, out1)
                    self.prompt_send.send(ctl, out2)
                    
                    self.layer1_dynamic_recv.release(X)
                    self.state = False
                else:
                    #TODO get hidden next_token state compute layer2, if eos, turn on eos, if not send next_token to prompt and layer1
                    ctl, X = self.layer1_fixed_recv.recv() # get next state token

                    if ctl['end']:
                        self.layer1_fixed_recv.release(X)
                        break
                    if ctl['eop']:
                        self.state = True
                        self.layer1_fixed_recv.release(X)
                        continue
                    
                    input2 = X.to(self.model_device)
                    cache_position = torch.tensor([cache_len], device=input2.device)
                    cache_len = cache_len + 1
                    
                    _res = self.layer2(
                        input_ids = input2,
                        past_key_values = past_key_values,
                        cache_position = cache_position,
                        use_cache = True
                    )

                    past_key_values = _res["past_key_values"]

                    _out = _res["logits"][:,-1].argmax(dim=-1, keepdim=True)

                    out1 = self.layer1_send.get_buffer()
                    out2 = self.prompt_send.get_buffer()

                    if self.layer1_send_data_device_is_cpu:
                        _out1 = _out.to("cpu")
                    else:
                        _out1 = _out
                    
                    out1.copy_(_out1)

                    if self.prompt_send_data_device_is_cpu:
                        _out2 = _out.to("cpu")
                    else:
                        _out2 = _out
                    
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

                data_dtype : torch.dtype = torch.float32,

                queue_size: int = 4
                ):
            
            self.layer1_send = PipeSender.dynamic(
                dest=layer1_node,
                extra_control_keys=extra_control_keys,
                queue_size=queue_size,
                data_group=layer1_send_data_group,
                data_device=layer1_send_data_device,
                data_dtype=data_dtype
            )

            self.layer2_recv = PipeReceiver.fixed(
                source=layer2_node,
                data_dim=layer2_next_receiving_dim,
                queue_size=queue_size,
                data_group=layer2_recv_data_group,
                data_device=layer2_recv_data_device,
                data_dtype=data_dtype,
            )

            
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            self.layer_data_is_cpu = layer1_send_data_device == "cpu"
            self.state = True


        def run(self)->None:
            prompt = ""
            while True:
                if self.state:
                    p = input("prompt: ")
                    if p.lower() in ("q","quit"):
                        break
                    prompt += "\nuser : "+ p + "\nprompt: "

                    inputs = self.tokenizer(prompt, return_tensors="pt")

                    # attention_mask = inputs.get("attention_mask", None) 
                    # ##sxxt...... I didn't thought about sending.. this.... 
                    # ##but since my implementation receive only one prompt, it's okay for now.

                    # if attention_mask is not None:
                    #     attention_mask = attention_mask.to(device)
                    if not self.layer_data_is_cpu:
                        input_ids = inputs['input_ids'].to("cuda")
                    else:
                        input_ids = inputs['input_ids']
                    
                    self.layer1_send.send({'end':False,'eop':False}, input_ids)

                    self.state = False
                    print("prompt: ", end="")
                else:
                    ctl, next_token = self.layer2_recv.recv()

                    if ctl['eop']:
                        self.state = True
                        continue
                    
                    token_text = self.tokenizer.decode(next_token[0])
                    print(token_text, end="")
                    prompt += token_text
                    self.layer2_recv.release(next_token)


            self.layer1_send.send({'end':True})

            

                    





                
            
#####################################################
# dist.init_process_group("gloo")
# rank = dist.get_rank()

# if rank==0:
#     sender = PipeSender.fixed(1, [1,1])
#     ten = sender.get_buffer()
#     ten[0,0] = 12
#     sender.send({'end':False, 'data':True}, ten)
#     sender.send({'end':True})
#     sender.close()

# elif rank==1:
#     recver = PipeReceiver.fixed(0, [1,1])
#     ctrl, ten = recver.recv()
#     print(ctrl, ten)
#     recver.recv()
#     recver.release(ten)

#     recver.close()

# dist.destroy_process_group()


import torch
import torch.nn as nn

from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)



