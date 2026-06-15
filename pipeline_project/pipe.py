import torch
import torch.distributed as dist
from .buffer import *
# import torch.nn as nn

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