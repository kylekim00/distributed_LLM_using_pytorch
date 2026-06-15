import torch
import torch.distributed as dist


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