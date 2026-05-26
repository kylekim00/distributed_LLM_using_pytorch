import torch
import torch.distributed as dist
import time
dist.init_process_group('gloo')
rank = dist.get_rank()



class Buffer_Send:
    def __init__(self, 
                 tensor_dim:list | torch.Size,
                 target:int, 
                 queue_size:int=4
                ):
        self.pending_queue = list()
        self.target = target
        self.free_tensor = [torch.empty(tensor_dim) for _ in range(queue_size)]
        self.queue_size = queue_size

    def get_empty_tensor(self):
        if not self.free_tensor:
            req, ten= self.pending_queue.pop(0)
            req.wait()
            self.free_tensor.append(ten)
            
        return self.free_tensor.pop(0)
    


    def send_tensor(self, ten:torch.Tensor):
        req = dist.isend(ten, self.target)
        self.pending_queue.append((req, ten))


    def drain(self):
        while self.pending_queue:
            req, ten = self.pending_queue.pop(0)
            req.wait()
            self.free_tensor.append(ten)
        

class Buffer_Recv:
    def __init__(self, 
                 tensor_dim:list | torch.Size, 
                 target:int=1, 
                 queue_size:int=4
                 ):
        self.pending_queue = list()
        # self.free_tensor = list()
        self.queue_size = queue_size
        self.target = target
        for _ in range(queue_size):#fill pending queue
            ten = torch.empty(tensor_dim)
            res = dist.irecv(ten, src=self.target)
            self.pending_queue.append((res, ten))

    #when starting computation, it gets the next tensor from pending queue to get data.
    def get_next_tensor(self)->torch.Tensor:
        res, ten = self.pending_queue.pop(0)
        res.wait()
        return ten

    #when computation is done, it posts used tensor back to pending_queue
    def free_sent_tensor(self, ten:torch.Tensor)->None:
        res = dist.irecv(ten,src=self.target)
        self.pending_queue.append((res, ten))

    def drain(self):
        while self.pending_queue:
            req, ten = self.pending_queue.pop(0)
            # req.wait()
        
    
tensor_dim = [1,1]

    
if rank == 0:
    send_buf = Buffer_Send(tensor_dim, 1)
    recv_buf = Buffer_Recv(tensor_dim, 2)
    for i in range(10):
        ten = send_buf.get_empty_tensor()
        ten[0,0] = i
        print(f"Node 0 -{ten}-> Node 1")
        send_buf.send_tensor(ten=ten)
        out = recv_buf.get_next_tensor()
        print(f"result :{i} -> {out}")
        recv_buf.free_sent_tensor(out)
        time.sleep(1)



elif rank == 1:
    recv_buf = Buffer_Recv(tensor_dim, 0)
    send_buf = Buffer_Send(tensor_dim, 2)
    for _ in range(10):
        ten = recv_buf.get_next_tensor()
        out = send_buf.get_empty_tensor()
        # out = ten + 1
        # out.copy_(ten+1)
        torch.add(ten, 1, out=out)
        print(f"Node 1 -{out}-> Node 2")
        recv_buf.free_sent_tensor(ten)
        send_buf.send_tensor(out)
    recv_buf.drain()
    send_buf.drain()
        

elif rank == 2:
    recv_buf = Buffer_Recv(tensor_dim, 1)
    send_buf = Buffer_Send(tensor_dim, 0)
    for _ in range(10):
        ten = recv_buf.get_next_tensor()
        out = send_buf.get_empty_tensor()
        # out = ten + 1
        # out.copy_(ten+1)
        torch.add(ten, 1, out=out)
        print(f"Node 2 -{out}-> Node 0")
        recv_buf.free_sent_tensor(ten)
        send_buf.send_tensor(out)
    recv_buf.drain()
    send_buf.drain()



dist.barrier()
dist.destroy_process_group()





# if rank == 0:
#     ls = list()
#     free_buffer = [torch.empty([1,1]) for _ in range(4)]
#     weight = torch.tensor(3)
#     for i in range(10):
#        if len(ls) == 4:
#            req, ten = ls.pop(0)
#            req.wait()
#            free_buffer.append(ten)
           
#        ten = free_buffer.pop(0)
#        ten[0,0] = i
#        ls.append((dist.isend(ten, 1), ten))
#        time.sleep(0.8)

#     for req, ten in ls:
#         req.wait()
#         free_buffer.append(ten)

#     dist.barrier()
#     dist.destroy_process_group()


# elif rank==1:
#     ls = list()
#     free_buffer = [torch.empty([1, 1]) for _ in range(4)]

#     for i in range(10):
#         if len(ls) == 4:
#             print("recv pending...")
#             res, ten = ls.pop(0)
#             res.wait()
#             # ten = buffer[0]
#             print(f"i : {ten} a : {ten} {len(ls)}")
#             free_buffer.append(ten)
            

#         ten = free_buffer.pop(0)
#         ls.append((dist.irecv(ten, 0), ten))

        

#     for res, ten in ls:
#         res.wait()
#         print(f"i : {ten} a : {ten} {len(ls)}")
#         free_buffer.append(ten)


#     dist.barrier()
#     dist.destroy_process_group()
        
    
# import torch
# import torch.distributed as dist
# import time
# dist.init_process_group('gloo')
# rank = dist.get_rank()
# # a = torch.empty([1,1])
# if rank == 0:
#     ls = list()
#     buffer = [torch.empty([1,1]) for _ in range(4)]

#     for i in range(10):
#        if len(ls) == 4:
#            ls.pop(0).wait()
           
#        ten = buffer.pop(0)
#        ten[0,0] = i
#        ls.append(dist.isend(ten, 1))
       
#        buffer.append(ten)
#     for req in ls:
#         req.wait()

#     dist.barrier()
#     dist.destroy_process_group()


# elif rank==1:
#     ls = list()
#     buffer = [torch.empty([1, 1]) for _ in range(4)]

#     for i in range(10):
#         if len(ls) == 4:
#             print("recv pending...")
#             ls.pop(0).wait()
#             ten = buffer[0]
#             print(f"i : {ten} a : {ten} {len(ls)}")
            

#         ten = buffer.pop(0)
#         ls.append(dist.irecv(ten, 0))
#         buffer.append(ten)
#         time.sleep(1)
#     for res in ls:
#         res.wait()
#         ten = buffer.pop(0)
#         print(f"i : {ten} a : {ten} {len(ls)}")


#     dist.barrier()
#     dist.destroy_process_group()

# MAX_INFLIGHT = 4
# N = 20

# def pipe1():
#     sender_queue = []
#     for i in range(N):

#         if len(sender_queue) >= MAX_INFLIGHT:
#             req = sender_queue.pop(0)
#             req.wait()                 #if finished, it will pop it. You don't need to handle this.
#         x = torch.tensor([i])
#         req = dist.isend(x, dst=1)
#         sender_queue.append(req)

#     for req in sender_queue:
#         req.wait()






# def pipe2():
#     receiver_queue = []
#     initial = min(MAX_INFLIGHT, N)

#     #first give irecv to receive from queue asynchronously. Keep the queue size
#     for _ in range(initial):
#         buf = torch.empty(1, dtype=torch.long)
#         req = dist.irecv(buf, src=0)
#         receiver_queue.append((req, buf))

#     received = 0
#     while received < N:
#         req, buf = receiver_queue.pop(0)
#         req.wait()

#         print(f"rank1 got {buf.item()}")
#         received+=1

#         if received + len(receiver_queue) < N:
#             new_buf = torch.empty(1, dtype=torch.long)
#             new_req = dist.irecv(new_buf, src=0)
#             receiver_queue.append((new_req, new_buf))



    # dist.init_process_group(backend="gloo")
    
    # rank = dist.get_rank()
    # if rank == 0:
    #     # pipe1()
    #     pass

    # elif rank == 1:
    #     pass
    #     # pipe2()
    # pass