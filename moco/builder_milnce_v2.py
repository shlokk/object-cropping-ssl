# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder,args, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()
        print(dim)

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        # self.encoder_q = base_encoder(num_classes=dim)
        # self.encoder_k = base_encoder(num_classes=dim)
        self.encoder_q = base_encoder( dim_out=args.moco_dim, avg_pool_shape=args.avg_pool_shape,num_classes = dim)
        self.encoder_k = base_encoder( dim_out=args.moco_dim, avg_pool_shape=args.avg_pool_shape,num_classes = dim)
        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(args.avg_pool_shape[0]*args.avg_pool_shape[1],dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        # print("concat start")
        # print(keys.shape)

        batch_size = keys.shape[0]
        # print(batch_size)


        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity
        # print(ptr)
        # print(self.queue.shape)
        keys = keys.permute(1,2,0)
        # print(keys.shape)

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:,:, ptr:ptr + batch_size] = keys
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        # print("query 1st forawrd pass " + str(torch.cuda.memory_allocated()))
        q = self.encoder_q(im_q)  # queries: NxC
        # print("just after query 1st forawrd pass " + str(torch.cuda.memory_allocated()))
        q = nn.functional.normalize(q, dim=2)
        # print(im_q.shape,im_k.shape)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            # print("after momentum update " + str(torch.cuda.memory_allocated()))

            # shuffle for making use of BN

            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            # print("encoder k")
            # print(im_k.shape)
            # print(self.encoder_k)
            # import pdb
            # pdb.set_trace()

            # print("key forawrd pass "+str(torch.cuda.memory_allocated()))
            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=2)
            # k = nn.functional.normalize(k, dim=1)
            # second_dim = k.shape[0]/im_k.shape[0]
            # print(second_dim)
            # k = k.reshape(im_k.shape[0],int(second_dim),k.shape[1])
            # print(k.shape)
            # k = k.reshape(k.shape[0] * k.shape[1], k.shape[2])


            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)


        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        # print(q.shape,k.shape)

        # print("poistive sample lpos " + str(torch.cuda.memory_allocated()))
        # l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # print(l_pos)
        # l_pos = torch.matmul(q,k.T)
        k = k.permute(0,2,1)
        # print(q.shape, k.shape)
        # print("k")
        # # print(k)
        # print(k.shape)
        # print("q")
        # # print(q)
        # print(q.shape)


        l_pos = torch.bmm(q, k)
        # print("lpos shape")
        # print(l_pos.shape)
        k = k.permute(0, 2, 1)
        # print(k)
        # print(q)
        # print(l_pos)
        # print(stop)
        # l_pos = torch.einsum('nc,nc->n', [q, k])
        # print(l_pos.shape)
        # negative logits: NxK
        # print("negative sample l_neg " + str(torch.cuda.memory_allocated()))
        # l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        # print(self.queue.shape)
        l_neg = torch.einsum('ncp,dpk->ncdk', [q, self.queue.clone().detach()])
        # print("lneg shape")
        # print(l_neg.shape)
        # print(stop)

        # logits: Nx(1+K)
        # logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        # logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(l_pos.shape[0], dtype=torch.long).cuda()
        # print(labels)



        # dequeue and enqueue
        self._dequeue_and_enqueue(k)
        # print("last after queue dequeue " + str(torch.cuda.memory_allocated()))

        return l_pos,l_neg, labels


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
