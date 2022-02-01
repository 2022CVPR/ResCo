"""https://github.com/facebookresearch/moco"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import pdb
import torch.distributed as dist
import numpy as np


class ResCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, pos_size_per_cls=8, class_num=1000, dim=128, neg_size_per_cls=4):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(ResCo, self).__init__()
        
        self.class_num = class_num
        # create the encoders
        # num_classes is the output fc dimension
        self.encoder = base_encoder(num_classes=dim)
        dim_feat = 2048
        self.linear = nn.Linear(dim_feat, self.class_num)

        self.mlp = nn.Sequential(nn.Linear(dim_feat, dim_feat), nn.BatchNorm1d(dim_feat), nn.ReLU(), nn.Linear(dim_feat, dim))

        # create the queue
        self.neg_size_per_cls = neg_size_per_cls
        self.register_buffer("neg_queue_list", torch.randn(dim, self.neg_size_per_cls * self.class_num))
        self.neg_queue_list = nn.functional.normalize(self.neg_queue_list, dim=0)
        self.register_buffer("neg_queue_ptr", torch.zeros(self.class_num, dtype=torch.long))


        self.pos_size_per_cls = pos_size_per_cls
        self.register_buffer("pos_queue_list", torch.randn(dim, self.pos_size_per_cls * self.class_num))
        self.pos_queue_list = nn.functional.normalize(self.pos_queue_list, dim=0)
        self.register_buffer("pos_queue_ptr", torch.zeros(self.class_num, dtype=torch.long))


    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, key_c, c):
        # gather keys before updating queue

        # instance_size = 1 (put one into queue)
        instance_size = key_c.shape[0]
        neg_ptr = int(self.neg_queue_ptr[c])
        neg_real_ptr = neg_ptr + c * self.neg_size_per_cls
        # replace the keys at ptr (dequeue and enqueue)
        self.neg_queue_list[:, neg_real_ptr:neg_real_ptr + instance_size] = key_c.T
        neg_ptr = (neg_ptr + instance_size) % self.neg_size_per_cls  # move pointer
        self.neg_queue_ptr[c] = neg_ptr
    
        pos_ptr = int(self.pos_queue_ptr[c])
        pos_real_ptr = pos_ptr + c * self.pos_size_per_cls
        # replace the keys at ptr (dequeue and enqueue)
        self.pos_queue_list[:, pos_real_ptr:pos_real_ptr + instance_size] = key_c.T
        pos_ptr = (pos_ptr + instance_size) % self.pos_size_per_cls  # move pointer
        self.pos_queue_ptr[c] = pos_ptr



    def _train(self, im_q, im_k, labels_q, labels_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        mid_feat_q = self.encoder(im_q)  # queries: NxC
        feat_q = self.mlp(mid_feat_q)
        feat_q = nn.functional.normalize(feat_q, dim=1)

        # compute key features
        mid_feat_k = self.encoder(im_k)
        feat_k = self.mlp(mid_feat_k)
        feat_k = nn.functional.normalize(feat_k, dim=1)
        
        pos_cur_queue_list = self.pos_queue_list.clone().detach()
        neg_cur_queue_list = self.neg_queue_list.clone().detach()
        
        pos_list =  torch.zeros([feat_q.shape[0], self.pos_size_per_cls]).to(feat_q.get_device())
        neg_list = torch.zeros([feat_q.shape[0], self.neg_size_per_cls * (self.class_num - 1)]).to(feat_q.get_device())

        # pos_list = torch.Tensor([]).cuda()
        # neg_list = torch.Tensor([]).cuda()
        
        for i in range(feat_q.shape[0]):
            pos_sample = pos_cur_queue_list[:, labels_q[i]*self.pos_size_per_cls: (labels_q[i]+1)*self.pos_size_per_cls]
            pos_ith = feat_q[i:i+1] @ pos_sample
            pos_list[i, :] = pos_ith
            # pos_list = torch.cat((pos_list, pos_ith), dim=0)

            neg_sample = torch.cat([neg_cur_queue_list[:,0:labels_q[i]*self.neg_size_per_cls],
                                    neg_cur_queue_list[:,(labels_q[i]+1)*self.neg_size_per_cls:]],
                                   dim=1)
            neg_ith = feat_q[i:i+1] @ neg_sample
            neg_list[i, :] = neg_ith
            # neg_list = torch.cat((neg_list, neg_ith), dim=0)


        sim_batch_con = feat_q @ feat_k.T
        labels_batch_con = torch.eq(labels_q[:, None], labels_k[None, :]).float()
        sim_con = torch.cat([sim_batch_con, pos_list, neg_list], dim=1)
        labels_con = torch.cat([labels_batch_con, torch.ones_like(pos_list), torch.zeros_like(neg_list)], dim=1)
        # compute classification logits
        
        logit_cls_q = self.linear(mid_feat_q)
        logit_cls_k = self.linear(mid_feat_k)

        feat_k_gather = concat_all_gather(feat_k)
        labels_k_gather = concat_all_gather(labels_k)

        if self.neg_size_per_cls != 0:
            for i in range(feat_k_gather.shape[0]):
                self._dequeue_and_enqueue(feat_k_gather[i:i + 1], labels_k_gather[i])
                           
        return sim_con, labels_con, logit_cls_q, logit_cls_k

    def _inference(self, image):
        mid_feat = self.encoder(image)
        logit_cls = self.linear(mid_feat)
        return logit_cls

  
    def forward(self, im_q, im_k=None, labels_q=None, labels_k=None):
        if self.training:
            return self._train(im_q, im_k, labels_q, labels_k)  
        else:
           return self._inference(im_q)

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
