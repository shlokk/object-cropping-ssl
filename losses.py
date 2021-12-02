"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn
import math
import numpy as np
# Following eqn. (1) of paper https://arxiv.org/pdf/1912.06430.pdf
class MILSimCLR(nn.Module):
    def __init__(self, bag_size=1, temperature=1.0, temperature_decay=0.01):
        """
        :param num_group: number for feature descriptors/image (product of spatial dimensions before global pool)
        :param temperature: The temperature param for exp(f(x).f(y)/temperature)
        """
        super(MILSimCLR, self).__init__()
        self.bag_size = bag_size
        self.temperature = temperature

    def forward(self, features, temp=None):
        """
        :param features: [num_bags*bag_size*2,feat_dim] matrix with rows (i-1)*bag_size : i*bag_size containing features
        from the the i^th image and a similar matrix stacked below that contains features from transformed images
        :return:
        """
        # Split the feature matrix into two parts (original_features, transformed_features)
        import pdb

        org_feat, trans_feat = torch.split(features, features.shape[0]//2)
        num_instances = org_feat.shape[0]
        num_bags = num_instances//self.bag_size
        sim_mat = torch.matmul(org_feat, trans_feat.T)
        rem_diag_mat = ~torch.eye(num_instances, dtype=torch.bool) * 1.0
        sim_mat = sim_mat * rem_diag_mat.to(device="cuda:0")
        sim_mat = sim_mat[None,:,:]
        # compute group max for stability: num_instances/self.num_group sized max vector
        sim_mat_max = torch.squeeze(torch.nn.functional.max_pool2d(sim_mat, (self.bag_size, num_instances))).detach()
        # repeat and make mat: [num_instances, 1]
        sim_mat_max = sim_mat_max.repeat_interleave(self.bag_size).view((-1, 1))
        sim_mat = sim_mat - sim_mat_max
        sim_mat = sim_mat * rem_diag_mat.to(device="cuda:0")
        temp = temp if temp is not None else self.temperature
        exp_sim_mat = torch.exp(sim_mat/temp)
        #pooled mat of size [num_bags, num_bags]
        pooled_mat = torch.squeeze(torch.nn.functional.avg_pool2d(exp_sim_mat, (self.bag_size, self.bag_size)))
        positive_mask = torch.eye(num_bags)
        negative_mask = ~torch.eye(num_bags, dtype=torch.bool) * 1.0
        positive_mat = pooled_mat * positive_mask.to(device="cuda:0")
        negative_mat = pooled_mat * negative_mask.to(device="cuda:0")
        positive_vector = positive_mat.sum(dim=1)
        negative_vector = negative_mat.sum(dim=1)
        # We want to maximize log(P/(N+P))
        # Hence, loss = -log(P/(N+P)) = log((N + P)/P) = log(1 + N/P)
        pdb.set_trace()
        log_loss = torch.log(1.0 + torch.div(negative_vector,positive_vector)).mean()
        return log_loss

class MILSimCLR_moco(nn.Module):
    def __init__(self, bag_size=1, temperature=1.0, temperature_decay=0.01):
        """
        :param num_group: number for feature descriptors/image (product of spatial dimensions before global pool)
        :param temperature: The temperature param for exp(f(x).f(y)/temperature)
        """
        super(MILSimCLR_moco, self).__init__()
        self.bag_size = bag_size
        self.temperature = temperature

    def forward(self, lpos,lneg,args, temp=None):
        """
        :param features: [num_bags*bag_size*2,feat_dim] matrix with rows (i-1)*bag_size : i*bag_size containing features
        from the the i^th image and a similar matrix stacked below that contains features from transformed images
        :return:
        """
        # Split the feature matrix into two parts (original_features, transformed_features)
        # import pdb


        num_instances = lpos.shape[0]
        lpos = torch.squeeze(lpos)
        # lpos = lpos[None,None,:]
        # lpos = torch.unsqueeze(lpos,0)

        # lpos = torch.unsqueeze(lpos, 0)
        # print(self.temperature)
        # print(lpos.shape)
        exp_sim_mat = torch.exp(lpos/self.temperature)

        num_bags = num_instances // self.bag_size
        # mask = torch.eye(num_bags, dtype=torch.float32).to(device=args.gpu)
        # print(exp_sim_mat.shape)
        #pooled mat of size [num_bags, num_bags]
        # pooled_mat = torch.squeeze(torch.nn.functional.avg_pool2d(exp_sim_mat, (self.bag_size, self.bag_size)))
        # pooled_mat = torch.squeeze(torch.nn.functional.avg_pool2d(exp_sim_mat, (self.bag_size, self.bag_size)))
        # pooled_mat =  torch.squeeze(torch.nn.functional.avg_pool2d(exp_sim_mat, (self.bag_size, self.bag_size)))
        # pooled_mat = torch.squeeze(torch.nn.functional.avg_pool1d(exp_sim_mat, (self.bag_size)))
        pooled_mat = exp_sim_mat
        # print(pooled_mat.size())
        # print(mask.size())
        # pooled_mat = pooled_mat*mask
        # print("after pooled mat " + str(torch.cuda.memory_allocated()))
        # positive_mask = torch.ones_like(pooled_mat.size())
        # positive_vector = pooled_mat.sum(dim=1)
        # lneg = lneg.T
        # lneg = lneg[None,:,:]
        # lneg = torch.unsqueeze(lneg,0)
        # print(lneg.size(),self.temperature)
        exp_sim_mat_neg = torch.exp(lneg / self.temperature)
        # print(exp_sim_mat_neg.shape)
        # print(exp_sim_mat_neg.size(), temp)
        # negative_mat = torch.squeeze(torch.nn.functional.avg_pool2d(exp_sim_mat_neg, (self.bag_size, self.bag_size)))
        # negative_mat = torch.squeeze(torch.nn.functional.avg_pool1d(exp_sim_mat_neg, (self.bag_size)))
        # negative_mat = torch.squeeze(torch.nn.functional.avg_pool1d(exp_sim_mat_neg, (self.bag_size)))
        # print(pooled_mat.size(),negative_mat.size(),negative_mat.size())
        negative_mat = exp_sim_mat_neg


        # negative_mask = ~torch.eye(num_bags, dtype=torch.bool) * 1.0
        # positive_mat = pooled_mat * positive_mask.to(device="cuda:0")
        # negative_mat = pooled_mat * negative_mask.to(device="cuda:0")
        print(negative_mat)
        print(pooled_mat)
        positive_vector = pooled_mat.sum(dim=0)
        negative_vector = negative_mat.sum(dim=1)
        # print(positive_vector,negative_vector)

        # print("after")
        # print(pooled_mat.size(), negative_mat.size())
        # We want to maximize log(P/(N+P))
        # Hence, loss = -log(P/(N+P)) = log((N + P)/P) = log(1 + N/P)
        # pdb.set_trace()
        # print(torch.div(negative_vector, positive_vector))
        # print(positive_vector,negative_vector)
        print(positive_vector)
        # print(negative_vector)

        log_loss = torch.log(1.0 + torch.div(negative_vector,positive_vector)).mean()
        # print(log_loss)
        # print(log_loss)
        # import pdb
        # pdb.set_trace()
        return log_loss


class MILSimCLR_moco_v2(nn.Module):
    def __init__(self, bag_size=1, temperature=1.0, temperature_decay=0.01,max_feature=False):
        """
        :param num_group: number for feature descriptors/image (product of spatial dimensions before global pool)
        :param temperature: The temperature param for exp(f(x).f(y)/temperature)
        """
        super(MILSimCLR_moco_v2, self).__init__()
        self.bag_size = bag_size
        self.temperature = temperature
        self.max_feature = max_feature


    def forward(self, lpos,lneg,args, temp=None):
        """
        :param features: [num_bags*bag_size*2,feat_dim] matrix with rows (i-1)*bag_size : i*bag_size containing features
        from the the i^th image and a similar matrix stacked below that contains features from transformed images
        :return:
        """
        # Split the feature matrix into two parts (original_features, transformed_features)
        # import pdb

        # print(lpos.shape,lneg.shape)

        num_instances = lpos.shape[0]
        lpos = torch.squeeze(lpos)
        exp_sim_mat = torch.exp(lpos / self.temperature)
        if self.max_feature:
            # x1 = np.amax(exp_sim_mat,axis=2)
            # pooled_mat = np.amax(x1,axis=1)
            pooled_mat = torch.amax(exp_sim_mat,dim=(1, 2))
        else:
            pooled_mat = exp_sim_mat.mean(dim=2).mean(dim=1)

        # lpos = lpos[None,None,:]
        # lpos = torch.unsqueeze(lpos,0)

        # lpos = torch.unsqueeze(lpos, 0)
        # print(self.temperature)
        # print(lpos.shape)
        # exp_sim_mat = torch.exp(lpos/self.temperature)

        # num_bags = num_instances // self.bag_size
        # mask = torch.eye(num_bags, dtype=torch.float32).to(device=args.gpu)
        # print(exp_sim_mat.shape)
        #pooled mat of size [num_bags, num_bags]
        # pooled_mat = torch.squeeze(torch.nn.functional.avg_pool2d(exp_sim_mat, (self.bag_size, self.bag_size)))
        # pooled_mat = torch.squeeze(torch.nn.functional.avg_pool2d(exp_sim_mat, (self.bag_size, self.bag_size)))
        # pooled_mat =  torch.squeeze(torch.nn.functional.avg_pool2d(exp_sim_mat, (self.bag_size, self.bag_size)))
        # pooled_mat = torch.squeeze(torch.nn.functional.avg_pool1d(exp_sim_mat, (self.bag_size)))
        # print(pooled_mat.size())
        # print(mask.size())
        # pooled_mat = pooled_mat*mask
        # print("after pooled mat " + str(torch.cuda.memory_allocated()))
        # positive_mask = torch.ones_like(pooled_mat.size())
        # positive_vector = pooled_mat.sum(dim=1)
        # lneg = lneg.T
        # lneg = lneg[None,:,:]
        # lneg = torch.unsqueeze(lneg,0)
        # print(lneg.size(),self.temperature)
        exp_sim_mat_neg = torch.exp(lneg / self.temperature)
        # print(exp_sim_mat_neg.shape)
        # print(exp_sim_mat_neg.size(), temp)
        negative_mat = exp_sim_mat_neg.mean(dim=2).mean(dim=1)
        # print(negative_mat)
        # print(pooled_mat)
        # negative_mat = torch.squeeze(torch.nn.functional.avg_pool2d(exp_sim_mat_neg, (self.bag_size, self.bag_size)))
        # negative_mat = torch.squeeze(torch.nn.functional.avg_pool1d(exp_sim_mat_neg, (self.bag_size)))
        # negative_mat = torch.squeeze(torch.nn.functional.avg_pool1d(exp_sim_mat_neg, (self.bag_size)))
        # negative_mat = torch.squeeze(torch.nn.functional.avg_pool1d(exp_sim_mat_neg, (self.bag_size)))
        # print(pooled_mat.size(),negative_mat.size(),negative_mat.size())



        # negative_mask = ~torch.eye(num_bags, dtype=torch.bool) * 1.0
        # positive_mat = pooled_mat * positive_mask.to(device="cuda:0")
        # negative_mat = pooled_mat * negative_mask.to(device="cuda:0")
        # print(negative_mat.shape)
        # print(pooled_mat.shape,negative_mat.shape)

        # positive_vector = pooled_mat.sum(dim=0)
        positive_vector = pooled_mat
        negative_vector = negative_mat.sum(dim=1)
        print(positive_vector)
        print(negative_vector)
        # print(" positive_vector shape")
        # print(positive_vector.shape)
        # print(" negative_vector shape")
        # print(negative_vector.shape)
        # print(pooled_mat.shape, negative_mat.shape)
        # print(positive_vector.shape, negative_vector.shape)

        # print(positive_vector,negative_vector)

        # print("after")
        # print(pooled_mat.size(), negative_mat.size())
        # We want to maximize log(P/(N+P))
        # Hence, loss = -log(P/(N+P)) = log((N + P)/P) = log(1 + N/P)
        # pdb.set_trace()
        # print(torch.div(negative_vector, positive_vector))
        # print(positive_vector,negative_vector)
        # print(positive_vector)
        # print(negative_vector)

        log_loss = torch.log(1.0 + torch.div(negative_vector,positive_vector)).mean()
        # print(log_loss)
        # print(log_loss)
        # import pdb
        # pdb.set_trace()
        return log_loss

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
