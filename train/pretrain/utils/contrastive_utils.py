from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import numpy as np



class HardConLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super(HardConLoss, self).__init__()
        self.temperature = temperature
        self.eps = 1e-08

    def forward(self, features_1, features_2, pairsimi):
        losses = {}

        device = (torch.device('cuda') if features_1.is_cuda else torch.device('cpu'))
        batch_size = features_1.shape[0]

        features = torch.cat([features_1, features_2], dim=0)
        mask = torch.eye(batch_size, dtype=torch.bool).to(device)
        mask = mask.repeat(2, 2)
        mask = ~mask
        
        pos = torch.exp(torch.sum(features_1*features_2, dim=-1) / self.temperature)
        pos = torch.cat([pos, pos], dim=0)
        all_sim = torch.mm(features, features.t().contiguous())
        neg = torch.exp(all_sim / self.temperature).masked_select(mask).view(2*batch_size, -1)

        pairmask = torch.cat([pairsimi, pairsimi], dim=0)
        posmask = (pairmask == 1).detach()
        posmask = posmask.type(torch.int32)

        negimp = neg.log().exp()
        Ng = (negimp*neg).sum(dim = -1) / negimp.mean(dim = -1)
        loss_pos = (-posmask * torch.log(pos / (Ng+pos))).sum() / posmask.sum()
        losses["instdisc_loss"] = loss_pos
        return losses

class iMIXConLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super(iMIXConLoss, self).__init__()
        self.temperature = temperature
        self.eps = 1e-08
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, features_1, features_2, mix_rand_list, mix_lambda):
        losses = {}

        device = (torch.device('cuda') if features_1.is_cuda else torch.device('cpu'))
        batch_size = features_1.shape[0]
    
        all_sim = torch.mm(features_1, features_2.t().contiguous())/self.temperature
        pos = torch.exp(torch.sum(features_1*features_2, dim=-1) / self.temperature)
        pos_rand = torch.exp(torch.sum(features_1*features_2[mix_rand_list], dim=-1) / self.temperature)
        
        mask = torch.eye(batch_size, dtype=torch.bool).to(device)
        mask = ~mask
        neg = torch.exp(all_sim).masked_select(mask).view(batch_size, -1) 
        negimp = neg.log().exp()
        Ng = (negimp*neg).sum(dim = -1) / negimp.mean(dim = -1)
        
        output = torch.log(pos / (Ng+pos))
        output_rand = torch.log(pos_rand / (Ng+pos))
        loss_pos = -(mix_lambda * output + (1. - mix_lambda) * output_rand).mean()
        
        losses["instdisc_loss"] = loss_pos
        return losses