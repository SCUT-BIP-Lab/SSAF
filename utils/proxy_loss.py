import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AMSoftmax(nn.Module):
    def __init__(self,
                 in_feats,
                 n_classes,
                 m=0.5,
                 s=30):
        super(AMSoftmax, self).__init__()
        self.m = m
        self.s = s
        self.in_feats = in_feats
        self.W = torch.nn.Parameter(torch.randn(in_feats, n_classes), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.W, gain=1)

    def forward(self, x, lb):
        assert x.size()[0] == lb.size()[0], f'feature number is {x.size()[0]}, but label number is {lb.size()[0]}'
        assert x.size()[1] == self.in_feats, f'feature dim is {x.size()[1]}, but loss dim is {self.in_feats}'
        x_norm = torch.div(x, torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12))
        w_norm = torch.div(self.W, torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12))
        costh = torch.mm(x_norm, w_norm)
        lb_view = lb.view(-1, 1)
        if lb_view.is_cuda: lb_view = lb_view.cpu()
        delt_costh = torch.zeros(costh.size()).scatter_(1, lb_view, self.m)
        if x.is_cuda: delt_costh = delt_costh.cuda()
        costh_m = costh - delt_costh
        costh_m_s = self.s * costh_m
        loss = self.ce(costh_m_s, lb)
        return loss, costh


class Softmax(nn.Module):
    def __init__(self,
                 in_feats,
                 n_classes,
                ):
        super(Softmax, self).__init__()
        self.in_feats = in_feats
        self.W = torch.nn.Parameter(torch.randn(in_feats, n_classes), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.W, gain=1)

    def forward(self, x, lb):
        assert x.size()[0] == lb.size()[0]
        assert x.size()[1] == self.in_feats
        logit = torch.mm(x, self.W)
        loss = self.ce(logit, lb)
        score = nn.functional.softmax(logit)
        return loss, score