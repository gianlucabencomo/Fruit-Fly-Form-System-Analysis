import torch
import torch.nn as nn
from normalization import AdaptiveGroupNorm
import torch.nn.functional as F


class AdaptiveGroupNormLoss(nn.Module):
    def __init__(self, 
                 model=None, 
                 lam: float = 5e-3, 
                 target_q: float = 2.324, # weighted average of in entropy over 13 Dm cell types
                 target_v: float = 3.309 # weighted average of out entropy over 13 Dm cell types
                 ):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.model = model
        self.lam = lam
        self.target_q = target_q
        self.target_v = target_v

    def entropy(self, p, dim=0):
        H = -torch.sum(p * torch.log(p + 1e-8), dim=dim)
        return H.mean()

    def regularize(self):
        loss = 0.0
        for module in self.model.modules():
            if isinstance(module, AdaptiveGroupNorm):
                A = F.softmax(module.Q, dim=0)
                V = F.softmax(module.V, dim=1)
                loss += (self.entropy(A, dim=0) - self.target_q) ** 2
                loss += (self.entropy(V, dim=1) - self.target_v) ** 2
        return loss

    def forward(self, logits, targets):
        loss = self.ce(logits, targets)
        if self.lam > 0.0 or self.model != None:
            loss += self.lam * self.regularize()
        return loss
