import torch
import torch.nn as nn
from normalization import AdaptiveGroupNorm
import torch.nn.functional as F

class AdaptiveGroupNormLoss(nn.Module):
    def __init__(self, model = None, lam: float = 5e-3):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.model = model
        self.lam = lam

    def entropy(self, p):
        H = -torch.sum(p * torch.log(p + 1e-8), dim=0)
        return H.mean()

    def regularize_Q(self):
        loss = 0.0
        for module in self.model.modules():
            if isinstance(module, AdaptiveGroupNorm):
                A = F.softmax(module.Q, dim=0)
                loss += self.entropy(A)
        return loss

    def forward(self, logits, targets):
        loss = self.ce(logits, targets) 
        if self.lam > 0.0 or self.model == None:
            loss += self.lam * self.regularize_Q()
        return loss