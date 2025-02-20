import torch
import torch.nn as nn
import torch.nn.functional as F
from normalization import AdaptiveGroupNorm

EPS = 1e-8

def compute_entropy(p, dim=0):
    return (-torch.sum(p * torch.log(p + EPS), dim=dim)).mean()

class AdaptiveGroupNormLoss(nn.Module):
    def __init__(self, model=None, lam: float = 1e-3, target_q: float = 2.324, target_v: float = 3.309):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.model = model
        self.lam = lam
        self.target_q = target_q
        self.target_v = target_v
        
        self.adaptive_modules = []
        if self.model is not None:
            self.adaptive_modules = [m for m in self.model.modules() if isinstance(m, AdaptiveGroupNorm)]
    
    def regularize(self):
        loss = sum(
            (compute_entropy(F.softmax(m.Q, dim=0), dim=0) - self.target_q) ** 2 +
            (compute_entropy(F.softmax(m.V, dim=1), dim=1) - self.target_v) ** 2
            for m in self.adaptive_modules
        )
        return loss

    def forward(self, logits, targets):
        loss = self.ce(logits, targets)
        if self.lam > 0.0 and self.model is not None:
            loss = loss + self.lam * self.regularize()
        return loss