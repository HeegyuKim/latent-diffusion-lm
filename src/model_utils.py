
from dataclasses import dataclass

import torch


@dataclass
class WeightClipper:
    min_value: float = -1.0
    max_value: float = 1.0

    def __call__(self, module, param):
        if hasattr(module, param):
            self.clip(module, param)
        
    def clip(self, module, param):
        p = getattr(module, param).data
        p = p.clamp(self.min_value, self.max_value)
        getattr(module, param).data = p


@torch.no_grad()
def apply_weight_clipping(model, min_value, max_value):
    for param in model.parameters():
        if param.requires_grad:
            param.clamp_(min_value, max_value)