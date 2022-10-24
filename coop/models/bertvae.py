import torch
from torch import nn
from transformers import RobertaModel
from torch.distributions import Normal, kl_divergence
from typing import Dict, Optional, NamedTuple


class BertVAEOutput(NamedTuple):
    latent: Normal
    zkl: Optional[torch.Tensor]
    zkl_real: Optional[torch.Tensor]


class BertVAE(nn.Module):
    def __init__(
        self,
        encoder_config,
        latent_dim: int,
        free_bit: float = 0.5,
    ):  
        super().__init__()
        encoder = RobertaModel.from_pretrained(encoder_config)
        self.encoder = encoder
        self.latent_dim = latent_dim

        self.proj = nn.Linear(
            self.encoder.config.hidden_size, 2 * latent_dim, bias=False
        )

        self.free_bit = free_bit

    def forward(
        self,
        *args,
        **kwargs
    ):
        cls_vec = self.encoder(*args, **kwargs, return_dict=True).pooler_output
        mu, log_var = torch.chunk(self.proj(cls_vec), chunks=2, dim=-1)
        std = torch.exp(0.5 * log_var)
        q = Normal(mu, std)
        p = Normal(0, 1)
        zkl_real = kl_divergence(q, p)
        kl_mask = torch.gt(zkl_real, self.free_bit)
        bz = cls_vec.size(0)
        zkl = zkl_real[kl_mask].sum() / bz
        zkl_real = zkl_real.sum(dim=-1).mean()

        return BertVAEOutput(
            latent=q, zkl=zkl, zkl_real=zkl_real
        )
