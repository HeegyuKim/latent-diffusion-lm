import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence
from typing import Any, Callable, List, Optional, Union
from .sgpt import SentenceGPTOutput


class BowPredictor(torch.nn.Module):

    def __init__(self,
                 latent_dim: int,
                 vocab_size: int
                 ) -> None:
        super().__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, vocab_size),
            torch.nn.Sigmoid(),
        )

    def forward(self, X):
        return self.model(X)



class VAEMLP(torch.nn.Module):

    def __init__(self,
                 latent_dim: int,
                 free_bit: float = 0.5
                 ) -> None:
        super().__init__()

        hidden_dim = 4 * latent_dim
        self.model = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, latent_dim),
            # torch.nn.GELU(),
            # torch.nn.Linear(hidden_dim, hidden_dim),
            # torch.nn.GELU(),
            # torch.nn.Linear(hidden_dim, hidden_dim),
            # torch.nn.GELU(),
            # torch.nn.Linear(hidden_dim, hidden_dim),
            # torch.nn.GELU(),
            # torch.nn.Linear(hidden_dim, hidden_dim),
            # torch.nn.GELU(),
            # torch.nn.Linear(hidden_dim, hidden_dim),
            # torch.nn.GELU(),
            # torch.nn.Linear(hidden_dim, hidden_dim),
            # torch.nn.GELU(),
            # torch.nn.Linear(hidden_dim, hidden_dim),
            # torch.nn.GELU(),
            # torch.nn.Linear(hidden_dim, hidden_dim),
            # torch.nn.GELU(),
        )
        self.proj = nn.Linear(
            hidden_dim, 
            2 * latent_dim, 
            bias=False
        )
        self.free_bit = free_bit

    def reparameterize(self, x):
        mu, log_var = torch.chunk(self.proj(x), chunks=2, dim=-1)
        std = torch.exp(0.5 * log_var)
        return Normal(mu, std)

    def kldiv_loss(self, p: Normal, q: Normal, attention_mask: Optional[torch.Tensor] = None, use_free_bit: bool = True):
        zkl_real = kl_divergence(p, q).mean(-1)

        if attention_mask is not None:
            attn_sum = attention_mask.sum()
            zkl_real = zkl_real.masked_fill(attention_mask == 0, 0.0)
        else:
            attn_sum = p.loc.shape[0] * p.loc.shape[1]

        if use_free_bit:
            kl_mask = torch.gt(zkl_real, self.free_bit)
            zkl = zkl_real[kl_mask].sum() / attn_sum
            zkl_real = zkl_real.sum() / attn_sum

            return zkl, zkl_real
        else:
            zkl_real = zkl_real.sum() / attn_sum
            return zkl_real


    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        compute_kldiv_loss: bool = False
    ) -> SentenceGPTOutput:
        hidden = self.model(inputs_embeds)
        # p = self.reparameterize(hidden)
        if compute_kldiv_loss:
            zkl, zkl_real = self.kldiv_loss(p, Normal(0, 1), attention_mask)
        else:
            zkl, zkl_real = None, None

        return SentenceGPTOutput(
            latent=hidden, zkl=zkl, zkl_real=zkl_real
        )
