from typing import Any, Callable, List, Optional, Union
from .base import BaseTask
from omegaconf import DictConfig
from coop.models import Optimus
from coop.metric import levenshtein_batch
import pandas as pd
import torch
import torch.optim as optim
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from transformers import AutoTokenizer
from tokenizers import Tokenizer
from datasets import load_dataset

from omegaconf import OmegaConf

from ..dataset.optimus import OptimusPLMDataset, OptimusCollator
from tqdm import tqdm
import wandb


def switch_dict_tensor_device(d: dict, device: str):
    for k, v in d.items():
        if torch.is_tensor(v) and v.device != device:
            d[k] = v.to(device)


class OptimusTask(BaseTask):
    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)
        self.enc_tok = AutoTokenizer.from_pretrained(config.model.encoder)
        self.dec_tok = AutoTokenizer.from_pretrained(config.model.decoder)

        self.dec_tok.add_special_tokens(config.model.get("decoder_special_tokens", {}))

        self.model = Optimus(
            config.model.latent_dim,
            self.dec_tok.pad_token_id,
            self.dec_tok.bos_token_id,
            self.dec_tok.eos_token_id,
            config.model.encoder,
            config.model.decoder,
            config.model.free_bit
        )

    def get_train_collator(self) -> Callable:
        return OptimusCollator(self.enc_tok, self.dec_tok, self.model.decoder, max_length=self.config.model.max_seq_len)

    def get_eval_collator(self) -> Callable:
        return OptimusCollator(self.enc_tok, self.dec_tok, self.model.decoder, max_length=self.config.model.max_seq_len)

    def get_train_dataset(self) -> Dataset:
        return OptimusPLMDataset(
            self.config.dataset.train,
            self.enc_tok,
            self.dec_tok,
            self.config.model.max_seq_len,
            weights=self.config.dataset.get("train_weights"),
            decoder_bos_token=self.dec_tok.bos_token,
            decoder_eos_token=self.dec_tok.eos_token
        )

    def get_eval_dataset(self) -> Dataset:
        return OptimusPLMDataset(
            self.config.dataset.test,
            self.enc_tok,
            self.dec_tok,
            self.config.model.max_seq_len,
            split="test",
            decoder_bos_token=self.dec_tok.bos_token,
            decoder_eos_token=self.dec_tok.eos_token
        )

    @torch.no_grad()
    def encode(self, reviews: Union[List[str], str], device: str = None):
        if isinstance(reviews, str):
            reviews = [reviews]

        if device is None:
            self.to(self.device)

        src = self.enc_tok(reviews, return_tensors="pt", padding=True, truncation=True, max_length=self.config.model.max_seq_len)
        switch_dict_tensor_device(src, self.device)
        return self.model(src).latent.loc
        
    @torch.no_grad()
    def generate(self, latents: torch.Tensor, prompts: Optional[Union[str, List[str]]] = None, device: str = None, **kwargs):
        if device is None:
            self.to(self.device)

        if prompts is not None:
            input_ids = self.dec_tok(prompts, return_tensors="pt").input_ids.to(self.device)
        else:
            input_ids = None
            
        generations = self.model.generate(z=latents, input_ids=input_ids, **kwargs)
        generations = [[x for x in g if x >= 0] for g in generations]
        return self.dec_tok.batch_decode(generations, skip_special_tokens=True)


    def step(self, batch, batch_idx) -> dict:
        src = {
            "input_ids": batch["src_input_ids"],
            "attention_mask": batch["src_attention_mask"],
            "token_type_ids": batch["src_token_type_ids"],
        }
        tgt = {
            "input_ids": batch["tgt_input_ids"],
            "attention_mask": batch["tgt_attention_mask"],
            "labels": batch["tgt_labels"],
        }
        return self.model(src=src, tgt=tgt)

    def training_step(self, batch, batch_idx) -> dict:
        loss = self.step(batch, batch_idx)

        nll, zkl, zkl_real = loss.nll, loss.zkl, loss.zkl_real
        klw = self.model.klw(
            self.global_step, self.config.trainer.optimus_checkout_step
        )
        loss = nll + klw * zkl
    
        out = {"loss": loss, "nll": nll, "zkl": zkl, "zkl_real": zkl_real, "klw": klw}
        self.log_dict(out, prefix="train_", prog_bar=True)

        if self.config.model.get("is_vae", True):
            return out
        else:
            return out["nll"]

    def validation_step(self, batch, batch_idx) -> dict:
        loss = self.step(batch, batch_idx)
        nll, zkl, zkl_real = loss.nll, loss.zkl, loss.zkl_real

        out = {"loss": nll + zkl, "nll": nll, "zkl": zkl, "zkl_real": zkl_real}
        self.log_dict(out, prefix="eval_", on_epoch=True)
        
        input_sents = self.enc_tok.batch_decode(batch["src_input_ids"], skip_special_tokens=True)
        input_latents = self.encode(input_sents)
        output_sents = self.generate(input_latents, min_length=3, num_beams=4, max_length=self.config.model.max_seq_len)
        lev_dist_mean = levenshtein_batch(input_sents, output_sents)
        self.log("eval_levenshtein_dist", lev_dist_mean, on_epoch=True)

        return {
            "source": input_sents,
            "generated": output_sents
        }

    def validation_epoch_end(self, validation_step_outputs) -> None:
        table = wandb.Table(columns=["source", "generated"])

        for batch in validation_step_outputs:
            for src, gen in zip(batch["source"], batch["generated"]):
                table.add_data(src, gen)

        wandb.log({"sample": table})