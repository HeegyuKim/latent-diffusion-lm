from typing import Any, Callable, Optional
from .base import BaseTask
from omegaconf import DictConfig
from coop.models import Optimus
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

from ..dataset.optimus import OptimusDataset, OptimusCollator
import wandb


def switch_dict_tensor_device(d: dict, device: str):
    for k, v in d.items():
        if torch.is_tensor(v) and v.device != device:
            d[k] = v.to(device)


class OptimusTask(BaseTask):
    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)
        print(OmegaConf.to_yaml(config))
        self.enc_tok = AutoTokenizer.from_pretrained("bert-base-cased")
        self.dec_tok = AutoTokenizer.from_pretrained("gpt2")
        self.dec_tok.pad_token_id = self.dec_tok.eos_token_id
        self.dec_tok.bos_token_id = self.dec_tok.eos_token_id

        self.model = Optimus(
            config.model.latent_dim,
            -100,
            self.dec_tok.eos_token_id,
            self.dec_tok.eos_token_id,
        )

    def get_train_collator(self) -> Callable:
        return OptimusCollator(self.enc_tok, self.dec_tok, self.model.decoder)

    def get_eval_collator(self) -> Callable:
        return OptimusCollator(self.enc_tok, self.dec_tok, self.model.decoder)

    def get_train_dataset(self) -> Dataset:
        return OptimusDataset(
            load_dataset(**self.config.dataset.train),
            self.enc_tok,
            self.dec_tok,
            self.config.model.max_seq_len,
        )

    def get_eval_dataset(self) -> Dataset:
        return OptimusDataset(
            load_dataset(**self.config.dataset.eval),
            self.enc_tok,
            self.dec_tok,
            self.config.model.max_seq_len,
        )

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
        self.log_dict(out, prefix="train_")
        return out

    def validation_step(self, batch, batch_idx) -> dict:
        loss = self.step(batch, batch_idx)
        nll, zkl, zkl_real = loss.nll, loss.zkl, loss.zkl_real

        out = {"loss": nll + zkl, "nll": nll, "zkl": zkl, "zkl_real": zkl_real}
        self.log_dict(out, prefix="eval_")
        return out

    def on_validation_epoch_end(self) -> None:
        texts = """
        this food is amazing!!
        this food is delicious
        this food is disgusting
        omg what a wonderful restaurant 
        """.strip().split(
            "\n"
        )
        table = wandb.Table(columns=["source", "generated"])

        for text in texts:
            inputs = self.enc_tok(text.strip(), return_tensors="pt")
            switch_dict_tensor_device(inputs, self.device)

            latent = self.model(src=inputs).latent.mean
            generated = self.model.generate(
                latent, max_tokens=64, min_length=4, no_repeat_ngram_size=2, num_beams=5
            )[0]
            generated = self.dec_tok.decode(generated)
            # print(text.strip(), "->", self.dec_tok.decode(generated))
            table.add_data(text, generated)

        wandb.log({"sample": table})
        
