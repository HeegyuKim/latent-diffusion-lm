from typing import Any
from .base import BaseTask
from ..dataset.seq2seq import OptimusDataset
from omegaconf import DictConfig
from coop.models import Optimus

import torch
import torch.optim as optim
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from transformers import AutoTokenizer
from tokenizers import Tokenizer
from datasets import load_dataset


class OptimusTask(BaseTask):
    def setup(self, config: DictConfig):
        self.enc_tok = AutoTokenizer.from_pretrained("bert-base-cased")
        self.dec_tok = AutoTokenizer.from_pretrained("gpt2")
        self.dec_tok.pad_token = self.dec_tok.eos_token
        self.model = Optimus(
            config.model.latent_dim,
            -100,
            self.dec_tok.eos_token_id,
            self.dec_tok.eos_token_id,
        )

    def get_train_dataset(self) -> Dataset:
        return OptimusDataset(
            load_dataset(self.config.dataset.train.dataset_name, split="train[:10%]"),
            self.enc_tok,
            self.dec_tok,
            max_length=self.config.model.max_seq_len,
            encoder_bos_token_id=self.enc_tok.bos_token_id,
            encoder_eos_token_id=self.enc_tok.eos_token_id,
            encoder_pad_token_id=self.enc_tok.pad_token_id,
            decoder_bos_token_id=self.dec_tok.eos_token_id,
            decoder_eos_token_id=self.dec_tok.eos_token_id,
            decoder_pad_token_id=self.dec_tok.eos_token_id,
        )

    def get_eval_dataset(self) -> Dataset:
        return OptimusDataset(
            load_dataset(self.config.dataset.train.dataset_name, split="test[:10%]"),
            self.enc_tok,
            self.dec_tok,
            max_length=self.config.model.max_seq_len,
            encoder_bos_token_id=self.enc_tok.bos_token_id,
            encoder_eos_token_id=self.enc_tok.eos_token_id,
            encoder_pad_token_id=self.enc_tok.pad_token_id,
            decoder_bos_token_id=self.dec_tok.eos_token_id,
            decoder_eos_token_id=self.dec_tok.eos_token_id,
            decoder_pad_token_id=self.dec_tok.eos_token_id,
        )

    def step(self, batch, batch_idx) -> dict:
        src = {
            "input_ids": batch["source_input_ids"],
            "attention_mask": batch["source_attention_mask"],
            "labels": batch["source_labels"],
        }
        tgt = {
            "input_ids": batch["target_input_ids"],
            "attention_mask": batch["target_attention_mask"],
            "labels": batch["target_labels"],
        }
        loss = self.model(src=src, tgt=tgt)

        nll, zkl, zkl_real = loss.nll, loss.zkl, loss.zkl_real
        klw = self.model.klw(
            self.global_step, self.config.trainer.optimus_checkout_step
        )
        loss = nll + klw * zkl

        return {"loss": loss, "nll": nll, "zkl": zkl, "zkl_real": zkl_real}

    def training_step(self, batch, batch_idx) -> dict:
        return self.step(batch, batch_idx)

    def validation_step(self, batch, batch_idx) -> dict:
        return self.step(batch, batch_idx)

    # def on_validation_epoch_end(self) -> None:
    #     return super().on_validation_epoch_end()
