from dataclasses import dataclass
from typing import Dict
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer
from tokenizers import Tokenizer
from datasets import load_dataset

from coop.models import Optimus
from src.base import BaseTrainer, BaseConfig, switch_dict_tensor_device


@dataclass
class VAEConfig(BaseConfig):
    latent_dim: int = 512
    label_for_pad_id: int = -100
    max_seq_len: int = 128
    dataset_name: str = "yelp_polarity"


class VAETrainer(BaseTrainer):


    def setup(self, config: VAEConfig):
        self.enc_tok = AutoTokenizer.from_pretrained("bert-base-cased")
        self.dec_tok = AutoTokenizer.from_pretrained("gpt2")
        self.dec_tok.pad_token = self.dec_tok.eos_token
        self.train_dataset = load_dataset(config.dataset_name, split="train[:10%]")
        self.eval_dataset = load_dataset(config.dataset_name, split="test[:1%]")

        self.vae = Optimus(
            self.config.latent_dim,
            self.config.label_for_pad_id,
            50256,
            50256
        ).to(self.config.device)
        

    def get_optimizer(self):
        return torch.optim.Adam(self.vae.parameters(), self.config.learning_rate)

    def train_step(self, batch: Dict) -> Dict:
        texts = batch["text"]
        labels = batch["label"]

        src = self.enc_tok(texts, add_special_tokens=True, padding="max_length", truncation=True, max_length=self.config.max_seq_len, return_tensors="pt")
        tgt = self.dec_tok(texts, add_special_tokens=True, padding="max_length", truncation=True, max_length=self.config.max_seq_len, return_tensors="pt")
        tgt['labels'] = tgt["input_ids"]

        switch_dict_tensor_device(src, self.config.device)
        switch_dict_tensor_device(tgt, self.config.device)

        losses = self.vae(src=src, tgt=tgt)
        # losses = self.model(**batch)
        # nll, zkl, zkl_real = losses.nll, losses.zkl, losses.zkl_real
        # klw = self.vae.klw(self.step, self.checkout_step)
        # loss = nll + klw * zkl
        return losses.nll


if __name__ == "__main__":
    VAETrainer(VAEConfig()).train()