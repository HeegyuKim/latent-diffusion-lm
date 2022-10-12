from typing import Any, Callable, List, Optional, Union
from .base import BaseTask
from omegaconf import DictConfig
from coop.models.optimus_v2 import Optimus
from transformers import RobertaConfig, GPT2Config
import pandas as pd
import torch
import torch.optim as optim
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from tokenizers import Tokenizer
from datasets import load_dataset, interleave_datasets

from omegaconf import OmegaConf

from ..dataset.optimus_v2 import OptimusDataset, OptimusIterableDataset
from tqdm import tqdm
import wandb


def switch_dict_tensor_device(d: dict, device: str):
    for k, v in d.items():
        if torch.is_tensor(v) and v.device != device:
            d[k] = v.to(device)


class OptimusTask(BaseTask):
    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)
        print(OmegaConf.to_yaml(config))
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.tokenizer)
        
        self.encoder_config = RobertaConfig.from_json_file("../../../config/model/" + config.model.encoder)
        self.decoder_config = GPT2Config.from_json_file("../../../config/model/" + config.model.decoder)
        self.model = Optimus(
            config.model.latent_dim,
            -100,
            self.tokenizer.bos_token_id,
            self.tokenizer.eos_token_id,
            self.encoder_config,
            self.decoder_config,
            config.model.free_bit
        )

    def get_train_collator(self) -> Callable:
        return DataCollatorForSeq2Seq(
            self.tokenizer, 
            self.model.decoder,
            max_length=self.config.model.max_seq_len
            )

    def get_eval_collator(self) -> Callable:
        return DataCollatorForSeq2Seq(
            self.tokenizer, 
            self.model.decoder,
            max_length=self.config.model.max_seq_len
            )

    def get_train_dataset(self) -> Dataset:
        ds = []
        for name in self.config.dataset.train:
            d = load_dataset(name, split="train", streaming=True)
            ds.append(d)

        if len(ds) == 1:
            ds = ds[0]
            return OptimusDataset(
                ds,
                self.tokenizer,
                self.config.model.max_seq_len,
                "sentence"
            )
        else:
            ds = interleave_datasets(ds)
            return OptimusIterableDataset(
                ds,
                self.tokenizer,
                self.config.model.max_seq_len,
                "sentence"
            )

    def get_eval_dataset(self) -> Dataset:
        return OptimusDataset(
            load_dataset("csv", data_files="../../../data/optimus_v2_val.csv", split="train"),
            self.tokenizer,
            self.config.model.max_seq_len,
            "sentence"
        )

    @torch.no_grad()
    def encode(self, reviews: Union[List[str], str], device: str = None):
        if isinstance(reviews, str):
            reviews = [reviews]

        if device is None:
            self.to(self.device)

        src = self.tokenizer(reviews, return_tensors="pt", padding=True, truncation=True, max_length=self.config.model.max_seq_len)
        return self.model(src).latent.loc
        
    @torch.no_grad()
    def generate(self, latents: torch.Tensor, prompts: Optional[Union[str, List[str]]], device: str = None, **kwargs):
        if prompts is not None:
            input_ids = self.tokenizer(prompts, return_tensors="pt").input_ids
        else:
            input_ids = None
        generations = self.model.generate(z=latents, input_ids=input_ids, **kwargs)
        generations = [[x for x in g if x >= 0] for g in generations]
        return self.tokenizer.batch_decode(generations)


    def step(self, batch, batch_idx) -> dict:
        src = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "token_type_ids": batch["token_type_ids"],
        }
        tgt = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["labels"],
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

        if self.config.model.get("is_vae", True):
            return out
        else:
            return out["nll"]

    def validation_step(self, batch, batch_idx) -> dict:
        loss = self.step(batch, batch_idx)
        nll, zkl, zkl_real = loss.nll, loss.zkl, loss.zkl_real

        out = {"loss": nll + zkl, "nll": nll, "zkl": zkl, "zkl_real": zkl_real}
        self.log_dict(out, prefix="eval_")
        return out

    def on_validation_epoch_end(self) -> None:
        samples = pd.read_csv("../../../data/optimus_v2_val.csv").sentence
        if samples is None:
            return 

        table = wandb.Table(columns=["source", "generated"])

        for text in tqdm(samples, "generating samples..."):
            inputs = self.tokenizer(text.strip(), return_tensors="pt", padding=False, truncation=True, max_length=self.config.model.max_seq_len)
            switch_dict_tensor_device(inputs, self.device)

            latent = self.model(src=inputs).latent.mean
            generated = self.model.generate(
                latent, max_length=self.config.model.max_seq_len, min_length=3, no_repeat_ngram_size=2, num_beams=5
            )[0]
            generated = self.tokenizer.decode(generated)
            table.add_data(text, generated)

        wandb.log({"sample": table})
