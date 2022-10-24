from typing import Any, Callable, List, Optional, Union
from .base import BaseTask
from omegaconf import DictConfig
from coop.models.optimus_v2 import Optimus
from coop.models.mlp import BowPredictor
from coop.metric import levenshtein_batch
from coop.bow import create_bow
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
from hydra.utils import get_original_cwd, to_absolute_path

from ..dataset.optimus_v2 import OptimusDataset, OptimusIterableDataset, OptimusPLMDataset
from tqdm import tqdm
import wandb


def switch_dict_tensor_device(d: dict, device: str):
    for k, v in d.items():
        if torch.is_tensor(v) and v.device != device:
            d[k] = v.to(device)


class OptimusTask(BaseTask):
    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.tokenizer)
        
        self.encoder_config = RobertaConfig.from_json_file(to_absolute_path("config/model/" + config.model.encoder))
        self.decoder_config = GPT2Config.from_json_file(to_absolute_path("config/model/" + config.model.decoder))
        if self.config.model.get("bow_loss", True):
            self.bow_model = BowPredictor(config.model.latent_dim, self.encoder_config.vocab_size)   
        self.model = Optimus(
            config.model.latent_dim,
            self.tokenizer.pad_token_id,
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
        return OptimusPLMDataset(
            self.config.dataset.train,
            self.tokenizer,
            self.config.model.max_seq_len,
            weights=self.config.dataset.get("train_weights")
        )

    def get_eval_dataset(self) -> Dataset:
        return OptimusPLMDataset(
            self.config.dataset.test,
            self.tokenizer,
            self.config.model.max_seq_len,
            split="test"
        )

    @torch.no_grad()
    def encode(self, reviews: Union[List[str], str], device: str = None, return_distribution: bool = False):
        if isinstance(reviews, str):
            reviews = [reviews]

        if device is None:
            self.to(self.device)

        src = self.tokenizer(reviews, return_tensors="pt", padding=True, truncation=True, max_length=self.config.model.max_seq_len)
        switch_dict_tensor_device(src, self.device)

        if return_distribution:
            return self.model(src).latent
        else:
            return self.model(src).latent.loc
        
    @torch.no_grad()
    def generate(self, latents: torch.Tensor, prompts: Optional[Union[str, List[str]]] = None, device: str = None, **kwargs):
        if device is None:
            self.to(self.device)

        if prompts is not None:
            input_ids = self.tokenizer(prompts, return_tensors="pt").input_ids.to(self.device)
        else:
            input_ids = None
            
        generations = self.model.generate(z=latents, input_ids=input_ids, **kwargs)
        generations = [[x for x in g if x >= 0] for g in generations]
        return self.tokenizer.batch_decode(generations, skip_special_tokens=True)


    def step(self, batch, batch_idx) -> dict:
        out = {}
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
        model_out = self.model(src=src, tgt=tgt)
        loss = model_out.nll

        if self.config.model.get("is_vae", True):
            zkl, zkl_real = model_out.zkl, model_out.zkl_real 
            klw = self.model.klw(
                self.global_step, self.config.trainer.optimus_checkout_step
            )
            out["nll"] = loss.clone()
            out["zkl"] = zkl
            out["zkl_real"] = zkl_real
            out["klw"] = klw
            loss += klw * zkl

        if self.config.model.get("bow_loss", True):
            bow = create_bow(
                batch["input_ids"].cpu(),
                self.encoder_config.vocab_size,
                [self.tokenizer.pad_token_id]
                ).to(self.device)
            bow_pred = self.bow_model(model_out.latent.rsample())
            bow_kl = F.binary_cross_entropy(bow_pred, bow)
            out["bow_loss"] = bow_kl
            loss += bow_kl
        out["loss"] = loss
        return out

    def training_step(self, batch, batch_idx) -> dict:
        out = self.step(batch, batch_idx)
        self.log_dict(out, prefix="train_", prog_bar=True)
        return out

    def validation_step(self, batch, batch_idx) -> dict:
        out = self.step(batch, batch_idx)
        self.log_dict(out, prefix="eval_", prog_bar=True)
        
        input_sents = self.tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
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
