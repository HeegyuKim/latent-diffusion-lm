from typing import Any, Callable, List, Optional, Union

from ..model_utils import apply_weight_clipping
from .base import BaseTask
from omegaconf import DictConfig
from coop.models.optimus_v2 import Optimus
import pandas as pd
import torch
import torch.optim as optim
from torch.distributions import Normal, kl_divergence
from torch import kl_div, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from transformers import GPT2Model, GPT2Config, RobertaConfig, AutoTokenizer
from coop.models.sgpt import SentenceGPT
from coop.models.mlp import VAEMLP
from tokenizers import Tokenizer
from datasets import load_dataset

from hydra.utils import get_original_cwd, to_absolute_path

from ..dataset.sgpt import SGPTDataset
from .optimus_v2 import OptimusTask
from tqdm import tqdm
from collections import defaultdict
import wandb


def switch_dict_tensor_device(d: dict, device: str):
    for k, v in d.items():
        if torch.is_tensor(v) and v.device != device:
            d[k] = v.to(device)

def freeze_model(model):
    for p in model.parameters():
        p.requires_grad = False


def stack_normals(normals):
    locs = [x.loc for x in normals]
    scales = [x.scale for x in normals]
    return Normal(
        torch.stack(locs),
        torch.stack(scales),
    )

def pad_sentence_latents(sents: Normal, max_seq_len: int, epsilon=1e-6):
    """
        sents: [s, dim]

        out: [max_s, dim]
    """
    pad_size = max_seq_len - sents.loc.shape[0]
    if pad_size == 0:
        return sents
    elif pad_size < 0:
        return Normal(
            sents.loc[:max_seq_len],
            sents.scale[:max_seq_len]
        )
    else:
        pad = torch.zeros(
            (pad_size, sents.loc.shape[1]),
            device = sents.loc.device,
            dtype=sents.loc.dtype
        )
        pad = Normal(pad, pad + epsilon)

        return Normal(
            torch.cat([sents.loc, pad.loc], dim=0),
            torch.cat([sents.scale, pad.scale], dim=0)
        )

class NoCollator():
    def __call__(self, x: Any) -> Any:
        return x
        

class SGPTTaskV2(BaseTask):

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)
        gpt2_config = GPT2Config.from_json_file(to_absolute_path("config/model/" + config.model.model))
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.tokenizer)
        self.encoder_config = RobertaConfig.from_json_file(to_absolute_path("config/model/" + config.model.encoder))
        self.decoder_config = GPT2Config.from_json_file(to_absolute_path("config/model/" + config.model.decoder))
        self.autoencoder = Optimus(
            config.model.latent_dim,
            self.tokenizer.pad_token_id,
            self.tokenizer.bos_token_id,
            self.tokenizer.eos_token_id,
            self.encoder_config,
            self.decoder_config,
            config.model.free_bit
        )

        # freeze_model(self.autoencoder)
        self.sgpt = SentenceGPT(
            gpt2_config,
            config.model.latent_dim,
            config.model.free_bit
            )
        # self.model = VAEMLP(
        #     config.model.latent_dim,
        #     config.model.free_bit
        # )

    
    def get_train_collator(self) -> Callable:
        return NoCollator()

    def get_eval_collator(self) -> Callable:
        return NoCollator()

    def get_train_dataset(self) -> Dataset:
        return SGPTDataset(
            # self.autoencoder,
            self.config.dataset.train,
            self.config.model.max_seq_len,
            column="utterances",
            weights=self.config.dataset.get("train_weights")
        )

    def get_eval_dataset(self) -> Dataset:
        return SGPTDataset(
            # self.autoencoder,
            self.config.dataset.test,
            self.config.model.max_seq_len,
            column="utterances",
            split="train"
        )

    def _item_for_train(self, sents):
        if len(sents) > self.config.model.max_sentence_len:
            sents = sents[:self.config.model.max_sentence_len]

        inputs = self.tokenizer(sents, padding="max_length", truncation=True, max_length=self.config.model.max_seq_len)
        sents = self.model.encoder(**inputs).pooler_output

        inputs = Normal(
            sents.loc[:-1],
            sents.scale[:-1]
        )
        labels = Normal(
            sents.loc[1:],
            sents.scale[1:]
        )
        # labels = Normal(
        #     sents.loc[:-1],
        #     sents.scale[:-1]
        # )
        return {
            "inputs": inputs,
            "attention_mask": torch.LongTensor(attention_mask).to(self.device),
            "labels": labels
        }, out_sents


    def _compute_nll_loss(self, output, sents, masks, sample=False):
        if sample:
            latent = output.latent.rsample()
        else:
            latent = output.latent.loc

        latent = latent.view(-1, output.latent.loc.shape[2]) # [batch * seq_len, latent_dim]
        latent = latent[masks == 1]
        inputs = self.autoencoder.tokenizer(
            sents, 
            padding=True, 
            truncation=True, 
            max_length=self.autoencoder.config.model.max_seq_len,
            return_tensors="pt"
            )
        switch_dict_tensor_device(inputs, self.device)

        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]
        inputs["labels"] = inputs["input_ids"]

        return self.autoencoder.model.decoder(
            **inputs,
            past_key_values=(latent,),
            latent_as_gpt_memory=True,
            latent_as_gpt_emb=True,
            return_dict=True
        ).loss

    def step(self, batch, batch_idx, sample=False) -> dict:
        new_batch = defaultdict(list)
        sents = []
        
        for b in batch:
            item, s = self._item_for_train(b["utterances"])
            sents.extend(s)
            for k, v in item.items():
                new_batch[k].append(v)
        
        batch = {
            "inputs": stack_normals(new_batch["inputs"]),
            "labels": stack_normals(new_batch["labels"]),
            "attention_mask": torch.stack(new_batch["attention_mask"])
        }

        if sample:
            inputs = batch["inputs"].rsample()
        else:
            inputs = batch["inputs"].loc

        all_masks = torch.cat(new_batch["attention_mask"])

        output = self.model(
            inputs_embeds=inputs,
            attention_mask=batch["attention_mask"],
            compute_kldiv_loss=True
            )

        loss = self._compute_kldiv_loss(output, batch)
        # nll_loss = self._compute_nll_loss(output, sents, all_masks, sample)
        # loss = nll_loss / 100 + kldiv_loss
        # loss = self._compute_mse_loss(output, batch)
        return loss, output

    def training_step(self, batch, batch_idx) -> dict:
        loss, output = self.step(batch, batch_idx, False)

        out = {"loss": loss, "zkl": output.zkl, "zkl_real": output.zkl_real}
        self.log_dict(out, prefix="train_", prog_bar=True)
        
        return out

    def validation_step(self, batch, batch_idx) -> dict:
        loss, output = self.step(batch, batch_idx)

        out = {"loss": loss, "zkl": output.zkl, "zkl_real": output.zkl_real}
        self.log_dict(out, prefix="val_", on_epoch=True, batch_size=len(batch))

        if batch_idx == 0:
            table = wandb.Table(["dialog", "index", "input", "output"])
            for i, item in enumerate(batch):
                sents = item["utterances"]
                seq_len = min(self.config.model.max_seq_len, len(sents))

                latents = output.latent.loc[i, :seq_len, :]
                nexts = self.autoencoder.generate(latents, max_length=64, min_length=3)

                if i == 0:
                    print(sents)
                    print(nexts)

                for j in range(seq_len):
                    table.add_data(i, j, sents[j], nexts[j])
            
            if wandb.run is not None:
                wandb.log({"eval_samples": table})