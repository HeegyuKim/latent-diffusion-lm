from typing import Any, Callable, Dict, List, Optional, Union

from transformers import AutoTokenizer

from .base import BaseTask
from omegaconf import DictConfig
import pandas as pd
import torch
import torch.optim as optim
from torch.distributions import Normal, kl_divergence
from torch import kl_div, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from tokenizers import Tokenizer

from hydra.utils import get_original_cwd, to_absolute_path
from coop.metric import levenshtein_batch

from ..dataset.latent_dialog import LatentDialogDataset
from ..dataset import dataset_utils
from .. import model_utils
from coop.models.bertvae import BertVAE
from .optimus_v2 import OptimusTask

from tqdm import tqdm
from collections import defaultdict
import wandb


class ListCollator():
    def __call__(self, x: List[Dict]) -> Dict:
        out = defaultdict(list)
        for item in x:
            for k, v in item.items():
                out[k].append(v)

        return out
        

class LatentDialogTask(BaseTask):

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.model)
        self.model = BertVAE(config.model.model, config.model.latent_dim)
        self.autoencoder = OptimusTask.load_from_checkpoint(to_absolute_path(config.model.autoencoder))

        model_utils.freeze_model(self.autoencoder)
    
    def get_train_collator(self) -> Callable:
        return ListCollator()

    def get_eval_collator(self) -> Callable:
        return ListCollator()

    def get_train_dataset(self) -> Dataset:
        return LatentDialogDataset(
            self.config.dataset.train,
            weights=self.config.dataset.get("train_weights")
        )

    def get_eval_dataset(self) -> Dataset:
        return LatentDialogDataset(
            self.config.dataset.test,
            split="train"
        )

    def encode_text(self, texts: Union[str, List[str]]):
        pass

    def _truncate_and_pad(self, inputs):
        pad_token_id = self.tokenizer.pad_token_id
        pad_len = min(2 + dataset_utils.get_longest_length(inputs["input_ids"]), self.config.model.max_seq_len)
        pad_len = dataset_utils.pad_to_multiple_of(pad_len, 8)

        for k, values in inputs.items():
            new_values = []
            for v in values:
                if k == "input_ids":
                    nv = dataset_utils.truncate(
                        v, 
                        max_len=pad_len - 2, 
                        truncation_side='left',
                        prefix_value=self.tokenizer.cls_token_id,
                        postfix_value=self.tokenizer.sep_token_id
                        )
                    nv = dataset_utils.pad(
                        nv, 
                        max_len=pad_len, 
                        padding_value=self.tokenizer.pad_token_id
                    )
                else: # token_type_ids, attention_mask
                    nv = dataset_utils.truncate(
                        v, 
                        max_len=pad_len,
                        truncation_side='left'
                        )
                    nv = dataset_utils.pad(
                        nv, 
                        max_len=pad_len, 
                        padding_value=0
                    )
                new_values.append(nv)
                
            inputs[k] = torch.LongTensor(new_values).to(self.device)

        return inputs

    def step(self, batch):
        context = self.tokenizer(batch["context"], padding=False, truncation=False, add_special_tokens=False)
        context = self._truncate_and_pad(context)

        generated = self.model(**context).latent
        response = self.autoencoder.encode(batch["response"], return_distribution=True)

        loss = kl_divergence(generated, response).mean()
        return loss, generated

    def training_step(self, batch, batch_idx) -> dict:
        loss, _ = self.step(batch)
        out = {"loss": loss}
        self.log_dict(out, prefix="train_", prog_bar=True)
        return out

    def validation_step(self, batch, batch_idx) -> dict:
        batch_size = len(batch["context"])
        loss, response = self.step(batch)
        out = {"loss": loss}
        self.log_dict(out, prefix="eval_", on_epoch=True, prog_bar=True, batch_size=batch_size)
        
        if batch_idx < 1:
            input_sents = batch["context"]
            label_sents = batch["response"]
            output_sents = self.autoencoder.generate(response.loc, min_length=3, num_beams=4, max_length=64)
            lev_dist_mean = levenshtein_batch(label_sents, output_sents)
            self.log("eval_levenshtein_dist", lev_dist_mean, on_epoch=True, prog_bar=True, batch_size=batch_size)

            return {
                "context": input_sents,
                "response": label_sents,
                "generated": output_sents
            }
        else:
            return None

    def validation_epoch_end(self, validation_step_outputs) -> None:
        table = wandb.Table(columns=["context", "response", "generated"])

        for batch in validation_step_outputs:
            if batch is not None:
                for ctx, res, gen in zip(batch["context"], batch["response"], batch["generated"]):
                    table.add_data(ctx, res, gen)

        if wandb.run is not None:
            wandb.log({"sample": table})
        else:
            print(table)