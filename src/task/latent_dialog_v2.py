from typing import Any, Callable, Dict, List, Optional, Union

from transformers import AutoTokenizer, GPT2Config

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
from coop.models.optimus_v2 import OptimusDecoder
from .optimus_v2 import OptimusTask, switch_dict_tensor_device

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

def constrasive_mse_loss(X, Y, lamb: float = 0.1):
    pair_mse = F.mse_loss(X, Y)

    bs = X.shape[0]
    X = X.unsqueeze(1).repeat(1,bs,1)
    mse = ((X - Y) ** 2).sum(-1).sqrt()

    # diag_mse = mse.diag().mean()
    neg_mse = mse.masked_fill(torch.eye(bs, device=X.device) == 1, 0).sum() / (bs * bs - bs)

    loss = pair_mse - lamb * neg_mse        
    return loss, pair_mse, neg_mse
    # return diag_mse, diag_mse, 0

class LatentDialogTaskV2(BaseTask):

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.model)
        self.context_model = BertVAE(config.model.model, config.model.latent_dim)
        self.response_model = BertVAE(config.model.model, config.model.latent_dim)

        dec_config = GPT2Config.from_json_file(to_absolute_path("config/model/" + config.model.decoder))
        self.decoder = OptimusDecoder(dec_config, config.model.latent_dim, self.tokenizer.pad_token_id)

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

    @torch.no_grad()
    def generate(
        self,
        z: torch.Tensor,
        input_ids: torch.Tensor=None,
        **kwargs
    ):
        bz, _ = z.size()

        if input_ids is None:
            input_ids = z.new_full((bz, 1), dtype=torch.long, fill_value=self.tokenizer.bos_token_id)

        generated = self.decoder.generate(
            input_ids,
            bos_token_id=self.tokenizer.cls_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.sep_token_id,
            past_key_values=(z,),
            latent_as_gpt_memory=True,
            latent_as_gpt_emb=True,
            **kwargs
        ).tolist()

        generated = self.tokenizer.batch_decode(generated)
        
        return generated

    def step(self, batch, generate = False):
        context = batch["context"]
        context = self.tokenizer(context, padding=False, truncation=False, add_special_tokens=False)
        context = self._truncate_and_pad(context)
        context = self.context_model(**context).latent.loc

        response = batch["response"]
        response_inputs = self.tokenizer(response, padding=False, truncation=False, add_special_tokens=False)
        response = self._truncate_and_pad(response_inputs)
        response = self.response_model(**response).latent.loc

        loss, pos_loss, cont_loss = constrasive_mse_loss(context, response)

        del response_inputs["token_type_ids"]
        res_ids = response_inputs["input_ids"]
        response_inputs["labels"] = res_ids.masked_fill(res_ids == self.tokenizer.pad_token_id, -100)
        outputs = self.decoder(
            **response_inputs,
            past_key_values=(context,),
            latent_as_gpt_memory=True,
            latent_as_gpt_emb=True,
            return_dict=True
        )
        nll_loss = outputs.loss
        loss = loss + nll_loss

        if generate:
            gen = self.generate(
                context,
                max_length=self.config.model.decoder_max_seq_len,
                min_length=3,
                repetition_penalty=2.0,
                num_beams=4
            )
            return loss, pos_loss, cont_loss, nll_loss, gen
        else:
            return loss, pos_loss, cont_loss, nll_loss, None

    def training_step(self, batch, batch_idx) -> dict:
        loss, pos_loss, cont_loss, nll_loss, _ = self.step(batch)
        out = {"loss": loss, "mse_loss": pos_loss, "cont_mse_loss": cont_loss, "nll_loss": nll_loss}
        self.log_dict(out, prefix="train_", prog_bar=True)
        return out

    def validation_step(self, batch, batch_idx) -> dict:
        batch_size = len(batch["context"])
        loss, pos_loss, cont_loss, nll_loss, gen = self.step(batch, generate=batch_idx < 5)
        out = {"loss": loss, "mse_loss": pos_loss, "cont_mse_loss": cont_loss, "nll_loss": nll_loss}
        self.log_dict(out, prefix="eval_", on_epoch=True, prog_bar=True, batch_size=batch_size)
        
        if gen is not None:
            input_sents = batch["context"]
            label_sents = batch["response"]

            return {
                "context": input_sents,
                "response": label_sents,
                "prediction": gen
            }
        else:
            return None

    def validation_epoch_end(self, validation_step_outputs) -> None:
        table = wandb.Table(columns=["context", "response", "prediction"])

        for batch in validation_step_outputs:
            if batch is not None:
                for ctx, res, loss in zip(batch["context"], batch["response"], batch["prediction"]):
                    table.add_data(ctx, res, loss)

        if wandb.run is not None:
            wandb.log({"sample": table})
        else:
            print(table)