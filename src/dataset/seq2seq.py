from typing import Any, Optional, Sequence, Union
from pydantic import BaseModel

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from tokenizers import Tokenizer


class BaseSeq2SeqDataset(Dataset, BaseModel):
    dataset: Any
    tokenizer: Any
    max_length: int
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    pad_token_id: int = 0
    label_for_pad_token_id: int = -100
    padding: bool = True
    padding_side: str = "right"
    truncation: bool = True
    truncation_side: str = "right"

    class Config:
        arbitrary_types_allowed = True

    def __len__(self):
        return len(self.dataset)

    def pad(self, ids: list[int], pad_value: Optional[int] = None) -> list[int]:
        if pad_value is None:
            pad_value = self.pad_token_id

        pad_len = self.max_length - len(ids)
        if pad_len > 0:
            pad_ids = [pad_value] * pad_len

            if self.padding_side == "right":
                ids = ids + pad_ids
            else:
                ids = pad_ids + ids

        return ids

    def truncate(self, ids: list[int]) -> list[int]:
        max_length = self.max_length

        if self.bos_token_id is not None:
            max_length -= 1
        if self.eos_token_id is not None:
            max_length -= 1

        if len(ids) >= max_length:
            if self.truncation_side == "left":
                ids = ids[max_length:]
            else:
                ids = ids[:max_length]

        if self.bos_token_id is not None:
            ids.insert(0, self.bos_token_id)

        if self.eos_token_id is not None:
            ids.append(self.eos_token_id)

        return ids

    def encode(self, text: str, prefix: str = "") -> list[int]:
        ids = self.tokenizer.encode(text, add_special_tokens=False)

        if self.truncation:
            ids = self.truncate(ids)

        masks = [1] * len(ids)
        labels = ids.copy()

        if self.padding:
            ids = self.pad(ids)
            masks = self.pad(masks)
            labels = self.pad(labels, self.label_for_pad_token_id)

        return {
            f"{prefix}input_ids": ids,
            f"{prefix}attention_mask": masks,
            f"{prefix}labels": labels,
        }


class Seq2SeqDataset(BaseSeq2SeqDataset):
    prefix: str = ""

    def __getitem__(self, index) -> dict:
        text = self.dataset[index]["text"]
        return self.encode(text, prefix=self.prefix)


class OptimusDataset(Dataset):
    def __init__(
        self,
        dataset: Sequence,
        encoder_tokenizer: PreTrainedTokenizer,
        decoder_tokenizer: PreTrainedTokenizer,
        max_length: int,
        encoder_bos_token_id: Optional[int] = None,
        encoder_eos_token_id: Optional[int] = None,
        encoder_pad_token_id: int = 0,
        decoder_bos_token_id: Optional[int] = None,
        decoder_eos_token_id: Optional[int] = None,
        decoder_pad_token_id: int = 0,
        label_for_pad_token_id: int = -100,
        padding: bool = True,
        padding_side: str = "right",
        truncation: bool = True,
        truncation_side: str = "right",
    ) -> None:
        super().__init__()

        self.encoder_dataset = Seq2SeqDataset(
            dataset=dataset,
            tokenizer=encoder_tokenizer,
            max_length=max_length,
            bos_token_id=encoder_bos_token_id,
            eos_token_id=encoder_eos_token_id,
            pad_token_id=encoder_pad_token_id,
            label_for_pad_token_id=label_for_pad_token_id,
            padding=padding,
            padding_side=padding_side,
            truncation=truncation,
            truncation_side=truncation_side,
            prefix="source_",
        )

        self.decoder_dataset = Seq2SeqDataset(
            dataset=dataset,
            tokenizer=decoder_tokenizer,
            max_length=max_length,
            bos_token_id=decoder_bos_token_id,
            eos_token_id=decoder_eos_token_id,
            pad_token_id=decoder_pad_token_id,
            label_for_pad_token_id=label_for_pad_token_id,
            padding=padding,
            padding_side=padding_side,
            truncation=truncation,
            truncation_side=truncation_side,
            prefix="target_",
        )

    def __len__(self) -> int:
        return len(self.encoder_dataset)

    def __getitem__(self, index) -> dict:
        source = self.encoder_dataset[index]
        target = self.decoder_dataset[index]

        return {**source, **target}
