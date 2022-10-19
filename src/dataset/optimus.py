from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union, List
from datasets import load_dataset, interleave_datasets
from transformers import DataCollatorForSeq2Seq, DataCollatorWithPadding

import torch
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
from tokenizers import Tokenizer


def normalize_weights(weights):
    if weights is None:
        return None
        
    s = sum(weights)
    return [w / s for w in weights]


def map_to_tokenizer(
    dataset, column, tokenizer, max_seq_len: int, add_bos_eos_token: bool = False
):
    def map_fn(x):
        if add_bos_eos_token:
            text = [tokenizer.bos_token + t + tokenizer.eos_token for t in x[column]]
        else:
            text = x[column]

        return tokenizer(text, padding=False, truncation=True, max_length=max_seq_len)

    return dataset.map(map_fn, batched=True)

@dataclass
class OptimusPLMDataset(IterableDataset):
    dataset_paths: List[str]
    encoder_tokenizer: PreTrainedTokenizer
    decoder_tokenizer: PreTrainedTokenizer
    max_seq_len: int
    column: str = "sentence"
    weights: Optional[List] = None
    split: str = "train"
    decoder_special_token: str = "<|endoftext|>"

    def __iter__(self) -> dict:
        ds = []
        def filter_text(x):
            return len(x) >= 10 and len(x) <= 256 and "뉴시스" not in x and "재배포" not in x

        for name in self.dataset_paths:
            d = load_dataset(name, split=self.split, streaming=True, use_auth_token=True)
            d = d.filter(lambda x: filter_text(x["sentence"]))
            ds.append(d)

        ds = interleave_datasets(ds, probabilities=normalize_weights(self.weights))

        for item in ds:
            text = item[self.column]
            dec_text = self.decoder_special_token + text + self.decoder_special_token
            enc = self.encoder_tokenizer(text, padding=False, truncation=True, max_length=self.max_seq_len)
            dec = self.decoder_tokenizer(dec_text, padding=False, truncation=True, max_length=self.max_seq_len)
            
            out = {}
            for prefix, item in zip(["src_", "tgt_"], [enc, dec]):
                for key in ["input_ids", "attention_mask", "labels", "token_type_ids"]:
                    if key in item:
                        out[f"{prefix}{key}"] = item[key]

            yield out


@dataclass
class OptimusDataset(Dataset):
    def __init__(
        self,
        dataset,
        encoder_tokenizer,
        decoder_tokenizer,
        max_seq_len: int,
        column: str = "text",
        decoder_add_bos_token: bool = True,
    ) -> None:
        super().__init__()

        self.dataset = dataset
        self.enc_data = map_to_tokenizer(
            dataset, column, encoder_tokenizer, max_seq_len, False
        )
        self.dec_data = map_to_tokenizer(
            dataset, column, decoder_tokenizer, max_seq_len, decoder_add_bos_token
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: Any) -> dict:
        enc = self.enc_data[index]
        dec = self.dec_data[index]
        out = {}

        for prefix, item in zip(["src_", "tgt_"], [enc, dec]):
            for key in ["input_ids", "attention_mask", "labels", "token_type_ids"]:
                if key in item:
                    out[f"{prefix}{key}"] = item[key]

        return out


@dataclass
class OptimusCollator:
    def __init__(
        self,
        encoder_tokenizer,
        decoder_tokenizer,
        decoder_model=None,
        padding: Union[bool, str, PaddingStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        label_pad_token_id: Optional[int] = -100,
        return_tensors: str = "pt",
    ) -> None:
        self.enc_collator = DataCollatorWithPadding(
            encoder_tokenizer,
            padding=padding,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
        )
        self.dec_collator = DataCollatorForSeq2Seq(
            decoder_tokenizer,
            decoder_model,
            padding=padding,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            label_pad_token_id=label_pad_token_id,
            return_tensors=return_tensors,
        )

    def __call__(self, features: dict, return_tensors=None) -> dict:
        src_features = []
        tgt_features = []

        for feature in features:
            src = {}
            tgt = {}
            for k, v in feature.items():
                if k.startswith("src_"):
                    src[k[4:]] = v
                elif k.startswith("tgt_"):
                    tgt[k[4:]] = v
            tgt["labels"] = feature["tgt_input_ids"]
            src_features.append(src)
            tgt_features.append(tgt)

        enc_out = self.enc_collator(src_features)
        dec_out = self.dec_collator(tgt_features)

        out = {}

        for prefix, item in zip(["src_", "tgt_"], [enc_out, dec_out]):
            for k, v in item.items():
                out[f"{prefix}{k}"] = v

        return out
