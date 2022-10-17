from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Union
from transformers import PreTrainedTokenizer

from torch.utils.data import Dataset, IterableDataset
from transformers.utils import PaddingStrategy
from datasets import load_dataset, interleave_datasets
import torch


def normalize_weights(weights):
    if weights is None:
        return None
        
    s = sum(weights)
    return [w / s for w in weights]

def pad_sentence_latents(sents: torch.Tensor, max_seq_len: int):
    """
        sents: [s, dim]

        out: [max_s, dim]
    """
    pad_size = max_seq_len - sents.shape[0]
    if pad_size == 0:
        return sents
    elif pad_size < 0:
        return sents[:max_seq_len]
    else:
        pad = torch.zeros(
            (pad_size, sents.shape[1]),
            device = sents.device,
            dtype=sents.dtype
        )
        return torch.cat([sents, pad], dim=0)

@dataclass
class SGPTDataset(Dataset):
    encoder: Any
    dataset_paths: List[str]
    max_seq_len: int
    column: str = "sentences"
    weights: Optional[List] = None
    split: str = "train"
    dataset = None

    def _get_dataset(self):
        if self.dataset is None:
            ds = []
            for name in self.dataset_paths:
                d = load_dataset(name, split=self.split, use_auth_token=True)
                ds.append(d)
    
            ds = interleave_datasets(ds, probabilities=normalize_weights(self.weights))
            self.dataset = ds

        return self.dataset

    def __len__(self) -> int:
        return len(self._get_dataset())

    def __getitem__(self, index) -> dict:
        dataset = self._get_dataset()

        item = dataset[index]
        sents = self.encoder.encode(item[self.column])
        mask_len = min(len(sents), self.max_seq_len)
        attention_mask = [1] * mask_len + [0] * (self.max_seq_len - mask_len)

        sents = pad_sentence_latents(sents, self.max_seq_len + 1)
        
        return {
            "inputs": sents[:-1],
            "attention_mask": torch.LongTensor(attention_mask),
            "labels": sents[1:]
        }

@dataclass
class SGPTIterableDataset(IterableDataset):
    encoder: Any
    dataset_paths: List[str]
    max_seq_len: int
    column: str = "sentences"
    weights: Optional[List] = None
    split: str = "train"

    def __iter__(self) -> dict:
        ds = []

        for name in self.dataset_paths:
            d = load_dataset(name, split=self.split, streaming=True, use_auth_token=True)
            ds.append(d)

        ds = interleave_datasets(ds, probabilities=normalize_weights(self.weights))

        for item in ds:
            sents = self.encoder.encode(item[self.column])
            mask_len = min(len(sents), self.max_seq_len)
            attention_mask = [1] * mask_len + [0] * (self.max_seq_len - mask_len)

            sents = pad_sentence_latents(sents, self.max_seq_len + 1)
            
            yield {
                "inputs": sents[:-1],
                "attention_mask": torch.LongTensor(attention_mask),
                "labels": sents[1:]
            }