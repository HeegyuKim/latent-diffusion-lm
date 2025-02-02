from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union, List
from datasets import load_dataset, interleave_datasets
from transformers import DataCollatorForSeq2Seq, DataCollatorWithPadding

import torch
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
from tokenizers import Tokenizer

from .dataset_utils import normalize_weights

@dataclass
class LatentDialogDataset(Dataset):
    dataset_paths: List[str]
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
        return item

