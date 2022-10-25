from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Union
from transformers import PreTrainedTokenizer

from torch.utils.data import Dataset, IterableDataset
from transformers.utils import PaddingStrategy
from datasets import load_dataset, interleave_datasets


def normalize_weights(weights):
    if weights is None:
        return None
        
    s = sum(weights)
    return [w / s for w in weights]


@dataclass
class OptimusDataset(Dataset):
    def __init__(
        self,
        dataset,
        tokenizer,
        max_seq_len: int,
        column: str = "text"
    ) -> None:
        super().__init__()

        self.dataset = dataset
        self.max_seq_len = max_seq_len
        self.column = column
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: Any) -> dict:
        item = self.dataset[index]
        out = self.tokenizer(item[self.column], padding=False, truncation=True, max_length=self.max_seq_len)
        out["labels"] = out["input_ids"]
        return out


@dataclass
class OptimusIterableDataset(IterableDataset):
    def __init__(
        self,
        dataset,
        tokenizer,
        max_seq_len: int,
        column: str = "text"
    ) -> None:
        super().__init__()

        self.dataset = dataset
        self.max_seq_len = max_seq_len
        self.column = column
        self.tokenizer = tokenizer

    def __iter__(self) -> dict:
        for item in self.dataset:
            out = self.tokenizer(item[self.column], padding=False, truncation=True, max_length=self.max_seq_len)
            out["labels"] = out["input_ids"]
            yield out


@dataclass
class OptimusPLMDataset(IterableDataset):
    dataset_paths: List[str]
    tokenizer: PreTrainedTokenizer
    max_seq_len: int
    column: str = "sentence"
    weights: Optional[List] = None
    split: str = "train"
    streaming: bool = True

    def __iter__(self) -> dict:
        ds = []
        def filter_text(x):
            return len(x) >= 10 and len(x) <= 256 and "뉴시스" not in x and "재배포" not in x

        for name in self.dataset_paths:
            d = load_dataset(name, split=self.split, streaming=self.streaming, use_auth_token=True)
            d = d.filter(lambda x: filter_text(x["sentence"]))
            ds.append(d)

        ds = interleave_datasets(ds, probabilities=normalize_weights(self.weights))

        for item in ds:
            out = self.tokenizer(item[self.column], padding=False, truncation=True, max_length=self.max_seq_len)
            out["labels"] = out["input_ids"]
            yield out
