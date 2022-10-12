from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union
from transformers import DataCollatorForSeq2Seq, DataCollatorWithPadding

from torch.utils.data import Dataset, IterableDataset
from transformers.utils import PaddingStrategy

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

