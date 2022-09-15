import torch
from torch.utils.data import Dataset

from transformers import Seq2SeqCollator


class Seq2SeqDataset(Dataset):
    def __init__(
        self, 
        dataset, 
        collate: bool = False
        ):
        super().__init__()
        self.dataset = dataset

        if collate:
            self.collator = 
    
    def __