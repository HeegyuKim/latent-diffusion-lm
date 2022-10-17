from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union
from transformers import DataCollatorForSeq2Seq, DataCollatorWithPadding

from torch.utils.data import Dataset, IterableDataset
from transformers.utils import PaddingStrategy

