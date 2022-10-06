from src.task.optimus import OptimusTask
from faiss import IndexFlatL2, write_index
from datasets import load_dataset, load_from_disk

from tqdm import tqdm
import numpy as np


task = OptimusTask.load_from_checkpoint("checkpoint/optimus-aihub_conv-396k.ckpt", map_location='cuda:0')

dataset = load_dataset("json", data_files="data/dialog/aihub_daily_conv_train.jsonl", split="train")
dataset = dataset.filter(lambda x: len(x['text']) > 0)

faiss = IndexFlatL2(task.config.model.latent_dim)


for i in tqdm(range(len(dataset))):
    text = dataset[i]['text']
    latent = task.encode(text).numpy()
    faiss.add(latent)
    
write_index(faiss, "ckpt396-512d.faiss")