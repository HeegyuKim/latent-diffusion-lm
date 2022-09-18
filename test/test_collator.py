from transformers import AutoTokenizer
from datasets import load_dataset
from tokenizers.processors import TemplateProcessing
from coop.models import Optimus
from torch.utils.data import DataLoader

from src.dataset.optimus import OptimusDataset, OptimusCollator


enc_tok = AutoTokenizer.from_pretrained("bert-base-cased")
dec_tok = AutoTokenizer.from_pretrained("gpt2")
dec_tok.bos_token_id = dec_tok.eos_token_id
dec_tok.pad_token_id = dec_tok.eos_token_id
print(dec_tok.bos_token)
dataset = load_dataset("imdb", split="train")
dataset = OptimusDataset(dataset, enc_tok, dec_tok, 128)

for i in range(10):
    print(dec_tok.decode(dataset[i]["tgt_input_ids"], skip_special_tokens=False))

# model = Optimus(512, -100, dec_tok.eos_token_id, dec_tok.eos_token_id,)
# collator = OptimusCollator(enc_tok, dec_tok, model.decoder)
# dataloader = DataLoader(dataset, 4, collate_fn=collator)
# batch = next(iter(dataloader))
# print(batch)
# print(enc_tok.decode(batch["src_input_ids"][0], skip_special_tokens=False))
# print(dec_tok.decode(batch["tgt_input_ids"][0], skip_special_tokens=False))
