print("hi?")

from transformers import AutoTokenizer
from datasets import load_dataset
# from src.dataset.seq2seq import OptimusDataset
from pydantic import BaseModel

print("hi?")

enc_tok = AutoTokenizer.from_pretrained("bert-base-cased")
dec_tok = AutoTokenizer.from_pretrained("gpt2")
dataset = load_dataset("imdb", split="train")
dataset = OptimusDataset(
    dataset,
    encoder_tokenizer = enc_tok,
    decoder_tokenizer=dec_tok,
    max_length=32,
    
    encoder_bos_token_id=enc_tok.cls_token_id,
    encoder_eos_token_id=enc_tok.sep_token_id,
    encoder_pad_token_id=enc_tok.pad_token_id,

    decoder_bos_token_id=enc_tok.eos_token_id,
    decoder_eos_token_id=enc_tok.eos_token_id,
    decoder_pad_token_id=enc_tok.eos_token_id,
)

for i in range(3):
    item = dataset[i]
    print(item)