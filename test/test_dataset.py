print("hi?")

from transformers import AutoTokenizer
from datasets import load_dataset
from src.dataset.seq2seq import OptimusDataset

# from pydantic import BaseModel

print("hi?")

enc_tok = AutoTokenizer.from_pretrained("bert-base-cased")
dec_tok = AutoTokenizer.from_pretrained("gpt2")
dataset = load_dataset("imdb", split="train")

print(type(enc_tok), type(dataset))
dataset = OptimusDataset(
    dataset,
    encoder_tokenizer=enc_tok,
    decoder_tokenizer=dec_tok,
    max_length=8,
    encoder_bos_token_id=enc_tok.cls_token_id,
    encoder_eos_token_id=enc_tok.sep_token_id,
    encoder_pad_token_id=enc_tok.pad_token_id,
    decoder_bos_token_id=dec_tok.eos_token_id,
    decoder_eos_token_id=dec_tok.eos_token_id,
    decoder_pad_token_id=dec_tok.eos_token_id,
)

for i in range(3):
    item = dataset[i]
    # print(item)
    print(enc_tok.decode(item["source_input_ids"], skip_special_tokens=False))
    print(item["source_labels"])
    # print(enc_tok.decode(item['source_labels'], skip_special_tokens=False))
    print(dec_tok.decode(item["target_input_ids"], skip_special_tokens=False))
    # print(dec_tok.decode(item['target_labels'], skip_special_tokens=False))
    print(item["target_labels"])
    print(item["source_attention_mask"])
