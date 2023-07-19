import argparse

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

from train import gen_dataset

def main():
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("test_trainer/checkpoint-20000")

    def decompile_cod(cod: str) -> str:
        input_ids = tokenizer(cod, return_tensors="pt").input_ids
        generated_ids = model.generate(input_ids, max_length=1000)
        return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    # def compute_loss(asm, text):
    #     asm_ids = tokenizer(asm, return_tensors="pt").input_ids
    #     generated_ids = model.generate(input_ids, max_length=1000)
    #     return generated_ids

    ds = gen_dataset(tokenizer)
    test_set = ds['test']

    num_test = 4
    for i, (asm, text) in enumerate(zip(test_set['asm'], test_set['text'])):
        decomped = decompile_cod(asm)
        print("----------------------------------------------")
        print(f"Original Cod:\n {text}\n Decomped Cod:\n {decomped}")
        print("----------------------------------------------")
        if i == num_test:
            break



if __name__ == '__main__':
    main()