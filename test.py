import argparse

import tqdm
import torch
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

from train import gen_dataset

def main():
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("test_trainer/checkpoint-15000")
    metric = evaluate.load('bleu')
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def decompile_cod(cod: str) -> str:
        input_ids = tokenizer(cod, return_tensors="pt").input_ids.to(device)
        generated_ids = model.generate(input_ids, max_length=1000)
        return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    def compute_metric(orig, decomped):
        if isinstance(orig, str):
            orig = [orig]
        if isinstance(decomped, str):
            decomped = [decomped]
        m = metric.compute(predictions=decomped, references=orig)
        return m

    model = model.to(device)
    ds = gen_dataset(tokenizer)
    test_set = ds['test']

    decomped = []
    orig_txt = []
    print("Decompiling test set...")
    for asm, text in tqdm.tqdm(zip(test_set['asm'], test_set['text']), total=len(test_set)):
        decomp = decompile_cod(asm)
        decomped.append(decomp)
        orig_txt.append(text)

    print(f"Computing eval metric on test set...")
    m = compute_metric(orig_txt, decomped)
    print(m)

    # Output a few samples to visually compare
    num_test = 4
    for i, (decomped, text) in enumerate(zip(decomped, orig_txt)):
        print("----------------------------------------------")
        print(f"Original Cod:\n {text}\nDecomped Cod:\n {decomped}")
        print("----------------------------------------------")
        if i == num_test:
            break


if __name__ == '__main__':
    main()