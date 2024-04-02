import os
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
)

from datasets import Dataset


def tokenize_function(examples, tokenizer, padding="max_length", max_tok_len=None):
    # tokenize inputs
    model_inputs = tokenizer(
        examples["asm"],
        max_length=max_tok_len,
        padding=padding,
        truncation=True,
    )

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(
        text_target=examples["text"],
        max_length=max_tok_len,
        padding=padding,
        truncation=True,
    )

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label]
            for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def gen_dataset(tokenizer, max_asm_seq_len=512):
    """
    TODO: OOP-ify this cod and get max_asm_seq_len from the model
    """
    S_TEXTS = []
    C_TEXTS = []

    MAX_SOURCE_LENGTH = 0
    MAX_TARGET_LENGTH = 0

    num_seq_exceed_len = 0
    data_total = 0

    for root, dirs, files in os.walk("harvest"):
        for dir in dirs:
            with open(Path(root) / dir / "sanitized.s") as f:
                asm_text = f.read()
            with open(Path(root) / dir / "sanitized.c") as f:
                c_text = f.read()

            data_total += 1
            # Only keep data with token sequence length within max model sequence input length
            num_asm_toks = len(tokenizer(asm_text, return_tensors="pt").input_ids[0])
            if num_asm_toks > max_asm_seq_len:
                num_seq_exceed_len += 1
                continue

            S_TEXTS.append(asm_text)
            C_TEXTS.append(c_text)

            if len(asm_text) > MAX_SOURCE_LENGTH:
                MAX_SOURCE_LENGTH = len(asm_text)

            if len(c_text) > MAX_TARGET_LENGTH:
                MAX_TARGET_LENGTH = len(c_text)

    print(
        f"Number of sequences skipped due to exceeded length: {num_seq_exceed_len}, suitable samples: {len(S_TEXTS)}"
    )

    ds = Dataset.from_dict({"asm": S_TEXTS, "text": C_TEXTS})

    tokenized_datasets = ds.map(
        lambda x: tokenize_function(x, tokenizer=tokenizer), batched=True
    )

    num_test = 100
    num_train = len(tokenized_datasets) - num_test
    final_ds = tokenized_datasets.train_test_split(
        train_size=num_train, test_size=num_test, seed=1334
    )
    return final_ds


def main():
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-small")

    final_ds = gen_dataset(tokenizer)

    training_args = TrainingArguments(
        output_dir="test_trainer",
        evaluation_strategy="epoch",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=10,
        save_total_limit=10,
        save_steps=5000,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=final_ds["train"],
        eval_dataset=final_ds["test"],
    )

    trainer.train()


if __name__ == "__main__":
    main()
