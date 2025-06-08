#!/usr/bin/env python3


import json

from transformers import Trainer, TrainingArguments
from transformers import T5Tokenizer, Qwen3ForCausalLM, AutoConfig
from datasets import load_dataset, DatasetDict
from transformers import DataCollatorForLanguageModeling

ds_train = load_dataset("./datasets/train", split="train")
ds_valid = load_dataset("./datasets/validation", split="train")

raw_datasets = DatasetDict(
    {
        "train": ds_train,  # .shuffle().select(range(50000)),
        "valid": ds_valid,  # .shuffle().select(range(500))
                            }
    )


context_length = 2048
tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt-1b")

TOKENS = []

with open('words.json', 'r') as file:
    for token in json.load(file):
        TOKENS.append(token)

tokenizer.add_tokens(TOKENS)

print(raw_datasets)

def tokenize(element):
    outputs = tokenizer(
        element["text"],
    )
    input_batch = []
    n = context_length

    for input_ids in outputs["input_ids"]:
        for input_id in input_ids:
            input_batch.append(input_id)
        input_batch.append(tokenizer.eos_token_id)
    batch = [input_batch[i:i + n] for i in range(0, len(input_batch), n)]
    return {"input_ids": [item for item in batch if len(item) == n]}

tokenized_datasets = raw_datasets.map(
    tokenize, batched=True, remove_columns=raw_datasets["train"].column_names
)

print(tokenized_datasets)

config = AutoConfig.from_pretrained(
    "Qwen/Qwen3-0.6B",
    vocab_size=len(tokenizer),
    n_ctx=context_length,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

model = Qwen3ForCausalLM(config)
model_size = sum(t.numel() for t in model.parameters())
print(f"Qwen3 size: {model_size/1000**2:.1f}M parameters")

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

args = TrainingArguments(
    output_dir="qwen3-0.6b-vericava-posts-v4",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    eval_strategy="steps",
    eval_steps=100,
    logging_steps=100,
    gradient_accumulation_steps=8,
    num_train_epochs=75,
    weight_decay=0.1,
    warmup_steps=300,
    lr_scheduler_type="cosine",
    learning_rate=2e-4,
    save_steps=100,
    bf16=True,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"],
)

trainer.train()

trainer.push_to_hub()

