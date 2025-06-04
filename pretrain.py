#!/usr/bin/env python3

from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig
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


context_length = 128
tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt2-small", trust_remote_code=True)

TOKENS = ["Swarm", "tvk", "Cisco"]
tokenizer.add_tokens(TOKENS)

print(raw_datasets)
outputs = tokenizer(
    raw_datasets["train"][:2]["text"],
    truncation=True,
    max_length=context_length,
    return_overflowing_tokens=True,
    return_length=True,
)

print(f"Input IDs length: {len(outputs['input_ids'])}")
print(f"Input chunk lengths: {(outputs['length'])}")
print(f"Chunk mapping: {outputs['overflow_to_sample_mapping']}")

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
    "gpt2",
    vocab_size=len(tokenizer),
    n_ctx=context_length,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

model = GPT2LMHeadModel(config)
model_size = sum(t.numel() for t in model.parameters())
print(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

