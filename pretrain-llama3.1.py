#!/usr/bin/env python3


import json

from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, T5Tokenizer, Qwen3ForCausalLM, AutoConfig, AutoModelForCausalLM
from datasets import load_dataset, DatasetDict, concatenate_datasets
from transformers import DataCollatorForLanguageModeling


ds_train = load_dataset("./datasets/train", split="train")
ds_valid = load_dataset("./datasets/validation", split="train")

raw_datasets = DatasetDict(
    {
        "train": concatenate_datasets([ds_train, ds_valid]),  # .shuffle().select(range(50000)),
    }
)


context_length = 128000
tokenizer = AutoTokenizer.from_pretrained("tokyotech-llm/Llama-3.3-Swallow-70B-v0.4")

#TOKENS = []

#with open('words-old.json', 'r') as file:
#    for token in json.load(file):
#        TOKENS.append(token)

#tokenizer.add_tokens(TOKENS)

print(raw_datasets)

def tokenize(element):
    return {"input_ids": tokenizer(
        element["text"],
    ) }
#    input_batch = []
#    n = context_length

#    for input_ids in outputs["input_ids"]:
#        for input_id in input_ids:
#            input_batch.append(input_id)
#        input_batch.append(tokenizer.eos_token_id)
#    batch = [input_batch[i:i + n] for i in range(0, len(input_batch), n)]
#    return {"input_ids": [item for item in batch if len(item) == n]}

tokenized_datasets = raw_datasets.map(
    tokenize, batched=False, remove_columns=raw_datasets["train"].column_names
)

print(tokenized_datasets)

#config = AutoConfig.from_pretrained(
#    "Qwen/Qwen3-1.7B",
#    vocab_size=len(tokenizer),
#    n_ctx=context_length,
#    bos_token_id=tokenizer.bos_token_id,
#    eos_token_id=tokenizer.eos_token_id,
#)

model = AutoModelForCausalLM.from_pretrained("tokyotech-llm/Llama-3.1-Swallow-8B-v0.5")

model_size = sum(t.numel() for t in model.parameters())
print(f"Llama3.1 size: {model_size/1000**2:.1f}M parameters")

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

args = TrainingArguments(
    output_dir="Llama-3.1-vericava-posts-v1",
    auto_find_batch_size=True,
    eval_strategy="no",
    logging_steps=100,
    gradient_accumulation_steps=8,
    num_train_epochs=400,
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
)

trainer.train()

trainer.push_to_hub()


