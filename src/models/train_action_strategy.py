# train_action_strategy.py
"""
Fine-tune T5 model to generate Action Strategy
Input: Headline + Summary + Sentiment
Output: Short/Mid/Long action strategy
"""

import os
import json
import torch
from datasets import load_dataset
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)

# Load dataset (JSONL with {"input": ..., "output": ...})
def load_custom_dataset(path="data/training/train.jsonl"):
    dataset = load_dataset("json", data_files={"train": path, "validation": path})
    return dataset

# Preprocess
def preprocess_function(examples, tokenizer, max_input=256, max_output=64):
    model_inputs = tokenizer(examples["input"], max_length=max_input, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["output"], max_length=max_output, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main():
    model_name = "t5-small"  # or "t5-base"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    dataset = load_custom_dataset("data/training/train.jsonl")
    tokenized_datasets = dataset.map(
        lambda batch: preprocess_function(batch, tokenizer),
        batched=True,
        remove_columns=["input", "output"]
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = TrainingArguments(
        output_dir="./checkpoints",
        eval_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=3,
        logging_dir="./logs",
        logging_steps=50
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()

    # Save model
    model.save_pretrained("./models/action-strategy")
    tokenizer.save_pretrained("./models/action-strategy")
    print("✅ Model saved to ./models/action-strategy")

if __name__ == "__main__":
    main()
