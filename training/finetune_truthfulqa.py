import os
from argparse import ArgumentParser

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def tokenize(example, tokenizer, max_length):
    messages = [
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["best_answer"]},
    ]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=False)
    tokenized = tokenizer(
        text, truncation=True, max_length=max_length, return_tensors="pt"
    )
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized


def main():
    parser = ArgumentParser(description="Fine-tune DeepSeek-V3 on TruthfulQA")
    parser.add_argument("--model", required=True, help="Model checkpoint")
    parser.add_argument("--output-dir", required=True, help="Where to save weights")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    )
    model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_cfg)

    dataset = load_dataset("truthful_qa", "generation", split="train")
    max_length = tokenizer.model_max_length
    tokenized_ds = dataset.map(
        lambda ex: tokenize(ex, tokenizer, max_length),
        remove_columns=dataset.column_names,
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_total_limit=2,
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
