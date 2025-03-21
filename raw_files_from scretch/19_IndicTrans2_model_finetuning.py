from itertools import batched

import torch
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from IndicTransToolkit import IndicProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# Define source and target languages
src_lang, tgt_lang = "eng_Latn", "guj_Gujr"

# Base model to fine-tune
base_model_name = "ai4bharat/indictrans2-indic-indic-1B"
tokenizer = AutoTokenizer.from_pretrained(base_model_name,trust_remote_code= True)
model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name,trust_remote_code=True).to(device)
ip = IndicProcessor(inference=True)
output_model_dir = "./indictrans2-poetry-finetuned"

raw_datasets = load_dataset("json", data_files="./simplified_data_for_sutra.json")


def tokenize_function(example):
    return tokenizer(
        example["source"],
        text_target=example["target"],
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors="pt",
        return_attention_mask=True
    )


raw_datasets = raw_datasets["train"].map(lambda x: {"source": x["gujarati"]["source"], "target": x["gujarati"]["target"]})
tokenized_datasets = raw_datasets.map(tokenize_function,batched=True)
tokenized_datasets = tokenized_datasets.train_test_split(test_size=0.1)

train_dataset, val_dataset = tokenized_datasets["train"],tokenized_datasets["test"]

training_args = Seq2SeqTrainingArguments(
    output_dir=output_model_dir,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    max_grad_norm=1,
    num_train_epochs=3,
    learning_rate=5e-5,
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_strategy="epoch",
    report_to="all",
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset= train_dataset,
    eval_dataset= val_dataset,
    data_collator=data_collator
)

trainer.train()

model.save_pretrained(output_model_dir)
tokenizer.save_pretrained(output_model_dir)