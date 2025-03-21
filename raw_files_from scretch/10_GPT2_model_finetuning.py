import json
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset

# Load dataset
with open("gpt2_train.json", "r", encoding="utf-8") as f:
    train_data = json.load(f)
with open("gpt2_val.json", "r", encoding="utf-8") as f:
    val_data = json.load(f)

# Convert data into GPT-2 format
formatted_data = []
for data in train_data:
    formatted_data.append(data)

# Convert to Hugging Face dataset format
dataset = Dataset.from_list(formatted_data)

# Load GPT-2 tokenizer and model
model_name = "gpt2-medium"  # Use "gpt2-medium" or "gpt2-large" for better results
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad token by default

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Load model
model = GPT2LMHeadModel.from_pretrained(model_name)

# Training arguments
training_args = TrainingArguments(
    output_dir="./gpt2_finetuned",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_steps=10,
    save_total_limit=2,
    fp16=True if torch.cuda.is_available() else False,
    push_to_hub=False
)
# training_args = TrainingArguments(
#     output_dir="./gpt2_finetuned",
#     evaluation_strategy="no",  # ✅ Disable evaluation
#     save_strategy="epoch",
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     num_train_epochs=3,
#     logging_dir="./logs",
# )

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # GPT-2 uses causal modeling, not masked language modeling
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset= tokenized_dataset,
    data_collator=data_collator
)

# Fine-tune the model
trainer.train()

# Save fine-tuned model
model.save_pretrained("./gpt2_finetuned")
tokenizer.save_pretrained("./gpt2_finetuned")

print("Training Complete!")
