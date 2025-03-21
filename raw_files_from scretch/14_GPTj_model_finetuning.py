from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset

# Load GPT-J model & tokenizer
model_name = "EleutherAI/gpt-j-6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load dataset (Ensure JSON has "text" column)
dataset = load_dataset("json", data_files="formatted_poetry_data.json", split="train")
# Set PAD token to EOS token (common practice for GPT models)
tokenizer.pad_token = tokenizer.eos_token

# If eos_token is not available, add a custom pad token
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

# Apply tokenization
dataset = dataset.map(tokenize_function, batched=True)

# Split into train/test
dataset = dataset.train_test_split(test_size=0.1)
train_dataset, val_dataset = dataset["train"], dataset["test"]

# Training Arguments
training_args = TrainingArguments(
    output_dir="./gptj_gujarati",
    per_device_train_batch_size=2,  # Reduce batch size if OOM
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,  # Accumulate gradients for stability
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    learning_rate=2e-5,
    weight_decay=0.01,
    num_train_epochs=3,
    logging_steps=100,
    fp16=True,  # Mixed precision for speedup
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer
)

# Train
trainer.train()

# Save model & tokenizer
model.save_pretrained("./gptj_gujarati")
tokenizer.save_pretrained("./gptj_gujarati")
# Save the model to disk
