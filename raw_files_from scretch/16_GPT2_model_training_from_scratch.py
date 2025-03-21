from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments, PreTrainedTokenizerFast

# Load custom tokenizer
tokenizer = PreTrainedTokenizerFast(tokenizer_file="./gujarati_bpe/vocab.json", unk_token="<unk>", pad_token="<pad>")

# Define model config
config = GPT2Config(
    vocab_size=30_000,  # Same as tokenizer
    n_positions=512,
    n_embd=768,
    n_layer=12,
    n_head=12
)

# Load GPT-2 model from scratch
model = GPT2LMHeadModel(config)

# Load dataset
dataset = load_dataset("json", data_files="formatted_poetry_data.json", split="train")

# Tokenization
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

dataset = dataset.map(tokenize_function, batched=True)
dataset = dataset.train_test_split(test_size=0.1)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./gpt2_gujarati",
    per_device_train_batch_size=4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    learning_rate=5e-5,
    logging_steps=100,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer
)

# Train
trainer.train()

# Save model
model.save_pretrained("./gpt2_gujarati")
tokenizer.save_pretrained("./gpt2_gujarati")
