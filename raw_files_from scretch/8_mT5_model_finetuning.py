from transformers import MT5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq, T5Tokenizer
from torch.utils.data import Dataset, DataLoader
import torch
import json

# Define a custom dataset
class GujaratiPoetryDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "input_ids": torch.tensor(item["source"], dtype=torch.long),
            "labels": torch.tensor(item["target"], dtype=torch.long)
        }

# Load tokenized data
with open("train_tokenized.json", "r", encoding="utf-8") as f:
    train_tokenized = json.load(f)
with open("val_tokenized.json", "r", encoding="utf-8") as f:
    val_tokenized = json.load(f)

# Create dataset instances
train_dataset = GujaratiPoetryDataset(train_tokenized)
val_dataset = GujaratiPoetryDataset(val_tokenized)

# Load model and tokenizer
model = MT5ForConditionalGeneration.from_pretrained("google/mt5-base")
tokenizer = T5Tokenizer.from_pretrained("google/mt5-base", legacy=False)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    save_total_limit=3,
    num_train_epochs=5,
    learning_rate=3e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    save_strategy="epoch"
)

# Initialize data collator with the tokenizer
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator  # Pass the data collator here
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("mt5-gujarati-poetry")
print("Fine-tuned model saved!")