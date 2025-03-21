import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, AutoModel
from datasets import Dataset
# Load your data
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Flatten the data into a list of dictionaries
    poems = [{"input": entry["source"], "output": entry["target"]} for entry in data["gujarati"]]
    return poems
# Prepare dataset for training
def prepare_dataset(poems):
    # Create a dataset from the poems
    dataset = [{'text': f"{poem['input']} {poem['output']}"} for poem in poems]
    return Dataset.from_list(dataset)
# Fine-tune the Sutra model
def fine_tune_model(dataset):
    # Load pre-trained Sutra model and tokenizer
    model_name = "TWO/sutra-mlt256-v2"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForCausalLM.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding="max_length", max_length=512)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=2,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
        logging_dir='./logs',
    )
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    # Start training
    trainer.train()
    model.save_pretrained('./fine_tuned_model')
    tokenizer.save_pretrained('./fine_tuned_model')
# Main function
def main():
    input_file = 'simplified_data_for_sutra.json'  # Path to your input data file
    poems = load_data(input_file)
    dataset = prepare_dataset(poems)
    fine_tune_model(dataset)
    print("Fine-tuning completed and model saved.")
if __name__ == "__main__":
    main()
