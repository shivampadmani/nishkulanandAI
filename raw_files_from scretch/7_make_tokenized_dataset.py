import json
import re
from transformers import T5Tokenizer

# Load pre-trained MT5 tokenizer
tokenizer = T5Tokenizer.from_pretrained("google/mt5-base", legacy=False)

# Step 1: Extract all unique <RHYME_x> tokens
def extract_rhyme_tokens(data):
    rhyme_tokens = set()
    for entry in data:
        matches = re.findall(r"<RHYME_\d+>", entry.get("target", ""))
        rhyme_tokens.update(matches)
    return list(rhyme_tokens)

# Load dataset splits
with open("train.json", "r", encoding="utf-8") as f:
    train_data = json.load(f)
with open("val.json", "r", encoding="utf-8") as f:
    val_data = json.load(f)
with open("test.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

# Extract special tokens from all datasets
rhyme_tokens = extract_rhyme_tokens(train_data) + extract_rhyme_tokens(val_data) + extract_rhyme_tokens(test_data)
rhyme_tokens = list(set(rhyme_tokens))  # Remove duplicates

# Add special tokens to tokenizer
special_tokens = {"additional_special_tokens": rhyme_tokens}
tokenizer.add_special_tokens(special_tokens)

# Step 2: Function to tokenize data (removing <RHYME_x> tokens)
def tokenize_data(data, tokenizer, max_length=128):
    tokenized_pairs = []
    for entry in data:
        # Add task prefix to the source text
        source = "generate poetry: " + entry["source"]

        # Remove <RHYME_x> tokens from target text
        target = re.sub(r"<RHYME_\d+>", "", entry["target"]).strip()

        # Tokenize source and target
        source_tokens = tokenizer(
            source, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt"
        )["input_ids"]
        target_tokens = tokenizer(
            target, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt"
        )["input_ids"]

        # Append tokenized data
        tokenized_pairs.append({
            "source": source_tokens.tolist()[0],  # Convert tensor to list
            "target": target_tokens.tolist()[0]
        })

    return tokenized_pairs

# Tokenize the splits
train_tokenized = tokenize_data(train_data, tokenizer)
val_tokenized = tokenize_data(val_data, tokenizer)
test_tokenized = tokenize_data(test_data, tokenizer)

# Save tokenized data
with open("train_tokenized.json", "w", encoding="utf-8") as f:
    json.dump(train_tokenized, f, ensure_ascii=False, indent=4)
with open("val_tokenized.json", "w", encoding="utf-8") as f:
    json.dump(val_tokenized, f, ensure_ascii=False, indent=4)
with open("test_tokenized.json", "w", encoding="utf-8") as f:
    json.dump(test_tokenized, f, ensure_ascii=False, indent=4)

print("Tokenized data saved with proper padding!")
print("Length of vocab: ", len(tokenizer))
# print("Added special tokens:", tokenizer.additional_special_tokens)
