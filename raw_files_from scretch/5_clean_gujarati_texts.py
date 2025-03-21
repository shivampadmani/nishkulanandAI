import json
import os
import re
input_dir = "json_files_context"
output_dir = "json_files_context"
os.makedirs(output_dir, exist_ok=True)

def clean_gujarati_text(text):
    # Remove Gujarati punctuation and unnecessary symbols but kep rhyme_tags
    cleaned_text = re.sub(r"[,\.;:!?\-\—\“\”\‘\’\(\)\"\']", "", text)  
    return cleaned_text


def clean_text(text):
    # Remove all <RHYME_X> tags but keep the words
    text = re.sub(r"<RHYME_\d+>", "", text)  # Remove <RHYME_X> tags
    # text = re.sub(r"[^\u0A80-\u0AFF\s]", "", text)  # Keep only Gujarati characters
    return text.strip()


for filename in os.listdir(input_dir):
    if filename.endswith(".json"):
        with open(os.path.join(input_dir, filename), "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Clean all couplets
        for entry in data["gujarati"]:
            entry["couplet"] = clean_text(entry["couplet"])
        
        # Save the cleaned file
        with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

print("Cleaning completed for all JSON files!")
