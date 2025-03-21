import os
import json

# Input and output directories
input_dir = "json_files"
output_dir = "processed_json_files"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Loop through all JSON files in input_dir
for filename in os.listdir(input_dir):
    if filename.endswith(".json"):  # Process only JSON files
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # Read JSON file
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Process poetry data (extracting couplets and context)
        processed_data = {
            "prakaran": data["prakaran"],
            "title_gu": data["title_gu"],
            "title_trans": data["title_trans"],
            "poetry": []  # Storing processed poetic data
        }

        for entry in data["gujarati"]:
            couplet_text = entry["couplet"]
            context_text = entry["context"]
            rhyme_tag = couplet_text.split("\n")[-1].split()[-1]  # Extract rhyme tag

            processed_data["poetry"].append({
                "couplet": couplet_text,
                "rhyme_tag": rhyme_tag,  # Store the rhyme pattern
                "context": context_text
            })

        # Save processed data
        with open(output_path, "w", encoding="utf-8") as f_out:
            json.dump(processed_data, f_out, ensure_ascii=False, indent=4)

        print(f"Processed: {filename}")

print("✅ All files processed and saved!")
