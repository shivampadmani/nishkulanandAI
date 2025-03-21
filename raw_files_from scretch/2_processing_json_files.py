import os
import re
import json
os.chdir("D:\\OneDrive - Indian Institute of Science\\Projects\\NishkulanandAI")
# Function to remove numbers between spaces
def remove_numbers_in_couplet(couplet):
    # Regular expression pattern to match numbers (Gujarati and English) between spaces
    pattern = r'(?<=\s)[0-9]+(?=\s)|(?<=\s)[\u0AE6-\u0AEF0-9]+(?=\s)'
    
    # Replace the numbers with an empty string
    return re.sub(pattern, '', couplet).strip()

# Function to process a single JSON file
def process_json_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile:
        data = json.load(infile)
    
    # Iterate through "gujarati" and "transliteration" keys
    for key in ["gujarati", "transliteration"]:
        if key in data:
            for item in data[key]:
                if "couplet" in item:
                    # Apply the number removal function to each couplet
                    item["couplet"] = remove_numbers_in_couplet(item["couplet"])
    
    # Save the modified data to the output JSON file
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=4)

# Function to process all JSON files in the "json_files" directory
def process_all_json_files(directory):
    # Get all the JSON files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            input_file = os.path.join(directory, filename)
            output_file = os.path.join(directory,  filename)
            
            # Process each file
            process_json_file(input_file, output_file)
            print(f"Processed {filename} and saved to {output_file}")

# Directory containing the JSON files
json_directory = 'json_files'

# Process all files in the directory
process_all_json_files(json_directory)
