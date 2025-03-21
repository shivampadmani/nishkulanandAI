import os
import json
import re

# File to store all rhyming pairs
rhyme_pairs_file = "rhyme_pairs.txt"

# Load or initialize global rhyme counter
counter_file = "rhyme_counter.txt"
if os.path.exists(counter_file):
    with open(counter_file, "r") as f:
        global_rhyme_counter = int(f.read().strip())
else:
    global_rhyme_counter = 1

# Function to extract rhyming words (strictly before `,`, `;`, `.`)
def extract_rhyming_words(couplet):
    words = re.findall(r"([\u0A80-\u0AFF]+\u0ABE?[\u0A80-\u0AFF]*)(?=[,;.])", couplet)  # Captures last word before punctuation
    return words

# Function to check if two words rhyme (match last consonant + matra)
def words_rhyme(word1, word2):
    return word1[-1] == word2[-1]  # Compare last character (consonant + matra)

# Function to tag rhyming words in a couplet
def tag_rhyming_words(data):
    global global_rhyme_counter
    rhyme_pairs = []
    rhyme_dict = {}  # Stores rhyme groups

    for item in data["gujarati"]:
        couplet = item["couplet"]
        words = extract_rhyming_words(couplet)

        if len(words) >= 1:
            matched = set()
            for i in range(len(words)):
                for j in range(i + 1, len(words)):
                    if words_rhyme(words[i], words[j]) and (words[i], words[j]) not in matched:
                        if words[i] in rhyme_dict:
                            tag = rhyme_dict[words[i]]
                        elif words[j] in rhyme_dict:
                            tag = rhyme_dict[words[j]]
                        else:
                            tag = f"<RHYME_{global_rhyme_counter}>"
                            global_rhyme_counter += 1

                        rhyme_dict[words[i]] = tag
                        rhyme_dict[words[j]] = tag
                        matched.add((words[i], words[j]))
                        rhyme_pairs.append(f"{words[i]} - {words[j]}")

            # Apply tags
            for word, tag in rhyme_dict.items():
                couplet = re.sub(rf"\b{word}\b(?=[,;.])", f"{word}{tag}", couplet, count=1)

        item["couplet"] = couplet

    return data, rhyme_pairs

# Directory paths
input_dir = "json_files_with_context"
output_dir = "processed_json_files"
os.makedirs(output_dir, exist_ok=True)

all_rhyme_pairs = []

# Process all JSON files
for filename in os.listdir(input_dir):
    if filename.endswith(".json"):
        with open(os.path.join(input_dir, filename), "r", encoding="utf-8") as f:
            data = json.load(f)

        processed_data, rhyme_pairs = tag_rhyming_words(data)
        all_rhyme_pairs.extend(rhyme_pairs)

        with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=4)

# Save updated global rhyme counter
with open(counter_file, "w") as f:
    f.write(str(global_rhyme_counter))

# Save all rhyme pairs to a file
with open(rhyme_pairs_file, "w", encoding="utf-8") as f:
    f.write("\n".join(all_rhyme_pairs))

print("Gujarati rhyming words tagged successfully!")
print(f"Saved rhyming pairs to {rhyme_pairs_file}")
