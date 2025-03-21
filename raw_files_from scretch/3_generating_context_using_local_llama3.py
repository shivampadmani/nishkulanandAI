import json
import subprocess
import os

# Change to the working directory
os.chdir("D:\\OneDrive - Indian Institute of Science\\Projects\\NishkulanandAI\\json_files_dataset")

def summarize_text(couplet):
    # print("Generating context for:", couplet)
    prompt = f"Translate this Gujarati couplet to English statement. Don't add any extra words in your response (not even words about \"translation\") other than words used in this couplet. Dont do just transliteration but do translate the sentences actually:  {couplet}"
    
    try:
        # Run Ollama with LLaMA 3
        result = subprocess.run(
            ["ollama", "run", "llama3", prompt],
            capture_output=True,
            text=True,
            encoding="utf-8"  # Ensure output is read correctly
        )
        # print("Running Ollama...")

        # Check if subprocess execution failed
        if result.returncode != 0:
            print(f"Error running Ollama:\nSTDERR: {result.stderr}")
            return "Error: Unable to generate summary."

        # Ensure result.stdout is not None before stripping
        return result.stdout.strip() if result.stdout else "Error: No output from model."

    except Exception as e:
        print(f"Exception in summarize_text: {e}")
        return "Error: Exception occurred."

# Directory containing JSON files
json_directory = "new_json_files"

for filename in os.listdir(json_directory):
    print(f"Processing file: {filename}")
    if filename.endswith(".json"):
        file_path = os.path.join(json_directory, filename)

        try:
            # Read the JSON file
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)

            # Process the "gujarati" and "transliteration" keys
            for key in ["gujarati"]:
                if key in data:
                    print(f"Processing key: {key}")
                    for i in range(len(data[key])):
                        couplet = data[key][i].get('couplet', '')
                        if couplet:
                            data[key][i]['context'] = summarize_text(couplet)
                        else:
                            print(f"Skipping empty couplet in {filename}")

            # Save the updated data back to the file
            with open(file_path, "w", encoding="utf-8") as file:
                json.dump(data, file, ensure_ascii=False, indent=4)
                print(f"Processed and updated file: {filename}")

        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON file: {filename}")
        except Exception as e:
            print(f"Unexpected error processing {filename}: {e}")
