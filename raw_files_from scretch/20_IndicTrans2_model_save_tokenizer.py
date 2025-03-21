from transformers import AutoTokenizer
import os
os.chdir("D:\\OneDrive - Indian Institute of Science\Projects\\NishkulanandAI")
# Define the model name
model_name = "ai4bharat/indictrans2-indic-indic-1B"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)

# Define the directory to save the tokenizer
save_directory = "./indictrans2-poetry-finetuned"

# Save the tokenizer
tokenizer.save_pretrained(save_directory)

print(f"Tokenizer saved successfully in {save_directory}")
    
