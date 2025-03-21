from transformers import MT5ForConditionalGeneration, T5Tokenizer
import torch

# Load the fine-tuned model
model = MT5ForConditionalGeneration.from_pretrained("mt5-gujarati-poetry")

# Load the tokenizer (ensure it matches the one used during training)
tokenizer = T5Tokenizer.from_pretrained("google/mt5-base", legacy=False)

def generate_poetry(prompt):
    # Add a task prefix
    input_text = "generate poetry: " + prompt

    # Tokenize the input
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    # Generate output
    # output_ids = model.generate(input_ids, max_length=50, num_beams=3, early_stopping=True)
    # output_ids = model.generate(input_ids, 
    #                             max_length=500,
    #                             min_length = 200,
    #                             do_sample=True, 
    #                             temperature=0.9, 
    #                             top_k=50, 
    #                             top_p=0.9, 
    #                             repetition_penalty=1.5)

    # output_ids = model.generate(input_ids, 
    #                         max_length=50, 
    #                         num_beams=3, 
    #                         repetition_penalty=1.5)

    output_ids = model.generate(
        input_ids, 
        max_length=50, 
        min_length=10, 
        do_sample=True, 
        temperature=0.7, 
        top_k=50, 
        top_p=0.9,
        repetition_penalty=1.2,
        early_stopping=False
    )
    # Decode the output
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return output_text

# Example prompts
print(generate_poetry("Describe the beauty of Akshardham"))
print(generate_poetry("Explain virtues of Sadhu."))
print(generate_poetry("Who is Bhagwan Swaminarayan"))