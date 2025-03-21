from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load fine-tuned model
model_path = "./gpt2_finetuned"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

def generate_poetry(prompt, max_length=100,min_length= 50):
    input_text = f"Context: {prompt} \n Poetry:"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    output_ids = model.generate(
        input_ids,
        do_sample=True,
        max_length=max_length,
        min_length = min_length,
        num_return_sequences=5,
        temperature=0.9,
        top_p=0.9
    )
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text

# Example test
english_context = "The beauty of shreeji maharaj"
generated_poetry = generate_poetry(english_context)
print("Generated Gujarati Poetry:\n", generated_poetry)
