from tokenizers import ByteLevelBPETokenizer

# Initialize a BPE tokenizer
tokenizer = ByteLevelBPETokenizer()

# Train tokenizer on your dataset (Gujarati text)
tokenizer.train(
    files=["formarted_final_data.json"],  # Provide a large corpus of Gujarati text
    vocab_size=60_000,  # Adjust as needed
    min_frequency=2,  # Minimum frequency for a subword to be included
    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
)

# Save tokenizer
tokenizer.save_model("./gujarati-bpe-tokenizer")

# Load into Hugging Face format
from transformers import PreTrainedTokenizerFast

custom_tokenizer = PreTrainedTokenizerFast(tokenizer_file="./gujarati-bpe-tokenizer/vocab.json",
    bos_token="<s>",
    eos_token="</s>",
    unk_token="<unk>",
    pad_token="<pad>",
    mask_token="<mask>"
)

# Save the tokenizer in Hugging Face format
custom_tokenizer.save_pretrained("./gujarati-bpe-tokenizer")
