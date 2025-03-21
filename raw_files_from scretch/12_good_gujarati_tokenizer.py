# Load model directly
from transformers import AutoTokenizer, AutoModelForMaskedLM
import re
import json

with open("cleaned_final_data.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

tokenizer = AutoTokenizer.from_pretrained("ashwani-tanwar/Gujarati-XLM-R-Base")
model = AutoModelForMaskedLM.from_pretrained("ashwani-tanwar/Gujarati-XLM-R-Base")

rhyme_tokens = set()
for data in dataset["gujarati"]:
    matches = re.findall(r"<RHYME_\d+>", data.get("target", ""))
    rhyme_tokens.update(matches)

rhyme_tokens = list(rhyme_tokens)

special_tokens = {"additional_special_tokens": rhyme_tokens}
tokenizer.add_special_tokens(special_tokens)

model.resize_token_embeddings(len(tokenizer))

text = "સર્વે સંત સુજાણને, હું પ્રથમ લાગી પાય<RHYME_208975>; \nઆદરું આ ગ્રંથને, જેમાં વિઘન કોઈ ન થાય<RHYME_208975>."
tokens = tokenizer.tokenize(text)
numbers = tokenizer.encode(text)
print("Tokenized:", tokens)
print("Numbers:", numbers)
print("Detokenized:", tokenizer.decode(numbers))




