from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from IndicTransToolkit import IndicProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"

# src_lang, tgt_lang = "guj_Gujr", "en_Latn"
src_lang, tgt_lang = "eng_Latn","guj_Gujr"
model_name = "ai4bharat/indictrans2-indic-indic-1B"

# THis model can translate any language to english
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16, # performance might slightly vary for bfloat16
    attn_implementation="flash_attention_2"
).to(device)

ip = IndicProcessor(inference=True)

input_sentences = [
    "કુંભ મેળો, વિશ્વનો સૌથી મોટો આધ્યાત્મિક મેળાવડો, ચાર પવિત્ર નદીઓ",
    " ओडिशा का ओडिसी मंदिर की मूर्तियों से प्रेरित तरल मुद्राओं को प्रदर्शित करता है। ",
    "Akshardham is the divine abode of Bhagwan Swaminarayan.",
    "The ball is red",
]

batch = ip.preprocess_batch(
    input_sentences,
    src_lang=src_lang,
    tgt_lang=tgt_lang,
)

# Tokenize the sentences and generate input encodings
inputs = tokenizer(
    batch,
    truncation=True,
    padding="longest",
    return_tensors="pt",
    return_attention_mask=True,
).to(device)

# Generate translations using the model
with torch.no_grad():
    generated_tokens = model.generate(
        **inputs,
        use_cache=True,
        min_length=0,
        max_length=512,
        num_beams=5,
        num_return_sequences=1,
    )

# Decode the generated tokens into text
with tokenizer.as_target_tokenizer():
    generated_tokens = tokenizer.batch_decode(
        generated_tokens.detach().cpu().tolist(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

# Postprocess the translations, including entity replacement
translations = ip.postprocess_batch(generated_tokens, lang=tgt_lang)

for input_sentence, translation in zip(input_sentences, translations):
    print(f"{src_lang}: {input_sentence}")
    print(f"{tgt_lang}: {translation}")