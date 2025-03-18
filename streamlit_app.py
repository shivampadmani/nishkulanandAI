import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor
import torch
import os
import re
device = "cuda" if torch.cuda.is_available() else "cpu"
# Set page configuration (MUST BE THE FIRST STREAMLIT COMMAND)
st.set_page_config(
	page_title="NishkulanandAI - Gujarati Poetry Generator",
	page_icon="🪷",
	layout="wide"
)

# Initialize session state for input if it doesn't exist
if "input" not in st.session_state:
	st.session_state.input = ""

# Initialize session state for example button clicks
if "example_clicked" not in st.session_state:
	st.session_state.example_clicked = False
	st.session_state.selected_example = ""

# Initialize chat history if it doesn't exist
if "chat_history" not in st.session_state:
	st.session_state.chat_history = []


# Callback function for example buttons
def set_example_prompt(prompt):
	st.session_state.example_clicked = True
	st.session_state.selected_example = prompt
	st.session_state.input = prompt  # Update the input field directly


# Load the fine-tuned Gujarati poetry model and tokenizer
@st.cache_resource
def load_model():
	try:
		model_name = "./indictrans2-poetry-finetuned"
		base_model_path = "ai4bharat/indictrans2-indic-indic-1B"  # Original model instead of fine-tuned
		tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
		# THis model can translate any language to english
		# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

		model = AutoModelForSeq2SeqLM.from_pretrained(
			model_name,
			trust_remote_code=True,
			torch_dtype=torch.float16,  # performance might slightly vary for bfloat16
			attn_implementation="flash_attention_2"
		).to(device)
		return tokenizer, model, None
	except Exception as e:
		return None, None, str(e)


# Function to generate poetry
def generate_poem(prompt):
	try:
		ip = IndicProcessor(inference=True)

		# input_ids = tokenizer.encode(prompt, return_tensors="pt")
		src_lang, tgt_lang = "eng_Latn", "guj_Gujr"

		# Add safety check for device availability
		device = "cuda" if torch.cuda.is_available() else "cpu"
		model.to(device)
		batch = ip.preprocess_batch(
			[prompt,prompt,prompt],
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

		# with torch.no_grad():
		# 	output = model.generate(
		# 		**inputs,
		# 		use_cache=True,
		# 		min_length=0,
		# 		max_length=256,
		# 		num_beams=5,
		# 		num_return_sequences=2,
		# 		repetition_penalty=2.
		# 	)
		with torch.no_grad():
			output = model.generate(
				**inputs,
				use_cache=True,
				min_length=0,
				max_length=256,
				do_sample=True,  # Enable sampling
				top_k=50,  # Use top-k sampling
				num_return_sequences=1,
				repetition_penalty=1.5,
			)
		with tokenizer.as_target_tokenizer():
			generated_tokens = tokenizer.batch_decode(
				output.detach().cpu().tolist(),
				skip_special_tokens=True,
				clean_up_tokenization_spaces=True,
			)

		# Postprocess the translations, including entity replacement
		translations = ip.postprocess_batch(generated_tokens, lang=tgt_lang)
		string = " \n".join(translations[i] for i in range(len(batch))) + " \n"
		return string
		# return translations[0]
	except Exception as e:
		return f"Error generating poem: {str(e)}"


# Example prompts
example_prompts = [
	"Description of Akshardham",
	"The characteristics of a true saint",
	"Ramanand swami met Nilkanth Varni"
]

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-family: 'Serif';
        text-align: center;
        color: #722F37;
    }
    .poem-output {
        background-color: #FFF9E6;
        border-left: 5px solid #722F37;
        padding: 20px;
        border-radius: 5px;
        font-family: 'Serif';
        line-height: 1.6;
    }
    .sidebar-content {
        background-color: #F5F5F5;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .example-button {
        width: 100%;
        margin: 5px 0;
    }
    .stButton > button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# Check if image directory exists
image_dir = "images"
if not os.path.exists(image_dir):
	os.makedirs(image_dir, exist_ok=True)
	st.warning(
		f"Image directory '{image_dir}' was created. Please add nishkulanand.jpg and swaminarayan.jpg to this directory.")

# Load model
tokenizer, model, model_error = load_model()

# Layout with three columns
left_col, main_col, right_col = st.columns([1, 2, 1])

# Left sidebar with Nishkulanand Swami image and info
with left_col:
	try:
		st.image("images/nishkulanand.jpg", caption="Nishkulanand Swami (1766-1848)", width=400)
	except Exception:
		st.error("Could not load nishkulanand.jpg. Please ensure the image exists in the 'images' directory.")

	st.markdown("""
    <div class="sidebar-content">
    <h4>About Nishkulanand Swami</h4>
    <p>A renowned poet-saint and disciple of Bhagwan Swaminarayan. Known for his poetic works like "Bhakta Chintamani" and "Nishkulanand Kaavya" that express devotion and spiritual knowledge.</p>
    </div>
    """, unsafe_allow_html=True)

# Main content
with main_col:
	st.markdown("<h1 class='main-header'>🪷 NishkulanandAI 🪷</h1>", unsafe_allow_html=True)
	st.markdown("<h3 style='text-align: center;'>Generate Gujarati Poetry in the Style of Nishkulanand Swami</h3>",
				unsafe_allow_html=True)

	if model_error:
		st.error(f"Error loading model: {model_error}")
		st.warning("Demo mode: Poetry generation is simulated.")

	st.markdown("---")

	# Example prompts
	st.markdown("### ✨ Try one of these examples:")
	cols = st.columns(3)
	for i, prompt in enumerate(example_prompts):
		cols[i].button(prompt, key=f"example_{i}", on_click=set_example_prompt, args=(prompt,))

	# Input section with pre-filled value if example was clicked
	st.subheader("🙏 Enter your prompt in English")
	user_input = st.text_area("The topic or theme of your poem:",
							  value=st.session_state.input,
							  height=100,
							  key="text_input")

	# Update session state based on text area
	st.session_state.input = user_input

	# Generate button
	if st.button("🪷 Generate Divine Poetry 🪷", use_container_width=True, key="generate_button"):
		if user_input:
			with st.spinner("🕉️ Channeling the divine wisdom of Nishkulanand Swami..."):
				if model and tokenizer:
					generated_poem = generate_poem(user_input)
				else:
					# Demo mode - generate a simulated poem
					generated_poem = f"""
                    [Simulated Gujarati poem based on: '{user_input}']

                    સ્વામિનારાયણના દિવ્ય પ્રેમમાં,
                    ભક્તિની જ્યોત જલે છે અંતરમાં,
                    નિષ્કુલાનંદ કહે, સાંભળો સહુ જન,
                    વૈરાગ્ય વિના નહિં મળે મોક્ષનું ધન.
                    """

				# Store in session history
				st.session_state.chat_history.append(("You", user_input))
				# Replace all periods with periods followed by a new line
				formatted_poem = re.sub(r'\.(?=\s|$)', '.\n', generated_poem)
				# Add to session state
				st.session_state.chat_history.append(("Nishkulanand", formatted_poem))
				# st.session_state.chat_history.append(("Nishkulanand", generated_poem))

				# Clear input after generating
				st.session_state.input = ""
				st.rerun()

	# Display the most recent poem
	if st.session_state.chat_history:
		st.markdown("---")
		st.subheader("🪔 Your Divine Poem:")
		last_prompt = next((msg[1] for msg in reversed(st.session_state.chat_history) if msg[0] == "You"), None)
		last_poem = next((msg[1] for msg in reversed(st.session_state.chat_history) if msg[0] == "Nishkulanand"), None)

		if last_prompt and last_poem:
			st.markdown(f"**Prompt:** {last_prompt}")
			st.markdown(f"<div class='poem-output'>{last_poem}</div>", unsafe_allow_html=True)

			# Share buttons
			share_cols = st.columns(3)
			share_cols[0].download_button(
				"📥 Download Poem",
				last_poem,
				file_name="nishkulanand_poem.txt",
				mime="text/plain"
			)
			if share_cols[1].button("📱 Share", key="share_button"):
				st.info("Sharing functionality would be implemented here in a production app.")

			if share_cols[2].button("❤️ Save as Favorite", key="favorite_button"):
				st.success("Poem saved to favorites!")
			# In a real app, you'd save to a database or file

	# Chat history
	if len(st.session_state.chat_history) > 2:
		st.markdown("---")
		with st.expander("📜 Previous Poems"):
			history_pairs = [(st.session_state.chat_history[i][1], st.session_state.chat_history[i + 1][1])
							 for i in range(0, len(st.session_state.chat_history) - 2, 2)]

			for i, (prompt, poem) in enumerate(reversed(history_pairs)):
				st.markdown(f"**Prompt:** {prompt}")
				st.markdown(f"<div class='poem-output'>{poem}</div>", unsafe_allow_html=True)
				st.markdown("---")

		# Clear history button
		if st.button("🗑️ Clear History", use_container_width=True, key="clear_history"):
			st.session_state.chat_history = []
			st.rerun()

# Right sidebar with Bhagwan Swaminarayan image and context
with right_col:
	try:
		st.image("images/swaminarayan.jpg", caption="Bhagwan Swaminarayan (1781-1830)", width=400)
	except Exception:
		st.error("Could not load swaminarayan.jpg. Please ensure the image exists in the 'images' directory.")

	st.markdown("""
    <div class="sidebar-content">
    <h4>About Bhagwan Swaminarayan</h4>
    <p>Founder of the Swaminarayan Sampradaya, a Hindu denomination. His teachings emphasized devotion (bhakti), knowledge (jnana), detachment (vairagya), and dharma (righteousness).</p>
    </div>
    """, unsafe_allow_html=True)

	# Additional information
	st.markdown("""
    <div class="sidebar-content">
    <h4>About This App</h4>
    <p>This AI model has been trained on the poetic works of Nishkulanand Swami to generate Gujarati poetry that reflects his unique style, spiritual themes, and devotional essence.</p>
	<p> Please contact Shivam Padmani on LinkedIn for more details.
<a href="https://www.linkedin.com/in/shivampadmani/">LinkedIn Profile</a>
</p>
    </div>
    """, unsafe_allow_html=True)
