# Gujarati AI Poetry Generator

## 🌟 Introduction
This project explores the intersection of **Artificial Intelligence** and **Satsang literature** by generating **Gujarati poetry** inspired by the works of great poets like **Nishkulanand Swami**. Using AI models, this tool transforms English prompts into **meaningful, poetic Gujarati verses**.

## ✨ Features
- **Gujarati BPE Tokenizer**: Custom-built tokenizer for efficient text generation.
- **Fine-tuned GPT Model**: Leveraging GPT-J/GPT-2 for Gujarati poetry generation.
- **Dataset**: Based on structured English-Gujarati poetry pairs.
- **Use Cases in Satsang**: AI-driven tools for **kirtan composition, scripture analysis, and poetic expressions**.

## 🛠️ Installation
```bash
# Clone the repository
git clone https://github.com/shivampadmani/NishkulanandAI.git
cd NishkulanandAI

# Install dependencies
pip install -r requirements.txt
```

## 📜 Dataset Structure
The training dataset follows this format:
```json
{
    "gujarati": [
        {
            "title_gu": "પ્રકરણ ૧: મંગળાચરણ",
            "title_trans": "Prakaran 1: Mangaḷācharaṇ",
            "target": "મંગલમૂર્તિ મહાપ્રભુ, શ્રીસહજાનંદ સુખરૂપ...",
            "source": "The divine Lord is the embodiment of joy and beauty..."
        }
    ]
}
```




## 🔥 Inference Example
```python
from model import generate_poetry
prompt = "The divine Lord is the embodiment of joy and beauty."
response = generate_poetry(prompt)
print(response)
```

## 🎯 Future Enhancements
- Improving **rhyme and meter** matching.
- Expanding dataset with **more scriptures & kirtans**.
- Deploying a **web-based interface** for real-time poetry generation.

## 🌟 Contribute
Feel free to open issues and pull requests! Your contributions can help refine AI-driven Gujarati poetry generation.

## 💬 Contact
For any queries or collaboration, message me on LinkedIn: https://www.linkedin.com/in/shivampadmani/

## 📜 License
This project is licensed under the **MIT License**.

---
🔗 *For more AI & Satsang insights,Follow me on GitHub and connect with me on LinkedIn!*

