import os
import json
import re
from bs4 import BeautifulSoup

os.chdir("D:\\OneDrive - Indian Institute of Science\\Projects\\NishkulanandAI\\json_files_dataset")

def extract_text_with_line_breaks(html_content):
    # Convert <br> tags to newline
    html_content = html_content.replace("<br />", "\n")
    
    # Remove all other HTML tags
    clean_text = re.sub(r'<.*?>', '', html_content)
    
    # Split into lines
    lines = [line.strip() for line in clean_text.split('\n') if line.strip()]
    return lines

def process_couplets(lines):
    couplets = []
    for i in range(0, len(lines) - 1, 2):  # Process in pairs
        if i + 1 < len(lines):
            couplet_number = (i // 2) + 1  # Start from 0 instead of 1
            
            # Extract last words of each line for RHYME_X tagging
            last_word_1 = lines[i].split()[-1]
            last_word_2 = lines[i+1].split()[-1]
            
            # rhyme_tag = f"<RHYME_{couplet_number}>"
            #
            # # Append rhyme tags
            line1 = f"{lines[i]} "
            line2 = f"{lines[i+1]}"
            
            couplets.append({
                "couplet": f"{line1}\n{line2}",
                "couplet_number": couplet_number
            })
    return couplets

def process_html_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")
    
    prakaran_number = re.search(r'prakaran_(\d+)', file_path).group(1)
    
    # Extract titles
    title_gu = soup.find("h1", {"id": "title_gu"}).get_text(strip=True)
    # title_trans = soup.find("h1", {"id": "title_trans"}).get_text(strip=True)
    
    # Extract Gujarati and transliteration sections using find_all
    gu_divs = soup.find_all("span", {"class": "main_text_uni"})
    # trans_divs = soup.find_all("span", {"class": "main_text_trans"})
    
    # Concatenate text from all found divs and flatten the list of lines
    gu_text = "\n".join([line for gu_div in gu_divs for line in extract_text_with_line_breaks(str(gu_div))]) if gu_divs else ""
    # trans_text = "\n".join([line for trans_div in trans_divs for line in extract_text_with_line_breaks(str(trans_div))]) if trans_divs else ""
    
    gu_couplets = process_couplets(gu_text.split('\n'))
    # trans_couplets = process_couplets(trans_text.split('\n'))
    
    return {
        "prakaran": int(prakaran_number),
        "title_gu": title_gu,
        # "title_trans": title_trans,
        "gujarati": gu_couplets,
        # "transliteration": trans_couplets
    }

def main():
    input_dir = "./html_files"  # Change to your actual directory
    output_dir = "./new_json_files"
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.startswith("bhaktachintamani_prakaran_") and filename.endswith(".html"):
            file_path = os.path.join(input_dir, filename)
            data = process_html_file(file_path)
            
            output_file = os.path.join(output_dir, f"bhaktachintamani_prakaran_{data['prakaran']}.json")
            with open(output_file, "w", encoding="utf-8") as json_file:
                json.dump(data, json_file, ensure_ascii=False, indent=4)
            print(f"Processed {filename} -> {output_file}")

if __name__ == "__main__":
    main()
