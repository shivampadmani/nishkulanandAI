import requests
import os
os.chdir("D:\\OneDrive - Indian Institute of Science\\Projects\\NishkulanandAI")

# Function to save the HTML content of a page to a file
def save_html_content(url, output_dir):
    # Send a GET request to the URL
    response = requests.get(url)
    
    # If the request is successful
    if response.status_code == 200:
        # Get the page name based on prakaran and lang parameters for naming
        prakaran = url.split("prakaran=")[-1]  # Extract prakaran value from URL
        page_name = f"bhaktachintamani_prakaran_{prakaran}"
        
        # Define the path for the file
        file_path = os.path.join(output_dir, f"{page_name}.html")
        
        # Save the HTML content to the file
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(response.text)
        
        print(f"Saved HTML content of {url} to {file_path}")
    else:
        print(f"Failed to retrieve the page: {url}")

# Main function to scrape all Bhaktachintamani pages
def crawl_and_save_bhaktachintamani(base_url, output_dir, start_prakaran, end_prakaran):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Crawl each page with prakaran values in the given range
    for prakaran in range(start_prakaran, end_prakaran + 1):
        # Construct the URL for each page
        url = f"{base_url}?lang=gu&prakaran={prakaran}"
        
        # Save the HTML content of the page
        save_html_content(url, output_dir)

# Base URL for Bhaktachintamani pages
base_url = "https://www.anirdesh.com/chintamani/index.php"

# Directory to save the HTML files
output_dir = "bhaktachintamani_html_files"

# Define the range of prakaran values to scrape
start_prakaran = 1  # Starting prakaran
end_prakaran = 164  # Ending prakaran

# Start the scraping process
crawl_and_save_bhaktachintamani(base_url, output_dir, start_prakaran, end_prakaran)

print(f"All Bhaktachintamani pages have been saved in the '{output_dir}' directory.")
