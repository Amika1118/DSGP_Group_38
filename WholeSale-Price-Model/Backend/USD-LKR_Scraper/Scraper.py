import requests
from bs4 import BeautifulSoup
import json
import re
import os

def scrape_next_week_forecast(url):
    """
    Scrapes the forecast paragraph under the H3 heading 'USD to LKR Forecast for Next Week'
    and extracts both the full text and the numeric rate after 'Rs'.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Find the H3 heading
        target_h3 = soup.find('h3', string=lambda text: text and "USD to LKR Forecast for Next Week" in text)
        if not target_h3:
            return {"error": "Could not find the H3 heading 'USD to LKR Forecast for Next Week'."}

        # Find the following paragraph
        target_paragraph = target_h3.find_next_sibling('p')
        if not target_paragraph:
            return {"error": "Found the H3 heading but no following paragraph was found."}

        # Extract full text
        forecast_text = target_paragraph.get_text(strip=True)
        forecast_text = ' '.join(forecast_text.split())  # Clean whitespace

        # Extract the numeric rate after "Rs"
        # Pattern: Rs followed by optional space/non-breaking space, then a number (with optional decimal)
        rate_match = re.search(r'Rs[\s\u00A0]*(\d+\.?\d*)', forecast_text)
        if rate_match:
            predicted_rate = float(rate_match.group(1))
        else:
            predicted_rate = None

        return {
            "forecast_text": forecast_text,
            "predicted_rate": predicted_rate
        }

    except requests.exceptions.RequestException as e:
        return {"error": f"Error fetching the webpage: {e}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {e}"}

def save_to_json(data, filepath):
    """
    Saves the provided data to a JSON file at the specified path.
    Creates the directory if it does not exist.
    """
    try:
        # Ensure the directory exists
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"Data successfully saved to {filepath}")
    except IOError as e:
        print(f"Error saving to JSON file: {e}")

if __name__ == "__main__":
    target_url = "https://coincodex.com/forex/usd-lkr/forecast/"
    result = scrape_next_week_forecast(target_url)

    print("Scraped Result:")
    print(json.dumps(result, indent=4, ensure_ascii=False))

    if "error" not in result:
        # Save to the specified directory with the filename
        output_path = "../USD-LKR_Scraper/usd_lkr_forecast.json"
        save_to_json(result, output_path)
    else:
        print("Scraping failed. Check the error above.")