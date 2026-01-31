
import io
import os
import argparse
from typing import Optional
import json
import requests
from markitdown import MarkItDown

def request_to_json(content: str) -> dict:
    if isinstance(content, str) and "Markdown Content:\n" in content:
        # If the content starts with "Markdown Content:\n", extract only the part after it (from JINA)
        content = content.split("Markdown Content:\n")[1]
    return json.loads(content)

def scrape_by_jina(url: str) -> tuple[Optional[str], Optional[str]]:
    """Scrape website content using Jina API.

    Args:
        url: The URL of the website to scrape.

    Returns:
        tuple: A tuple of (content, error_message). Content is None if there's an error.
    """
    jina_api_key = os.getenv("JINA_API_KEY", "")
    if not jina_api_key:
        return (
            None,
            "JINA_API_KEY is not set, JINA scraping is not available.",
        )
    jina_reader_url = os.getenv("JINA_READER_URL", "https://r.jina.ai")
    if not jina_reader_url:
        return (
            None,
            "JINA_READER_URL is not set, JINA scraping is not available.",
        )
    jina_url = f"{jina_reader_url}/{url}"
    jina_headers = {
        "Authorization": f"Bearer {jina_api_key}",
        "X-Base": "final",
        "X-Engine": "browser",
        "X-With-Generated-Alt": "true",
        "X-With-Iframe": "true",
        "X-With-Shadow-Dom": "true",
    }
    try:
        response = requests.get(jina_url, headers=jina_headers, timeout=120)
        if response.status_code == 422:
            # Return as error to allow fallback to other tools and retries
            return (
                None,
                "Tool execution failed with Jina 422 error, which may indicate the URL is a file. This tool does not support files. If you believe the URL might point to a file, you should try using other applicable tools, or try to process it in the sandbox.",
            )
        response.raise_for_status()
        content = response.text
        if (
            "Warning: This page maybe not yet fully loaded, consider explicitly specify a timeout."
            in content
        ):
            # Try with longer timeout
            response = requests.get(jina_url, headers=jina_headers, timeout=300)
            if response.status_code == 422:
                return (
                    None,
                    "Tool execution failed with Jina 422 error, which may indicate the URL is a file. This tool does not support files. If you believe the URL might point to a file, you should try using other applicable tools, or try to process it in the sandbox.",
                )
            response.raise_for_status()
            content = response.text
        return content, None
    except Exception as e:
        return None, f"Failed to get content from Jina.ai: {str(e)}\n"
    
def scrape_request(url: str) -> tuple[Optional[str], Optional[str]]:
    """This function uses requests to scrape a website.
    Args:
        url: The URL of the website to scrape.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        try:
            stream = io.BytesIO(response.content)
            md = MarkItDown()
            content = md.convert_stream(stream).text_content
            return content, None
        except Exception:
            # If MarkItDown conversion fails, return raw response text
            return response.text, None

    except Exception as e:
        return None, f"{str(e)}"
    
def scrape_website(url: str) -> tuple[Optional[str], Optional[str]]:
    """Scrape website content using Jina API with fallback to requests.

    Args:
        url: The URL of the website to scrape.

    Returns:
        tuple: A tuple of (content, error_message). Content is None if there's an error.
    """
    if not url:
        return None, "[ERROR] URL is empty. Please provide a valid URL."
    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"
    
    if "huggingface.co/datasets" in url or "huggingface.co/spaces" in url:
        return None, "[WARNING] You are trying to scrape a Hugging Face dataset for answers, please do not use the scrape tool for this purpose."
    
    retry_count = 0
    max_retries = 3

    while retry_count < max_retries:
        try:
            content, error = scrape_by_jina(url)
            if content is not None:
                return content, None
            # Fallback to requests scraping
            content, error_requests = scrape_request(url)
            if content is not None:
                return content, None
            # If both methods failed, return combined error messages
            combined_error = f"Jina scraping error: {error}\nRequests scraping error: {error_requests}"
            return None, combined_error
        except Exception as e:
            retry_count += 1
            if retry_count == max_retries:
                return None, f"[Error] Failed to get content from Jina.ai: {str(e)}\n"
    
    return None, "[Error] Exceeded maximum retries to scrape the website."

def main():
    parser = argparse.ArgumentParser(
        description="Scrape website content using Jina API with fallback to requests"
    )
    parser.add_argument(
        "--url",
        type=str,
        help="The URL of the website to scrape",
    )
    args = parser.parse_args()
    url = args.url

    content, error = scrape_website(url)
    if error:
        print(f"Error: {error}")
    else:
        print(f"Scraped Content:\n{content}")


if __name__ == "__main__":
    main()