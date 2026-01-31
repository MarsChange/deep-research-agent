---
name: scrape-website
description: Scrape and extract readable content from webpages. Use when you need to read the full content of a webpage URL, convert HTML to clean markdown text, or extract information from web articles.
---

# Scrape Website

## Goal

Extract readable content from webpages and convert them to clean markdown text. This skill is useful when:

- You have a URL and need to read its full content
- You need to extract article text, tables, or structured data from a webpage
- The browser tools are unavailable or you need a lightweight alternative
- You want to convert HTML content to clean, readable markdown

## Prerequisites

This skill requires the following environment variables:

- `JINA_API_KEY`: Jina Reader API key (required)
- `JINA_READER_URL`: Jina Reader URL (optional, defaults to `https://r.jina.ai`)

## Script

Execute the scrape script with the target URL:

```shell
python scripts/scrape_website.py --url "https://example.com/article"
```

### Options

- `--url URL`: The URL of the website to scrape (required)

### Examples

Scrape a news article:

```shell
python scripts/scrape_website.py --url "https://www.bbc.com/news/technology-12345678"
```

Scrape documentation:

```shell
python scripts/scrape_website.py --url "https://docs.python.org/3/library/asyncio.html"
```

Scrape a Wikipedia page:

```shell
python scripts/scrape_website.py --url "https://en.wikipedia.org/wiki/Artificial_intelligence"
```

## How It Works

1. **Primary Method (Jina Reader)**: Uses Jina AI's Reader API to render JavaScript and extract clean content
   - Handles dynamic content (JavaScript-rendered pages)
   - Extracts images with generated alt text
   - Processes iframes and shadow DOM

2. **Fallback Method (Requests + MarkItDown)**: If Jina fails, falls back to direct HTTP request
   - Uses MarkItDown to convert HTML to markdown
   - Lighter weight but may miss dynamic content

3. **Auto-retry**: Automatically retries up to 3 times on failure

## Output Format

The script returns the webpage content as clean markdown text, including:

- Article text and paragraphs
- Headers and structure
- Links (preserved as markdown links)
- Tables (converted to markdown tables)
- Lists and formatting

## Limitations

- Does not support file downloads (PDF, Excel, etc.) - use sandbox tools for files
- May not work well with heavily JavaScript-dependent single-page applications
- Some websites may block scraping attempts
- Hugging Face datasets/spaces URLs are not supported

## Notes

- For file processing (PDF, Excel, CSV), use the E2B sandbox tools instead
- If scraping fails, consider using browser tools (`browser_navigate` + `browser_snapshot`) as an alternative
- The script automatically adds `https://` prefix if the URL doesn't have a protocol
