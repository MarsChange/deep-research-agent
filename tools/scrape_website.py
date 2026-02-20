import io
import agent_logging
import os
import random
import time
from typing import Optional

import requests
from markitdown import MarkItDown
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

logger = agent_logging.getLogger(__name__)

# 1. 引入 User-Agent 轮换池，降低被封禁风险
_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Mobile/15E148 Safari/604.1"
]

# Patterns that indicate Jina returned a blocked/invalid page
_JINA_BLOCKED_PATTERNS = [
    "Target URL returned error 403",
    "requiring CAPTCHA",
    "Just a moment...",
    "Verify you are human",
    "Access denied",
    "Enable JavaScript and cookies to continue",
]


def _is_blocked_content(content: str) -> bool:
    if len(content.strip()) < 200:
        return True
    return any(pattern in content for pattern in _JINA_BLOCKED_PATTERNS)


def _get_retry_session():
    """创建一个带有自动重试逻辑的请求会话"""
    session = requests.Session()
    retry = Retry(
        total=3,  # 最大重试次数
        backoff_factor=1,  # 指数退避间隔
        status_forcelist=[500, 502, 503, 504],  # 针对这些状态码进行重试
        raise_on_status=False
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _scrape_by_jina(url: str) -> tuple[Optional[str], Optional[str]]:
    jina_api_key = os.getenv("JINA_API_KEY", "")
    if not jina_api_key:
        return None, "JINA_API_KEY is not set"

    jina_reader_url = os.getenv("JINA_READER_URL", "https://r.jina.ai")
    jina_url = f"{jina_reader_url}/{url}"

    # 保持 Jina 推荐的高级渲染配置
    jina_headers = {
        "Authorization": f"Bearer {jina_api_key}",
        "X-Base": "final",
        "X-Engine": "browser",
        "X-With-Generated-Alt": "true",
        "X-With-Iframe": "true",
        "X-With-Shadow-Dom": "true",
    }

    try:
        logger.info(f"Jina scraping: {url}")
        session = _get_retry_session()
        response = session.get(jina_url, headers=jina_headers, timeout=45)

        if response.status_code == 422:
            return None, "Jina 422: URL may point to a file, not supported"

        response.raise_for_status()
        content = response.text

        if "Warning: This page maybe not yet fully loaded" in content:
            logger.info(f"Jina: page not fully loaded, retrying with longer timeout: {url}")
            time.sleep(2)  # 稍微等待
            response = session.get(jina_url, headers=jina_headers, timeout=60)
            response.raise_for_status()
            content = response.text

        if _is_blocked_content(content):
            return None, f"Jina returned blocked/CAPTCHA page for {url}"

        return content, None

    except Exception as e:
        return None, f"Jina scraping failed: {str(e)}"


def _scrape_request(url: str) -> tuple[Optional[str], Optional[str]]:
    """增强版回退抓取：使用随机 UA 和重试 Session"""
    try:
        headers = {
            "User-Agent": random.choice(_USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }

        logger.info(f"Requests fallback scraping: {url}")
        session = _get_retry_session()
        response = session.get(url, headers=headers, timeout=60, allow_redirects=True)

        # 专门处理 403 错误，引导 Agent 使用浏览器工具
        if response.status_code == 403:
            return None, "Error 403: Access Forbidden. This site might require browser_navigate to bypass anti-bot."

        response.raise_for_status()

        try:
            stream = io.BytesIO(response.content)
            md = MarkItDown()
            content = md.convert_stream(stream).text_content
            if content and len(content.strip()) > 50:
                return content, None
        except Exception as e:
            logger.warning(f"MarkItDown conversion failed: {e}")

        if response.text and len(response.text.strip()) > 50:
            return response.text, None

        return None, "Page content is empty or too short"

    except Exception as e:
        return None, str(e)


def scrape_website(url: str) -> str:
    if not url:
        return "Error: URL is empty."

    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"

    # 屏蔽已知无法抓取的域名
    if any(domain in url for domain in ["huggingface.co/datasets", "huggingface.co/spaces"]):
        return "Error: Cannot scrape this specific platform. Use other tools."

    # 优先尝试 Jina
    content, jina_error = _scrape_by_jina(url)
    if content is not None:
        return content

    logger.warning(f"Jina failed ({jina_error}), falling back to requests: {url}")

    # 回退到增强版 requests
    content, req_error = _scrape_request(url)
    if content is not None:
        return content

    return f"Error: All scraping methods failed.\nJina: {jina_error}\nRequests: {req_error}"


SCRAPE_WEBSITE_TOOLS = []
if os.getenv("JINA_API_KEY"):
    SCRAPE_WEBSITE_TOOLS = [scrape_website]