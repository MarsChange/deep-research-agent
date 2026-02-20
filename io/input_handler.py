# io/input_handler.py
import os
import base64
import json
import pdfminer.high_level
import mammoth
import openpyxl
from typing import Tuple
from openai import OpenAI
from markitdown import MarkItDown  # 作为通用兜底处理


def _extract_task_relevant_info_from_image(image_path: str, task_description: str) -> str:
    """参考 MiroThinker：结合任务描述，针对性提取图像关键信息"""
    client = OpenAI()  # 自动读取环境变量 OPENAI_API_KEY
    try:
        with open(image_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode("utf-8")

        prompt = f"""分析此图片并提取与以下任务直接相关的细节：
        任务：{task_description}
        请输出简洁的事实摘要。"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[图像解析失败: {e}]"


def process_input(task_description: str, task_file_name: str = None) -> Tuple[str, str]:
    """规范化输入：将附件内容转为 Markdown 拼接到 Prompt 末尾"""
    file_content_section = ""

    if task_file_name and os.path.exists(task_file_name):
        ext = task_file_name.split(".")[-1].lower()

        # 图像处理
        if ext in {"jpg", "jpeg", "png", "gif", "webp"}:
            relevant_info = _extract_task_relevant_info_from_image(task_file_name, task_description)
            file_content_section = f"\n\n## 附件图像分析\n{relevant_info}"

        # PDF 处理
        elif ext == "pdf":
            text = pdfminer.high_level.extract_text(task_file_name)
            file_content_section = f"\n\n## PDF 文本内容\n```text\n{text[:30000]}\n```"

        # Excel 处理 (转为 Markdown 表格)
        elif ext in ["xlsx", "xls"]:
            wb = openpyxl.load_workbook(task_file_name, data_only=True)
            sheet = wb.active
            rows = list(sheet.rows)
            # 简单转换前 15 行
            md_table = "| " + " | ".join([str(c.value or "") for c in rows[0]]) + " |\n| " + "--- |" * len(
                rows[0]) + "\n"
            for row in rows[1:15]:
                md_table += "| " + " | ".join([str(c.value or "") for c in row]) + " |\n"
            file_content_section = f"\n\n## Excel 表格预览\n{md_table}"

        # 其他文本/代码文件
        elif ext in ["py", "txt", "md", "json"]:
            with open(task_file_name, "r", encoding="utf-8") as f:
                content = f.read()
                file_content_section = f"\n\n## 文件内容 ({task_file_name})\n```\n{content[:20000]}\n```"

    # 组合最终 Prompt，并强制要求 \boxed{} 格式
    full_prompt = f"{task_description}\n{file_content_section}\n\n请将最终答案包裹在 \\boxed{{}} 中。"
    return full_prompt, full_prompt