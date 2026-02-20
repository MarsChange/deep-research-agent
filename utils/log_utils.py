# utils/log_utils.py
import os
import json
from pathlib import Path
from typing import Optional


def get_latest_failure_experience(log_dir: str = "logs") -> Optional[str]:
    """
    扫描日志目录，获取最近一次任务生成的失败总结。
    """
    try:
        log_path = Path(log_dir)
        if not log_path.exists():
            return None

        # 按修改时间倒序排列所有 JSON 日志
        log_files = sorted(log_path.glob("*.json"), key=os.path.getmtime, reverse=True)

        for log_file in log_files:
            with open(log_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                # 检查日志中是否存有 AnswerGenerator 生成的总结
                # 通常存放在 'error' 字段或特定的 'failure_summary' 字段
                summary = data.get("error")
                if summary and len(summary) > 50:  # 确保是有意义的总结
                    return summary
    except Exception as e:
        print(f"读取历史经验失败: {e}")
    return None