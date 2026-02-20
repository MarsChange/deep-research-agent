# agent_logging/task_logger.py
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
from colorama import Fore, Style, init

init(autoreset=True)


def get_utc_plus_8_time() -> str:
    """获取东八区时间"""
    utc_plus_8 = timezone(timedelta(hours=8))
    return datetime.now(utc_plus_8).strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class StepLog:
    """记录单个步骤的详细信息"""
    step_name: str
    message: str
    timestamp: str
    info_level: str = "info"
    metadata: dict = field(default_factory=dict)


@dataclass  # <--- 核心修复：必须将主类也标记为 dataclass
class TaskLog:
    """主任务日志类，负责内存记录与持久化"""
    task_id: str
    log_dir: str = "logs"
    # 使用 field(default_factory=...) 确保每个实例都有独立的时间和列表
    start_time: str = field(default_factory=get_utc_plus_8_time)
    step_logs: List[StepLog] = field(default_factory=list)
    error: Optional[str] = None  # 用于存储 AnswerGenerator 生成的失败总结

    def log_step(self, info_level: str, step_name: str, message: str, metadata: dict = None):
        """核心函数：根据步骤名自动匹配图标并打印彩色日志"""
        icons = {
            "LLM": "🧠 ", "Tool": "🔧 ", "Main Agent": "👑 ",
            "Error": "❌ ", "Success": "✅ ", "Sub-Agent": "🤖 ",
            "Worker": "📝 ", "Rollback": "🔄 "
        }
        # 匹配图标
        icon = next((v for k, v in icons.items() if k in step_name), "📝 ")
        step_name_with_icon = f"{icon}{step_name}"

        log_entry = StepLog(
            step_name=step_name_with_icon,
            message=message,
            timestamp=get_utc_plus_8_time(),
            info_level=info_level,
            metadata=metadata or {}
        )
        self.step_logs.append(log_entry)

        # 终端彩色输出
        color_map = {
            "info": Fore.GREEN,
            "warning": Fore.YELLOW,
            "error": Fore.RED,
            "debug": Fore.CYAN
        }
        color = color_map.get(info_level.lower(), Fore.WHITE)

        print(
            f"[{log_entry.timestamp}][{color}{info_level.upper()}{Style.RESET_ALL}] - {step_name_with_icon}: {message}")

    def save(self):
        """保存为 JSON 文件"""
        os.makedirs(self.log_dir, exist_ok=True)
        filename = os.path.join(self.log_dir, f"task_{self.task_id}.json")
        with open(filename, "w", encoding="utf-8") as f:
            # 现在 self 是 dataclass 实例，asdict 可以正常工作了
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)