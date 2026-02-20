# core/tool_executor.py

import asyncio
import json
import logging
from inspect import iscoroutinefunction
from typing import List, Dict, Any


class ToolExecutor:
    def __init__(self, task_log):
        self.task_log = task_log
        self.used_queries = {}  # 用于存储已执行的查询 key: 1

    # 确保方法签名包含 agent_name 和 turn_count
    async def execute_tool_batch(
            self,
            tool_calls_data: List[dict],
            tool_functions_map: dict,
            agent_name: str = "Agent",
            turn_count: int = 0
    ) -> List[dict]:
        """
        并行化执行工具调用，增加了日志追踪与重复检查逻辑。
        """
        tasks = []
        call_ids = []

        for tc_data in tool_calls_data:
            call_id = tc_data["id"]
            func_name = tc_data["function"]["name"]

            try:
                # 解析参数
                args_str = tc_data["function"]["arguments"]
                args = json.loads(args_str)

                # 1. 参数修正逻辑
                args = self.fix_tool_call_arguments(func_name, args)

                # 2. 重复调用检查
                query_key = f"{func_name}:{json.dumps(args, sort_keys=True)}"
                if query_key in self.used_queries:
                    self.task_log.log_step("warning", f"{agent_name} | Turn {turn_count}", f"拦截重复调用: {func_name}")
                    tasks.append(asyncio.sleep(0, result=f"Error: 已经执行过相同的 {func_name}，请更换参数。"))
                elif func_name not in tool_functions_map:
                    tasks.append(asyncio.sleep(0, result=f"Error: 工具 {func_name} 未定义"))
                else:
                    self.used_queries[query_key] = True
                    func = tool_functions_map[func_name]

                    # 3. 记录带编号的结构化日志
                    self.task_log.log_step("info", f"{agent_name} | Turn {turn_count}", f"🔧 执行工具: {func_name}")

                    # 4. 封装异步/同步任务
                    if iscoroutinefunction(func):
                        tasks.append(func(**args))
                    else:
                        tasks.append(asyncio.to_thread(func, **args))

                call_ids.append(call_id)
            except Exception as e:
                self.task_log.log_step("error", f"{agent_name} | Executor", f"解析失败: {e}")
                # 即使解析失败也要占位，保证 call_ids 对应
                tasks.append(asyncio.sleep(0, result=f"Error: 无法解析工具参数 - {str(e)}"))
                call_ids.append(call_id)

        # 并行执行所有工具
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 格式化结果
        formatted_results = []
        for call_id, res in zip(call_ids, results):
            content = str(res) if not isinstance(res, Exception) else f"Error: {res}"
            formatted_results.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": content
            })

        return formatted_results

    def fix_tool_call_arguments(self, tool_name: str, arguments: dict) -> dict:
        """自动纠正参数名错误"""
        fixed_args = arguments.copy()
        if tool_name in ["scrape_website", "analyze_webpage"]:
            for mistake in ["description", "introduction", "content"]:
                if mistake in fixed_args and "info_to_extract" not in fixed_args:
                    fixed_args["info_to_extract"] = fixed_args.pop(mistake)
        return fixed_args