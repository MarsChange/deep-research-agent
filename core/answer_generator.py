# core/answer_generator.py
import agent_logging


class AnswerGenerator:
    def __init__(self, client, model, task_log):
        self.client = client
        self.model = model
        self.task_log = task_log

    async def generate_failure_summary(self, task_description, message_history):
        self.task_log.log_step("warning", "Main Agent", "正在生成失败复盘总结...")

        # 构建反思提示词
        prompt = f"""
        任务未能在规定步骤内完成。原始任务：{task_description}
        请根据对话历史总结：
        1. 已经确认的关键事实。
        2. 尝试过但确认无效的路径。
        3. 导致目前卡壳的核心原因。
        结果将用于下次重试，请保持简洁。
        """

        history = message_history + [{"role": "user", "content": prompt}]
        try:
            response = await self.client.chat.completions.create(model=self.model, messages=history)
            summary = response.choices[0].message.content or "未能生成总结"
            self.task_log.log_step("info", "Failure Summary", f"总结生成完毕: {summary[:100]}...")
            return summary
        except Exception as e:
            return f"总结生成出错: {str(e)}"