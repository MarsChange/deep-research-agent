import re


class OutputFormatter:
    def _extract_boxed_content(self, text: str) -> str:
        """支持嵌套大括号的 \boxed{} 提取逻辑"""
        if not text: return ""
        # 寻找最后一个 \boxed
        matches = list(re.finditer(r"\\boxed\s*\{", text))
        if not matches: return ""

        start_idx = matches[-1].end() - 1
        content = ""
        count = 0
        for i in range(start_idx, len(text)):
            if text[i] == "{":
                count += 1
            elif text[i] == "}":
                count -= 1

            if count == 0:
                content = text[start_idx + 1: i]
                break
        return content.strip()