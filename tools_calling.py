"""
Prompt builders and tool calling utilities for the agent.

Provides system prompt generation for main agent and sub-agent, tool function
descriptions, and summarize prompt generation.
"""

import datetime
import logging
from typing import List

logger = logging.getLogger(__name__)


def build_tool_functions_prompt(tool_functions: list) -> str:
    """
    Build system prompt section describing tool functions.

    Args:
        tool_functions: List of tool function objects

    Returns:
        System prompt text describing tool functions by category
    """
    if not tool_functions:
        return ""

    # Group tools by category based on function name
    categories = {}
    for func in tool_functions:
        name = func.__name__
        if name.startswith("browser_"):
            category = "browser"
        elif name == "analyze_webpage":
            category = "webpage_analyzer"
        elif name == "search_engine":
            category = "search"
        elif name.startswith("search_wikipedia") or name == "list_wikipedia_revisions":
            category = "wiki"
        elif name == "scrape_website":
            category = "scrape"
        else:
            category = "general"

        if category not in categories:
            categories[category] = []
        categories[category].append(name)

    lines = ["# Available Tools"]
    lines.append("")

    if "search" in categories:
        lines.append("**Google/Bing Search** (`search_engine`):")
        lines.append("- Search the web using Google or Bing. Returns titles, URLs, snippets, answer boxes, and knowledge graphs.")
        lines.append("- Use Chinese keywords for Chinese questions, English for English questions.")
        lines.append("- Craft SHORT queries: 3-7 discriminative keywords targeting ONE specific aspect.")
        lines.append("")

    if "wiki" in categories:
        lines.append("**Wikipedia Tools** (`search_wikipedia`, `search_wikipedia_revision`, `list_wikipedia_revisions`):")
        lines.append("- `search_wikipedia(entity)` — Get current Wikipedia page content")
        lines.append("- `search_wikipedia_revision(entity, date)` — Get historical page content at a specific date")
        lines.append("- `list_wikipedia_revisions(entity)` — List available revisions (use FIRST when exploring history)")
        lines.append("")

    if "scrape" in categories:
        lines.append("**Website Scraper** (`scrape_website`):")
        lines.append("- Extract readable content from any webpage, converts HTML to markdown")
        lines.append("- Use as alternative to analyze_webpage for full page content")
        lines.append("")

    if "webpage_analyzer" in categories:
        lines.append("**Webpage Analyzer** (`analyze_webpage`):")
        lines.append("- `analyze_webpage(url, question)` — Fetches a webpage and uses AI to extract information relevant to your question")
        lines.append("- Only use when search snippets are insufficient to answer the question — if snippets already clearly answer it, skip webpage analysis")
        lines.append("- Select the most promising 2-3 URLs from search results based on title and snippet relevance")
        lines.append("")

    if "browser" in categories:
        lines.append("**Browser Tools** (persistent browser session):")
        lines.append("- For web page interaction: navigation, clicking, typing, screenshots")
        lines.append("- Browser state persists across calls (cookies, sessions maintained)")
        lines.append("- Use `browser_snapshot` to get page structure and element references")
        lines.append(f"- Available: {', '.join(categories['browser'])}")
        lines.append("")

    if "general" in categories:
        lines.append("**Other Tools:**")
        lines.append(f"- {', '.join(categories['general'])}")
        lines.append("")

    return "\n".join(lines)


def build_main_agent_system_prompt(
    tool_functions: list,
    chinese_context: bool = False,
    max_parallel: int = 10,
) -> str:
    import datetime
    formatted_date = datetime.datetime.today().strftime("%Y-%m-%d")

    prompt = f"""You are a high-level Research Coordinator. Today is: {formatted_date}

# Core Objective
Your goal is to provide precise, fact-based answers (usually a single noun or number). You orchestrate parallel workers to gather raw data and MANDATORY use a Python sandbox for any data synthesis or logical filtering.

## Operational Workflow

1. **Reasoning Ledger (STATE MANAGEMENT)**:
   Before each tool call, you must maintain an internal ledger:
   - **CONFIRMED**: Facts already established (e.g., a specific person's name, a date).
   - **NEXT STEP**: The immediate specific data point or calculation needed.
   - **REMAINING**: Pending nodes in the reasoning chain.

2. **Parallel Research**: Use `execute_subtasks` to dispatch up to {max_parallel} queries. Focus on gathering RAW information (e.g., "list of all members", "all edit timestamps") rather than pre-processed summaries.

3. **Mandatory Sandbox Processing (CRITICAL)**:
   - **Logic & Math**: For any task involving **counting**, **sorting**, **ranking**, **filtering** (e.g., "who is the youngest"), or **decryption**, you are FORBIDDEN from reasoning mentally.
   - **Workflow**: Collect raw snippets/data into a list -> Call `run_python_code` -> Write a script to find the answer -> Extract the result from `stdout`[cite: 10].
   - **Example**: To find a co-founder from 10 search results, create a Python list of biographies and filter by "founded" keyword and date.

4. **Answer Extraction**: Output the result derived from the sandbox.

## Forward Progression Rule
Once a fact is CONFIRMED in your ledger, it is an immutable constant. Never re-search it; use it as a bridge to the next node immediately.

## Output Format Requirement
- Final answer must be a JSON dictionary: {{"answer": "result"}}.
- The result should be ONLY the noun or number, wrapped in \\boxed{{}} for extraction.
"""

    if chinese_context:
        prompt += """
## 中文处理策略
1. **术语精准**：确保子任务使用标准中文译名或专有名词。
2. **文本分析**：对于涉及中文文本的逻辑排除、成员比对或字数统计，必须搜集原文后利用 Python 代码进行字符串匹配和处理。
"""
    return prompt


def build_sub_agent_system_prompt(
    tool_functions: list,
    chinese_context: bool = False,
) -> str:
    """
    Build the system prompt for the sub-agent worker (research execution).

    The sub-agent has direct access to search, scrape, analyze tools and executes
    specific research subtasks.

    Args:
        tool_functions: List of tool function objects available to the sub-agent
        chinese_context: Whether the question involves CJK content

    Returns:
        System prompt text for the sub-agent worker
    """
    formatted_date = datetime.datetime.today().strftime("%Y-%m-%d")
    tool_prompt = build_tool_functions_prompt(tool_functions)

    prompt = f"""You are a research worker agent that executes specific research subtasks. Today is: {formatted_date}

# Agent Specific Objective

You complete well-defined, single-scope research objectives efficiently and accurately.
Do not infer, speculate, or attempt to fill in missing parts yourself. Only return factual content.

Critically assess the reliability of all information:
- If the credibility of a source is uncertain, clearly flag it.
- Do NOT treat information as trustworthy just because it appears — cross-check when necessary.
- If you find conflicting or ambiguous information, include all relevant findings and flag the inconsistency.

Be cautious and transparent in your output:
- Always return all related information. If information is incomplete or weakly supported, still share partial excerpts, and flag any uncertainty.
- Never assume or guess — if an exact answer cannot be found, say so clearly.
- Prefer quoting or excerpting original source text rather than interpreting or rewriting it, and provide the URL if available.

{tool_prompt}

# Research Strategy

## Early Answer Rule (IMPORTANT)
If the search result snippets (titles, descriptions, answer boxes, knowledge graphs) already clearly and unambiguously answer your question, report the answer immediately WITHOUT analyzing individual webpages. Only proceed to webpage analysis when snippets are insufficient or ambiguous.

## Phase 1: Search
- Use `search_engine` to find relevant sources
- Craft SHORT queries (3-7 keywords) targeting the specific subtask
- Each new query MUST be substantially different from all previous queries — never repeat or accumulate keywords
- For each aspect, perform at most 2-3 searches before moving to Phase 2
- **If snippets already answer the question clearly, skip to Phase 4 (Report)**

## Phase 2: Analyze Pages (when snippets are insufficient)
After getting search results, use `analyze_webpage`, `scrape_website`, or browser tools to read the most relevant URLs:
- Select the top 2-3 most promising URLs from search results (based on title and snippet relevance)
- Call `analyze_webpage(url, question)` for each URL
- Review each analysis result before deciding next steps

## Phase 3: Cross-Validation
- Only needed when findings conflict or are ambiguous
- Compare findings across sources to resolve contradictions

## Phase 4: Report
- Synthesize findings with supporting evidence
- Present all candidate answers with confidence levels
- Document conflicting information or uncertainties

## Tool-Use Guidelines

1. **Each step must involve exactly ONE tool call only.**
2. Craft precise search queries: 3-7 discriminative keywords, targeting ONE specific aspect.
3. **Query construction rule**: Search queries must be concise keyword phrases. Do NOT dump reasoning context into the query string.
4. **Tool diversity requirement**: If you have used the same tool type 3 times in a row, you MUST switch to a different tool (e.g., from search to analyze_webpage, or vice versa).
5. **For historical or time-specific content**: Use `search_wikipedia_revision` or `list_wikipedia_revisions` for Wikipedia history.
6. Even if a tool result does not directly answer the question, thoroughly extract all partial information that may help guide future steps.
7. After issuing ONE tool call, STOP immediately. Wait for the result.
"""

    if chinese_context:
        prompt += """
## 中文内容处理

处理中文相关的子任务时：
- **搜索关键词**：使用中文关键词进行搜索，获取更准确的中文资源
- **思考过程**：分析、推理、判断等内部思考过程应使用中文表达
- **信息摘录**：保持中文原文的准确性，避免不必要的翻译或改写
- **各种输出**：包括状态说明、过程描述、结果展示等所有输出都应使用中文
- **回应格式**：对中文子任务的回应应使用中文，保持语境一致性

"""

    return prompt


# tools_calling.py

def generate_summarize_prompt(
    task_description: str,
    task_failed: bool = False,
    is_main_agent: bool = True,
    chinese_context: bool = False,
) -> str:
    """
    针对 Main Agent 强化了 Few-Shot 约束和格式严整性，严禁任何解释。
    """
    prompt = "This is a direct instruction to the assistant. STOP all tool use immediately. Report the results NOW.\n\n"

    if is_main_agent:
        prompt += (
            f"Original Question: {task_description}\n\n"
            "Based on all gathered research, synthesize the FINAL ANSWER according to these STRICT RULES:\n"
            "1. Output ONLY the answer wrapped in \\boxed{}.\n"
            "2. NO introductory text, NO conversational filler, NO bold text, and NO explanations outside the box.\n"
            "3. For people, provide the full name. For dates, use YYYY-MM-DD. For lists, use comma-separated values.\n"
            "4. If the answer is a specific entity name, ensure it is the complete, official name.\n\n"
            "### FEW-SHOT EXAMPLES:\n"
            "User Question: What is the volume number of the journal mentioned?\n"
            "Assistant: \\boxed{3}\n\n"
            "User Question: 十余年后，他创立的这家出版公司的名字是什么？\n"
            "Assistant: \\boxed{阿诺尔多·蒙达多利出版社}\n\n"
            "User Question: Who are the co-founders of the collective established in the early 1990s?\n"
            "Assistant: \\boxed{Harald Hauswald}\n\n"
            "User Question: What is the name of the significant military operation?\n"
            "Assistant: \\boxed{Operation Desert Shield}\n\n"
            "### FINAL TASK:\n"
            "Now, provide the final answer for the original question. Remember: ONLY the \\boxed{} content is allowed."
        )
    else:
        # 子代理（Sub-Agent）需要详尽的报告，以便主代理分析
        prompt += (
            "Provide a comprehensive, structured research report covering:\n"
            "- All confirmed key facts and raw data found.\n"
            "- Direct quotes, numbers, and dates from source materials.\n"
            "- URLs for every piece of information.\n"
            "- A clear statement of any conflicting data or missing details."
        )

    if chinese_context:
        prompt += "\n\n注意：对于中文语境的问题，最终答案必须使用中文（除非答案是数字或固有的英文专有名词）。"

    return prompt

