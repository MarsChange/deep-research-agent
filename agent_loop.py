import asyncio
import inspect
import json
import logging
import os
import re
from dataclasses import dataclass
from inspect import iscoroutinefunction
import uuid
from agent_logging.task_logger import TaskLog
from core.answer_generator import AnswerGenerator
from typing import (
    Any,
    AsyncIterator,
    Callable,
    List,
    Literal,
    Optional,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from core.tool_executor import ToolExecutor

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionChunk
from tools_calling import (
    build_main_agent_system_prompt,
    build_sub_agent_system_prompt,
    generate_summarize_prompt,
)

# Import sub-agent tools; fallback to empty if unavailable
try:
    from tools import SUB_AGENT_TOOLS
except ImportError:
    SUB_AGENT_TOOLS = []

logger = logging.getLogger(__name__)

# --- Constants ---

MAX_MAIN_AGENT_TURNS = 30
MAX_SUB_AGENT_TURNS = 25

QUESTION_ANALYSIS_PROMPT = """Carefully analyze the given task description (question) without attempting to solve it directly. Your role is to identify potential challenges and areas that require special attention during the solving process, and provide practical guidance for someone who will solve this task by actively gathering and analyzing information from the web.

Identify and concisely list key points in the question that could potentially impact subsequent information collection or the accuracy and completeness of the problem solution, especially those likely to cause mistakes, carelessness, or confusion during problem-solving.

The question author does not intend to set traps or intentionally create confusion. Interpret the question in the most common, reasonable, and straightforward manner, without speculating about hidden meanings or unlikely scenarios. However, be aware that mistakes, imprecise wording, or inconsistencies may exist due to carelessness or limited subject expertise, rather than intentional ambiguity.

Additionally, when considering potential answers or interpretations, note that question authors typically favor more common and familiar expressions over overly technical, formal, or obscure terminology. They generally prefer straightforward and common-sense interpretations rather than being excessively cautious or academically rigorous in their wording choices.

Also, consider additional flagging issues such as:
- Potential mistakes or oversights introduced unintentionally by the question author due to his misunderstanding, carelessness, or lack of attention to detail.
- Terms or instructions that might have multiple valid interpretations due to ambiguity, imprecision, outdated terminology, or subtle wording nuances.
- Numeric precision, rounding requirements, formatting, or units that might be unclear, erroneous, or inconsistent with standard practices or provided examples.
- Contradictions or inconsistencies between explicit textual instructions and examples or contextual clues provided within the question itself.

Avoid overanalyzing or listing trivial details that would not materially affect the task outcome.

## Knowledge-Based Entity Hypothesis Generation (CRITICAL)

For each descriptive element, indirect reference, or oblique characterization in the question, you MUST leverage your internal knowledge to generate the most likely candidate entities, **especially targeting "long-tail" or niche entities that might not be immediately obvious**. This step transforms vague multi-hop questions into concrete, targeted research tasks.

For each descriptive element in the question:
1. **Extract the descriptive clue** from the question text (e.g., "A specific essay from 1834").
2. **Propose the most likely candidate entity** based on your broad knowledge. **Prioritize specific, less-common titles or names** that match the temporal and contextual markers.
3. **Multi-hop Linkage Analysis**: Identify the specific "bridge" connecting this entity to the next part of the question. (e.g., "The essay is mentioned in the 'Who Was?' series").
4. **Explain the match** — briefly state why this candidate fits the description.
5. **Assign confidence** — high / medium / low.
6. **List alternatives** — if multiple candidates are plausible, list the top 2-3 with reasoning for each.

**Chain derivation**: Once you identify a high-confidence candidate for one element, immediately use it as a known condition to derive candidates for subsequent elements. Build a complete candidate reasoning chain from the question's starting point to its final target.

After generating all hypotheses, clearly categorize:
- **High-confidence hypotheses** — can be treated as near-facts, only need quick verification.
- **Medium/low-confidence hypotheses** — require dedicated search effort to confirm or eliminate.
- **Unknown elements** — no strong candidate available, require open-ended search from scratch.

## Reasoning Chain Decomposition (CRITICAL)

If the question involves multi-hop reasoning (requiring multiple information retrieval steps to reach an answer):

1. **Identify the reasoning chain** — Decompose the question into independent sub-questions, each involving exactly ONE specific information retrieval task.
2. **Handle Multi-hop Dependencies**: Explicitly state which sub-question results are needed to unlock the search query for the next node.
3. **Pre-judge each node using internal knowledge** — For each sub-question, provide the most likely candidate answer based on your existing knowledge, with confidence level.
4. **Provide a search plan for each sub-question**:
   - Recommended search keywords (3-7 keywords, short and precise). **Include the specific "long-tail" entity names discovered in the Hypothesis phase**.
   - Alternative keywords (to use if the first search fails).
5. **Mark dependencies between sub-questions** — Which sub-questions can be searched independently (parallelizable), and which require prior sub-questions to be resolved first (sequential).
6. **Specify the key fact each sub-question must confirm** — What information must be established at each step before proceeding to the next.
7. **Identify skippable nodes** — If a node's hypothesis has high confidence, recommend the main agent adopt it directly rather than spending time on verification.

Note: Search keywords for each sub-question must be independent — NEVER mix keywords from multiple sub-questions into a single query.

## 中文分析指导

如果问题涉及中文语境，请特别注意：

- **语言理解**：识别可能存在的中文表达歧义、方言差异或特定语境下的含义。
- **文化背景**：考虑可能需要中文文化背景知识才能正确理解的术语或概念。
- **信息获取**：标注需要使用中文搜索关键词才能获得准确信息的方面。
- **格式要求**：识别中文特有的格式要求、表达习惯或答案形式。
- **翻译风险**：标记直接翻译可能导致误解或信息丢失的关键术语。对于长尾实体，优先使用其中文标准译名进行搜索。
- **分析输出**：使用中文进行分析和提示，确保语言一致性。


## Final Output Format Prediction
Predict if the final answer should be a single number, a specific date, a noun, or a list. 
Explicitly flag this for the Main Agent to ensure formatting compliance.


Here is the question:

"""

DEFAULT_SYSTEM_PROMPT = """You are a highly precise research assistant. Your goal is to gather information from the internet and provide direct, concise answers to user questions.

Follow these instructions strictly:
1. **Research Thoroughly**: Use available tools to collect comprehensive information before concluding.
2. **Be Transparent**: Document all evidence, reasoning steps, and uncertainties internally.
3. **Format Requirement (CRITICAL)**: 
   - **DO NOT** wrap your response in a JSON object or any code blocks (e.g., do not use {"answer": "..."}).
   - **Direct Output Only**: Provide only the raw answer string.
   - For numerical questions, output only the integer.
   - For multiple entities, separate them with a comma and a space (", ").
4. **Standardization**:
   - Convert all English letters to lowercase.
   - Remove any leading or trailing whitespace.
5. **No Explanations**: Provide just the answer itself without conversational filler or introductory phrases.

Example:
User: "What is the capital of France?"
Assistant: paris
"""


# --- Data Classes ---


@dataclass
class ToolCall:
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None
    tool_arguments: Optional[dict] = None


@dataclass
class Chunk:
    step_index: int
    type: Literal["text", "tool_call", "tool_call_result"]
    content: Optional[str] = None
    tool_call: Optional[ToolCall] = None
    tool_result: Optional[Any] = None


def python_type_to_json_type(t):
    """Map Python types to JSON types."""
    if t is str:
        return "string"
    elif t is int:
        return "integer"
    elif t is float:
        return "number"
    elif t is bool:
        return "boolean"
    elif t is list or get_origin(t) is list:
        return "array"
    elif t is dict or get_origin(t) is dict:
        return "object"
    return "string"


def parse_docstring(docstring: str) -> dict:
    """
    Parse a docstring to extract description and parameter descriptions.

    Supports Google-style docstrings:
        Args:
            param_name: Description of the parameter
            param_name (type): Description of the parameter

    Returns:
        dict with 'description' and 'params' keys
    """
    if not docstring:
        return {"description": "", "params": {}}

    lines = docstring.strip().split("\n")
    description_lines = []
    params = {}
    current_section = "description"
    current_param = None
    current_param_desc = []

    for line in lines:
        stripped = line.strip()

        # Check for section headers
        if stripped.lower() in ("args:", "arguments:", "parameters:", "params:"):
            current_section = "args"
            continue
        elif stripped.lower() in (
                "returns:", "return:", "yields:", "raises:", "examples:", "example:", "note:", "notes:"):
            # Save current param if any
            if current_param and current_param_desc:
                params[current_param] = " ".join(current_param_desc).strip()
            current_section = "other"
            continue

        if current_section == "description":
            description_lines.append(stripped)
        elif current_section == "args":
            # Check if this is a new parameter definition
            # Patterns: "param_name: description" or "param_name (type): description"
            param_match = re.match(r"^(\w+)(?:\s*\([^)]*\))?\s*:\s*(.*)$", stripped)
            if param_match:
                # Save previous param
                if current_param and current_param_desc:
                    params[current_param] = " ".join(current_param_desc).strip()
                current_param = param_match.group(1)
                current_param_desc = [param_match.group(2)] if param_match.group(2) else []
            elif current_param and stripped:
                # Continuation of current param description
                current_param_desc.append(stripped)

    # Save last param
    if current_param and current_param_desc:
        params[current_param] = " ".join(current_param_desc).strip()

    # Clean up description - remove empty lines at end
    while description_lines and not description_lines[-1]:
        description_lines.pop()

    return {
        "description": " ".join(description_lines).strip(),
        "params": params,
    }


def function_to_schema(func: Callable) -> dict:
    """
    Convert a Python function to an OpenAI API Tool Schema.

    Extracts function description and parameter descriptions from docstring.
    """
    type_hints = get_type_hints(func)
    signature = inspect.signature(func)

    # Parse docstring for descriptions
    docstring_info = parse_docstring(func.__doc__ or "")

    parameters = {"type": "object", "properties": {}, "required": []}

    for name, param in signature.parameters.items():
        if name in ("self", "cls"):
            continue

        annotation = type_hints.get(name, str)
        param_type = python_type_to_json_type(annotation)

        param_info = {"type": param_type}

        # Add parameter description from docstring if available
        if name in docstring_info["params"]:
            param_info["description"] = docstring_info["params"][name]

        if get_origin(annotation) == Literal:
            param_info["enum"] = list(get_args(annotation))
            param_info["type"] = python_type_to_json_type(type(get_args(annotation)[0]))

        parameters["properties"][name] = param_info
        if param.default == inspect.Parameter.empty:
            parameters["required"].append(name)

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": docstring_info["description"],
            "parameters": parameters,
        },
    }


# --- CJK Detection ---
def _contains_cjk(text: str) -> bool:
    """Check if text contains CJK (Chinese/Japanese/Korean) characters."""
    for char in text:
        if '\u4e00' <= char <= '\u9fff' or '\u3400' <= char <= '\u4dbf':
            return True
    return False


async def run_sub_agent(
        client: AsyncOpenAI,
        model: str,
        subtask: str,
        sub_agent_tool_functions: list,
        chinese_context: bool = False,
) -> str:
    """Run the sub-agent worker to complete a research subtask.

    Non-streaming, bounded turns.

    Args:
        client: OpenAI-compatible async client
        model: Model name
        subtask: The research subtask description
        sub_agent_tool_functions: Tool functions available to the sub-agent
        chinese_context: Whether CJK context is detected

    Returns:
        Summary report string from the sub-agent
    """
    # Build sub-agent system prompt
    system_prompt = build_sub_agent_system_prompt(
        sub_agent_tool_functions, chinese_context
    )

    # Build tool schema and map
    tool_schema = [function_to_schema(f) for f in sub_agent_tool_functions]
    tool_functions_map = {f.__name__: f for f in sub_agent_tool_functions}

    # Initialize messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": subtask},
    ]

    logger.info(f"[Sub-Agent] Starting subtask: {subtask[:200]}...")

    task_failed = False
    turn = 0

    for turn in range(MAX_SUB_AGENT_TURNS):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tool_schema if tool_schema else None,
            )
        except Exception as e:
            logger.error(f"[Sub-Agent] LLM call failed at turn {turn}: {e}")
            break

        choice = response.choices[0]
        assistant_message = choice.message

        # If no tool calls, this is the final response
        if not assistant_message.tool_calls:
            content = assistant_message.content or ""
            logger.info(f"[Sub-Agent] Completed at turn {turn} with {len(content)} chars")
            messages.append({"role": "assistant", "content": content})
            break

        # Build assistant message for history
        tool_calls_data = []
        for tc in assistant_message.tool_calls:
            args_str = tc.function.arguments
            # Ensure arguments is valid JSON for DashScope API compatibility
            try:
                json.loads(args_str)
            except (json.JSONDecodeError, TypeError):
                args_str = json.dumps({})
            tool_calls_data.append({
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": args_str,
                },
            })
        assistant_msg = {"role": "assistant", "tool_calls": tool_calls_data}
        if assistant_message.content:
            assistant_msg["content"] = assistant_message.content
        messages.append(assistant_msg)

        # Execute tool calls
        for tc in assistant_message.tool_calls:
            func_name = tc.function.name
            try:
                parsed_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError as e:
                tool_result = f"Error: Failed to parse arguments: {e}"
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": tool_result}
                )
                continue

            # Execute the tool (sync tools run in thread to avoid blocking event loop)
            try:
                if func_name in tool_functions_map:
                    func = tool_functions_map[func_name]
                    if iscoroutinefunction(func):
                        result = await func(**parsed_args)
                    else:
                        result = await asyncio.to_thread(func, **parsed_args)
                    tool_result = str(result)
                else:
                    tool_result = f"Error: Tool '{func_name}' not found."
            except Exception as e:
                tool_result = f"Error: Execution failed - {str(e)}"

            logger.info(
                f"[Sub-Agent] Turn {turn}: {func_name} -> {len(tool_result)} chars"
            )
            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": tool_result}
            )
    else:
        # Loop completed without break = max turns exhausted
        task_failed = True
        logger.warning(
            f"[Sub-Agent] Reached max turns ({MAX_SUB_AGENT_TURNS})"
        )

    # Generate summary via summarize prompt
    summarize = generate_summarize_prompt(
        task_description=subtask,
        task_failed=task_failed,
        is_main_agent=False,
        chinese_context=chinese_context,
    )
    messages.append({"role": "user", "content": summarize})

    try:
        summary_response = await client.chat.completions.create(
            model=model,
            messages=messages,
            # No tools parameter — force text-only response
        )
        summary = summary_response.choices[0].message.content or ""
    except Exception as e:
        logger.error(f"[Sub-Agent] Summary generation failed: {e}")
        # Fall back to last assistant message
        summary = "Error generating summary. "
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                summary += msg["content"]
                break

    logger.info(f"[Sub-Agent] Summary: {len(summary)} chars")
    return summary


# --- Main Agent Loop ---


async def agent_loop(
        input_messages: list,
        tool_functions: List[Callable],
        skill_directories: Optional[List[str]] = None,
) -> AsyncIterator[Chunk]:
    """
    Main agent loop with multi-agent architecture.

    The main agent decomposes tasks and delegates research to sub-agent workers
    via execute_subtask. The sub-agent has direct access to search, scrape, and
    analyze tools.

    Args:
        input_messages: List of chat messages
        tool_functions: List of tool functions for the MAIN agent (e.g., sandbox tools)
        skill_directories: Deprecated, ignored. Kept for API compatibility.
    """

    task_id = str(uuid.uuid4())[:8]
    task_log = TaskLog(task_id=task_id)
    executor = ToolExecutor(task_log)
    task_log.log_step("info", "Main Agent", f"--- 任务开始: {task_id} ---")
    user_question = next((msg["content"] for msg in reversed(input_messages) if msg["role"] == "user"), "")
    task_log.log_step("info", "Main Agent", f"任务描述预览: {user_question[:100]}...")
    assert os.getenv("DASHSCOPE_API_KEY"), "DASHSCOPE_API_KEY is not set"

    model = os.getenv("QWEN_MODEL") or "qwen-max"

    client = AsyncOpenAI(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
    )

    # --- Extract user question ---
    user_question = ""
    for msg in reversed(input_messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                user_question = content
            break

    task_log.log_step("info", "Main Agent", f"收到任务描述: {user_question[:100]}...")
    # --- Detect Chinese context ---
    chinese_context = _contains_cjk(user_question)

    # Yield an initial chunk immediately so the SSE connection has data
    # and the client won't idle-timeout during Phase 0 pre-analysis
    yield Chunk(type="text", content="", step_index=0)

    # --- Phase 0: Question Pre-Analysis (optional, controlled by ENABLE_QUESTION_ANALYSIS) ---
    question_analysis = ""
    enable_analysis = os.getenv("ENABLE_QUESTION_ANALYSIS", "true").lower() in ("true", "1", "yes")
    if enable_analysis and user_question:
        try:
            analysis_task = asyncio.create_task(
                client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": f"{QUESTION_ANALYSIS_PROMPT}{user_question}",
                        }
                    ],
                )
            )
            # Wait with periodic keepalive to prevent SSE timeout
            pending = {analysis_task}
            while pending:
                _, pending = await asyncio.wait(pending, timeout=30)
                if pending:
                    yield Chunk(type="text", content="", step_index=0)

            analysis_response = analysis_task.result()
            question_analysis = analysis_response.choices[0].message.content or ""
            task_log.log_step("info", "Main Agent | Phase 0", "预分析完成。")
        except Exception as e:
            logger.warning(f"Phase 0 question pre-analysis failed: {e}, skipping")
            task_log.log_step("warning", "Main Agent | Phase 0", f"预分析失败: {e}")

    # --- Sub-agent tool functions come from SUB_AGENT_TOOLS (imported from tools/) ---
    sub_agent_tool_functions = list(SUB_AGENT_TOOLS)
    max_parallel = int(os.getenv("SUB_AGENT_NUM", "10"))

    # --- Create execute_subtasks closure (parallel sub-agent dispatch) ---
    async def execute_subtasks(subtasks_json: str) -> str:
        """
        Delegate one or more research subtasks to worker agents, executed in parallel. Each worker has independent access to web search, webpage analysis, Wikipedia, website scraping, and browser tools. Workers run concurrently and return structured research reports.

        Args:
            subtasks_json: A JSON array of subtask description strings. Each element is a self-contained research question that includes ALL relevant context (workers have no shared memory). For a single subtask, use a one-element array.
        """
        try:
            questions = json.loads(subtasks_json)
            if isinstance(questions, str):
                questions = [questions]
        except json.JSONDecodeError:
            # Fallback: treat as a single question
            questions = [subtasks_json]

        if not questions:
            return "Error: No subtasks provided."

        # Cap parallelism
        questions = questions[:max_parallel]

        logger.info(
            f"[Main Agent] Dispatching {len(questions)} subtask(s) in parallel"
        )
        task_log.log_step(
            "info",
            "Main Agent | Dispatch",
            f"正在并行分发 {len(questions)} 个研究子任务"
        )

        async def run_wrapped_sub_agent(idx, question):
            sub_task_id = f"Sub-{idx + 1}"
            task_log.log_step("info", f"Worker | {sub_task_id}", f"开始处理: {question[:50]}...")
            try:
                # 执行原有的 run_sub_agent
                result = await run_sub_agent(
                    client=client,
                    model=model,
                    subtask=question,
                    sub_agent_tool_functions=sub_agent_tool_functions,
                    chinese_context=chinese_context,
                )
                task_log.log_step("info", f"Worker | {sub_task_id}", "任务完成")
                return result
            except Exception as e:
                task_log.log_step("error", f"Worker | {sub_task_id}", f"执行失败: {str(e)}")
                raise e

        # Run all sub-agents concurrently
        tasks = [run_wrapped_sub_agent(i, q) for i, q in enumerate(questions)]

        # 按照 MiroThinker 风格，使用 return_exceptions=True 保证部分失败不影响整体
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Format combined results
        output_parts = []
        for i, (q, r) in enumerate(zip(questions, results)):
            if isinstance(r, Exception):
                # 记录异常详情
                error_msg = f"Error - {type(r).__name__}: {str(r)}"
                output_parts.append(f"## Subtask {i + 1}\n**Question**: {q}\n**Result**: {error_msg}")
            else:
                # 成功结果
                output_parts.append(f"## Subtask {i + 1}\n**Question**: {q}\n**Result**:\n{r}")

        # 最终保存一次日志状态
        task_log.save()

        return "\n\n---\n\n".join(output_parts)

    # --- Main agent tools: [execute_subtasks] + passed-in tool_functions ---
    main_agent_tools = [execute_subtasks] + list(tool_functions or [])

    # --- Build main agent system prompt ---
    system_prompt = build_main_agent_system_prompt(
        main_agent_tools, chinese_context, max_parallel=max_parallel
    )
    # Append the default system prompt (JSON answer format requirement)
    system_prompt = f"{system_prompt}\n\n{DEFAULT_SYSTEM_PROMPT}"

    # --- Prepare messages ---
    prompt_messages = input_messages.copy()
    if prompt_messages and prompt_messages[0].get("role") == "system":
        original_content = prompt_messages[0].get("content", "")
        prompt_messages[0] = {
            "role": "system",
            "content": f"{original_content}\n\n{system_prompt}",
        }
    else:
        prompt_messages.insert(
            0,
            {"role": "system", "content": system_prompt},
        )

    # Inject question pre-analysis as context
    if question_analysis:
        prompt_messages.append(
            {
                "role": "user",
                "content": (
                    f"<question_analysis>\n{question_analysis}\n</question_analysis>\n\n"
                    "Based on the above analysis, now solve the original question. "
                    "Start by outlining your decomposition plan, then delegate subtasks step by step."
                ),
            }
        )

    # --- Build tool schema and function map ---
    tool_schema = [function_to_schema(f) for f in main_agent_tools]
    tool_functions_map = {f.__name__: f for f in main_agent_tools}

    params = {
        "model": model,
        "stream": True,
        "tools": tool_schema,
    }

    step_index = 0
    consecutive_rollbacks = 0  # 连续回退计数器
    MAX_CONSECUTIVE_ROLLBACKS = 5  # 最大允许回退次数，防止死循环
    # --- Main Agent Loop (bounded) ---
    for turn in range(MAX_MAIN_AGENT_TURNS):
        # Make the streaming request
        task_log.log_step("info", f"Main Agent | Turn: {turn + 1}", "正在请求 LLM 推理...")
        stream = await client.chat.completions.create(
            messages=prompt_messages,
            **params,
            # extra_body={"enable_thinking": True},
        )

        tool_calls_buffer = {}
        full_assistant_text = ""
        # Process the stream
        async for chunk in stream:  # type: ChatCompletionChunk
            chunk = cast(ChatCompletionChunk, chunk)

            delta = chunk.choices[0].delta

            # Case A: Standard text content
            if delta.content:
                full_assistant_text += delta.content
                yield Chunk(
                    type="text", content=delta.content, step_index=step_index
                )

            # Case B: Tool call fragments (accumulate them)
            if delta.tool_calls:
                for tc_chunk in delta.tool_calls:
                    idx = tc_chunk.index
                    if idx not in tool_calls_buffer:
                        tool_calls_buffer[idx] = {
                            "id": tc_chunk.id,
                            "function": {
                                "name": tc_chunk.function.name,
                                "arguments": "",
                            },
                        }
                    # Append tool arguments fragment
                    if tc_chunk.function.arguments:
                        tool_calls_buffer[idx]["function"]["arguments"] += (
                            tc_chunk.function.arguments
                        )

        if full_assistant_text:
            task_log.log_step("debug", "Main Agent | LLM Response", f"文本回复内容预览: {full_assistant_text[:100]}...")
        invalid_tags = ["<mcp_call>", "</mcp_call>", "<mcp_result>"]
        is_format_error = any(tag in full_assistant_text for tag in invalid_tags)
        is_refusal = any(kw in full_assistant_text for kw in ["无法回答", "对不起", "I cannot", "sorry"])

        if is_format_error or is_refusal:
            if consecutive_rollbacks < MAX_CONSECUTIVE_ROLLBACKS:
                consecutive_rollbacks += 1
                task_log.log_step("warning", "Rollback", f"检测到异常输出，执行第 {consecutive_rollbacks} 次回退重试")

                continue
            else:
                task_log.log_step("error", "Main Agent", "连续回退次数过多，强制终止当前分支")
                break

        # 3. 校验通过，重置计数器并处理后续逻辑
        consecutive_rollbacks = 0
        # If no tool calls, the model returned a final text response
        if not tool_calls_buffer:
            task_log.log_step("info", "Main Agent", "模型给出了最终回复，循环结束。")
            break

        assistant_tool_calls_data = []
        sorted_indices = sorted(tool_calls_buffer.keys())

        for idx in sorted_indices:
            raw_tool = tool_calls_buffer[idx]
            assistant_tool_calls_data.append(
                {
                    "id": raw_tool["id"],
                    "type": "function",
                    "function": {
                        "name": raw_tool["function"]["name"],
                        "arguments": raw_tool["function"]["arguments"],
                    },
                }
            )

        # Append the assistant's tool call request to history
        prompt_messages.append(
            {
                "role": "assistant",
                "tool_calls": assistant_tool_calls_data,
            }
        )

        # # Execute tools and yield results — parallel for async tools
        # # Phase 1: Parse all tool calls and yield tool_call notifications
        # parsed_tool_calls = []  # (call_id, func_name, parsed_args, tool_call, error_msg)
        # for tool_data in assistant_tool_calls_data:
        #     call_id = tool_data["id"]
        #     func_name = tool_data["function"]["name"]
        #     func_args_str = tool_data["function"]["arguments"]
        #     task_log.log_step("info", "Main Agent | Tool Call Start",
        #                       f"准备执行工具: {func_name} | 参数: {func_args_str}")
        #     tool_call = ToolCall(
        #         tool_call_id=call_id,
        #         tool_name=func_name,
        #         tool_arguments={},
        #     )
        #
        #     try:
        #         parsed_args = json.loads(func_args_str)
        #         tool_call.tool_arguments = parsed_args
        #         yield Chunk(
        #             step_index=step_index,
        #             type="tool_call",
        #             tool_call=tool_call,
        #         )
        #         parsed_tool_calls.append(
        #             (call_id, func_name, parsed_args, tool_call, None)
        #         )
        #     except json.JSONDecodeError as e:
        #         error_msg = f"Error: Failed to parse tool arguments JSON: {func_args_str}. Error: {e}"
        #         yield Chunk(
        #             step_index=step_index,
        #             type="tool_call",
        #             tool_call=tool_call,
        #         )
        #         parsed_tool_calls.append(
        #             (call_id, func_name, {}, tool_call, error_msg)
        #         )
        #
        # # Phase 2: Launch all tools — async ones run concurrently
        # async_tasks = {}  # call_id -> asyncio.Task
        # sync_results = {}  # call_id -> result string
        #
        # for call_id, func_name, parsed_args, tool_call, error_msg in parsed_tool_calls:
        #     if error_msg:
        #         sync_results[call_id] = error_msg
        #         continue
        #
        #     if func_name not in tool_functions_map:
        #         sync_results[call_id] = f"Error: Tool '{func_name}' not found."
        #         continue
        #
        #     func = tool_functions_map[func_name]
        #     task_log.log_step("debug", f"Main Agent | Tool Executing", f"正在调用 API/沙箱: {func_name}")
        #     if iscoroutinefunction(func):
        #         async_tasks[call_id] = asyncio.create_task(func(**parsed_args))
        #     else:
        #         # Run sync tools in thread to avoid blocking event loop
        #         async_tasks[call_id] = asyncio.create_task(
        #             asyncio.to_thread(func, **parsed_args)
        #         )
        #
        # # Wait for all async tasks with periodic keepalive
        # if async_tasks:
        #     pending = set(async_tasks.values())
        #     while pending:
        #         _, pending = await asyncio.wait(pending, timeout=30)
        #         if pending:
        #             yield Chunk(
        #                 type="text", content="", step_index=step_index
        #             )
        #
        #     # Collect async results
        #     for call_id, task in async_tasks.items():
        #         try:
        #             sync_results[call_id] = str(task.result())
        #         except Exception as e:
        #             sync_results[call_id] = f"Error: Execution failed - {str(e)}"
        #
        # # Phase 3: Yield all results and update message history
        # for call_id, func_name, parsed_args, tool_call, error_msg in parsed_tool_calls:
        #     tool_result_content = sync_results[call_id]
        #     task_log.log_step("info", "Main Agent | Tool Call Success",
        #                       f"工具 {func_name} 返回长度: {len(tool_result_content)}")
        #     yield Chunk(
        #         type="tool_call_result",
        #         tool_result=tool_result_content,
        #         step_index=step_index,
        #         tool_call=tool_call,
        #     )
        #
        #     prompt_messages.append(
        #         {
        #             "role": "tool",
        #             "tool_call_id": call_id,
        #             "content": tool_result_content,
        #         }
        #     )
        tool_results = await executor.execute_tool_batch(
            tool_calls_data=assistant_tool_calls_data,
            tool_functions_map=tool_functions_map,
            agent_name="Main Agent",
            turn_count=turn + 1
        )

        # 4. 更新消息历史并持久化
        # tool_results 已经是格式化好的 [{"role": "tool", ...}, ...] 列表
        for result_msg in tool_results:
            prompt_messages.append(result_msg)

            # 为了兼容原有的 Chunk 输出，可以从 result_msg 中提取数据 yield
            yield Chunk(
                type="tool_call_result",
                tool_result=result_msg["content"],
                step_index=step_index,
                # tool_call 信息可从 assistant_tool_calls_data 匹配
            )
        step_index += 1
        task_log.save()
    else:
        # Reached max turns — inject summarize prompt for final answer
        logger.warning(
            f"Main agent reached max turns ({MAX_MAIN_AGENT_TURNS}), generating summary"
        )
        task_log.log_step("warning", "Main Agent", f"已达到最大轮次 ({MAX_MAIN_AGENT_TURNS})，开始生成失败分析报告")
        ans_gen = AnswerGenerator(client, model, task_log)
        ans_gen = AnswerGenerator(client, model, task_log)
        # 生成结构化复盘内容
        summary = await ans_gen.generate_failure_summary(
            task_description=user_question,
            message_history=prompt_messages
        )

        # 将总结存入 TaskLog 的 error 字段并持久化
        task_log.error = summary
        task_log.save()

        yield Chunk(
            type="text",
            content=f"\n\n### 💡 任务失败经验复盘\n{summary}",
            step_index=step_index
        )
    task_log.log_step("info", "Main Agent", "任务流程全部结束。")
    task_log.save()


