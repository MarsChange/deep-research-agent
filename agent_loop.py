import inspect
import json
import os
import re
from dataclasses import dataclass
from inspect import iscoroutinefunction
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

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionChunk
from skills import (
    SkillIntegrationTools,
    SkillMetadata,
    build_unified_tools_prompt,
    discover_skills,
)

DEFAULT_SYSTEM_PROMPT = """You are an AI assistant designed to help with a variety of tasks. You have access to several tools that can assist you in providing accurate and relevant information.

Your task is to actively gather detailed information from the internet and generate answers to users' questions. Your goal is not to rush to a definitive answer or conclusion, but to collect complete information and present all reasonable candidate answers you find, accompanied by clearly documented supporting evidence, reasoning steps, uncertainty factors, and explicit intermediate findings.

The user has no intention of deliberately setting traps or creating confusion. Please use the most common, reasonable, and direct explanation to handle the task, and do not overthink or focus on rare or far-fetched explanations.

Important Note: - Gather comprehensive information from reliable sources to fully understand all aspects of the issue.
- Present all possible candidate answers you identified during the information gathering process, regardless of uncertainty, ambiguity, or incomplete verification. Avoid jumping to conclusions or omitting any discovered possibilities.
- Clearly record the detailed facts, evidence, and reasoning steps supporting each candidate answer, and carefully preserve the intermediate analysis results.
- During the information collection process, clearly mark and retain all uncertainties, conflicting interpretations, or different understandings that are discovered. Do not arbitrarily discard or resolve these issues on your own.
- In cases where there is inconsistency, ambiguity, errors, or potential mismatches with general guidelines or provided examples in the explicit instructions of a problem (such as numerical accuracy, formatting, specific requirements), all reasonable explanations and corresponding candidate answers should be clearly documented and presented.   

Recognize that the original task description itself may inadvertently contain errors, imprecision, inaccuracy, or conflicts due to user carelessness, misunderstanding, or limited professional knowledge. Do not attempt to internally question or "correct" these instructions; instead, present the survey results transparently based on every reasonable interpretation.

Your goal is to achieve the highest degree of completeness, transparency, and detailed documentation, enabling users to make independent judgments and choose their preferred answers. Even in the presence of uncertainty, explicitly recording the existence of possible answers can significantly enhance the user experience, ensuring that no reasonable solutions are irreversibly omitted due to early misunderstandings or premature filtering.

When generating responses, it is crucial to pay attention to the following points:
1. Keep your response as concise as possible. The response content should be returned as a JSON dictionary, with the answer to the question corresponding to the key "answer". For example, if the user inputs {"question": "Where is the capital of France?"}, you only need to respond with {"answer": "Paris"}
2. To minimize misjudgments caused by format differences, the following preprocessing is applied to the output responses:
- Convert English letters to lowercase;
- Remove leading and trailing spaces;
- All numerical questions involve integers;
- If the answer contains multiple entities, please follow English grammar, with a comma or semicolon followed by a space. The specific symbol should be based on the user's question.
"""


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
        elif stripped.lower() in ("returns:", "return:", "yields:", "raises:", "examples:", "example:", "note:", "notes:"):
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


async def agent_loop(
    input_messages: list,
    tool_functions: List[Callable],
    skill_directories: Optional[List[str]] = ["skills"],
) -> AsyncIterator[Chunk]:
    """
    Main agent loop with skills support.

    Args:
        input_messages: List of chat messages
        tool_functions: List of tool functions available to the agent
        skill_directories: Optional list of directories to scan for skills.
                          Skills are folders containing SKILL.md files.
    """

    assert os.getenv("DASHSCOPE_API_KEY"), "DASHSCOPE_API_KEY is not set"

    client = AsyncOpenAI(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
    )

    # Discover and load skills metadata
    skills: List[SkillMetadata] = (
        discover_skills(skill_directories) if skill_directories else []
    )

    # Build unified tools system prompt (tool functions + skills)
    tools_prompt = build_unified_tools_prompt(tool_functions or [], skills)

    # Prepare messages with tools context injected
    prompt_messages = input_messages.copy()
    if tools_prompt and prompt_messages:
        # Inject tools prompt into system message if present, otherwise prepend
        if prompt_messages[0].get("role") == "system":
            original_content = prompt_messages[0].get("content", "")
            prompt_messages[0] = {
                "role": "system",
                "content": f"{original_content}\n\n{tools_prompt}",
            }
        else:
            prompt_messages.insert(
                0,
                {
                    "role": "system",
                    "content": f"{DEFAULT_SYSTEM_PROMPT}\n\n{tools_prompt}",
                },
            )

    # Build a mapping from function name to function for quick lookup

    llm_tools = (tool_functions or []).copy()

    if skills:
        skill_tools = SkillIntegrationTools(skills)
        llm_tools.extend([skill_tools.load_skill_file, skill_tools.execute_script])

    step_index = 0
    tool_schema = [function_to_schema(tool_function) for tool_function in llm_tools]

    tool_functions_map = {func.__name__: func for func in llm_tools}

    params = {
        "model": os.getenv("QWEN_MODEL"),
        "stream": True,
        "tools": tool_schema,
    }

    # Main Agent Loop: continues as long as the model requests tool executions
    while True:
        # Make the streaming request
        stream = await client.chat.completions.create(
            messages=prompt_messages, **params,
            extra_body={"enable_thinking":True}
        )

        tool_calls_buffer = {}

        # Process the stream
        async for chunk in stream:  # type: ChatCompletionChunk
            chunk = cast(ChatCompletionChunk, chunk)

            delta = chunk.choices[0].delta

            # Case A: Standard text content
            if delta.content:
                yield Chunk(type="text", content=delta.content, step_index=step_index)

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
        if not tool_calls_buffer:
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

        # Execute tools and yield results
        for tool_data in assistant_tool_calls_data:
            call_id = tool_data["id"]
            func_name = tool_data["function"]["name"]
            func_args_str = tool_data["function"]["arguments"]

            tool_result_content = ""
            parsed_args = {}
            tool_call = ToolCall(
                tool_call_id=call_id,
                tool_name=func_name,
                tool_arguments={},
            )

            try:
                # Parse JSON arguments
                parsed_args = json.loads(func_args_str)
                tool_call.tool_arguments = parsed_args

                # Notify caller that we are about to execute a tool
                yield Chunk(
                    step_index=step_index,
                    type="tool_call",
                    tool_call=tool_call,
                )

                # Execute the function if it exists
                if func_name in tool_functions_map:
                    func = tool_functions_map[func_name]
                    # Note: If tools are async, use await
                    if iscoroutinefunction(func):
                        result = await func(**parsed_args)
                    else:
                        result = func(**parsed_args)
                    tool_result_content = str(result)
                else:
                    tool_result_content = f"Error: Tool '{func_name}' not found."

            except json.JSONDecodeError as e:
                tool_result_content = f"Error: Failed to parse tool arguments JSON: {func_args_str}. Error: {e}"
                yield Chunk(
                    step_index=step_index,
                    type="tool_call",
                    tool_call=tool_call,
                )
            except Exception as e:
                tool_result_content = f"Error: Execution failed - {str(e)}"

            yield Chunk(
                type="tool_call_result",
                tool_result=tool_result_content,
                step_index=step_index,
                tool_call=tool_call,
            )

            prompt_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": tool_result_content,
                }
            )
        step_index += 1
